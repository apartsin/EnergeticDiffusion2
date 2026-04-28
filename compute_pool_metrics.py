"""Compute per-pool MOSES-style metrics for Fig 13.

For each .txt pool of generated SMILES, sample N=5000 RDKit-canonical SMILES
(matching FCD's protocol) and compute:
  - validity      : RDKit-parseable / total in raw .txt
  - uniqueness    : distinct canonical / parseable
  - novelty       : fraction of rows whose max-Tanimoto to labelled-master is < 0.7
                    (Morgan-FP r=2, 2048 bits)
  - scaffold_nn   : mean of per-row max-Tanimoto on Bemis-Murcko scaffold FPs
  - chem_pass     : fraction of rows with no chem_redflags reject (optional 7th panel)

Aggregate per condition (mean +/- std across seeds). Writes results/pool_metrics.json.

Mirrors compute_fcd.py / novelty_filter.py I/O conventions.
"""
from __future__ import annotations
import json, glob, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

# Make the chem filter importable
sys.path.insert(0, "m6_postprocess_bundle")
try:
    import chem_redflags as crf
    HAVE_CHEM = True
except Exception as e:
    print(f"[pool] chem_redflags import failed ({e}); chem_pass panel will be skipped")
    HAVE_CHEM = False

OUT = Path("results"); OUT.mkdir(exist_ok=True, parents=True)
SAMPLE_N = 5000
NOVEL_THRESHOLD = 0.7        # frac with max-tani < this is "novel"
random.seed(0)

print("[pool] loading reference (labelled-master)")
lm = pd.read_csv("m6_postprocess_bundle/labelled_master.csv",
                  usecols=["smiles"], low_memory=False)
ref_smis = lm["smiles"].dropna().tolist()
print(f"[pool] labelled-master rows: {len(ref_smis)}")


def morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def scaffold_fp(mol):
    try:
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        if sc is None or sc.GetNumAtoms() == 0:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(sc, 2, 1024)
    except Exception:
        return None


print("[pool] building reference fingerprint banks")
t0 = time.time()
ref_fps_morgan = []
ref_fps_scaffold = []
for s in ref_smis:
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    ref_fps_morgan.append(morgan_fp(m))
    sf = scaffold_fp(m)
    if sf is not None:
        ref_fps_scaffold.append(sf)
print(f"[pool] morgan ref: {len(ref_fps_morgan)}, scaffold ref: {len(ref_fps_scaffold)}, "
      f"build {time.time()-t0:.1f}s")


def chem_pass(smi: str) -> bool:
    """True if SMILES survives the chem_redflags screen (no 'reject' alerts)."""
    if not HAVE_CHEM: return True
    try:
        r = crf.screen(smi)
        # screen() returns a dict; reject if any alert has severity 'reject'
        # or disallowed atoms present. Conservative interpretation:
        if r.get("disallowed_atoms"): return False
        for alert in r.get("alerts", []):
            if isinstance(alert, (list, tuple)) and len(alert) >= 2 and alert[1] == "reject":
                return False
        return True
    except Exception:
        return False


def process_pool(path: str) -> dict:
    raw = Path(path).read_text().splitlines()
    raw = [s.strip() for s in raw if s.strip()]
    n_raw = len(raw)
    canon = []
    for s in raw:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        canon.append((s, m, Chem.MolToSmiles(m)))
    n_valid = len(canon)
    n_unique = len({c for _, _, c in canon})
    if n_valid == 0:
        return {"n_raw": n_raw, "n_valid": 0, "validity": 0, "uniqueness": 0,
                "novelty": None, "scaffold_nn": None, "chem_pass": None}

    # Sample for the heavy Tanimoto loop
    sample = random.sample(canon, k=min(SAMPLE_N, len(canon)))
    novelty_count = 0
    scaffold_max_taus = []
    chem_pass_count = 0
    for raw_s, m, canon_s in sample:
        # Morgan novelty
        fp = morgan_fp(m)
        max_tau = max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps_morgan))
        if max_tau < NOVEL_THRESHOLD:
            novelty_count += 1
        # Scaffold-NN
        sf = scaffold_fp(m)
        if sf is not None and ref_fps_scaffold:
            sc_max = max(DataStructs.BulkTanimotoSimilarity(sf, ref_fps_scaffold))
            scaffold_max_taus.append(sc_max)
        # Chem pass
        if chem_pass(canon_s):
            chem_pass_count += 1

    return {
        "n_raw": n_raw,
        "n_valid": n_valid,
        "n_sampled": len(sample),
        "validity":   n_valid / n_raw,
        "uniqueness": n_unique / n_valid,
        "novelty":    novelty_count / len(sample),
        "scaffold_nn": float(np.mean(scaffold_max_taus)) if scaffold_max_taus else None,
        "chem_pass":  chem_pass_count / len(sample),
    }


files = sorted(
    glob.glob("m6_postprocess_bundle/m1_3seed_*.txt") +
    glob.glob("results/m1_sweep_*.txt") +
    glob.glob("baseline_bundle/results/smiles_lstm_samples.txt"))
print(f"[pool] {len(files)} pool files to process")

per_pool = {}
for i, f in enumerate(files, 1):
    name = Path(f).stem
    t1 = time.time()
    metrics = process_pool(f)
    per_pool[name] = metrics
    print(f"[pool] {i}/{len(files)} {name}: "
          f"validity={metrics['validity']:.3f} uniq={metrics['uniqueness']:.3f} "
          f"novelty={metrics['novelty'] and round(metrics['novelty'],3)} "
          f"scaf={metrics['scaffold_nn'] and round(metrics['scaffold_nn'],3)} "
          f"chem={metrics['chem_pass'] and round(metrics['chem_pass'],3)}  "
          f"({time.time()-t1:.1f}s)")
    # incremental persistence
    (OUT / "pool_metrics.json").write_text(json.dumps(
        {"per_pool": per_pool, "per_condition": {}}, indent=2))


def cond_of(name):
    for prefix in ("m1_3seed_", "m1_sweep_"):
        if name.startswith(prefix):
            stem = name.replace(prefix, "")
            if "_seed" in stem:
                return stem.rsplit("_seed", 1)[0]
            return stem
    if name == "smiles_lstm_samples":
        return "smiles_lstm"
    return name


per_condition = {}
for name, m in per_pool.items():
    c = cond_of(name)
    per_condition.setdefault(c, []).append(m)

agg = {}
for c, ms in sorted(per_condition.items()):
    cell = {"n_seeds": len(ms)}
    for k in ("validity", "uniqueness", "novelty", "scaffold_nn", "chem_pass"):
        vals = [m[k] for m in ms if m.get(k) is not None]
        if vals:
            cell[f"{k}_mean"] = float(np.mean(vals))
            cell[f"{k}_std"]  = float(np.std(vals))
    agg[c] = cell

(OUT / "pool_metrics.json").write_text(json.dumps(
    {"per_pool": per_pool, "per_condition": agg}, indent=2))
print(f"\n[pool] -> {OUT/'pool_metrics.json'}")
print(f"\n{'condition':<28} {'n':>3} {'valid':>7} {'uniq':>7} {'novel':>7} {'scaf-NN':>8} {'chem':>7}")
for c in sorted(agg):
    cell = agg[c]
    def fmt(k):
        v = cell.get(f"{k}_mean")
        return f"{v:.3f}" if v is not None else "  -  "
    print(f"{c:<28} {cell['n_seeds']:>3} "
          f"{fmt('validity'):>7} {fmt('uniqueness'):>7} "
          f"{fmt('novelty'):>7} {fmt('scaffold_nn'):>8} {fmt('chem_pass'):>7}")
print("[pool] === DONE ===")
