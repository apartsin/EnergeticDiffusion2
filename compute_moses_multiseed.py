"""
Compute MOSES-style distribution metrics for all 12 sweep files
(4 conditions x 3 seeds) and aggregate into mean +/- std per condition.

Metrics computed (all via RDKit, no moses pip package required):
  - validity         : fraction of lines that parse as valid SMILES
  - uniqueness       : fraction of canonical SMILES that are unique
  - novelty_vs_LM    : fraction not in labelled master (exact canonical match)
  - int_div1         : 1 - mean pairwise Tanimoto on 500-mol subsample
  - snn_to_LM        : mean nearest-neighbour Tanimoto to LM (500 gen, 1000 LM)
  - scaffold_count   : number of unique Bemis-Murcko scaffolds

Outputs:
  results/moses_multiseed_per_file.json    -- per-file detailed metrics
  results/moses_multiseed_summary.json     -- per-condition mean +/- std
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Silence RDKit warnings about valence ([N+1] notation triggers them)
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# ── constants ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
SWEEP_PATTERN = "m1_sweep_C*.txt"
LM_CSV_CANDIDATES = [
    PROJECT_ROOT / "m4_bundle" / "labelled_master.csv",
    PROJECT_ROOT / "m6_postprocess_bundle" / "labelled_master.csv",
    PROJECT_ROOT / "data" / "labelled_master.csv",
]
OUT_PER_FILE = RESULTS_DIR / "moses_multiseed_per_file.json"
OUT_SUMMARY = RESULTS_DIR / "moses_multiseed_summary.json"

DIV_SAMPLE = 500          # molecules sampled for IntDiv1 computation
SNN_GEN_SAMPLE = 500      # generated molecules sampled for SNN
SNN_LM_SAMPLE = 1000      # LM molecules sampled for SNN reference
LM_MAX_ROWS = 50_000      # cap on LM rows loaded (for novelty / SNN)

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ── helpers ──────────────────────────────────────────────────────────────────

def canon(smi: str) -> str | None:
    """Canonicalise a SMILES string; return None if invalid."""
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def morgan_fp(mol):
    return MORGAN_GEN.GetFingerprint(mol)


def bulk_tanimoto_row(fp, fps_list) -> np.ndarray:
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps_list)
    return np.array(sims, dtype=np.float32)


def int_div1(fps: list) -> float:
    """1 - mean pairwise Tanimoto on provided fingerprints."""
    n = len(fps)
    if n < 2:
        return float("nan")
    total = 0.0
    count = 0
    for i in range(n):
        sims = bulk_tanimoto_row(fps[i], fps[:i])
        total += sims.sum()
        count += len(sims)
    mean_sim = total / count if count > 0 else 0.0
    return 1.0 - mean_sim


def snn_to_ref(gen_fps: list, ref_fps: list) -> float:
    """Mean nearest-neighbour Tanimoto: for each gen mol, max sim to ref set."""
    if not gen_fps or not ref_fps:
        return float("nan")
    nn_sims = []
    for fp in gen_fps:
        sims = bulk_tanimoto_row(fp, ref_fps)
        nn_sims.append(float(sims.max()))
    return float(np.mean(nn_sims))


def bemis_murcko_smiles(mol) -> str | None:
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return None


# ── load labelled master ─────────────────────────────────────────────────────

def load_lm_smiles(max_rows: int = LM_MAX_ROWS) -> set[str]:
    """Load canonical SMILES from the labelled master CSV."""
    lm_path = None
    for p in LM_CSV_CANDIDATES:
        if p.exists():
            lm_path = p
            break
    if lm_path is None:
        print("[WARN] Labelled master CSV not found; novelty/SNN set to NaN")
        return set()

    print(f"  Loading LM from {lm_path} (max {max_rows} rows) ...", flush=True)
    import csv
    lm_smiles = []
    with open(lm_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            raw = row.get("smiles") or row.get("SMILES") or row.get("smi") or ""
            c = canon(raw.strip())
            if c:
                lm_smiles.append(c)

    print(f"  LM loaded: {len(lm_smiles)} valid canonical SMILES", flush=True)
    return lm_smiles  # keep as list so we can subsample for SNN


# ── per-file metrics ─────────────────────────────────────────────────────────

def compute_file_metrics(
    filepath: Path,
    lm_set: set[str],
    lm_fps_pool: list,
    rng: random.Random,
) -> dict:
    t0 = time.time()
    raw_lines = [l.strip() for l in filepath.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_raw = len(raw_lines)

    # --- validity ---
    canon_smiles = []
    for smi in raw_lines:
        c = canon(smi)
        if c is not None:
            canon_smiles.append(c)
    n_valid = len(canon_smiles)
    validity = n_valid / n_raw if n_raw > 0 else 0.0

    # --- uniqueness ---
    unique_smiles = list(dict.fromkeys(canon_smiles))  # preserves order, dedupes
    n_unique = len(unique_smiles)
    uniqueness = n_unique / n_valid if n_valid > 0 else 0.0

    # --- novelty vs LM ---
    if lm_set:
        novel = [s for s in unique_smiles if s not in lm_set]
        n_novel = len(novel)
        novelty = n_novel / n_unique if n_unique > 0 else 0.0
    else:
        n_novel = None
        novelty = float("nan")

    # --- build fingerprints for unique set ---
    unique_mols = []
    unique_fps = []
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = morgan_fp(mol)
            unique_mols.append(mol)
            unique_fps.append(fp)

    # --- IntDiv1 on subsample ---
    sub_size = min(DIV_SAMPLE, len(unique_fps))
    idx_sub = rng.sample(range(len(unique_fps)), sub_size)
    sub_fps = [unique_fps[i] for i in idx_sub]
    id1 = int_div1(sub_fps)

    # --- SNN to LM ---
    if lm_fps_pool:
        gen_sub_size = min(SNN_GEN_SAMPLE, len(unique_fps))
        gen_idx = rng.sample(range(len(unique_fps)), gen_sub_size)
        gen_fps_sub = [unique_fps[i] for i in gen_idx]
        ref_fps_sub = rng.sample(lm_fps_pool, min(SNN_LM_SAMPLE, len(lm_fps_pool)))
        snn = snn_to_ref(gen_fps_sub, ref_fps_sub)
    else:
        snn = float("nan")

    # --- scaffold count ---
    scaffolds = set()
    for mol in unique_mols:
        sc = bemis_murcko_smiles(mol)
        if sc is not None:
            scaffolds.add(sc)
    scaffold_count = len(scaffolds)

    elapsed = time.time() - t0
    def _f(v):
        """Convert to plain Python float, rounded."""
        if v is None:
            return None
        fv = float(v)
        return None if (fv != fv) else round(fv, 6)  # NaN check

    metrics = {
        "file": filepath.name,
        "n_raw": n_raw,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "n_novel_vs_LM": int(n_novel) if n_novel is not None else None,
        "validity": _f(validity),
        "uniqueness": _f(uniqueness),
        "novelty_vs_LM": _f(novelty),
        "int_div1": _f(id1),
        "snn_to_LM": _f(snn),
        "scaffold_count": int(scaffold_count),
        "elapsed_s": round(float(elapsed), 1),
    }
    return metrics


# ── aggregate across seeds ────────────────────────────────────────────────────

def condition_label(filename: str) -> str:
    """Extract condition name from filename, stripping _seed{N} suffix."""
    # e.g. m1_sweep_C0_unguided_seed1.txt -> m1_sweep_C0_unguided
    return re.sub(r"_seed\d+$", "", Path(filename).stem)


def aggregate_conditions(per_file: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in per_file:
        cond = condition_label(m["file"])
        groups[cond].append(m)

    summary = []
    float_keys = ["validity", "uniqueness", "novelty_vs_LM", "int_div1", "snn_to_LM", "scaffold_count"]
    for cond, entries in sorted(groups.items()):
        row = {"condition": cond, "n_seeds": len(entries)}
        for k in float_keys:
            vals = [float(e[k]) for e in entries if e.get(k) is not None]
            if vals:
                row[f"{k}_mean"] = round(float(np.mean(vals)), 6)
                row[f"{k}_std"] = round(float(np.std(vals, ddof=0)), 6)
            else:
                row[f"{k}_mean"] = None
                row[f"{k}_std"] = None
        summary.append(row)
    return summary


# ── pretty table ─────────────────────────────────────────────────────────────

def print_summary_table(summary: list[dict]):
    header = f"{'Condition':<35} {'Valid':>6} {'Uniq':>6} {'Novel':>6} {'ID1':>6} {'SNN':>6} {'Scaff':>6}"
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in summary:
        cond = row["condition"]
        def fmt(k):
            m = row.get(f"{k}_mean")
            s = row.get(f"{k}_std")
            if m is None:
                return "  N/A"
            if k == "scaffold_count":
                return f"{m:5.0f}"
            return f"{m:.3f}"
        print(
            f"{cond:<35} "
            f"{fmt('validity'):>6} "
            f"{fmt('uniqueness'):>6} "
            f"{fmt('novelty_vs_LM'):>6} "
            f"{fmt('int_div1'):>6} "
            f"{fmt('snn_to_LM'):>6} "
            f"{fmt('scaffold_count'):>6}"
        )
    print("=" * len(header))
    print()

    # Also print with std
    print("Mean +/- std detail:")
    print()
    for row in summary:
        print(f"  {row['condition']}")
        for k in ["validity", "uniqueness", "novelty_vs_LM", "int_div1", "snn_to_LM", "scaffold_count"]:
            m = row.get(f"{k}_mean")
            s = row.get(f"{k}_std")
            if m is None:
                print(f"    {k:<20} N/A")
            elif k == "scaffold_count":
                print(f"    {k:<20} {m:.1f} +/- {s:.1f}")
            else:
                print(f"    {k:<20} {m:.4f} +/- {s:.4f}")
        print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(42)
    RESULTS_DIR.mkdir(exist_ok=True)

    # collect sweep files
    sweep_files = sorted(RESULTS_DIR.glob(SWEEP_PATTERN))
    if not sweep_files:
        print(f"[ERROR] No sweep files found matching {RESULTS_DIR}/{SWEEP_PATTERN}")
        sys.exit(1)
    print(f"Found {len(sweep_files)} sweep files:")
    for f in sweep_files:
        print(f"  {f.name}")
    print()

    # load labelled master
    print("Loading labelled master ...")
    lm_smiles_list = load_lm_smiles()
    lm_set = set(lm_smiles_list)

    # precompute LM fingerprints (pool for SNN subsampling)
    print(f"  Computing LM fingerprints for SNN (pool of {min(SNN_LM_SAMPLE*10, len(lm_smiles_list))}) ...", flush=True)
    lm_fp_pool_size = min(SNN_LM_SAMPLE * 10, len(lm_smiles_list))
    lm_smiles_sub = rng.sample(lm_smiles_list, lm_fp_pool_size) if len(lm_smiles_list) > lm_fp_pool_size else lm_smiles_list
    lm_fps_pool = []
    for smi in lm_smiles_sub:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            lm_fps_pool.append(morgan_fp(mol))
    print(f"  LM FP pool: {len(lm_fps_pool)} fingerprints ready")
    print()

    # process each file
    per_file_results = []
    for i, fpath in enumerate(sweep_files, 1):
        print(f"[{i:2d}/{len(sweep_files)}] Processing {fpath.name} ...", end=" ", flush=True)
        metrics = compute_file_metrics(fpath, lm_set, lm_fps_pool, rng)
        per_file_results.append(metrics)
        print(
            f"valid={metrics['validity']:.3f}  uniq={metrics['uniqueness']:.3f}  "
            f"novel={metrics['novelty_vs_LM'] or 'N/A':.3f}  "
            f"id1={metrics['int_div1'] or 'N/A':.3f}  "
            f"snn={metrics['snn_to_LM'] or 'N/A':.3f}  "
            f"scaff={metrics['scaffold_count']}  ({metrics['elapsed_s']}s)"
        )

    # aggregate
    summary = aggregate_conditions(per_file_results)

    # save outputs
    with open(OUT_PER_FILE, "w") as fh:
        json.dump({"per_file": per_file_results}, fh, indent=2)
    print(f"\nPer-file results saved to {OUT_PER_FILE}")

    with open(OUT_SUMMARY, "w") as fh:
        json.dump({"per_condition": summary}, fh, indent=2)
    print(f"Summary results saved to {OUT_SUMMARY}")

    # print table
    print_summary_table(summary)


if __name__ == "__main__":
    main()
