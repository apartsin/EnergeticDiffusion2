"""Compute novelty (max Tani to labelled-master) for each pool's top-1 SMILES.

m6_post.json only stores top-1 per run. We compute the max Tanimoto of each
top-1 SMILES against the 65 980-row labelled-master, giving a single
"memorization vs extrapolation" diagnostic per (condition, seed).

For per-pool scatter data we read each .txt SMILES pool, score with the
existing top-1 D/composite (from m6_post per-run), and pick the top-N by
predicted D directly from the .txt (we don't have per-row 3D-CNN scores
locally, but we can plot just the top-1 per (condition, seed) which is
plenty for the figure).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

OUT = Path("results"); OUT.mkdir(exist_ok=True, parents=True)
LM = pd.read_csv("m6_postprocess_bundle/labelled_master.csv",
                  usecols=["smiles"], low_memory=False)
ref_smis = LM["smiles"].dropna().tolist()
print(f"[novelty] labelled-master rows: {len(ref_smis)}")


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


print("[novelty] building reference FP bank ...")
ref_fps = [f for s in ref_smis if (f := fp(s)) is not None]
print(f"[novelty] ref_fps={len(ref_fps)}")


def max_tani(smi):
    f = fp(smi)
    if f is None: return None
    return float(max(DataStructs.BulkTanimotoSimilarity(f, ref_fps)))


post = json.load(open("results/m6_post.json"))

# MolMIM (separate run)
molmim_post_path = Path("results/molmim_post.json")
molmim = json.load(open(molmim_post_path)) if molmim_post_path.exists() else None

per_run_data = list(post.get("per_run", []))
if molmim:
    per_run_data.extend(molmim.get("per_run", []))

rows = []
for r in per_run_data:
    cond = r.get("condition", "?")
    seed = r.get("seed", 0)
    smi = r.get("top1_smiles")
    if not smi: continue
    t = max_tani(smi)
    rows.append({
        "condition": cond, "seed": seed,
        "top1_smiles": smi,
        "max_tani_to_LM": t,
        "novelty": (1 - t) if t is not None else None,
        "is_memorized": (t is not None and t >= 0.55),
        "top1_D_kms": r.get("top1_D_kms"),
        "top1_rho": r.get("top1_rho"),
        "top1_P_GPa": r.get("top1_P_GPa"),
        "top1_composite": r.get("top1_composite"),
        "topN_mean_D": r.get("topN_mean_D"),
        "topN_n_scaffolds": r.get("topN_n_scaffolds"),
        "n_validated": r.get("n_validated"),
    })

# Per-condition aggregation
agg = {}
for r in rows:
    c = r["condition"]
    agg.setdefault(c, []).append(r)

print()
print(f"{'condition':<28} {'n':>3} {'top1_D':>7} {'top1_rho':>8} {'top1_P':>7} {'top1_comp':>9} {'maxTani':>8} {'mem?':>5}")
for c in sorted(agg):
    rs = agg[c]
    Ds = [r["top1_D_kms"] for r in rs if r["top1_D_kms"]]
    Rs = [r["top1_rho"] for r in rs if r["top1_rho"]]
    Ps = [r["top1_P_GPa"] for r in rs if r["top1_P_GPa"]]
    Cs = [r["top1_composite"] for r in rs if r["top1_composite"] is not None]
    Ts = [r["max_tani_to_LM"] for r in rs if r["max_tani_to_LM"] is not None]
    mems = sum(r["is_memorized"] for r in rs)
    print(f"{c:<28} {len(rs):>3} "
           f"{(np.mean(Ds) if Ds else 0):>7.3f} "
           f"{(np.mean(Rs) if Rs else 0):>8.3f} "
           f"{(np.mean(Ps) if Ps else 0):>7.2f} "
           f"{(np.mean(Cs) if Cs else 0):>9.3f} "
           f"{(np.mean(Ts) if Ts else 0):>8.3f} "
           f"{mems}/{len(rs):>3}")

(OUT / "novelty_top1.json").write_text(json.dumps({"rows": rows}, indent=2))
print(f"\n[novelty] -> {OUT / 'novelty_top1.json'}")
print("[novelty] === DONE ===")
