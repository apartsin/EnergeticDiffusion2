"""2.4 Novelty stratification for the merged top-100.

For each merged-top-100 candidate, compute:
- max Tanimoto similarity to the labelled-master corpus (~66k)
- max Tanimoto similarity to the augmented training corpus (diffusion_pretrain
  train+validation+test, ~694k)

Outputs:
- experiments/novelty_stratification.json (per-candidate)
- experiments/novelty_stratification_summary.json (medians, fractions)
- prints a markdown table for paper inclusion

Run: /c/Python314/python scripts/diffusion/novelty_stratification.py
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

LEADS_MD = Path("experiments/final_merged_topN.md")
LABELLED = Path("data/training/master/labeled_master.csv")
AUG_DIR = Path("data/training/diffusion_pretrain")
OUT_DIR = Path("experiments")
N_BITS = 2048
RADIUS = 2


def parse_leads(md_path: Path):
    rows = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or "rank" in line or "---" in line:
            continue
        cells = [c.strip(" `") for c in line.split("|")]
        if len(cells) < 14:
            continue
        try:
            rows.append({
                "rank": int(cells[1]),
                "comp": float(cells[2]),
                "source": cells[13],
                "smiles": cells[14],
            })
        except ValueError:
            continue
    return rows


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)


def build_fp_bank(smiles_iter, name=""):
    fps = []
    valid = 0
    skipped = 0
    for smi in smiles_iter:
        if not isinstance(smi, str):
            skipped += 1; continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1; continue
        fps.append(fp(mol))
        valid += 1
    print(f"  [{name}] built FP bank: {valid} valid / {valid + skipped} attempted")
    return fps


def max_tanimoto(query_fp, bank):
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, bank)
    arr = np.asarray(sims, dtype=np.float32)
    return float(arr.max()), int(arr.argmax())


def main():
    t0 = time.time()
    leads = parse_leads(LEADS_MD)
    print(f"Loaded {len(leads)} merged top candidates")

    print("Loading labelled master ...")
    lm = pd.read_csv(LABELLED, usecols=["smiles"])
    print(f"  rows: {len(lm)}")
    lm_fps = build_fp_bank(lm["smiles"].tolist(), name="labelled")
    lm_smis = lm["smiles"].tolist()

    print("Loading augmented diffusion-pretrain corpus (train+val+test) ...")
    aug_smis = []
    for split in ("train.csv", "validation.csv", "test.csv"):
        df = pd.read_csv(AUG_DIR / split, usecols=["smiles"])
        aug_smis.extend(df["smiles"].tolist())
    print(f"  rows: {len(aug_smis)}")
    aug_fps = build_fp_bank(aug_smis, name="augmented")

    out = []
    for L in leads:
        mol = Chem.MolFromSmiles(L["smiles"])
        if mol is None:
            out.append({**L, "lm_max": None, "aug_max": None,
                       "lm_nn": None, "aug_nn": None})
            continue
        q = fp(mol)
        lm_max, lm_idx = max_tanimoto(q, lm_fps)
        aug_max, aug_idx = max_tanimoto(q, aug_fps)
        out.append({**L,
                    "lm_max": lm_max,
                    "aug_max": aug_max,
                    "lm_nn": lm_smis[lm_idx],
                    "aug_nn": aug_smis[aug_idx]})

    # Summary
    lm_arr = np.array([r["lm_max"] for r in out if r["lm_max"] is not None])
    aug_arr = np.array([r["aug_max"] for r in out if r["aug_max"] is not None])
    summary = {
        "n_candidates": len(out),
        "labelled_master_corpus_size": len(lm_smis),
        "augmented_corpus_size": len(aug_smis),
        "labelled_master_max_tanimoto": {
            "median": float(np.median(lm_arr)),
            "mean": float(lm_arr.mean()),
            "p25": float(np.percentile(lm_arr, 25)),
            "p75": float(np.percentile(lm_arr, 75)),
            "frac_above_0p55": float((lm_arr > 0.55).mean()),
            "frac_above_0p70": float((lm_arr > 0.70).mean()),
            "frac_exact_match": float((lm_arr >= 0.999).mean()),
        },
        "augmented_max_tanimoto": {
            "median": float(np.median(aug_arr)),
            "mean": float(aug_arr.mean()),
            "p25": float(np.percentile(aug_arr, 25)),
            "p75": float(np.percentile(aug_arr, 75)),
            "frac_above_0p55": float((aug_arr > 0.55).mean()),
            "frac_above_0p70": float((aug_arr > 0.70).mean()),
            "frac_exact_match": float((aug_arr >= 0.999).mean()),
        },
        "delta_aug_minus_lm_median": float(np.median(aug_arr) - np.median(lm_arr)),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "novelty_stratification.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    (OUT_DIR / "novelty_stratification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWall: {time.time() - t0:.1f}s")
    print(f"Outputs: {OUT_DIR / 'novelty_stratification.json'}")
    print(f"         {OUT_DIR / 'novelty_stratification_summary.json'}")


if __name__ == "__main__":
    main()
