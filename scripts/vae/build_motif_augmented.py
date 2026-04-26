"""Build a motif-oversampled SMILES corpus for LIMO v2 continuation.

Reads labeled + unlabeled master CSVs, scans each SMILES for rare-motif
SMARTS, and creates an augmented dataset where rare-motif rows are
duplicated. Output: data/training/master/labeled_motif_aug.csv (replaces
the labeled_master input for LIMO v2).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

BASE = Path("E:/Projects/EnergeticDiffusion2")

RARE = [
    ("furazan",   "c1nonc1"),
    ("tetrazole", "c1nnnn1"),
    ("tetrazole2", "C1=NN=NN1"),
    ("triazole",  "c1nncn1"),
    ("triazole2", "c1cnnn1"),
    ("azide",     "[N-]=[N+]=N"),
    ("azide2",    "N=[N+]=[N-]"),
    ("tetrazine", "c1nncnn1"),
    ("furoxan",   "c1nonc1=O"),
]
POLYNITRO_SMARTS = "[N+](=O)[O-]"

RARE_PATS = [(n, Chem.MolFromSmarts(s)) for n, s in RARE]
NITRO_PAT = Chem.MolFromSmarts(POLYNITRO_SMARTS)


def motif_count(mol):
    if mol is None: return None
    rare_hit = any(p is not None and mol.HasSubstructMatch(p) for _, p in RARE_PATS)
    n_nitro = len(mol.GetSubstructMatches(NITRO_PAT)) if NITRO_PAT else 0
    return rare_hit, n_nitro


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    paths = [BASE / "data/training/master/labeled_master.csv",
             BASE / "data/training/master/unlabeled_master.csv"]
    rows = []
    for p in paths:
        if not p.exists(): continue
        print(f"reading {p.name} …")
        df = pd.read_csv(p, low_memory=False)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        for s in df[col].astype(str).tolist():
            rows.append(s)
    print(f"total rows in: {len(rows):,}")
    rows = list(dict.fromkeys(rows))
    print(f"unique:        {len(rows):,}")

    # Categorise
    out_smis = []
    counts = {"rare": 0, "polynitro": 0, "neither": 0}
    t0 = time.time()
    for i, s in enumerate(rows):
        if i and i % 50_000 == 0:
            print(f"  {i:,}/{len(rows):,}  ({time.time()-t0:.0f}s)")
        m = Chem.MolFromSmiles(s)
        if m is None:
            out_smis.append(s)
            counts["neither"] += 1
            continue
        rare, nnitro = motif_count(m)
        if rare:
            out_smis.extend([s] * 5)        # 5× for rare motifs
            counts["rare"] += 1
        elif nnitro >= 3:
            out_smis.extend([s] * 2)        # 2× for polynitro (≥3 NO2)
            counts["polynitro"] += 1
        else:
            out_smis.append(s)
            counts["neither"] += 1

    print(f"\ncomposition (uniques):")
    print(f"  rare motif (5×):      {counts['rare']:>6,d}")
    print(f"  polynitro (2×):       {counts['polynitro']:>6,d}")
    print(f"  baseline (1×):        {counts['neither']:>6,d}")
    print(f"  augmented total:      {len(out_smis):>6,d}")

    out = BASE / "data/training/master/labeled_motif_aug.csv"
    pd.DataFrame({"smiles": out_smis}).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    sys.exit(main())
