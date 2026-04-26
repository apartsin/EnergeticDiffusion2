"""
Add a three-tier reliability label to labeled_master.csv.

  tier == "A"  : experimentally measured / from literature
                  (label_source_type == "compiled_observed")

  tier == "B"  : DFT or quantum-simulation derived
                  (source_dataset == "3DCNN", i.e. 3D-CNN surrogate of DFT)

  tier == "C"  : Kamlet-Jacobs empirical formula
                  (label_source_type == "kj_calculated")

  tier == NaN  : pure-ML predictions (de novo generators + QSAR); excluded
                 from the reliability tiers but retained in the file.

Dataset semantics (cumulative):
  Dataset A = rows where tier == "A"
  Dataset B = rows where tier in ("A","B")
  Dataset C = rows where tier in ("A","B","C")
"""
import shutil
import pandas as pd
from pathlib import Path

BASE   = Path("E:/Projects/EnergeticDiffusion2")
MASTER = BASE / "data/training/master/labeled_master.csv"
BACKUP = BASE / "data/training/master/labeled_master_pretier.csv.bak"

print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)
print(f"  {len(lm):,} rows, {len(lm.columns)} columns")

def _tier(row):
    if row["label_source_type"] == "compiled_observed":
        return "A"
    if row["label_source_type"] == "model_predicted" and row["source_dataset"] == "3DCNN":
        return "B"
    if row["label_source_type"] == "kj_calculated":
        return "C"
    return None

lm["tier"] = lm.apply(_tier, axis=1)

print("\nTier distribution:")
print(lm["tier"].value_counts(dropna=False))

print("\nDataset sizes (cumulative):")
print(f"  Dataset A (experimental):          {(lm['tier']=='A').sum():,}")
print(f"  Dataset B (A + DFT/quantum):       {lm['tier'].isin(['A','B']).sum():,}")
print(f"  Dataset C (B + K-J):               {lm['tier'].isin(['A','B','C']).sum():,}")
print(f"  Excluded (QSAR + ML generators):   {lm['tier'].isna().sum():,}")

print(f"\nSaving to {MASTER} …")
lm.to_csv(MASTER, index=False)
print(f"Done. Backup at {BACKUP}")
