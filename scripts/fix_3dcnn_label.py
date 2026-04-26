"""
Fix 3DCNN label_source_type bug in labeled_master.csv.

Root cause: the 3DCNN raw file (EMDP/Data/3DCNN.csv, actually xlsx) contains
DFT-computed quantum-chemistry properties — electronic_energy (Hartree),
HOMO_LUMO_gap (eV), dipole_moment (Debye), plus DFT-derived density and HOF.
These were incorrectly tagged as compiled_observed (experimental).

Fix: for rows where source_dataset == "3DCNN", set label_source_type to
"model_predicted". Affects 26,254 rows. After fix, real experimental count
drops from ~33,391 to ~7,137.
"""
import shutil
import pandas as pd
from pathlib import Path

BASE   = Path("E:/Projects/EnergeticDiffusion2")
MASTER = BASE / "data/training/master/labeled_master.csv"
BACKUP = BASE / "data/training/master/labeled_master_pre3dcnn_fix.csv.bak"

print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)
print(f"  {len(lm):,} rows, {len(lm.columns)} columns")

mask = (lm["source_dataset"] == "3DCNN") & (lm["label_source_type"] == "compiled_observed")
n_affected = int(mask.sum())

print("\nBefore:")
print(lm[lm['source_dataset']=='3DCNN']['label_source_type'].value_counts())

lm.loc[mask, "label_source_type"] = "model_predicted"

# Also update per-property source_type columns if they exist
for col in ["density_source_type","heat_of_formation_source_type",
            "detonation_velocity_source_type","detonation_pressure_source_type",
            "explosion_heat_source_type"]:
    if col in lm.columns:
        sub_mask = mask & (lm[col] == "compiled_observed")
        lm.loc[sub_mask, col] = "model_predicted"

print(f"\nRows updated: {n_affected:,}")
print("\nAfter:")
print(lm[lm['source_dataset']=='3DCNN']['label_source_type'].value_counts())
print()
print("Overall label_source_type distribution:")
print(lm["label_source_type"].value_counts())

print(f"\nSaving to {MASTER} …")
lm.to_csv(MASTER, index=False)
print(f"Done. Backup at {BACKUP}")
