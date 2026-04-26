"""
Fix 3DCNN heat_of_formation unit bug in labeled_master.csv.

Root cause: raw 3DCNN.csv header reads "heat_of_formation (kcal/mol)" but the
master column stores all other sources in kJ/mol. Ingestion did not apply the
unit conversion, leaving 26,254 rows ~4.18× too small.

Fix: multiply 3DCNN heat_of_formation by 4.184 (1 kcal/mol = 4.184 kJ/mol).

Verification: after fix, 3DCNN HOF range should become ~(-1,740, +1,180) kJ/mol,
matching the scale of train_set (EXPLO5, -1,073 to +1,550) and det_dataset
(experimental, -1,297 to +1,834).
"""
import shutil
import pandas as pd
from pathlib import Path

BASE   = Path("E:/Projects/EnergeticDiffusion2")
MASTER = BASE / "data/training/master/labeled_master.csv"
BACKUP = BASE / "data/training/master/labeled_master_pre_3dcnn_hof_units.csv.bak"

KCAL_TO_KJ = 4.184

print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)

mask = (lm["heat_of_formation_source_dataset"] == "3DCNN")
n_affected = int(mask.sum())
before = lm.loc[mask, "heat_of_formation"].describe()

lm.loc[mask, "heat_of_formation"] = lm.loc[mask, "heat_of_formation"] * KCAL_TO_KJ

after = lm.loc[mask, "heat_of_formation"].describe()

print(f"\nRows fixed: {n_affected:,}")
print("\nBefore (kcal/mol):")
print(before)
print("\nAfter (kJ/mol):")
print(after)

print(f"\nSaving to {MASTER} …")
lm.to_csv(MASTER, index=False)
print(f"Done. Backup at {BACKUP}")
