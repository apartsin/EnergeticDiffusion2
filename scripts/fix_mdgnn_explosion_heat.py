"""
Fix MDGNN explosion_heat unit conversion bug in labeled_master.csv.

Root cause:
  MDGNN raw file stores detonation heat as Q(cal/g).
  Correct conversion: explosion_heat (MJ/kg) = Q(cal/g) × 0.004184
  Bug: for some rows Q(cal/g) was stored directly as explosion_heat
       without the ×0.004184 factor, giving values 239× too high.

Safe threshold: no correctly-converted MDGNN row can exceed 12 MJ/kg
  (raw Q max = 1545 cal/g × 0.004184 = 6.46 MJ/kg).
  All MDGNN rows with explosion_heat > 12 are therefore unambiguously buggy.

Fix: for MDGNN rows where explosion_heat > 12, multiply by 0.004184.
"""
import shutil
import pandas as pd
from pathlib import Path

BASE   = Path("E:/Projects/EnergeticDiffusion2")
MASTER = BASE / "data/training/master/labeled_master.csv"
BACKUP = BASE / "data/training/master/labeled_master_prebug2.csv.bak"

CONVERSION = 0.004184   # cal/g  →  MJ/kg
THRESHOLD  = 12.0       # MJ/kg — physical ceiling for correctly-converted values

print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)

mask = (lm["explosion_heat_source_dataset"] == "MDGNN") & \
       (lm["explosion_heat"] > THRESHOLD)

n_affected = mask.sum()
before = lm.loc[mask, "explosion_heat"].copy()

lm.loc[mask, "explosion_heat"] = lm.loc[mask, "explosion_heat"] * CONVERSION

after = lm.loc[mask, "explosion_heat"]

print(f"\nRows fixed: {n_affected:,}")
print(f"Before — min: {before.min():.3f}  max: {before.max():.3f}  mean: {before.mean():.3f}")
print(f"After  — min: {after.min():.4f}  max: {after.max():.4f}  mean: {after.mean():.4f}")

# Verify no MDGNN rows remain above threshold
remaining = lm[(lm["explosion_heat_source_dataset"] == "MDGNN") &
               (lm["explosion_heat"] > THRESHOLD)]
print(f"\nMDGNN rows still above {THRESHOLD} MJ/kg: {len(remaining)}")

# Overall explosion_heat after fix
all_eh = lm["explosion_heat"].dropna()
print(f"\nOverall explosion_heat after fix:")
print(f"  max: {all_eh.max():.3f} MJ/kg")
print(f"  rows > 12 MJ/kg: {(all_eh > 12).sum()}")

print(f"\nSaving to {MASTER} …")
lm.to_csv(MASTER, index=False)
print(f"Done. Backup at {BACKUP}")
