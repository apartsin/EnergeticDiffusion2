"""
Fix density provenance for cm4c01978_si_001 (Ma et al. 2022 CSD crystal database).

Root cause: density values in this source come from CSD single-crystal X-ray
diffraction — experimental measurements. At ingestion, because the overall row
label_source_type was set to "kj_calculated" (D/P/Q come from K-J formula),
the density_source_type column was left as "unknown" instead of
"compiled_observed". Density is an INPUT to K-J, never an output.

Fix:
  - density_source_type  → "compiled_observed"
  - density_tier         → "A"
  - row-level tier recomputed (now Tier A for all 12,040 rows)
"""
import shutil
import pandas as pd
from pathlib import Path

BASE   = Path("E:/Projects/EnergeticDiffusion2")
MASTER = BASE / "data/training/master/labeled_master.csv"
BACKUP = BASE / "data/training/master/labeled_master_pre_cm4c_density.csv.bak"

PROPS = ["density", "heat_of_formation", "detonation_velocity",
         "detonation_pressure", "explosion_heat"]

print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)

mask = (lm["density_source_dataset"] == "cm4c01978_si_001") & lm["density"].notna()
n = int(mask.sum())
print(f"Rows to fix: {n:,}")

lm.loc[mask, "density_source_type"] = "compiled_observed"
if "density_tier" in lm.columns:
    lm.loc[mask, "density_tier"] = "A"

# recompute row-level tier
rank = {"A": 0, "B": 1, "C": 2, None: 99}
def row_min_tier(row):
    cands = [row[f"{p}_tier"] for p in PROPS if f"{p}_tier" in row.index]
    cands = [c for c in cands if c in ("A", "B", "C")]
    if not cands:
        return None
    return min(cands, key=lambda c: rank[c])
lm["tier"] = lm.apply(row_min_tier, axis=1)

print("\nRow-level tier distribution (after fix):")
print(lm["tier"].value_counts(dropna=False))

print("\nPer-property tier coverage (after fix):")
for p in PROPS:
    tc = f"{p}_tier"
    if tc in lm.columns:
        a = (lm[tc] == "A").sum()
        b = (lm[tc] == "B").sum()
        c = (lm[tc] == "C").sum()
        print(f"  {p}: A={a:,}  B={b:,}  C={c:,}")

print(f"\nSaving to {MASTER} …")
lm.to_csv(MASTER, index=False)
print(f"Done. Backup at {BACKUP}")
