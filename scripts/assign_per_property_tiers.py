"""
Add per-property reliability tier columns to labeled_master.csv.

Each numeric property gets a <property>_tier column with values:
  "A"  — experimental / from literature
  "B"  — DFT or quantum-simulation derived (incl. 3DCNN, EXPLO5 detonation code)
  "C"  — Kamlet-Jacobs empirical formula (cm4c01978_si_001)
  None — pure ML prediction (de novo generators, MDGNN) or QSAR (q-RASPR)

Mapping rules (by <property>_source_dataset):

  3DCNN                                     → B (DFT surrogate CNN)
  cm4c01978_si_001                          → C (K-J formula)
  denovo_sampling_rl / tl / MDGNN /
    generation / q-RASPR                    → None (excluded)
  train_set / test_set (EMDP):
       density                              → A  (XRD experimental)
       detonation_{velocity,pressure} /
       explosion_heat / heat_of_formation   → B  (EXPLO5 thermochemistry)
  5039, det_dataset_08-02-2022, Dm,
    combined_data, emdb_v21_molecules_*,
    Huang_Massa_*, CHNOClF_dataset,
    nist_cameo_enrichment                   → A (experimental literature)

The row-level `tier` column is preserved and replaced with the minimum-reliability
(i.e. most reliable) tier across populated properties — "A" if any property is A,
else "B" if any is B, else "C" if any is C, else None.
"""
import shutil
import pandas as pd
from pathlib import Path

BASE   = Path("E:/Projects/EnergeticDiffusion2")
MASTER = BASE / "data/training/master/labeled_master.csv"
BACKUP = BASE / "data/training/master/labeled_master_pre_prop_tier.csv.bak"

PROPS = ["density", "heat_of_formation", "detonation_velocity",
         "detonation_pressure", "explosion_heat"]

B_SOURCES   = {"3DCNN"}
C_SOURCES   = {"cm4c01978_si_001"}
EXCLUDED    = {"denovo_sampling_rl.predict.0_filtered",
               "denovo_sampling_tl.predict.0_filtered",
               "MDGNN", "generation", "q-RASPR"}
EXPLO5_SRC  = {"train_set", "test_set"}

def tier_for(property_name, source_dataset):
    if pd.isna(source_dataset) or source_dataset == "":
        return None
    if source_dataset in B_SOURCES:
        return "B"
    if source_dataset in C_SOURCES:
        return "C"
    if source_dataset in EXCLUDED:
        return None
    if source_dataset in EXPLO5_SRC:
        return "A" if property_name == "density" else "B"
    return "A"   # any other source is treated as experimental literature

print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)
print(f"  {len(lm):,} rows, {len(lm.columns)} columns")

# Assign per-property tier
for p in PROPS:
    src_col = f"{p}_source_dataset"
    tier_col = f"{p}_tier"
    lm[tier_col] = lm.apply(
        lambda r: tier_for(p, r[src_col]) if pd.notna(r[p]) else None,
        axis=1)
    counts = lm[tier_col].value_counts(dropna=False).to_dict()
    print(f"  {tier_col}: {counts}")

# Row-level tier: minimum (most reliable) among populated property tiers
rank = {"A": 0, "B": 1, "C": 2, None: 99}

def row_min_tier(row):
    cands = [row[f"{p}_tier"] for p in PROPS]
    cands = [c for c in cands if c in ("A", "B", "C")]
    if not cands:
        return None
    return min(cands, key=lambda c: rank[c])

lm["tier"] = lm.apply(row_min_tier, axis=1)

print("\nRow-level tier distribution (updated):")
print(lm["tier"].value_counts(dropna=False))

print("\nDataset sizes (cumulative, by row):")
print(f"  A     : {(lm['tier']=='A').sum():,}")
print(f"  A+B   : {lm['tier'].isin(['A','B']).sum():,}")
print(f"  A+B+C : {lm['tier'].isin(['A','B','C']).sum():,}")

print("\nPer-property tier coverage:")
rows = []
for p in PROPS:
    tc = f"{p}_tier"
    rows.append({
        "Property": p,
        "A (exp.)":     (lm[tc] == "A").sum(),
        "B (DFT/EXPLO5)":(lm[tc] == "B").sum(),
        "C (K-J)":      (lm[tc] == "C").sum(),
        "Excluded":     ((lm[p].notna()) & (lm[tc].isna())).sum(),
        "Total filled": lm[p].notna().sum(),
    })
cov = pd.DataFrame(rows)
print(cov.to_string(index=False))

print(f"\nSaving to {MASTER} …")
lm.to_csv(MASTER, index=False)
print(f"Done. Backup at {BACKUP}")
