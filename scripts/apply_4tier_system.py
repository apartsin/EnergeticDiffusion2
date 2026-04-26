"""
Apply 4-tier reliability system to labeled_master.csv (supersedes 3-tier).

Tiers (most → least reliable):
  A — Experimentally measured / from peer-reviewed literature (XRD density,
      measured detonation velocity, etc.)
  B — Quantum / physics-based simulation. Currently: EXPLO5 thermochemistry
      code outputs (D, P, Q from train_set/test_set).
      [True DFT outputs would go here — we have none yet.]
  C — Kamlet-Jacobs empirical formula (cm4c01978_si_001 detonation properties)
  D — Data-driven ML / QSAR models: 3DCNN surrogate, MDGNN/de novo generators
      with their own property predictors, q-RASPR QSAR.

Also fixes: density_source_type for cm4c01978_si_001 (was "unknown", is
actually experimental XRD from the CSD).

Row-level tier = most reliable tier across populated properties.
"""
import shutil
import pandas as pd
from pathlib import Path

BASE   = Path("E:/Projects/EnergeticDiffusion2")
MASTER = BASE / "data/training/master/labeled_master.csv"
BACKUP = BASE / "data/training/master/labeled_master_pre_4tier.csv.bak"

PROPS = ["density", "heat_of_formation", "detonation_velocity",
         "detonation_pressure", "explosion_heat"]

# Source mapping rules
ML_D_SOURCES = {
    "3DCNN",
    "denovo_sampling_rl.predict.0_filtered",
    "denovo_sampling_tl.predict.0_filtered",
    "MDGNN", "generation", "q-RASPR",
}
KJ_C_SOURCES = {"cm4c01978_si_001", "kj_from_explo5_hof"}
EXPLO5_SRC   = {"train_set", "test_set"}

def tier_for(property_name, source_dataset):
    if pd.isna(source_dataset) or source_dataset == "":
        return None
    # cm4c01978 density is XRD experimental (CSD crystallography);
    # only its D/P/Q/HOF come from K-J formula.
    if source_dataset == "cm4c01978_si_001":
        return "A" if property_name == "density" else "C"
    if source_dataset in ML_D_SOURCES:
        return "D"
    if source_dataset in KJ_C_SOURCES:
        return "C"
    if source_dataset in EXPLO5_SRC:
        return "A" if property_name == "density" else "B"
    return "A"

print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)

# 1. Fix cm4c01978 density provenance (XRD experimental)
mask_cm4c_dens = ((lm["density_source_dataset"] == "cm4c01978_si_001") &
                  lm["density"].notna())
n_density_fix = int(mask_cm4c_dens.sum())
lm.loc[mask_cm4c_dens, "density_source_type"] = "compiled_observed"
print(f"cm4c01978 density rows re-tagged as experimental: {n_density_fix:,}")

# 2. Assign per-property tier via source_dataset rule
for p in PROPS:
    src_col  = f"{p}_source_dataset"
    tier_col = f"{p}_tier"
    lm[tier_col] = lm.apply(
        lambda r: tier_for(p, r[src_col]) if pd.notna(r[p]) else None,
        axis=1)

# 3. Row-level tier = most reliable populated property
rank = {"A": 0, "B": 1, "C": 2, "D": 3, None: 99}
def row_min_tier(row):
    cands = [row[f"{p}_tier"] for p in PROPS]
    cands = [c for c in cands if c in ("A", "B", "C", "D")]
    if not cands:
        return None
    return min(cands, key=lambda c: rank[c])
lm["tier"] = lm.apply(row_min_tier, axis=1)

print("\nRow-level tier distribution:")
print(lm["tier"].value_counts(dropna=False))

print("\nPer-property tier coverage:")
for p in PROPS:
    tc = f"{p}_tier"
    d = lm[tc].value_counts(dropna=False).to_dict()
    print(f"  {p}: A={d.get('A',0):,}  B={d.get('B',0):,}  C={d.get('C',0):,}  D={d.get('D',0):,}")

print(f"\nSaving to {MASTER} …")
lm.to_csv(MASTER, index=False)
print(f"Done. Backup at {BACKUP}")
