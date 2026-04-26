"""
Fix DetoV column mapping bug in labeled_master.csv.

Affected sources:
  denovo_sampling_rl.predict.0_filtered
  denovo_sampling_tl.predict.0_filtered

Problem: detonation_velocity was populated from predict_DetoV (volume of gaseous
detonation products, L/kg) instead of predict_DetoD (Chapman-Jouguet detonation
velocity, km/s).

Fix: re-join each raw file on SMILES and replace detonation_velocity with predict_DetoD.
"""
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

BASE   = Path("E:/Projects/EnergeticDiffusion2")
MASTER = BASE / "data/training/master/labeled_master.csv"
BACKUP = BASE / "data/training/master/labeled_master.csv.bak"

SOURCES = {
    "denovo_sampling_rl.predict.0_filtered":
        BASE / "data/raw/energetic_external/EMDP/Data/denovo_sampling_rl.predict.0_filtered.csv",
    "denovo_sampling_tl.predict.0_filtered":
        BASE / "data/raw/energetic_external/EMDP/Data/denovo_sampling_tl.predict.0_filtered.csv",
}

# ── backup ────────────────────────────────────────────────────────────────────
print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

# ── load master ───────────────────────────────────────────────────────────────
print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)
print(f"  {len(lm):,} rows, {len(lm.columns)} columns")

original_dv = lm["detonation_velocity"].copy()

# ── apply fix per source ──────────────────────────────────────────────────────
total_fixed = 0
total_newly_filled = 0

for src_name, raw_path in SOURCES.items():
    print(f"\nProcessing: {src_name}")
    raw = pd.read_csv(raw_path, usecols=["smiles", "predict_DetoD"])
    raw = raw.rename(columns={"predict_DetoD": "correct_dv"})
    raw = raw.drop_duplicates("smiles")

    mask = lm["source_dataset"] == src_name
    n_rows = mask.sum()
    print(f"  {n_rows:,} rows in master for this source")

    # join on smiles to get correct predict_DetoD
    before = lm.loc[mask, "detonation_velocity"].copy()
    lm_sub = lm.loc[mask, ["smiles"]].merge(raw, on="smiles", how="left")
    lm_sub.index = lm.index[mask]

    lm.loc[mask, "detonation_velocity"] = lm_sub["correct_dv"].values

    after = lm.loc[mask, "detonation_velocity"]
    n_fixed  = (before.notna() & after.notna()).sum()
    n_new    = (before.isna() & after.notna()).sum()
    n_lost   = (before.notna() & after.isna()).sum()
    n_no_match = after.isna().sum()

    print(f"  Replaced existing values: {n_fixed:,}")
    print(f"  Newly filled (was NaN):   {n_new:,}")
    print(f"  Lost (no SMILES match):   {n_lost:,}")
    print(f"  Still NaN (no raw match): {n_no_match:,}")
    print(f"  New range: {after.min():.3f} – {after.max():.3f} km/s")

    total_fixed        += n_fixed
    total_newly_filled += n_new

# ── verify the fix ────────────────────────────────────────────────────────────
print("\n=== Verification ===")
for src_name in SOURCES:
    sub = lm[lm["source_dataset"] == src_name]["detonation_velocity"].dropna()
    print(f"{src_name}: n={len(sub):,}  mean={sub.mean():.3f}  "
          f"min={sub.min():.3f}  max={sub.max():.3f}")

# Confirm no more values below 1 km/s in these two sources
buggy_remaining = lm[
    lm["source_dataset"].isin(SOURCES.keys()) &
    lm["detonation_velocity"].notna() &
    (lm["detonation_velocity"] < 1.0)
]
print(f"\nRows still below 1.0 km/s in fixed sources: {len(buggy_remaining)}")

# Overall detonation_velocity below 1 km/s after fix
all_below1 = lm[lm["detonation_velocity"].notna() & (lm["detonation_velocity"] < 1.0)]
print(f"Total rows below 1.0 km/s across entire master: {len(all_below1)}")
if len(all_below1):
    print(all_below1[["smiles","detonation_velocity","source_dataset"]].head(10).to_string())

# ── save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving fixed master to {MASTER} …")
lm.to_csv(MASTER, index=False)

print(f"\nDone. Fixed {total_fixed:,} values, newly filled {total_newly_filled:,}.")
print(f"Backup preserved at {BACKUP}")
