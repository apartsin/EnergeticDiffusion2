"""
Comprehensive audit-fix pass on labeled_master.csv.

Issues found by audit and fixed here:

  A. heat_of_formation from cm4c01978_si_001 (n=12,029):
     - Unit bug: raw file "Hf solid" is in kcal/mol, stored in master as if
       kJ/mol.  FIX: multiply by 4.184.
     - Method bug: tagged source_type="unknown" and tier=C ("K-J formula")
       but HOF is an INPUT to K-J, not output. Actual method is Benson-
       style group-contribution (Ma et al. 2022 methodology).
       FIX: source_type = "group_contribution"; keep tier=C (empirical
       formula class, sibling of K-J for HOF).

  B. density source_type "unknown" on experimental rows (n=109):
     - emdb_v21_molecules_pubchem (108) and nist_cameo_enrichment (1).
     - These are literature-compiled experimental densities.
     - FIX: source_type = "compiled_observed". Tier A preserved.

  C. heat_of_formation source_type "unknown" on experimental rows (n=112):
     - emdb_v21_molecules_pubchem. Literature HOF values.
     - FIX: source_type = "compiled_observed". Tier A preserved.

  D. heat_of_formation source_type "unknown" on QSAR rows (n=2,438):
     - q-RASPR. QSAR-predicted HOF.
     - FIX: source_type = "qsar_predicted". Tier D preserved.

  E. detonation_velocity / detonation_pressure source_type "unknown" on
     cm4c01978_si_001 (n=11,956 DV + 11,982 DP):
     - K-J-formula-derived detonation properties.
     - FIX: source_type = "kj_calculated". Tier C preserved.

  F. detonation_velocity rows with NaN source_dataset (n=500):
     - 497 from denovo_sampling_rl, 3 from denovo_sampling_tl.
     - Row-level label_source_type="model_predicted" but per-property
       provenance was not populated.
     - FIX: propagate source_dataset and source_type from row-level fields,
       assign tier D.

Backups prior state to labeled_master_pre_label_audit.csv.bak.
"""
import shutil
from pathlib import Path
import pandas as pd

BASE   = Path("E:/Projects/EnergeticDiffusion2")
MASTER = BASE / "data/training/master/labeled_master.csv"
BACKUP = BASE / "data/training/master/labeled_master_pre_label_audit.csv.bak"

print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)
print(f"  {len(lm):,} rows")

# ── A. cm4c HOF: unit + tier/provenance ──────────────────────────────────────
KCAL_TO_KJ = 4.184
mask_a = (lm["heat_of_formation_source_dataset"] == "cm4c01978_si_001") & \
         lm["heat_of_formation"].notna()
n_a = int(mask_a.sum())
before_a = lm.loc[mask_a, "heat_of_formation"].describe()[["min","mean","max"]]
lm.loc[mask_a, "heat_of_formation"] = lm.loc[mask_a, "heat_of_formation"] * KCAL_TO_KJ
lm.loc[mask_a, "heat_of_formation_source_type"] = "group_contribution"
# tier stays "C"
after_a = lm.loc[mask_a, "heat_of_formation"].describe()[["min","mean","max"]]
print(f"\nA. cm4c HOF  fixed {n_a:,} rows")
print(f"   before (kcal/mol labelled kJ/mol): min={before_a['min']:.1f} mean={before_a['mean']:.1f} max={before_a['max']:.1f}")
print(f"   after   (kJ/mol):                  min={after_a['min']:.1f} mean={after_a['mean']:.1f} max={after_a['max']:.1f}")

# ── B. density: "unknown" → "compiled_observed" on experimental sources ──────
exp_density_sources = {"emdb_v21_molecules_pubchem", "nist_cameo_enrichment"}
mask_b = lm["density_source_dataset"].isin(exp_density_sources) & \
         (lm["density_source_type"] == "unknown")
n_b = int(mask_b.sum())
lm.loc[mask_b, "density_source_type"] = "compiled_observed"
print(f"B. density source_type 'unknown' → 'compiled_observed':  {n_b:,} rows")

# ── C. HOF: "unknown" → "compiled_observed" on emdb ──────────────────────────
mask_c = (lm["heat_of_formation_source_dataset"] == "emdb_v21_molecules_pubchem") & \
         (lm["heat_of_formation_source_type"] == "unknown")
n_c = int(mask_c.sum())
lm.loc[mask_c, "heat_of_formation_source_type"] = "compiled_observed"
print(f"C. HOF source_type 'unknown' → 'compiled_observed' (emdb):  {n_c:,} rows")

# ── D. HOF: "unknown" → "qsar_predicted" on q-RASPR ──────────────────────────
mask_d = (lm["heat_of_formation_source_dataset"] == "q-RASPR") & \
         (lm["heat_of_formation_source_type"] == "unknown")
n_d = int(mask_d.sum())
lm.loc[mask_d, "heat_of_formation_source_type"] = "qsar_predicted"
print(f"D. HOF source_type 'unknown' → 'qsar_predicted' (q-RASPR):  {n_d:,} rows")

# ── E. D / P: "unknown" → "kj_calculated" on cm4c ────────────────────────────
n_e_dv = 0
n_e_dp = 0
for prop in ("detonation_velocity", "detonation_pressure"):
    m = (lm[f"{prop}_source_dataset"] == "cm4c01978_si_001") & \
        (lm[f"{prop}_source_type"] == "unknown")
    n = int(m.sum())
    lm.loc[m, f"{prop}_source_type"] = "kj_calculated"
    if prop == "detonation_velocity":
        n_e_dv = n
    else:
        n_e_dp = n
print(f"E. cm4c D/P source_type 'unknown' → 'kj_calculated':  DV={n_e_dv:,}  DP={n_e_dp:,}")

# ── F. DV rows with NaN source_dataset (denovo predictors) ───────────────────
mask_f = lm["detonation_velocity"].notna() & \
         lm["detonation_velocity_source_dataset"].isna() & \
         (lm["label_source_type"] == "model_predicted")
n_f = int(mask_f.sum())
# Copy from row-level source_dataset
lm.loc[mask_f, "detonation_velocity_source_dataset"] = lm.loc[mask_f, "source_dataset"]
lm.loc[mask_f, "detonation_velocity_source_type"]    = "model_predicted"
if "detonation_velocity_tier" in lm.columns:
    lm.loc[mask_f, "detonation_velocity_tier"]       = "D"
print(f"F. DV NaN source_dataset filled from row-level (denovo):  {n_f:,} rows")

# ── recompute row-level tier ─────────────────────────────────────────────────
PROPS = ["density", "heat_of_formation", "detonation_velocity",
         "detonation_pressure", "explosion_heat"]
rank = {"A":0, "B":1, "C":2, "D":3, None:99}
def row_tier(r):
    cands = [r[f"{p}_tier"] for p in PROPS if f"{p}_tier" in r.index]
    cands = [c for c in cands if c in ("A","B","C","D")]
    if not cands: return None
    return min(cands, key=lambda c: rank[c])
lm["tier"] = lm.apply(row_tier, axis=1)

print("\nRow-level tier distribution (post-fix):")
print(lm["tier"].value_counts(dropna=False))

print("\nPer-property source_type 'unknown' remaining:")
for p in PROPS:
    c = (lm[f"{p}_source_type"] == "unknown").sum()
    if c:
        print(f"  {p}: {c:,}")

print(f"\nSaving to {MASTER} …")
lm.to_csv(MASTER, index=False)
print(f"Done. Backup at {BACKUP}")
