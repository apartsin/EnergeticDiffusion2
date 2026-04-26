"""
Generalized Kamlet-Jacobs imputation:

For every row in labeled_master that has
  density  (any tier ≥ A/B/C/D)
  heat_of_formation  (any tier)
and is MISSING detonation_velocity / detonation_pressure / explosion_heat,
compute the missing targets from K-J formula using that row's own rho and HOF.

The resulting values are tagged:
  *_source_dataset = "kj_row_impute"
  *_source_type    = "kj_calculated"
  *_tier           = "C"  (regardless of input tiers — formula dominates)

Rationale: for a novel generated molecule, K-J from its own (rho, HOF) gives a
physics-based detonation estimate that is more trustworthy than any trained
ML predictor evaluated off-distribution. See project docs for the tier
reasoning (K-J > ML for extrapolation).

CHNO only — skips rows with halogens/metals (product hierarchy breaks).
Works in input units:
  rho   g/cm³
  HOF kJ/mol (master-wide convention after 3DCNN unit fix)
  D   km/s (output)
  P   GPa  (output)
  Q   MJ/kg (output; internal calc in cal/g × 4.184e-3)
"""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

BASE   = Path("E:/Projects/EnergeticDiffusion2")
MASTER = BASE / "data/training/master/labeled_master.csv"
BACKUP = BASE / "data/training/master/labeled_master_pre_kj_general.csv.bak"

HOF_H2O = -57798.0   # cal/mol
HOF_CO2 = -94051.0
HOF_CO  = -26416.0
KCAL_TO_CAL = 1000
KJ_TO_CAL   = 239.006

PROPS = ["density", "heat_of_formation", "detonation_velocity",
         "detonation_pressure", "explosion_heat"]

def chno_counts(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    counts = {"C":0, "H":0, "N":0, "O":0, "OTHER":0}
    for atom in m.GetAtoms():
        s = atom.GetSymbol()
        counts[s if s in counts else "OTHER"] += 1
    return counts

def kj_params(a, b, g, d, hof_cal_mol):
    MW = 12.011*a + 1.008*b + 14.007*g + 15.999*d
    if MW <= 0:
        return None
    n2 = g / 2.0
    h_rem, o_rem, c_rem = b, d, a
    h2o = min(h_rem/2.0, o_rem); o_rem -= h2o; h_rem -= 2*h2o
    co2 = min(c_rem, o_rem/2.0); o_rem -= 2*co2; c_rem -= co2
    co  = min(c_rem, o_rem);     o_rem -= co;    c_rem -= co
    o2  = max(o_rem/2.0, 0.0)
    n_gas = n2 + h2o + co2 + co + o2
    if n_gas <= 0:
        return None
    N = n_gas / MW
    mass_gas = n2*28 + h2o*18 + co2*44 + co*28 + o2*32
    M = mass_gas / n_gas
    Q = (hof_cal_mol - (h2o*HOF_H2O + co2*HOF_CO2 + co*HOF_CO)) / MW
    if Q <= 0:
        return None
    phi = N * np.sqrt(M * Q)
    return N, M, Q, phi

def kj_D(phi, rho): return 1.01 * np.sqrt(phi) * (1 + 1.3*rho)
def kj_P(phi, rho): return 1.558 * rho**2 * phi

print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)

# Eligibility: has rho and HOF, valid density
elig = (lm["density"].notna() & (lm["density"] > 0.3) &
        lm["heat_of_formation"].notna())
print(f"Eligible rows (rho + HOF present): {elig.sum():,}")

missing_D = elig & lm["detonation_velocity"].isna()
missing_P = elig & lm["detonation_pressure"].isna()
missing_Q = elig & lm["explosion_heat"].isna()
print(f"  missing D: {missing_D.sum():,}")
print(f"  missing P: {missing_P.sum():,}")
print(f"  missing Q: {missing_Q.sum():,}")

n_D = n_P = n_Q = n_nonCHNO = n_math = 0

for idx in lm.index[elig]:
    if not (pd.isna(lm.at[idx,"detonation_velocity"]) or
            pd.isna(lm.at[idx,"detonation_pressure"]) or
            pd.isna(lm.at[idx,"explosion_heat"])):
        continue
    counts = chno_counts(lm.at[idx,"smiles"])
    if counts is None or counts["OTHER"] > 0:
        n_nonCHNO += 1
        continue
    a,b,g,d = counts["C"], counts["H"], counts["N"], counts["O"]
    hof_cal = lm.at[idx,"heat_of_formation"] * KJ_TO_CAL
    res = kj_params(a, b, g, d, hof_cal)
    if res is None:
        n_math += 1
        continue
    _, _, Q, phi = res
    rho = lm.at[idx, "density"]

    if pd.isna(lm.at[idx, "detonation_velocity"]):
        lm.at[idx, "detonation_velocity"]               = kj_D(phi, rho)
        lm.at[idx, "detonation_velocity_source_dataset"] = "kj_row_impute"
        lm.at[idx, "detonation_velocity_source_type"]    = "kj_calculated"
        lm.at[idx, "detonation_velocity_tier"]           = "C"
        n_D += 1
    if pd.isna(lm.at[idx, "detonation_pressure"]):
        lm.at[idx, "detonation_pressure"]               = kj_P(phi, rho)
        lm.at[idx, "detonation_pressure_source_dataset"] = "kj_row_impute"
        lm.at[idx, "detonation_pressure_source_type"]    = "kj_calculated"
        lm.at[idx, "detonation_pressure_tier"]           = "C"
        n_P += 1
    if pd.isna(lm.at[idx, "explosion_heat"]):
        lm.at[idx, "explosion_heat"]               = Q * 4.184e-3
        lm.at[idx, "explosion_heat_source_dataset"] = "kj_row_impute"
        lm.at[idx, "explosion_heat_source_type"]    = "kj_calculated"
        lm.at[idx, "explosion_heat_tier"]           = "C"
        n_Q += 1

print("\n=== Imputation summary ===")
print(f"D filled: {n_D:,}")
print(f"P filled: {n_P:,}")
print(f"Q filled: {n_Q:,}")
print(f"Skipped non-CHNO: {n_nonCHNO:,}")
print(f"Skipped math: {n_math:,}")

# Recompute row-level tier
rank = {"A":0,"B":1,"C":2,"D":3,None:99}
def row_tier(r):
    cands = [r[f"{p}_tier"] for p in PROPS if f"{p}_tier" in r.index]
    cands = [c for c in cands if c in ("A","B","C","D")]
    if not cands: return None
    return min(cands, key=lambda c: rank[c])
lm["tier"] = lm.apply(row_tier, axis=1)

print("\nRow-level tier distribution (post-impute):")
print(lm["tier"].value_counts(dropna=False))

print("\nPer-property coverage (post-impute):")
for p in PROPS:
    print(f"  {p:22s} filled: {lm[p].notna().sum():,}")

print(f"\nSaving to {MASTER} …")
lm.to_csv(MASTER, index=False)
print(f"Done. Backup at {BACKUP}")
