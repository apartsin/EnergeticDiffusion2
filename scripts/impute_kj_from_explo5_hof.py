"""
Task #5: compute Kamlet-Jacobs (D, P) from EXPLO5-derived HOF + density.

Scope: rows in labeled_master where
  - heat_of_formation is Tier B (from train_set/test_set = EXPLO5 thermochem)
  - density is Tier A (XRD experimental) or Tier B (EXPLO5)
  - detonation_velocity OR detonation_pressure is missing

Rationale: EXPLO5 runs its own thermochemistry to compute D/P but we already
have HOF from it — we can re-derive D/P via the K-J formula independently and
fill gaps.

Formulas for CHNO with oxygen balance hierarchy N2 → H2O → CO2 → CO → C(s):
  N (mol gas / g), M (avg MW), Q (cal/g) → φ = N·√(M·Q)
  D (km/s) = 1.01 · √φ · (1 + 1.3·ρ)
  P (GPa)  = 1.558 · ρ² · φ

Q from reactant HOF and product HOFs:
  Q (cal/g) = [Σ(n_i · ΔHf_i[product]) − ΔHf(reactant)] / MW
  Reference ΔHf (cal/mol): H2O(g)=−57798, CO2=−94051, CO=−26416, N2=0, C=0
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
BACKUP = BASE / "data/training/master/labeled_master_pre_kj_impute.csv.bak"

# reference HOFs (cal/mol)
HOF_H2O = -57798.0
HOF_CO2 = -94051.0
HOF_CO  = -26416.0

def chno_counts(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    counts = {"C": 0, "H": 0, "N": 0, "O": 0, "OTHER": 0}
    for atom in m.GetAtoms():
        s = atom.GetSymbol()
        counts[s if s in counts else "OTHER"] += 1
    return counts

def kj_params(a, b, g, d, hof_reactant_cal_per_mol):
    """Returns (N, M, Q, phi) for CaHbNgOd with given reactant HOF (cal/mol)."""
    MW = 12.011*a + 1.008*b + 14.007*g + 15.999*d
    if MW <= 0:
        return None

    # Product hierarchy: N→N2, H→H2O, C→CO2, then CO, then C(s); excess O as O2
    n2   = g / 2.0
    h_rem = b
    o_rem = d
    c_rem = a

    h2o = min(h_rem / 2.0, o_rem)
    o_rem -= h2o
    h_rem -= 2 * h2o

    co2 = min(c_rem, o_rem / 2.0)
    o_rem -= 2 * co2
    c_rem -= co2

    co = min(c_rem, o_rem)
    o_rem -= co
    c_rem -= co

    o2 = max(o_rem / 2.0, 0.0)

    n_gas_mol = n2 + h2o + co2 + co + o2
    if n_gas_mol <= 0:
        return None

    N = n_gas_mol / MW          # mol gas / g
    mass_gas = n2*28 + h2o*18 + co2*44 + co*28 + o2*32
    M = mass_gas / n_gas_mol    # average MW of gases

    # Heat released Q = ΔHf(reactant) − ΣΔHf(products). For exothermic: positive.
    Q_num = hof_reactant_cal_per_mol - (h2o*HOF_H2O + co2*HOF_CO2 + co*HOF_CO)
    Q = Q_num / MW              # cal/g
    if Q <= 0:
        return None

    phi = N * np.sqrt(M * Q)
    return N, M, Q, phi

def kj_D(phi, rho):
    return 1.01 * np.sqrt(phi) * (1 + 1.3 * rho)

def kj_P(phi, rho):
    return 1.558 * rho**2 * phi

# ── load master ───────────────────────────────────────────────────────────────
print(f"Backing up to {BACKUP} …")
shutil.copy2(MASTER, BACKUP)

print("Loading labeled_master.csv …")
lm = pd.read_csv(MASTER, low_memory=False)
print(f"  {len(lm):,} rows")

# target rows: EXPLO5 HOF (train_set/test_set source), valid CHNO, density present
mask_eligible = (
    lm["heat_of_formation_source_dataset"].isin(["train_set", "test_set"]) &
    lm["heat_of_formation"].notna() &
    lm["density"].notna() &
    (lm["density"] > 0.3)
)
print(f"Eligible rows (EXPLO5 HOF + density): {mask_eligible.sum():,}")

# HOF in master is kJ/mol — convert to cal/mol
# (verify by spot-checking range)
hof_range = lm.loc[mask_eligible, "heat_of_formation"].describe()
print(f"\nEXPLO5 HOF range (master units, assumed kJ/mol):")
print(hof_range)

KJ_TO_CAL = 239.006  # 1 kJ = 239.006 cal

n_D_filled = 0
n_P_filled = 0
n_Q_filled = 0
n_skip_nonCHNO = 0
n_skip_math = 0

for idx in lm.index[mask_eligible]:
    smi = lm.at[idx, "smiles"]
    counts = chno_counts(smi)
    if counts is None or counts["OTHER"] > 0:
        n_skip_nonCHNO += 1
        continue
    a, b, g, d = counts["C"], counts["H"], counts["N"], counts["O"]
    hof_cal_mol = lm.at[idx, "heat_of_formation"] * KJ_TO_CAL
    res = kj_params(a, b, g, d, hof_cal_mol)
    if res is None:
        n_skip_math += 1
        continue
    _, _, Q, phi = res
    rho = lm.at[idx, "density"]
    D = kj_D(phi, rho)
    P = kj_P(phi, rho)

    # Fill missing D
    if pd.isna(lm.at[idx, "detonation_velocity"]):
        lm.at[idx, "detonation_velocity"]               = D
        lm.at[idx, "detonation_velocity_source_dataset"] = "kj_from_explo5_hof"
        lm.at[idx, "detonation_velocity_source_type"]    = "kj_calculated"
        if "detonation_velocity_tier" in lm.columns:
            lm.at[idx, "detonation_velocity_tier"]       = "C"
        n_D_filled += 1

    # Fill missing P
    if pd.isna(lm.at[idx, "detonation_pressure"]):
        lm.at[idx, "detonation_pressure"]               = P
        lm.at[idx, "detonation_pressure_source_dataset"] = "kj_from_explo5_hof"
        lm.at[idx, "detonation_pressure_source_type"]    = "kj_calculated"
        if "detonation_pressure_tier" in lm.columns:
            lm.at[idx, "detonation_pressure_tier"]       = "C"
        n_P_filled += 1

    # Fill missing explosion_heat (Q is in cal/g; store as MJ/kg)
    if pd.isna(lm.at[idx, "explosion_heat"]):
        lm.at[idx, "explosion_heat"]               = Q * 4.184e-3   # cal/g → MJ/kg
        lm.at[idx, "explosion_heat_source_dataset"] = "kj_from_explo5_hof"
        lm.at[idx, "explosion_heat_source_type"]    = "kj_calculated"
        if "explosion_heat_tier" in lm.columns:
            lm.at[idx, "explosion_heat_tier"]       = "C"
        n_Q_filled += 1

print("\n=== Imputation summary ===")
print(f"D filled:  {n_D_filled:,}")
print(f"P filled:  {n_P_filled:,}")
print(f"Q filled:  {n_Q_filled:,}")
print(f"Skipped (non-CHNO): {n_skip_nonCHNO:,}")
print(f"Skipped (math): {n_skip_math:,}")

# Update row-level tier (since we added new property tiers)
rank = {"A": 0, "B": 1, "C": 2, None: 99}
PROPS = ["density", "heat_of_formation", "detonation_velocity",
         "detonation_pressure", "explosion_heat"]
if all(f"{p}_tier" in lm.columns for p in PROPS):
    def row_min_tier(row):
        cands = [row[f"{p}_tier"] for p in PROPS]
        cands = [c for c in cands if c in ("A", "B", "C")]
        if not cands:
            return None
        return min(cands, key=lambda c: rank[c])
    lm["tier"] = lm.apply(row_min_tier, axis=1)

print(f"\nSaving to {MASTER} …")
lm.to_csv(MASTER, index=False)
print(f"Done. Backup at {BACKUP}")
