"""D9: K-J residual stratified by N-fraction.

For each row with experimental (Tier-A) density + HOF + D, compute the
Kamlet-Jacobs predicted D from the formula, take residual = D_KJ - D_exp,
then stratify by atomic N-fraction.

Supports the paper §5.13 attribution: the K-J formula systematically
under-predicts D in the high-N regime where its empirical constants were
not fit.

Output: results/d9_kj_nfrac_table.json + d9_kj_nfrac.md
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True, parents=True)
LM = pd.read_csv("m6_postprocess_bundle/labelled_master.csv", low_memory=False)


def kamlet_jacobs(rho, hof_kjmol, n_C, n_H, n_N, n_O):
    """Kamlet-Jacobs D estimate (km/s).
    Uses the standard: D = 1.01 * sqrt(N * M^0.5 * Q^0.5) * (1 + 1.30 * rho)
    with N moles gas / g of explosive, M average MW of gas, Q heat of det cal/g.
    """
    if n_O >= 2 * n_C + n_H / 2:
        N_moles = (n_N / 2 + n_H / 2 + n_O - n_C) / (12 * n_C + n_H + 14 * n_N + 16 * n_O)
        M = (4 * n_C + 28 * n_N + 18 * (n_H / 2) + 32 * (n_O - 2 * n_C - n_H / 2)) / max(n_N + n_H / 2 + n_O - n_C, 1e-9)
    elif n_O >= n_C + n_H / 2:
        N_moles = (n_N / 2 + n_H / 2 + n_O / 2) / (12 * n_C + n_H + 14 * n_N + 16 * n_O)
        M = (4 * n_C + 28 * n_N + 18 * (n_H / 2) + 16 * (n_O - n_C - n_H / 2)) / max(n_N + n_H / 2 + n_O / 2, 1e-9)
    else:
        N_moles = (n_N / 2 + n_O / 2 + n_H / 4) / (12 * n_C + n_H + 14 * n_N + 16 * n_O)
        M = (12 * (n_C - n_O / 2 - n_H / 4) + 28 * n_N + 2 * (n_H / 4) + 18 * (n_O / 2)) / max(n_N / 2 + n_O / 2 + n_H / 4, 1e-9)
    M_total = 12 * n_C + n_H + 14 * n_N + 16 * n_O
    if M_total <= 0:
        return None
    Q = max(hof_kjmol * 1000 / M_total / 4.184, 100.0)  # cal/g, clamp
    arg = N_moles * np.sqrt(M) * np.sqrt(Q)
    if arg <= 0:
        return None
    D = 1.01 * np.sqrt(arg) * (1 + 1.30 * rho)
    return D


def count(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    nc = nh = nn = no = 0
    for a in m.GetAtoms():
        s = a.GetSymbol()
        if s == "C": nc += 1
        elif s == "N": nn += 1
        elif s == "O": no += 1
        nh += a.GetTotalNumHs()
    return nc, nh, nn, no


print(f"[d9] labelled-master rows: {len(LM)}")
m = LM[LM["detonation_velocity_tier"] == "A"][
    ["smiles", "density", "heat_of_formation", "detonation_velocity"]
].dropna().reset_index(drop=True)
print(f"[d9] Tier-A D + rho + HOF rows: {len(m)}")

m["counts"] = m["smiles"].map(count)
m = m.dropna(subset=["counts"]).reset_index(drop=True)
m["c_count"] = m["counts"].map(lambda x: x[0])
m["h_count"] = m["counts"].map(lambda x: x[1])
m["n_count"] = m["counts"].map(lambda x: x[2])
m["o_count"] = m["counts"].map(lambda x: x[3])
m["total"] = m[["c_count", "h_count", "n_count", "o_count"]].sum(axis=1)
m = m[m["total"] > 0].reset_index(drop=True)
m["N_frac_atom"] = m["n_count"] / m["total"]

# Kamlet-Jacobs prediction per row
def predict(row):
    return kamlet_jacobs(row["density"], row["heat_of_formation"],
                          row["c_count"], row["h_count"],
                          row["n_count"], row["o_count"])

m["D_kj"] = m.apply(predict, axis=1)
m = m.dropna(subset=["D_kj"]).reset_index(drop=True)
m["resid"] = m["D_kj"] - m["detonation_velocity"]
print(f"[d9] rows with valid K-J: {len(m)}")

# Stratify by N-fraction bins
bins = [0.0, 0.10, 0.20, 0.30, 0.40, 0.55, 1.01]
labels = ["[0.00-0.10)", "[0.10-0.20)", "[0.20-0.30)", "[0.30-0.40)",
          "[0.40-0.55)", "[0.55-1.00]"]
m["bin"] = pd.cut(m["N_frac_atom"], bins=bins, labels=labels, right=False, include_lowest=True)

table = []
for lbl in labels:
    sub = m[m["bin"] == lbl]
    if len(sub) == 0:
        continue
    table.append({
        "N_frac_bin": lbl,
        "n": int(len(sub)),
        "mean_D_exp_kms": float(sub["detonation_velocity"].mean()),
        "mean_D_kj_kms": float(sub["D_kj"].mean()),
        "mean_resid_kms": float(sub["resid"].mean()),
        "median_resid_kms": float(sub["resid"].median()),
        "std_resid_kms": float(sub["resid"].std()),
        "rmse_kms": float(np.sqrt((sub["resid"] ** 2).mean())),
    })
overall = {
    "N_frac_bin": "ALL",
    "n": int(len(m)),
    "mean_D_exp_kms": float(m["detonation_velocity"].mean()),
    "mean_D_kj_kms": float(m["D_kj"].mean()),
    "mean_resid_kms": float(m["resid"].mean()),
    "median_resid_kms": float(m["resid"].median()),
    "std_resid_kms": float(m["resid"].std()),
    "rmse_kms": float(np.sqrt((m["resid"] ** 2).mean())),
}
table.append(overall)

print()
print(f"{'bin':<14} {'n':>5} {'D_exp':>8} {'D_kj':>8} {'mean_r':>8} {'rmse':>8}")
for row in table:
    print(f"{row['N_frac_bin']:<14} {row['n']:>5} "
          f"{row['mean_D_exp_kms']:>8.3f} {row['mean_D_kj_kms']:>8.3f} "
          f"{row['mean_resid_kms']:>+8.3f} {row['rmse_kms']:>8.3f}")

# Pearson correlation: residual vs N-fraction
from scipy.stats import pearsonr, spearmanr
r, p = pearsonr(m["N_frac_atom"], m["resid"])
rs, ps = spearmanr(m["N_frac_atom"], m["resid"])
print(f"\nPearson r(N_frac, resid)  = {r:+.3f}  (p={p:.2e})")
print(f"Spearman r(N_frac, resid) = {rs:+.3f}  (p={ps:.2e})")

(OUT_DIR / "d9_kj_nfrac_table.json").write_text(json.dumps({
    "n_total": int(len(m)),
    "table": table,
    "pearson_resid_vs_Nfrac": {"r": r, "p": p},
    "spearman_resid_vs_Nfrac": {"r": rs, "p": ps},
}, indent=2))
print(f"[d9] -> {OUT_DIR / 'd9_kj_nfrac_table.json'}")
print("[d9] === DONE ===")
