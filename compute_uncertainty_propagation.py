"""Compute per-lead uncertainty propagation through the 6-anchor calibration → K-J chain.

Propagates LOO calibration errors (delta_rho = 0.078 g/cm³, delta_HOF = 64.6 kJ/mol)
through the K-J sensitivity slopes to obtain delta_D and delta_P for each lead.

Outputs:
  results/uncertainty_propagation.json
  Prints HTML <table> for pasting into the paper.
"""
from __future__ import annotations
import json, os, math, re
import numpy as np

ROOT      = os.path.dirname(__file__)
LEAD_DIR  = os.path.join(ROOT, "m2_bundle", "results")
CALIB_6A  = os.path.join(LEAD_DIR, "m2_calibration_6anchor.json")
OUT_JSON  = os.path.join(ROOT, "results", "uncertainty_propagation.json")

# LOO calibration errors (from m2_calibration_6anchor.json loo_residual)
LOO_RHO  = 0.07797   # g/cm³
LOO_HOF  = 64.599    # kJ/mol

# --------------------------------------------------------------------------
# K-J formula and analytic sensitivities
# --------------------------------------------------------------------------
def kj_phi_params(formula: str, hof_kJmol: float):
    """Return (N, M, Q, branch) for K-J computation."""
    cnt = {}
    for sym, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if sym: cnt[sym] = int(n) if n else 1
    a = cnt.get("C", 0); b = cnt.get("H", 0)
    c = cnt.get("N", 0); d = cnt.get("O", 0)
    M_total = 12.011*a + 1.008*b + 14.007*c + 15.999*d
    if M_total == 0: return None

    if d >= 2*a + b/2:          # oxygen-rich
        N = (b + 2*c + 2*d) / (4*M_total)
        M = (56*d + 88*c - 8*b) / (b + 2*c + 2*d)
        Q = (28900*b + 47000*(d - a - b/2) + 239.0*hof_kJmol) / M_total
        dQ_dHOF = 239.0 / M_total          # cal/g per kJ/mol
        branch = "O-rich"
    elif 2*a + b/2 > d >= b/2:  # carbon-rich (most CHNO energetics)
        N = (b + 2*c + 2*d) / (4*M_total)
        M = (2*b + 28*c + 32*d) / (b + 2*c + 2*d)
        raw = (57800*d - 239.0*hof_kJmol) / M_total
        Q = abs(raw)
        sign = -1.0 if raw >= 0 else +1.0  # Q = |raw|, so dQ/dHOF flips if raw<0
        dQ_dHOF = sign * (-239.0) / M_total
        branch = "C-rich"
    else:
        return None

    if N <= 0 or M <= 0 or Q <= 0: return None
    phi = N * math.sqrt(M) * math.sqrt(Q)
    return N, M, Q, phi, dQ_dHOF, branch


def kj_dp(rho, phi):
    D = 1.01 * math.sqrt(phi) * (1 + 1.30 * rho)
    P = 1.558 * rho**2 * phi
    return D, P


def kj_sensitivities(rho, hof_kJmol, formula):
    """Return dD/drho, dD/dHOF, dP/drho, dP/dHOF via analytic differentiation."""
    res = kj_phi_params(formula, hof_kJmol)
    if res is None: return None
    N, M, Q, phi, dQ_dHOF, branch = res

    D, P = kj_dp(rho, phi)

    # phi = N * sqrt(M) * sqrt(Q)
    # dphi/dQ = N * sqrt(M) * 0.5/sqrt(Q) = phi / (2*Q)
    dphi_dQ = phi / (2 * Q)
    dphi_dHOF = dphi_dQ * dQ_dHOF

    # dD/drho (at fixed phi): D = 1.01*sqrt(phi)*(1+1.3*rho) → dD/drho = 1.01*sqrt(phi)*1.3
    dD_drho = 1.01 * math.sqrt(phi) * 1.30

    # dD/dHOF (via phi): D = 1.01*sqrt(phi)*(1+1.3*rho) → dD/dphi = 0.5*D/phi
    dD_dHOF = (0.5 * D / phi) * dphi_dHOF

    # dP/drho (at fixed phi): P = 1.558*rho²*phi → dP/drho = 2*1.558*rho*phi = 2P/rho
    dP_drho = 2 * P / rho

    # dP/dHOF: P = 1.558*rho²*phi → dP/dphi = P/phi
    dP_dHOF = (P / phi) * dphi_dHOF

    return {
        "dD_drho": dD_drho,
        "dD_dHOF_per_kJmol": dD_dHOF,
        "dP_drho": dP_drho,
        "dP_dHOF_per_kJmol": dP_dHOF,
        "D_kms": D,
        "P_GPa": P,
        "branch": branch,
    }


def propagate(sens, delta_rho=LOO_RHO, delta_HOF=LOO_HOF):
    """Propagate calibration errors in quadrature."""
    dD = math.sqrt((sens["dD_drho"] * delta_rho)**2 +
                   (sens["dD_dHOF_per_kJmol"] * delta_HOF)**2)
    dP = math.sqrt((sens["dP_drho"] * delta_rho)**2 +
                   (sens["dP_dHOF_per_kJmol"] * delta_HOF)**2)
    return dD, dP


# --------------------------------------------------------------------------
# Load leads and calibration
# --------------------------------------------------------------------------
calib   = json.load(open(CALIB_6A))
a_rho   = calib["a_rho"]
b_rho   = calib["b_rho"]
c_hof   = calib["c_hof_kJmol"]

summary = json.load(open(os.path.join(LEAD_DIR, "m2_summary.json")))
by_id   = {r["id"]: r for r in summary}

import glob
rows = []
for f in sorted(glob.glob(os.path.join(LEAD_DIR, "m2_lead_L*.json"))):
    d      = json.load(open(f))
    lid    = d["id"]
    s      = by_id.get(lid, {})
    formula = d.get("formula", "")

    rho_dft = s.get("rho_dft")
    hof_raw = s.get("HOF_kJmol_wb97xd")
    if rho_dft is None or hof_raw is None:
        continue

    rho_cal = a_rho * rho_dft + b_rho
    hof_cal = hof_raw + c_hof

    sens = kj_sensitivities(rho_cal, hof_cal, formula)
    if sens is None:
        rows.append({"id": lid, "formula": formula, "note": "K-J branch undefined"})
        continue

    dD, dP = propagate(sens)

    rows.append({
        "id": lid,
        "formula": formula,
        "rho_cal": round(rho_cal, 3),
        "hof_cal": round(hof_cal, 1),
        "D_cal": round(sens["D_kms"], 2),
        "P_cal": round(sens["P_GPa"], 1),
        "dD_drho": round(sens["dD_drho"], 3),
        "dD_dHOF": round(sens["dD_dHOF_per_kJmol"], 4),
        "dP_drho": round(sens["dP_drho"], 2),
        "dP_dHOF": round(sens["dP_dHOF_per_kJmol"], 4),
        "delta_D_kms": round(dD, 3),
        "delta_P_GPa": round(dP, 2),
        "branch": sens["branch"],
    })

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(rows, open(OUT_JSON, "w"), indent=2)
print(f"Saved: {OUT_JSON}\n")

# --------------------------------------------------------------------------
# Print summary and HTML table
# --------------------------------------------------------------------------
header = ["ID", "Formula", "ρ_cal", "HOF_cal", "D_cal ± δD", "P_cal ± δP",
          "∂D/∂ρ", "∂D/∂HOF×100", "branch"]
print(f"{'ID':>4}  {'rho':>6}  {'HOF':>8}  {'D+/-dD':>14}  {'P+/-dP':>12}  branch")
print("-" * 65)
for r in rows:
    if "note" in r:
        print(f"{r['id']:>4}  {'':>6}  {'':>8}  {'undef':>14}  {'undef':>12}  undef")
        continue
    d_str = f"{r['D_cal']:.2f}+/-{r['delta_D_kms']:.2f}"
    p_str = f"{r['P_cal']:.1f}+/-{r['delta_P_GPa']:.1f}"
    print(f"{r['id']:>4}  {r['rho_cal']:>6.3f}  {r['hof_cal']:>8.1f}  {d_str:>14}  {p_str:>12}  {r['branch']}")

# HTML table
print("\n\n<!-- HTML TABLE FOR PAPER -->\n")
cap = (
    r'<caption><strong>Table 7c.</strong> Per-lead calibration-propagated uncertainty on '
    r'K&ndash;J detonation velocity \(D\) and pressure \(P\). \(\delta D\) and \(\delta P\) '
    r'are obtained by propagating the 6-anchor LOO calibration errors '
    r'(\(\delta\rho_\mathrm{LOO}=0.078\)&nbsp;g/cm<sup>3</sup>, '
    r'\(\delta\mathrm{HOF}_\mathrm{LOO}=64.6\)&nbsp;kJ/mol) through the K&ndash;J analytic '
    r'sensitivity slopes (\(\partial D/\partial\rho\), \(\partial D/\partial\mathrm{HOF}\)) '
    r'in quadrature. All values under the 6-anchor calibration (Table&nbsp;7b). '
    r'The propagated uncertainty is <em>calibration-only</em>; the K&ndash;J formula '
    r'systematic bias (0.3&ndash;2&nbsp;km/s at high N-fraction, &sect;5.2.2) and the '
    r'3D-CNN surrogate error are separate sources not included here.</caption>'
)
thead = (
    '  <thead>\n    <tr>\n'
    '      <th>ID</th>\n'
    '      <th>&rho;<sub>cal</sub> (g/cm<sup>3</sup>)</th>\n'
    '      <th>HOF<sub>cal</sub> (kJ/mol)</th>\n'
    '      <th><em>D</em><sub>cal</sub> (km/s)</th>\n'
    '      <th>&delta;<em>D</em> (km/s)</th>\n'
    '      <th><em>P</em><sub>cal</sub> (GPa)</th>\n'
    '      <th>&delta;<em>P</em> (GPa)</th>\n'
    '      <th>&part;<em>D</em>/&part;&rho;</th>\n'
    '    </tr>\n  </thead>\n  <tbody>'
)
print('<table style="font-size:0.92em">\n  ' + cap + '\n' + thead)

for r in rows:
    if "note" in r:
        print(f'    <tr><td>{r["id"]}</td><td colspan="7"><em>K&ndash;J branch undefined (very C-rich)</em></td></tr>')
        continue
    print(f'    <tr><td>{r["id"]}</td>'
          f'<td>{r["rho_cal"]:.3f}</td>'
          f'<td>{r["hof_cal"]:+.1f}</td>'
          f'<td>{r["D_cal"]:.2f}</td>'
          f'<td>&plusmn;{r["delta_D_kms"]:.2f}</td>'
          f'<td>{r["P_cal"]:.1f}</td>'
          f'<td>&plusmn;{r["delta_P_GPa"]:.1f}</td>'
          f'<td>{r["dD_drho"]:.2f}</td>'
          f'</tr>')

print("  </tbody>\n</table>")
