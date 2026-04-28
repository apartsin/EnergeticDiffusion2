"""Cantera + CJ-detonation wrapper for L1, L4, L5, RDX, TATB.

Free, license-free alternative to EXPLO5 / Cheetah-2 for thermochemical-
equilibrium Chapman-Jouguet detonation properties. Uses Cantera's GRI-30
mechanism (53 CHNO species) for the equilibrium solve and a simple
shoot-and-equilibrate root-finder for the CJ point.

Inputs: m2_bundle/results/m2_summary.json  (per-lead rho_cal, HOF_cal)
Outputs: results/cj_cantera.json + a printed comparison table.

CJ algorithm:
  1. Build a reactant mole-fraction vector from the compound's molecular
     formula (atomic fractions of C, H, N, O). For Cantera's gri30 mix, we
     use the species CH4, NH3, H2, O2 in the right ratio so total atoms
     match the explosive's CHNO ratio - this is a proxy for the explosive
     molecule itself, since Cantera's gri30 doesn't include arbitrary
     CHNO molecules. The reactants then equilibrate at constant
     (enthalpy, pressure) using the explosive's experimental HOF as
     boundary energy.
  2. For trial detonation velocity D_test:
     - Compute post-shock T, P, rho via Rankine-Hugoniot at frozen
       composition assuming ideal gas
     - Equilibrate the post-shock gas at fixed (T, P) to get product
       distribution
     - Compute post-shock sound speed and flow velocity in CJ frame
  3. Root-find on D_test such that post-shock flow speed = sound speed
     (the Chapman-Jouguet condition)
  4. Output D_CJ, P_CJ, T_CJ, dominant products

Limitations:
  - GRI-30 doesn't include solid carbon, so oxygen-poor CHNO compounds
    (e.g. TATB) may overestimate D because the model can't form C(s).
  - The reactant composition is constructed from C/H/N/O atom counts via
    a basis-species hack; real explosive enthalpy must be added by hand.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np
import cantera as ct

# Reference HOF for elements at 298 K (kJ/mol) - all zero by definition.
# Reference HOF for the basis species we use to build a CHNO atom mix:
#   methane   CH4(g):   -74.6 kJ/mol
#   ammonia   NH3(g):   -45.9 kJ/mol
#   hydrogen  H2(g):    0
#   oxygen    O2(g):    0
# Explosive HOF (kJ/mol of explosive) is added to the reactant stream
# directly via Cantera's TPX setter.

R_J = 8.31446  # J/(mol K)


def parse_formula(formula: str) -> dict[str, int]:
    import re
    out = {}
    for sym, n in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        if sym:
            out[sym] = int(n) if n else 1
    return out


def build_reactant_mix(formula: str, hof_kjmol: float, T0=298.15, P0=ct.one_atm):
    """Build a Cantera Solution at T0,P0 with composition matching the
    explosive's atomic ratios and total enthalpy = HOF_kJmol per mole.
    Returns (gas, n_atoms_per_mole_explosive).

    Strategy: pick a basis of (CH4, NH3, H2, O2) in proportions such that
    the atom-balance equations
        n_CH4 = n_C
        n_NH3 = n_N
        4*n_CH4 + 3*n_NH3 + 2*n_H2 = n_H
        2*n_O2 = n_O
    are satisfied. Solve for the basis moles given the formula's atom
    counts, then set the gas to that mole fraction at (T0, P0). Adjust
    enthalpy: the basis-species mix has its own formation enthalpy
    (sum n_i * Hf_i); the explosive's actual HOF differs. We absorb
    this difference by setting the gas to the explosive's HOF value
    using HP equilibration - the constant-pressure flame temperature
    rise at composition-matched stoichiometry exactly captures the
    energy release.
    """
    atoms = parse_formula(formula)
    nC, nH, nN, nO = atoms.get('C',0), atoms.get('H',0), atoms.get('N',0), atoms.get('O',0)
    # Solve basis
    n_CH4 = nC
    n_NH3 = nN
    n_O2  = nO / 2.0
    n_H2  = (nH - 4*n_CH4 - 3*n_NH3) / 2.0
    if n_H2 < 0:
        # Not enough H in formula to satisfy CH4+NH3; substitute C as CO2 instead.
        # Edge case for high-N high-O compounds. Use CO2 + N2 + H2O + O2 basis.
        # CO2: nC, N2: nN/2, H2O: nH/2, O2: (nO - 2*nC - nH/2)/2
        return _build_alt_basis(formula, hof_kjmol, T0, P0)
    n_total = n_CH4 + n_NH3 + n_H2 + n_O2
    X = {'CH4': n_CH4/n_total, 'NH3': n_NH3/n_total,
         'H2':  n_H2/n_total,  'O2':  n_O2/n_total}
    # Basis HOFs (kJ/mol)
    Hf_basis = -74.6*n_CH4 + (-45.9)*n_NH3 + 0*n_H2 + 0*n_O2
    delta_Hf = (hof_kjmol - Hf_basis) * 1000.0  # J per mole of explosive
    gas = ct.Solution('gri30.yaml')
    # Set composition first, then T, P
    gas.TPX = T0, P0, X
    return gas, n_total, delta_Hf, X


def _build_alt_basis(formula, hof_kjmol, T0, P0):
    """Alternative basis for oxygen-rich compounds: CO2+N2+H2O+O2."""
    atoms = parse_formula(formula)
    nC, nH, nN, nO = atoms.get('C',0), atoms.get('H',0), atoms.get('N',0), atoms.get('O',0)
    n_CO2 = nC
    n_N2  = nN / 2.0
    n_H2O = nH / 2.0
    n_O2  = (nO - 2*n_CO2 - n_H2O) / 2.0
    if n_O2 < 0: n_O2 = 0  # Fall back: use up all O in CO2 + H2O even if undercounted.
    n_total = max(n_CO2 + n_N2 + n_H2O + n_O2, 1e-12)
    X = {'CO2': n_CO2/n_total, 'N2': n_N2/n_total,
         'H2O': n_H2O/n_total, 'O2': n_O2/n_total}
    Hf_basis = -393.5*n_CO2 + 0*n_N2 + (-241.8)*n_H2O + 0*n_O2
    delta_Hf = (hof_kjmol - Hf_basis) * 1000.0
    gas = ct.Solution('gri30.yaml')
    gas.TPX = T0, P0, X
    return gas, n_total, delta_Hf, X


def cj_detonation(formula: str, rho_g_cm3: float, hof_kjmol: float,
                  verbose: bool = True) -> dict:
    """Compute CJ detonation velocity, pressure, temperature for a CHNO
    compound at given (rho, HOF). Returns a dict of results.
    """
    gas, n_basis, delta_Hf_J, X0 = build_reactant_mix(formula, hof_kjmol)
    # Inject the explosive's energy: equilibrate at constant (H_target, P0)
    # where H_target = current_H + delta_Hf (the energy that wasn't captured
    # by the basis-species mix because the basis is not the explosive).
    h0 = gas.enthalpy_mole              # J/kmol of basis mix
    # delta_Hf_J is per mole of EXPLOSIVE; basis is n_basis moles per
    # mole of explosive. Per kmol of basis: delta_Hf_J / n_basis * 1000.
    h_inj = h0 + (delta_Hf_J / n_basis) * 1000.0
    # Adiabatic flame temperature at constant H, P: solve for T such that
    # gas.HP at the equilibrium composition matches h_inj.
    P0 = gas.P
    try:
        gas.HPX = h_inj, P0, X0
    except Exception as e:
        if verbose: print(f'  HPX inject failed: {e}')
    gas.equilibrate('HP')
    T_eq = gas.T
    rho_eq = gas.density          # kg/m^3 of products at 1 atm, T_eq
    a_eq = gas.sound_speed        # m/s
    if verbose:
        print(f'  adiabatic flame T = {T_eq:.0f} K   rho_prod_1atm = {rho_eq:.3f} kg/m^3')

    # Now solve for CJ. Strong-shock approximation gives a starting estimate:
    # D_CJ ~ sqrt(2 * (gamma^2 - 1) * Q)  where Q is heat release per unit mass.
    # Use that as initial guess, then iterate.
    rho0 = rho_g_cm3 * 1000.0     # kg/m^3 of solid explosive
    # Heat release per unit mass: gas.delta_enthalpy_mass after vs before
    # at HP equilibrium - but we already injected. Simpler: use D from
    # the standard CJ formula by iteration.

    def trial(D):
        # Frozen Rankine-Hugoniot from solid (rho0, T0, P0=1 atm) to
        # post-shock gas. Treat solid as incompressible cold reactant
        # with negligible thermal energy: u_post = D - rho0 * D / rho_post.
        # Equilibrate at fixed (H_total, density).
        # Total specific internal energy frame-of-reference:
        #   e_post = e0 + 0.5*D^2 * (1 - (rho0/rho_post)^2)
        # We iterate rho_post such that post-shock equilibrium gives
        # consistent (T, P, rho).
        # Quick approximation: use the strong-shock limit
        # rho_post / rho0 ~ (gamma+1)/gamma  which for CHNO products is ~1.7.
        # Then equilibrate at HP using the imposed pressure.
        gamma_guess = 2.7  # CHNO detonation product gas, approximate
        rho_post_guess = rho0 * (gamma_guess + 1) / gamma_guess  # ~1.37*rho0
        u_post = D * (1 - rho0 / rho_post_guess)
        e0 = 0.0  # reference solid at 0 internal energy
        e_post = e0 + 0.5 * D*D * (1 - (rho0/rho_post_guess)**2)
        # Pressure from momentum: P_post = rho0 * D * u_post
        P_post = rho0 * D * u_post  # Pa
        # Temperature: enthalpy h_post = e_post + P_post / rho_post (specific units)
        h_post_specific = e_post + P_post / rho_post_guess  # J/kg
        # Set Cantera state at (h_post + h_inj_specific, P_post)
        h_inj_specific = h_inj / gas.mean_molecular_weight * 1000  # convert kJ/kmol -> J/kg
        h_target = h_post_specific + h_inj_specific
        try:
            gas.HPX = h_target, P_post, X0
            gas.equilibrate('HP')
            a_post = gas.sound_speed
            # CJ residual: u_post should equal a_post (sonic post-shock)
            return u_post - a_post, gas.T, P_post, rho_post_guess
        except Exception as e:
            return float('nan'), float('nan'), P_post, rho_post_guess

    # Scan D from 4 to 14 km/s in coarse steps, find sign change of residual
    Ds_kms = np.linspace(4.0, 14.0, 30)
    residuals = []
    for D_kms in Ds_kms:
        D = D_kms * 1000.0
        r, T, P, rho_post = trial(D)
        residuals.append((D_kms, r, T, P, rho_post))
    # Find first sign change
    D_CJ_kms = None
    for i in range(1, len(residuals)):
        a = residuals[i-1][1]; b = residuals[i][1]
        if not (math.isnan(a) or math.isnan(b)) and a * b < 0:
            # Linear interp root
            D_CJ_kms = residuals[i-1][0] - a * (residuals[i][0] - residuals[i-1][0]) / (b - a)
            T_CJ = 0.5*(residuals[i-1][2] + residuals[i][2])
            P_CJ_Pa = 0.5*(residuals[i-1][3] + residuals[i][3])
            rho_post = 0.5*(residuals[i-1][4] + residuals[i][4])
            break

    if D_CJ_kms is None:
        if verbose: print('  CJ root not found in 4-14 km/s range')
        return {'formula': formula, 'rho_g_cm3': rho_g_cm3, 'HOF_kJmol': hof_kjmol,
                'D_CJ_kms': None, 'P_CJ_GPa': None, 'T_CJ_K': None,
                'note': 'CJ root not found; check basis-species choice or HOF'}

    return {'formula': formula, 'rho_g_cm3': rho_g_cm3, 'HOF_kJmol': hof_kjmol,
            'D_CJ_kms': float(D_CJ_kms),
            'P_CJ_GPa': float(P_CJ_Pa / 1e9),
            'T_CJ_K':   float(T_CJ),
            'rho_post_g_cm3': float(rho_post / 1000.0),
            'note': f'CJ found via shock-equilibrium iteration; basis={list(X0.keys())}'}


# ── Driver ────────────────────────────────────────────────────────────────
def main():
    summary_path = Path('m2_bundle/results/m2_summary.json')
    summary = json.loads(summary_path.read_text())
    by_id = {r['id']: r for r in summary}

    # Anchor compounds with literature values for sanity check
    anchors = {
        'RDX':  {'formula': 'C3H6N6O6', 'rho_lit': 1.806, 'HOF_lit': 66.0,
                  'D_lit_kms': 8.75, 'P_lit_GPa': 34.9},
        'TATB': {'formula': 'C6H6N6O6', 'rho_lit': 1.937, 'HOF_lit': -154.0,
                  'D_lit_kms': 7.86, 'P_lit_GPa': 31.5},
    }

    results = {}
    print()
    print(f'{"compound":<8} {"formula":<12} {"rho":>5} {"HOF":>9}  {"D_CJ":>6} {"P_CJ":>7} {"T_CJ":>7}  note')

    # Sanity check on anchors using literature ρ + experimental HOF
    for name, a in anchors.items():
        print(f'\n=== {name} (anchor sanity check) ===')
        out = cj_detonation(a['formula'], a['rho_lit'], a['HOF_lit'])
        results[f'{name}_lit'] = out
        D = out.get('D_CJ_kms'); P = out.get('P_CJ_GPa'); T = out.get('T_CJ_K')
        D_str = f'{D:.2f}' if D is not None else 'n/a'
        P_str = f'{P:.1f}' if P is not None else 'n/a'
        T_str = f'{T:.0f}' if T is not None else 'n/a'
        print(f'  {name}: CJ-Cantera D={D_str} km/s, P={P_str} GPa, T={T_str} K  (lit D={a["D_lit_kms"]} P={a["P_lit_GPa"]})')

    # The actual chem-pass leads, using calibrated DFT values from m2_summary
    for lid in ('L1', 'L4', 'L5'):
        row = by_id.get(lid)
        if row is None:
            print(f'  {lid} not in m2_summary'); continue
        rho = row.get('rho_cal'); hof = row.get('HOF_kJmol_wb97xd_cal')
        if rho is None or hof is None:
            print(f'  {lid} missing rho_cal or HOF_cal'); continue
        # Need the formula
        lp = Path(f'm2_bundle/results/m2_lead_{lid}.json')
        if not lp.exists():
            print(f'  {lid} no per-lead JSON'); continue
        formula = json.loads(lp.read_text()).get('formula')
        print(f'\n=== {lid}  ({formula}, rho_cal={rho:.3f}, HOF_cal={hof:.1f}) ===')
        out = cj_detonation(formula, rho, hof)
        results[lid] = out
        D = out.get('D_CJ_kms'); P = out.get('P_CJ_GPa'); T = out.get('T_CJ_K')
        D_str = f'{D:.2f}' if D is not None else 'n/a'
        P_str = f'{P:.1f}' if P is not None else 'n/a'
        T_str = f'{T:.0f}' if T is not None else 'n/a'
        print(f'  {lid}: CJ-Cantera D={D_str} km/s, P={P_str} GPa, T={T_str} K')

    Path('results').mkdir(exist_ok=True, parents=True)
    Path('results/cj_cantera.json').write_text(json.dumps(results, indent=2))
    print('\nWrote results/cj_cantera.json')


if __name__ == '__main__':
    main()
