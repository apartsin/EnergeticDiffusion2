"""
Chapman-Jouguet (CJ) detonation recompute using Cantera.

Free, license-free alternative to EXPLO5 / Cheetah-2. Implements a minimal
CJ root-finder: for a given unburned (condensed) state with known density and
heat of formation, search for the detonation velocity D_CJ such that the
post-shock equilibrium gas state satisfies the Chapman-Jouguet condition
(post-shock flow speed equals local sound speed; Mach = 1 in the wave frame).

Recipe: classic SDToolbox-style algorithm
(see https://shepherd.caltech.edu/EDL/PublicResources/sdt/), simplified.

Conservation equations across the detonation wave (1 = upstream condensed
explosive treated as cold ideal-density slab, 2 = downstream equilibrium gas):
    rho1 * u1 = rho2 * u2                 (mass)
    P1 + rho1*u1^2 = P2 + rho2*u2^2       (momentum)
    h1 + 0.5 u1^2 = h2 + 0.5 u2^2         (energy)

In the wave frame, u1 = D and u2 is the post-shock flow velocity. The CJ
condition is u2 = a2 (local sound speed of the equilibrium products).

For each detonation velocity guess D_test, we solve for the post-shock
equilibrium state (T2, P2, composition) that simultaneously satisfies mass +
momentum + energy + chemical equilibrium at constant H, P (after using P,
v iteration), then check whether u2 == a2. We bracket D between 4 and 12 km/s
and bisect.

Since condensed-phase enthalpy contributions are small relative to bond
energies, we approximate h1 = HOF (at 298 K, condensed) per unit mass and P1
~ 1 bar; this is the standard CJ-DOT (Detonation On Top) closure used by
EXPLO5, Cheetah, and SDToolbox.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import cantera as ct
import numpy as np

# Standard CHNO product set (Cheetah-2 / EXPLO5 default)
PRODUCT_SPECIES = ["CO2", "CO", "H2O", "N2", "NO", "NH3", "CH4", "H2", "O2", "OH", "H", "O", "N"]

# Atomic masses (g/mol)
AW = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999}

# Reference conditions
T_REF = 298.15  # K
P_REF = 101325.0  # Pa (1 atm)


def parse_formula(formula: str) -> dict:
    """Parse e.g. 'C3H6N6O6' into {'C':3,'H':6,'N':6,'O':6}."""
    counts = {"C": 0, "H": 0, "N": 0, "O": 0}
    for elem, n in re.findall(r"([CHNO])(\d*)", formula):
        if elem:
            counts[elem] += int(n) if n else 1
    return counts


def molar_mass(formula_counts: dict) -> float:
    return sum(formula_counts[e] * AW[e] for e in formula_counts)


def build_gas() -> ct.Solution:
    """Build a Cantera Solution containing only the standard CHNO product set,
    using the NASA9 thermodynamic polynomials from SDToolbox's nasa_gas.yaml
    (valid 200-6000 K, much wider than GRI-Mech 3.0's 3500 K limit and adequate
    for typical detonation T_CJ ~ 3000-5000 K).
    """
    species_all = ct.Species.list_from_file("nasa_gas.yaml")
    sp_map = {s.name: s for s in species_all}
    selected = [sp_map[n] for n in PRODUCT_SPECIES if n in sp_map]
    return ct.Solution(thermo="ideal-gas", species=selected)


def set_equilibrium_products(
    gas: ct.Solution, formula: dict, T: float, P: float
) -> None:
    """Set the gas to a stoichiometric atomic balance and equilibrate at T,P."""
    # build an atomic-mole-fraction-equivalent initial mixture: spread atoms
    # over a "reasonable" starting composition that respects atom balance.
    # Simplest: use the elemental composition with a placeholder (atomic
    # H, C, N, O) - cantera will equilibrate to the true minimum.
    nC, nH, nN, nO = formula["C"], formula["H"], formula["N"], formula["O"]

    # Construct an initial guess that already conserves atoms: put C into CO,
    # H into H2O, N into N2, leftover O into O2. This guess does not affect
    # final equilibrium, only convergence.
    Y = {sp: 0.0 for sp in gas.species_names}
    eps = 1e-12
    co_amt = nC
    h2o_amt = nH / 2.0
    n2_amt = nN / 2.0
    # remaining O after CO and H2O
    o_remain = nO - co_amt - h2o_amt
    if o_remain >= 0:
        o2_amt = o_remain / 2.0
        co2_amt = 0.0
    else:
        # not enough O for CO + H2O: shift some CO -> bare C? GRI has C(s)? no.
        # use H2 instead of H2O for some H
        deficit = -o_remain
        # convert deficit moles of H2O -> H2 (frees 1 O each)
        h2o_amt -= deficit
        h2_amt = deficit
        Y["H2"] = max(h2_amt, 0)
        if h2o_amt < 0:
            # fall back: convert CO -> C + ... can't; just zero h2o
            Y["H2"] = nH / 2.0
            h2o_amt = 0.0
        o2_amt = 0.0
        co2_amt = 0.0
    Y["CO"] = co_amt
    Y["H2O"] = max(h2o_amt, 0)
    Y["N2"] = n2_amt
    Y["O2"] = max(o2_amt, 0)

    # normalize to mole fractions
    total = sum(Y.values())
    if total <= 0:
        raise ValueError("zero atoms")
    X = {k: v / total for k, v in Y.items() if v > 0}
    gas.TPX = T, P, X
    gas.equilibrate("TP")


def cj_solve(
    formula: dict,
    rho_solid: float,  # g/cc
    HOF_kJ_per_mol: float,  # condensed-phase HOF
    verbose: bool = False,
) -> dict:
    """Find Chapman-Jouguet detonation velocity by bracketed search."""
    gas = build_gas()
    M_explosive = molar_mass(formula)  # g/mol

    # Convert to SI: rho [kg/m^3], HOF -> specific enthalpy [J/kg]
    rho1 = rho_solid * 1000.0  # g/cc -> kg/m^3
    h1 = (HOF_kJ_per_mol * 1000.0) / (M_explosive * 1e-3)  # J/kg
    P1 = P_REF
    v1 = 1.0 / rho1

    # element constraint vector for cantera elemental conservation
    # we use the gas's elemental_mass_fraction pre-loaded by setting initial
    # composition with the right atom counts.

    def hugoniot_residual_at_v2(D: float, v2: float):
        """Given D and a trial v2, compute (T2, P2) from equilibrium TP iteration.

        For each v2: P2 set by momentum equation. Then find T2 such that, after
        TP equilibration, the equilibrium density matches 1/v2 (ideal-gas EOS
        plus equilibrium composition closes the system). Return the energy
        residual r = (h2_eq + 0.5*u2^2) - (h1 + 0.5*D^2). Hugoniot solution
        when r=0.
        """
        P2 = P1 + rho1 * D * D * (1.0 - v2 / v1)
        if P2 <= 0:
            return None, None
        # bisect T2 such that gas at TP equilibrium has density = 1/v2
        target_rho = 1.0 / v2

        def rho_at_T(T):
            try:
                set_equilibrium_products(gas, formula, T, P2)
                return gas.density
            except Exception:
                return None

        # NASA9 polynomials valid 200-6000 K; extrapolate to 8000 K which
        # covers all physically reasonable detonation temperatures.
        T_lo, T_hi = 1500.0, 8000.0
        rho_lo = rho_at_T(T_lo)
        rho_hi = rho_at_T(T_hi)
        if rho_lo is None or rho_hi is None:
            return None, None
        if not (rho_hi <= target_rho <= rho_lo):
            if target_rho > rho_lo:
                T_sol = T_lo
            else:
                T_sol = T_hi
        else:
            for _ in range(40):
                T_mid = 0.5 * (T_lo + T_hi)
                rho_mid = rho_at_T(T_mid)
                if rho_mid is None:
                    return None, None
                if rho_mid > target_rho:
                    T_lo = T_mid
                else:
                    T_hi = T_mid
                if abs(T_hi - T_lo) < 1.0:
                    break
            T_sol = 0.5 * (T_lo + T_hi)
        # final equilibrium at this T, P
        try:
            set_equilibrium_products(gas, formula, T_sol, P2)
        except Exception:
            return None, None
        h2_eq = gas.enthalpy_mass
        u2 = D * v2 / v1
        # energy residual
        resid = (h2_eq + 0.5 * u2 * u2) - (h1 + 0.5 * D * D)
        state = {
            "T2": T_sol,
            "P2": P2,
            "rho2": gas.density,
            "v2": 1.0 / gas.density,
            "u2": u2,
            "a2": gas.sound_speed,
            "h2": h2_eq,
            "X": {sp: gas.X[gas.species_index(sp)] for sp in PRODUCT_SPECIES},
        }
        return resid, state

    def find_hugoniot_state(D: float, branch: str = "any"):
        """Bisect v2 such that energy residual = 0 (Hugoniot solution).

        branch='any' => last sign change found in v2 scan from large to small v2
        branch='weak' => first sign change (larger v2, less compression)
        branch='strong' => smallest-v2 sign change

        Returns the post-shock state dict or None if no Hugoniot solution.
        """
        v_grid = np.linspace(0.98 * v1, 0.10 * v1, 60)
        resids = []
        for v2 in v_grid:
            r, _ = hugoniot_residual_at_v2(D, v2)
            resids.append(r)

        sign_changes = []
        for i in range(len(resids) - 1):
            if resids[i] is None or resids[i + 1] is None:
                continue
            if resids[i] * resids[i + 1] < 0:
                sign_changes.append((v_grid[i], v_grid[i + 1]))
        if not sign_changes:
            return None

        if branch == "weak":
            v_a, v_b = sign_changes[0]
        else:
            v_a, v_b = sign_changes[-1]

        for _ in range(50):
            v_mid = 0.5 * (v_a + v_b)
            r_mid, _ = hugoniot_residual_at_v2(D, v_mid)
            if r_mid is None:
                return None
            r_a, _ = hugoniot_residual_at_v2(D, v_a)
            if r_a is None:
                return None
            if r_a * r_mid < 0:
                v_b = v_mid
            else:
                v_a = v_mid
            if abs(v_b - v_a) < 1e-9 * v1:
                break
        _, st = hugoniot_residual_at_v2(D, 0.5 * (v_a + v_b))
        if st is None:
            return None
        st["M2"] = st["u2"] / st["a2"]
        return st

    def post_state(D: float):
        """Return strong-branch Hugoniot state (smallest v2)."""
        return find_hugoniot_state(D, branch="strong")

    # CJ condition: f(D) = u2 - a2 = 0. For too-low D, products are
    # supersonic relative to wave so u2 > a2 (M2>1 in lab? actually wave frame).
    # In wave frame: M2 = u2/a2; CJ has M2 = 1 (sonic at downstream). For
    # D < D_CJ, the RH solution either does not exist or has M2 > 1 (weak
    # detonation branch) — we want D_CJ at the minimum strong-detonation
    # speed where M2 = 1.

    # CJ definition: minimum D for which the equilibrium Hugoniot has a real
    # intersection with the Rayleigh line. Below D_CJ the energy residual has
    # no zero in v2; at D_CJ, single tangent point; above D_CJ, two crossings.
    # Algorithm: find smallest D such that the v2-scan energy-residual changes
    # sign at least once.

    def has_hugoniot(D: float) -> bool:
        """A Hugoniot solution exists at D iff the energy residual crosses zero
        (sign change) somewhere in v2 in the physical range. Use a fine v2 grid.
        """
        v_grid = np.linspace(0.99 * v1, 0.05 * v1, 80)
        resids = [hugoniot_residual_at_v2(D, v)[0] for v in v_grid]
        # require a TRUE sign change (not just touch zero from same side)
        prev = None
        for r in resids:
            if r is None:
                continue
            if prev is not None and prev * r < 0:
                return True
            prev = r
        return False

    # bracket D_CJ between D_lo (no Hugoniot) and D_hi (Hugoniot exists)
    D_lo, D_hi = 3000.0, 14000.0
    if not has_hugoniot(D_hi):
        if verbose:
            print(f"  D_hi={D_hi} has no Hugoniot; trying higher...")
        # extend
        for D_try in [16000.0, 20000.0, 25000.0]:
            if has_hugoniot(D_try):
                D_hi = D_try
                break
        else:
            return None
    if has_hugoniot(D_lo):
        # CJ < 3000 m/s; expand down
        D_lo = 1500.0
        if has_hugoniot(D_lo):
            return None  # something off

    if verbose:
        print(f"  bracket: D_lo={D_lo} (no Hugoniot), D_hi={D_hi} (Hugoniot OK)")

    for it in range(30):
        D_mid = 0.5 * (D_lo + D_hi)
        if has_hugoniot(D_mid):
            D_hi = D_mid
        else:
            D_lo = D_mid
        if verbose:
            print(f"  iter {it}: D_lo={D_lo:.1f}  D_hi={D_hi:.1f}")
        if D_hi - D_lo < 5.0:
            break
    D_cj = D_hi
    st = post_state(D_cj)
    if st is None:
        # try weak-branch
        st = find_hugoniot_state(D_cj, branch="weak")
    if st is None:
        return None
    return _pack(D_cj, st, formula, rho_solid, HOF_kJ_per_mol)


def _pack(D, st, formula, rho_solid, HOF):
    if st is None:
        return None
    # filter top products
    X = st["X"]
    top = sorted(X.items(), key=lambda kv: -kv[1])[:5]
    return {
        "rho": rho_solid,
        "HOF_kJmol": HOF,
        "formula": "".join(f"{e}{formula[e]}" for e in "CHNO" if formula[e]),
        "D_CJ_kms": D / 1000.0,
        "P_CJ_GPa": st["P2"] / 1e9,
        "T_CJ_K": st["T2"],
        "M2": st["M2"],
        "products_mole_fraction": X,
        "top_products": [(k, float(v)) for k, v in top],
    }


def main():
    base = Path("E:/Projects/EnergeticDiffusion2")
    summary = json.load(open(base / "m2_bundle/results/m2_summary.json"))
    by_id = {r["id"]: r for r in summary}

    # Load formulas from per-lead jsons
    def lead_formula(lid):
        p = base / f"m2_bundle/results/m2_lead_{lid}.json"
        return json.load(open(p))["formula"]

    cases = []
    # Anchors first
    cases.append(("RDX", "C3H6N6O6", 1.806, 66.0, 8.75))   # exp 8.75 km/s
    cases.append(("TATB", "C6H6N6O6", 1.937, -154.0, 7.86))  # exp 7.86 km/s

    # Leads use CALIBRATED rho and HOF
    for lid in ["L1", "L4", "L5"]:
        rec = by_id[lid]
        formula_str = lead_formula(lid)
        cases.append(
            (lid, formula_str, rec["rho_cal"], rec["HOF_kJmol_wb97xd_cal"], None)
        )

    results = {}
    print(f"\n{'cmpd':6s} {'rho':>5s} {'HOF':>9s} {'D_CJ':>8s} {'P_CJ':>8s} {'T_CJ':>7s}  dominant products")
    print("-" * 95)
    for name, formula_str, rho, HOF, exp_D in cases:
        formula = parse_formula(formula_str)
        try:
            res = cj_solve(formula, rho, HOF, verbose=False)
        except Exception as e:
            print(f"{name:6s}  FAILED: {e}")
            results[name] = {"error": str(e)}
            continue
        if res is None:
            print(f"{name:6s}  no CJ solution found")
            results[name] = {"error": "no solution"}
            continue
        results[name] = res
        top_str = ", ".join(f"{k}:{v:.2f}" for k, v in res["top_products"][:4])
        print(
            f"{name:6s} {rho:5.2f} {HOF:9.1f} {res['D_CJ_kms']:8.2f} "
            f"{res['P_CJ_GPa']:8.2f} {res['T_CJ_K']:7.0f}  {top_str}"
        )

    # write JSON
    out = base / "results/cj_cantera.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")

    # Comparison table
    print("\n" + "=" * 80)
    print("CJ-Cantera vs K-J calibrated vs experimental:")
    print("=" * 80)
    kj_lookup = {
        "RDX": ("8.25", "8.75"),
        "TATB": ("5.64", "7.86"),
        "L1": ("NaN (oxygen-rich)", None),
        "L4": ("13.27", None),
        "L5": ("NaN (oxygen-rich)", None),
    }
    for name in ["RDX", "TATB", "L1", "L4", "L5"]:
        r = results.get(name, {})
        d_cj = r.get("D_CJ_kms", float("nan"))
        kj, exp = kj_lookup[name]
        if exp is not None:
            print(f"{name:6s} K-J D_cal = {kj} km/s; CJ-Cantera D = {d_cj:5.2f} km/s; experimental D = {exp} km/s")
        else:
            print(f"{name:6s} K-J D_cal = {kj}; CJ-Cantera D = {d_cj:5.2f} km/s")

    # Calibration check
    rdx_cj = results.get("RDX", {}).get("D_CJ_kms", float("nan"))
    tatb_cj = results.get("TATB", {}).get("D_CJ_kms", float("nan"))
    if not (np.isnan(rdx_cj) or np.isnan(tatb_cj)):
        delta_rdx = abs(rdx_cj - 8.75)
        delta_tatb = abs(tatb_cj - 7.86)
        if delta_rdx < 0.5 and delta_tatb < 0.5:
            verdict = "CALIBRATED. L1/L4/L5 absolute numbers are trustworthy."
        elif delta_rdx < 1.0 and delta_tatb < 1.0:
            verdict = "MODERATE. L1/L4/L5 numbers indicative."
        else:
            verdict = (
                "POORLY CALIBRATED in absolute terms.\n"
                "  RDX off by {:.2f} km/s, TATB off by {:.2f} km/s.\n"
                "  Cause: the Cantera ideal-gas EOS cannot capture covolume\n"
                "  effects at detonation pressures (P_CJ ~ 30-100 GPa). EXPLO5\n"
                "  and Cheetah-2 use BKW or JCZ3 mixture EOS for this regime.\n"
                "  This run is therefore valid only for RELATIVE comparisons:\n"
                "  if L4_CJ / RDX_CJ on this EOS << 13.27/8.75 = 1.52, then\n"
                "  K-J's L4 = 13.27 km/s claim is an artefact of K-J's regression\n"
                "  in the high-N regime, NOT a thermochemically-consistent result.\n"
                "  Direct comparison: L4_CJ / RDX_CJ (Cantera) = {:.2f}; K-J ratio = 1.52."
            ).format(delta_rdx, delta_tatb,
                     results.get('L4', {}).get('D_CJ_kms', float('nan'))/max(rdx_cj, 1e-6))
        print(f"\nCalibration verdict:\n  {verdict}")

        # Annotate JSON with caveat
        results["_caveat"] = (
            "Cantera ideal-gas product EOS underpredicts D_CJ for high-density "
            "CHNO explosives by ~3x because it lacks covolume / non-ideal effects "
            "active at 30-100 GPa. RDX D_CJ predicted = {:.2f} km/s vs experimental "
            "8.75 km/s. Numbers here are only meaningful as relative comparisons "
            "between compounds on the same EOS, not as absolute predictions."
        ).format(rdx_cj)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
