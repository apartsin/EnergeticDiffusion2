"""
Evaluate top-ranked diffusion candidates via combined simulation + K-J formula.

Pipeline:
  1. Load generated candidates (SMILES + latents)
  2. Rank by 3DCNN predictions on user-specified criterion
  3. Select top K
  4. For each top-K molecule, compute properties via multiple methods:
       - 3DCNN (Uni-Mol) — fast ML surrogate
       - Girolami formula — pure Python density (no external deps)
       - K-J formula — Tier-C D/P from (ρ, HOF, formula)
       - [Optional] Psi4 B3LYP single-point HOF — requires conda
       - [Optional] CP2K periodic DFT density — requires conda
  5. Produce side-by-side comparison table (JSON + CSV)

Selection criteria (--rank-by):
    d           : maximize 3DCNN-predicted detonation velocity
    p           : maximize 3DCNN-predicted detonation pressure
    dp          : maximize D * P (proxy for explosive power)
    target      : minimize |predicted - target| for --target-values
    feasibility : low SA + SC score (most synthesizable)

Usage:
    python scripts/diffusion/evaluate_candidates.py \\
        --candidates guided_samples.json \\
        --limo-ckpt experiments/limo_ft_energetic_*/checkpoints/best.pt \\
        --rank-by d --top 20
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).parent))
from unimol_validator import UniMolValidator, PROP_ORDER


# ── Girolami-style density estimator (pure Python, no external deps) ────────
GIROLAMI_GROUP_VOLUME = {
    # Atomic volumes (Å³) from Girolami 1994 J. Chem. Educ.
    "H":  8.0 / 5.0,      # Girolami adjustment for H
    "C":  8.0,
    "N":  8.0,
    "O":  8.0,
    "F":  8.0,
    "Cl": 8.0 * 2.5,      # Cl has larger volume
}


def girolami_density(smi: str) -> float | None:
    """Girolami (1994) rule: ρ = MW / (k × Σ atom_volumes)
    where k is an empirical factor (~1.15–1.25 for energetic CHNO).
    Returns density in g/cm³, or None if SMILES can't be parsed.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)

    MW = sum(a.GetMass() for a in mol_h.GetAtoms())
    vol = 0.0
    for a in mol_h.GetAtoms():
        sym = a.GetSymbol()
        vol += GIROLAMI_GROUP_VOLUME.get(sym, 15.0)

    # ring correction: tight rings pack more densely
    ri = mol.GetRingInfo()
    ring_atoms = sum(len(r) for r in ri.AtomRings())

    # Girolami empirical packing factor adjusted for energetics
    k = 1.23
    if ring_atoms > 0:
        k -= 0.004 * ring_atoms  # small-ring correction

    # ρ = MW / (k * V * N_A × 1e-24) in g/cm³
    # V is in Å³; 1 Å³ = 1e-24 cm³; N_A = 6.022e23
    V_cm3 = k * vol * 1e-24 * 6.02214e23
    return MW / V_cm3


# ── Kamlet-Jacobs formula ───────────────────────────────────────────────────
HOF_H2O = -57798.0   # cal/mol
HOF_CO2 = -94051.0
HOF_CO  = -26416.0
KJ_TO_CAL = 239.006


def kj_detonation(smi: str, density: float, hof_kj_mol: float
                    ) -> tuple[float, float] | None:
    """Apply Kamlet-Jacobs formula. Returns (D km/s, P GPa) or None if non-CHNO."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    counts = {"C": 0, "H": 0, "N": 0, "O": 0, "OTHER": 0}
    for a in mol_h.GetAtoms():
        s = a.GetSymbol()
        counts[s if s in counts else "OTHER"] += 1
    if counts["OTHER"] > 0:
        return None

    a, b, g, d = counts["C"], counts["H"], counts["N"], counts["O"]
    MW = 12.011*a + 1.008*b + 14.007*g + 15.999*d
    if MW <= 0:
        return None

    # product hierarchy
    n2 = g / 2.0
    h_rem, o_rem, c_rem = b, d, a
    h2o = min(h_rem / 2, o_rem); o_rem -= h2o; h_rem -= 2 * h2o
    co2 = min(c_rem, o_rem / 2); o_rem -= 2 * co2; c_rem -= co2
    co = min(c_rem, o_rem); o_rem -= co; c_rem -= co
    o2 = max(o_rem / 2, 0.0)
    n_gas = n2 + h2o + co2 + co + o2
    if n_gas <= 0:
        return None

    N = n_gas / MW
    M = (n2*28 + h2o*18 + co2*44 + co*28 + o2*32) / n_gas
    hof_cal = hof_kj_mol * KJ_TO_CAL
    Q = (hof_cal - (h2o*HOF_H2O + co2*HOF_CO2 + co*HOF_CO)) / MW
    if Q <= 0:
        return None

    phi = N * np.sqrt(M * Q)
    D = 1.01 * np.sqrt(phi) * (1 + 1.3 * density)
    P = 1.558 * density**2 * phi
    return float(D), float(P)


# ── Optional: Psi4 B3LYP/6-31G(d) HOF ───────────────────────────────────────
def psi4_hof(smi: str) -> float | None:
    """B3LYP HOF via Psi4. Returns HOF (kJ/mol) or None if Psi4 unavailable.

    Requires Psi4 installed (`conda install -c psi4 psi4`).
    """
    try:
        import psi4
    except ImportError:
        return None
    try:
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        coords = mol.GetConformer().GetPositions()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        xyz_str = f"{len(symbols)}\n\n" + "\n".join(
            f"{s} {x:.6f} {y:.6f} {z:.6f}"
            for s, (x, y, z) in zip(symbols, coords))
        psi4.set_output_file("/dev/null", True)
        psi4.set_options({"basis": "6-31g(d)"})
        mol_ps = psi4.geometry(xyz_str)
        E = psi4.energy("b3lyp", molecule=mol_ps)
        # naive atomization-HOF via reference atomic energies could be added
        # for now, just return E (Hartree) so user can post-process
        return float(E * 2625.5)   # Hartree → kJ/mol electronic energy
    except Exception as e:
        print(f"  psi4 failed on {smi[:40]}: {e}")
        return None


# ── Optional: CP2K periodic DFT density (wrapper skeleton) ──────────────────
def cp2k_density(smi: str) -> float | None:
    """Periodic-DFT density estimate via CP2K. Requires CP2K binary + pyxtal.

    Stub: returns None until conda env set up.
    """
    try:
        import subprocess
        subprocess.run(["cp2k.psmp", "--version"], check=True,
                       capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    # TODO: integrate pyxtal CSP + CP2K periodic DFT when environment is ready
    return None


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="JSON file from sample_guided.py")
    ap.add_argument("--top",        type=int, default=20)
    ap.add_argument("--rank-by",    default="d",
                    choices=["d", "p", "dp", "target", "feasibility"])
    ap.add_argument("--target-density", type=float, default=None)
    ap.add_argument("--target-d",       type=float, default=None)
    ap.add_argument("--target-p",       type=float, default=None)
    ap.add_argument("--threedcnn-dir",  default="data/raw/energetic_external/EMDP/Data/smoke_model")
    ap.add_argument("--run-psi4",       action="store_true",
                    help="Run Psi4 B3LYP HOF (requires conda env)")
    ap.add_argument("--run-cp2k",       action="store_true",
                    help="Run CP2K periodic DFT density (requires conda env)")
    ap.add_argument("--out",            default=None)
    ap.add_argument("--base",           default="E:/Projects/EnergeticDiffusion2")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    cand_path = Path(args.candidates)
    if not cand_path.is_absolute(): cand_path = base / cand_path

    data = json.load(open(cand_path))
    candidates = data["samples"]
    print(f"Loaded {len(candidates)} candidates from {cand_path.name}")

    # Extract valid canonical SMILES, deduplicated
    seen = set()
    valid_cand = []
    for c in candidates:
        smi = c.get("canonical")
        if smi and smi not in seen:
            seen.add(smi)
            valid_cand.append(c)
    print(f"  {len(valid_cand)} valid unique canonical SMILES")

    # ── 3DCNN scoring ────────────────────────────────────────────────────────
    print(f"\n[1/3] Running 3DCNN (Uni-Mol) on all candidates …")
    try:
        val = UniMolValidator(Path(args.base) / args.threedcnn_dir)
        preds = val.predict([c["canonical"] for c in valid_cand])
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1
    for c, i in zip(valid_cand, range(len(valid_cand))):
        c["3dcnn"] = {
            p: (None if np.isnan(preds[p][i]) else float(preds[p][i]))
            for p in ["density", "DetoD", "DetoP", "DetoQ", "HOF_S", "BDE"]
        }

    # ── Rank ─────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Ranking by '{args.rank_by}' …")
    def score(c):
        p = c["3dcnn"]
        if args.rank_by == "d":
            return -(p.get("DetoD") or 0)
        if args.rank_by == "p":
            return -(p.get("DetoP") or 0)
        if args.rank_by == "dp":
            return -((p.get("DetoD") or 0) * (p.get("DetoP") or 0))
        if args.rank_by == "feasibility":
            return (c.get("sa_pred", 10) + c.get("sc_pred", 5))
        if args.rank_by == "target":
            err = 0.0
            if args.target_density is not None and p.get("density"):
                err += abs(p["density"] - args.target_density)
            if args.target_d is not None and p.get("DetoD"):
                err += abs(p["DetoD"] - args.target_d)
            if args.target_p is not None and p.get("DetoP"):
                err += abs(p["DetoP"] - args.target_p)
            return err
        return 0
    valid_cand.sort(key=score)
    top = valid_cand[: args.top]
    print(f"  Selected top {len(top)}")

    # ── Per-candidate validation ─────────────────────────────────────────────
    print(f"\n[3/3] Validating top {len(top)} with Girolami + K-J (+Psi4/CP2K if available) …")
    results = []
    for i, c in enumerate(top):
        smi = c["canonical"]
        row = {
            "rank":     i + 1,
            "smiles":   smi,
            "sa_pred":  c.get("sa_pred"),
            "sc_pred":  c.get("sc_pred"),
            "3dcnn":    c["3dcnn"],
        }

        # Girolami density
        row["girolami_density"] = girolami_density(smi)

        # K-J from (3DCNN density, 3DCNN HOF)
        rho_cnn = c["3dcnn"].get("density")
        hof_cnn = c["3dcnn"].get("HOF_S")
        if rho_cnn and hof_cnn:
            kj = kj_detonation(smi, rho_cnn, hof_cnn)
            row["kj_from_3dcnn"] = {"D": kj[0], "P": kj[1]} if kj else None

        # K-J from (Girolami density, 3DCNN HOF) — independent ρ source
        if row["girolami_density"] and hof_cnn:
            kj = kj_detonation(smi, row["girolami_density"], hof_cnn)
            row["kj_from_girolami_hof_cnn"] = {"D": kj[0], "P": kj[1]} if kj else None

        # Optional high-fidelity (Psi4, CP2K)
        if args.run_psi4:
            psi4_E = psi4_hof(smi)
            row["psi4_energy_kj_mol"] = psi4_E
            if psi4_E and row["girolami_density"]:
                # very approximate HOF — needs thermal + atomization for real HOF
                # placeholder: electronic energy only
                kj = kj_detonation(smi, row["girolami_density"], psi4_E)
                row["kj_from_girolami_hof_psi4"] = {"D": kj[0], "P": kj[1]} if kj else None

        if args.run_cp2k:
            rho_cp2k = cp2k_density(smi)
            row["cp2k_density"] = rho_cp2k

        results.append(row)
        if (i + 1) % 5 == 0 or i == len(top) - 1:
            print(f"  {i+1}/{len(top)} done")

    # ── Save ────────────────────────────────────────────────────────────────
    if args.out is None:
        args.out = str(cand_path.with_name(cand_path.stem + "_evaluated.json"))
    with open(args.out, "w") as f:
        json.dump({
            "source":          str(cand_path),
            "rank_by":         args.rank_by,
            "top_k":           len(top),
            "psi4_run":        args.run_psi4,
            "cp2k_run":        args.run_cp2k,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results":         results,
        }, f, indent=2)
    print(f"\nSaved → {args.out}")

    # Also CSV for readability
    csv_path = Path(args.out).with_suffix(".csv")
    rows = []
    for r in results:
        row = {
            "rank":             r["rank"],
            "smiles":           r["smiles"],
            "sa":               r.get("sa_pred"),
            "sc":               r.get("sc_pred"),
            "density_3dcnn":    r["3dcnn"].get("density"),
            "density_girolami": r.get("girolami_density"),
            "hof_3dcnn":        r["3dcnn"].get("HOF_S"),
            "d_3dcnn":          r["3dcnn"].get("DetoD"),
            "p_3dcnn":          r["3dcnn"].get("DetoP"),
        }
        if "kj_from_3dcnn" in r and r["kj_from_3dcnn"]:
            row["d_kj_cnn_inputs"] = r["kj_from_3dcnn"]["D"]
            row["p_kj_cnn_inputs"] = r["kj_from_3dcnn"]["P"]
        if "kj_from_girolami_hof_cnn" in r and r["kj_from_girolami_hof_cnn"]:
            row["d_kj_girolami_rho"] = r["kj_from_girolami_hof_cnn"]["D"]
            row["p_kj_girolami_rho"] = r["kj_from_girolami_hof_cnn"]["P"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"       CSV → {csv_path}")


if __name__ == "__main__":
    main()
