"""
Compute B3LYP/6-31G(d) heat of formation via Psi4 for a list of SMILES.

Usage:
    # MUST run inside the edft conda env:
    MAMBA_ROOT_PREFIX=~/micromamba ~/micromamba/micromamba.exe run -n edft \
        python scripts/simulation/psi4_hof.py --smi "O=[N+]([O-])N1CN(..." --smi "..."

    # Or take SMILES from a JSON (from sample_guided.py / evaluate_candidates.py):
    ... run -n edft python scripts/simulation/psi4_hof.py \
        --input guided_samples_evaluated.json \
        --field canonical \
        --out  psi4_hof_results.json

Computes HOF via atomization-enthalpy method:
    HOF(compound) = Σ HOF_atom(standard) - D_atomization(B3LYP)

Atomic reference HOFs (CODATA/NIST, kJ/mol):
    H   217.998
    C   716.68
    N   472.68
    O   249.18
    F    79.38
    Cl  121.30

Per-molecule cost:
    ~5-30 minutes on CPU (6-31G(d), ~30 atoms)
    ~2-10 minutes on 1 GPU (if Psi4-GPU build)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


HOF_ATOM_KJ = {
    "H":   217.998, "C": 716.68, "N": 472.68, "O": 249.18,
    "F":    79.38, "Cl": 121.30,
}

# Spin multiplicity of atoms in ground state
ATOM_SPIN = {"H": 2, "C": 3, "N": 4, "O": 3, "F": 2, "Cl": 2}

HARTREE_TO_KJ = 2625.4996


def build_molecule(smi: str, add_hs: bool = True):
    """SMILES → (symbols, coords) via RDKit ETKDG + UFF."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if add_hs:
        mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        return None
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords  = conf.GetPositions()
    return symbols, coords


def psi4_atom_energy_cache: dict[str, float] = {}  # filled per method/basis


def psi4_energy_molecule(symbols, coords, charge=0, multiplicity=1,
                          basis="6-31g(d)", method="b3lyp") -> float | None:
    """Single-point energy of a molecule at given level of theory (Hartree)."""
    import psi4
    psi4.core.clean()
    null_path = "NUL" if os.name == "nt" else "/dev/null"
    psi4.set_output_file(null_path, True)
    psi4.set_options({"basis": basis, "e_convergence": 1e-6,
                     "scf_type": "df", "reference": "uhf" if multiplicity > 1 else "rhf"})
    geom_str = f"{charge} {multiplicity}\n"
    geom_str += "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}"
                          for s, (x, y, z) in zip(symbols, coords))
    geom_str += "\nsymmetry c1\nno_reorient\nno_com\n"
    try:
        mol = psi4.geometry(geom_str)
        E = psi4.energy(method, molecule=mol)
        return float(E)
    except Exception as e:
        print(f"  psi4 failed: {e}")
        return None


def psi4_atom_energy(symbol: str, basis: str, method: str) -> float | None:
    """Isolated atom energy (needed for atomization → HOF). Cached."""
    key = f"{symbol}/{method}/{basis}"
    if key in psi4_atom_energy_cache:
        return psi4_atom_energy_cache[key]
    if symbol not in ATOM_SPIN:
        return None
    mult = ATOM_SPIN[symbol]
    coords = np.array([[0.0, 0.0, 0.0]])
    E = psi4_energy_molecule([symbol], coords, charge=0, multiplicity=mult,
                              basis=basis, method=method)
    if E is not None:
        psi4_atom_energy_cache[key] = E
    return E


def compute_hof(smi: str, basis: str = "6-31g(d)", method: str = "b3lyp"
                ) -> dict:
    """Compute HOF (kJ/mol) for a SMILES. Returns dict with energies + error info."""
    t0 = time.time()
    geom = build_molecule(smi, add_hs=True)
    if geom is None:
        return {"smi": smi, "error": "geometry_failed"}
    symbols, coords = geom
    if any(s not in HOF_ATOM_KJ for s in symbols):
        unk = [s for s in symbols if s not in HOF_ATOM_KJ]
        return {"smi": smi, "error": f"unsupported_atoms: {set(unk)}"}

    E_mol = psi4_energy_molecule(symbols, coords, basis=basis, method=method)
    if E_mol is None:
        return {"smi": smi, "error": "molecule_energy_failed"}

    # Atom reference energies at same level
    from collections import Counter
    counts = Counter(symbols)
    E_atoms = 0.0
    for sym, n in counts.items():
        e = psi4_atom_energy(sym, basis, method)
        if e is None:
            return {"smi": smi, "error": f"atom_energy_failed_{sym}"}
        E_atoms += n * e

    # Atomization energy and HOF
    D_atomization_kj = (E_atoms - E_mol) * HARTREE_TO_KJ   # positive for bound molecule
    HOF_atoms_kj = sum(HOF_ATOM_KJ[s] for s in symbols)
    HOF_mol_kj = HOF_atoms_kj - D_atomization_kj           # HOF of compound at 298K
    return {
        "smi":           smi,
        "E_mol_hartree": E_mol,
        "E_atoms_hartree": E_atoms,
        "D_atomization_kj_mol": D_atomization_kj,
        "HOF_kj_mol":    HOF_mol_kj,
        "n_atoms":       len(symbols),
        "method":        f"{method}/{basis}",
        "elapsed_sec":   time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smi",    action="append", default=[],
                    help="SMILES to process (may be repeated)")
    ap.add_argument("--input",  default=None, help="JSON file with candidates")
    ap.add_argument("--field",  default="canonical",
                    help="Key holding SMILES in each input item")
    ap.add_argument("--out",    default=None, help="Output JSON path")
    ap.add_argument("--basis",  default="6-31g(d)")
    ap.add_argument("--method", default="b3lyp")
    ap.add_argument("--limit",  type=int, default=None,
                    help="Only process first N molecules from input")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    smiles = list(args.smi)
    if args.input:
        data = json.load(open(args.input))
        items = data.get("results") or data.get("samples") or data
        if isinstance(items, list):
            for it in items:
                s = it.get(args.field) or it.get("smiles") or it.get("smi")
                if s: smiles.append(s)
    if args.limit:
        smiles = smiles[:args.limit]
    print(f"Processing {len(smiles)} SMILES at {args.method}/{args.basis}")

    results = []
    for i, s in enumerate(smiles):
        print(f"\n[{i+1}/{len(smiles)}]  {s[:60]}")
        r = compute_hof(s, basis=args.basis, method=args.method)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  HOF = {r['HOF_kj_mol']:.1f} kJ/mol  ({r['elapsed_sec']:.0f}s)")

    out = args.out or "psi4_hof_results.json"
    with open(out, "w") as f:
        json.dump({
            "method": args.method, "basis": args.basis,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
