"""M2 local CPU smoke test. Verifies:
    1. RDKit + PySCF imports
    2. SMILES -> 3D coords pipeline
    3. PySCF SCF on a tiny test case (H2O at HF/STO-3G, ~5s on a laptop)
    4. The opt+freq+SP+HOF pipeline shape on H2O at B3LYP/6-31G(d)
       (skipped if --hf-only is given to keep the test under 30 sec)

Does NOT require GPU4PySCF or a GPU. Use this to gate the runpod submission.

Run:
    /c/Python314/python scripts/dft/m2_smoke_test.py
    /c/Python314/python scripts/dft/m2_smoke_test.py --hf-only   # fastest
    /c/Python314/python scripts/dft/m2_smoke_test.py --full      # opt+freq+SP+HOF
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def step(name):
    print(f"[smoke] {name} ...", end=" ", flush=True)


def ok(t):
    print(f"OK ({t:.1f}s)")


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-only", action="store_true", help="Skip B3LYP, just HF/STO-3G")
    ap.add_argument("--full", action="store_true", help="Run full opt+freq+SP+HOF on H2O")
    args = ap.parse_args()

    t_total = time.time()

    # 1. Imports
    step("import RDKit")
    t0 = time.time()
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        ok(time.time() - t0)
    except Exception as e: fail(str(e))

    step("import PySCF")
    t0 = time.time()
    try:
        from pyscf import gto, scf, dft
        ok(time.time() - t0)
    except Exception as e: fail(str(e))

    step("import m2_dft_pipeline")
    t0 = time.time()
    try:
        from m2_dft_pipeline import (smiles_to_xyz, atoms_to_pyscf,
                                       molecular_formula, density_from_volume,
                                       kamlet_jacobs, hof_from_atomization,
                                       atomic_reference_energy, ATOMIC_HOF_298)
        ok(time.time() - t0)
    except Exception as e: fail(str(e))

    # 2. SMILES -> xyz on water
    step("SMILES O -> xyz")
    t0 = time.time()
    atoms, charge, n = smiles_to_xyz("O")
    if n != 3: fail(f"expected 3 atoms in H2O, got {n}")
    formula = molecular_formula(atoms)
    if formula != "H2O": fail(f"expected formula H2O, got {formula}")
    ok(time.time() - t0)

    # 3. PySCF HF/STO-3G on H2O (sanity)
    step("HF/STO-3G H2O")
    t0 = time.time()
    mol = gto.M(atom=atoms_to_pyscf(atoms), basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    e = mf.kernel()
    expected = -74.96
    if abs(e - expected) > 0.5: fail(f"HF/STO-3G H2O E={e:.3f}, expected ~{expected}")
    ok(time.time() - t0)

    if args.hf_only:
        print(f"[smoke] === HF-only smoke PASS ({time.time() - t_total:.1f}s) ===")
        return

    # 4. B3LYP/6-31G(d) single point on H2O (lightweight DFT verify)
    step("B3LYP/6-31G* H2O SP")
    t0 = time.time()
    mol = gto.M(atom=atoms_to_pyscf(atoms), basis="6-31g*", unit="Angstrom", verbose=0)
    mf = dft.RKS(mol, xc="b3lyp").density_fit()
    mf.verbose = 0
    e_b3lyp = mf.kernel()
    if abs(e_b3lyp + 76.4) > 0.2: fail(f"B3LYP/6-31G* H2O E={e_b3lyp:.3f}, expected ~-76.4")
    ok(time.time() - t0)

    # 5. Density estimate
    step("density_from_volume H2O")
    t0 = time.time()
    rho = density_from_volume(atoms, "H2O")
    if not (0.5 < rho < 2.0): fail(f"H2O density estimate {rho:.2f} out of plausible range")
    ok(time.time() - t0)

    # 6. K-J recomputation (will return None for H2O since it's not an explosive,
    #    but should not crash)
    step("Kamlet-Jacobs H2O (should return non-detonating)")
    t0 = time.time()
    kj = kamlet_jacobs(rho, -286.0, "H2O")     # -286 kJ/mol = HOF of liquid water
    if "D_kms" not in kj: fail("K-J output missing D_kms key")
    ok(time.time() - t0)

    if not args.full:
        print(f"[smoke] === Default smoke PASS ({time.time() - t_total:.1f}s) ===")
        print(f"[smoke] (use --full to run the opt+freq+SP+HOF pipeline on H2O)")
        return

    # 7. Full pipeline on H2O (CPU PySCF, ~30-60 s)
    step("opt B3LYP/6-31G* H2O")
    t0 = time.time()
    from m2_dft_pipeline import opt_b3lyp_6_31gss, freq_b3lyp_6_31gss, sp_wb97xd_def2tzvp
    opt = opt_b3lyp_6_31gss(atoms, charge=charge, use_gpu=False)
    if "atoms_opt" not in opt: fail("opt missing atoms_opt")
    ok(time.time() - t0)

    step("freq B3LYP/6-31G* H2O")
    t0 = time.time()
    freq = freq_b3lyp_6_31gss(opt["atoms_opt"], charge=charge, use_gpu=False)
    if freq["n_imag"] != 0: fail(f"H2O has {freq['n_imag']} imaginary freqs (should be 0)")
    if not (3000 < freq["freqs_wavenumber"][-1] < 4500):
        fail(f"H2O highest freq {freq['freqs_wavenumber'][-1]} cm-1 out of range (expect O-H stretch ~3700)")
    ok(time.time() - t0)

    step("SP wB97X-D/def2-TZVP H2O")
    t0 = time.time()
    sp = sp_wb97xd_def2tzvp(opt["atoms_opt"], charge=charge, use_gpu=False)
    ok(time.time() - t0)

    step("atomic refs B3LYP/6-31G* (H, O)")
    t0 = time.time()
    refs_b3lyp = {s: atomic_reference_energy(s, "6-31g*", "b3lyp", use_gpu=False)
                  for s in ("H", "O")}
    ok(time.time() - t0)

    step("HOF from atomization (H2O target ~ -286 kJ/mol experimental)")
    t0 = time.time()
    e_zpe = opt["E_b3lyp_631gss_hartree"] + freq["ZPE_kJmol"] / 2625.49963948
    hof = hof_from_atomization(opt["atoms_opt"], e_zpe, refs_b3lyp)
    print(f"computed HOF(H2O) = {hof:.1f} kJ/mol (expt -286)")
    if not (-400 < hof < -100):
        fail(f"HOF(H2O) {hof} kJ/mol way off from -286 expt")
    ok(time.time() - t0)

    print(f"\n[smoke] === Full smoke PASS ({time.time() - t_total:.1f}s) ===")


if __name__ == "__main__":
    main()
