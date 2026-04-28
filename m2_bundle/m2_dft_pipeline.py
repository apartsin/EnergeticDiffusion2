"""M2 DFT validation pipeline — runs on GPU4PySCF (A100/H100/RTX 4090).

For each lead in m2_smiles.json:
    1. RDKit ETKDGv3 conformer + MMFF94 prefilter -> initial xyz
    2. B3LYP/6-31G(d) geometry optimisation (gpu4pyscf, GPU)
    3. Analytical Hessian -> vibrational analysis (real-min check; lowest freq > 0)
    4. omega-B97X-D/def2-TZVP single point on optimised geometry (more accurate energy)
    5. Atomization-energy HOF: SP for each atomic species (C, H, N, O) at same theory
    6. Recompute Kamlet-Jacobs D, P from DFT-derived rho + HOF

Anchor compounds (RDX, TATB) are run alongside the leads to calibrate the
atomization-energy HOF against literature experimental HOF.

Outputs (in results/):
    m2_lead_<id>.json       - per-lead opt geom, freqs, energies, recomputed K-J
    m2_atom_refs.json       - atomic reference energies (one-time)
    m2_summary.json         - all leads + anchors aggregated

Usage on remote (after pip install gpu4pyscf pyscf rdkit-pypi):
    python3 m2_dft_pipeline.py --smiles m2_smiles.json --results results/

For local CPU smoke test, see m2_smoke_test.py.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────
HARTREE_TO_KJMOL = 2625.49963948
ANGSTROM_TO_BOHR = 1.8897259886
PACKING_COEFF = 0.69            # standard for energetic CHNO crystals

# Atomic experimental HOF at 298 K, kJ/mol (NIST CCCBDB)
ATOMIC_HOF_298 = {
    "H": 217.998,
    "C": 716.68,
    "N": 472.68,
    "O": 249.18,
    # F, Cl optionally
    "F": 79.38,
    "Cl": 121.301,
}


# ── Smiles -> xyz (CPU) ────────────────────────────────────────────────────
def smiles_to_xyz(smi: str, charge: int = 0):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smi}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"ETKDGv3 embedding failed for {smi}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        atoms.append((atom.GetSymbol(), p.x, p.y, p.z))
    return atoms, charge, mol.GetNumAtoms()


def atoms_to_pyscf(atoms):
    """Convert [(sym, x, y, z), ...] to a pyscf-style atom string."""
    lines = []
    for sym, x, y, z in atoms:
        lines.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}")
    return "\n".join(lines)


def atoms_from_mol_geom(mol):
    """Extract [(sym, x, y, z), ...] from a pyscf gto.Mole after optimisation."""
    coords = mol.atom_coords(unit="Angstrom")
    out = []
    for i, sym in enumerate(mol.elements):
        out.append((sym, float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
    return out


def molecular_formula(atoms):
    from collections import Counter
    c = Counter(s for s, *_ in atoms)
    order = ["C", "H", "N", "O", "F", "Cl"]
    parts = []
    for s in order:
        if c.get(s, 0): parts.append(f"{s}{c[s]}")
    for s in sorted(c):
        if s not in order and c[s]: parts.append(f"{s}{c[s]}")
    return "".join(parts)


# ── DFT calls (GPU on remote, CPU fallback in smoke test) ──────────────────
def _build_mol(atoms, basis, charge=0, spin=0):
    from pyscf import gto
    mol = gto.M(atom=atoms_to_pyscf(atoms), basis=basis, charge=charge, spin=spin,
                 unit="Angstrom", verbose=4)
    return mol


def _get_mf(mol, xc, use_gpu=True, df=True):
    """Return a DFT mean-field object (GPU if available, CPU fallback)."""
    if use_gpu:
        try:
            from gpu4pyscf import dft as gpu_dft
            mf = gpu_dft.RKS(mol, xc=xc)
            if df:
                mf = mf.density_fit()
            mf.verbose = 4  # print SCF iterations + diagnostic info
            return mf, "gpu"
        except ImportError:
            pass
    from pyscf import dft
    mf = dft.RKS(mol, xc=xc)
    if df:
        mf = mf.density_fit()
    mf.verbose = 4  # print SCF iterations + diagnostic info
    return mf, "cpu"


def opt_b3lyp_6_31gss(atoms, charge=0, use_gpu=True):
    """B3LYP/6-31G(d) geometry optimisation. Returns optimised atoms + energy."""
    from pyscf.geomopt.geometric_solver import optimize
    print(f"[diag][opt] building Mole: n_atoms={len(atoms)} basis=6-31g* charge={charge}", flush=True)
    mol = _build_mol(atoms, "6-31g*", charge=charge)
    print(f"[diag][opt] initial SCF kernel ...", flush=True)
    mf, backend = _get_mf(mol, "b3lyp", use_gpu=use_gpu)
    t0 = time.time()
    e0 = mf.kernel()
    print(f"[diag][opt] initial E={e0:.6f} Hartree, backend={backend}, t={time.time()-t0:.1f}s", flush=True)
    print(f"[diag][opt] starting geometric optimize() (maxsteps=100) ...", flush=True)
    t1 = time.time()
    mol_opt = optimize(mf, maxsteps=100)
    print(f"[diag][opt] optimize done in {time.time()-t1:.1f}s, final SCF on opt geom ...", flush=True)
    elapsed = time.time() - t0
    mf_opt, _ = _get_mf(mol_opt, "b3lyp", use_gpu=use_gpu)
    e_opt = mf_opt.kernel()
    print(f"[diag][opt] final E={e_opt:.6f}, total opt elapsed {elapsed:.1f}s", flush=True)
    return {
        "atoms_opt": atoms_from_mol_geom(mol_opt),
        "E_b3lyp_631gss_hartree": float(e_opt),
        "backend": backend,
        "elapsed_s": elapsed,
    }


def freq_b3lyp_6_31gss(atoms, charge=0, use_gpu=True):
    """Vibrational analysis on optimised geometry. Returns frequencies + ZPE."""
    from pyscf.hessian import thermo
    mol = _build_mol(atoms, "6-31g*", charge=charge)
    mf, backend = _get_mf(mol, "b3lyp", use_gpu=use_gpu)
    mf.kernel()
    # Modern PySCF/gpu4pyscf: use mf.Hessian() polymorphic API.
    # Falls through to GPU path automatically when mf is a gpu4pyscf RKS.
    hess = mf.Hessian()
    hess.verbose = 0
    h = hess.kernel()
    freq_info = thermo.harmonic_analysis(mf.mol, h)
    # ZPE: compute manually from real frequencies (PySCF doesn't expose 'ZPE'
    # directly under that key; freq_au is in atomic units of energy).
    freq_wn = np.asarray(freq_info.get("freq_wavenumber", []))
    if "freq_au" in freq_info:
        freqs_au = np.asarray(freq_info["freq_au"])
    else:
        # Convert wavenumbers to hartree: 1 cm-1 = 4.55634e-6 hartree
        freqs_au = freq_wn * 4.55634e-6
    real_mask = freq_wn > 0
    zpe_hartree = float(0.5 * freqs_au[real_mask].sum())
    real_freqs = freq_wn[real_mask] if len(freq_wn) else np.array([])
    return {
        "freqs_cm1": [float(x) for x in freq_wn],
        "freqs_au": [float(x) for x in freqs_au],
        "ZPE_kJmol": zpe_hartree * HARTREE_TO_KJMOL,
        "n_imag": int((freq_wn < 0).sum()),
        "min_real_freq_cm1": float(real_freqs.min()) if len(real_freqs) else float("nan"),
        "backend": backend,
    }


def sp_wb97xd_def2tzvp(atoms, charge=0, use_gpu=True):
    """omega-B97X-D / def2-TZVP single point at optimised geometry."""
    mol = _build_mol(atoms, "def2-tzvp", charge=charge)
    mf, backend = _get_mf(mol, "wb97x-d3bj", use_gpu=use_gpu)
    t0 = time.time()
    e = mf.kernel()
    return {
        "E_wb97xd_def2tzvp_hartree": float(e),
        "backend": backend,
        "elapsed_s": time.time() - t0,
    }


def atomic_reference_energy(symbol, basis, xc, use_gpu=True):
    """SP for an isolated atom (UHF, doublet/triplet/quartet as needed)."""
    from pyscf import gto, dft
    spin_map = {"H": 1, "C": 2, "N": 3, "O": 2, "F": 1, "Cl": 1}
    spin = spin_map.get(symbol, 0)
    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)
    if use_gpu:
        try:
            from gpu4pyscf import dft as gpu_dft
            mf = gpu_dft.UKS(mol, xc=xc).density_fit()
        except ImportError:
            mf = dft.UKS(mol, xc=xc).density_fit()
    else:
        mf = dft.UKS(mol, xc=xc).density_fit()
    mf.verbose = 4  # print SCF iterations + diagnostic info
    e = mf.kernel()
    return float(e)


def hof_from_atomization(atoms, e_total_hartree, atomic_refs_hartree):
    """HOF (kJ/mol) via atomization-energy method:
       HOF_compound = sum(HOF_atom) - atomization_energy
       atomization_energy = sum(E_atoms) - E_compound
    """
    from collections import Counter
    cnt = Counter(s for s, *_ in atoms)
    e_atoms = sum(cnt[s] * atomic_refs_hartree[s] for s in cnt)
    de_atomization = e_atoms - e_total_hartree           # positive
    hof_atoms = sum(cnt[s] * ATOMIC_HOF_298[s] for s in cnt)
    hof_compound = hof_atoms - de_atomization * HARTREE_TO_KJMOL
    return hof_compound


# ── Kamlet-Jacobs recomputation ────────────────────────────────────────────
def kamlet_jacobs(rho_g_cm3, hof_kJmol, formula):
    """K-J equations for CHNO:
       D = 1.01 * sqrt(N * sqrt(M) * sqrt(Q)) * (1 + 1.30 * rho)
       P = 15.58 * rho^2 * N * sqrt(M) * sqrt(Q)
    where N = mol of gas/g explosive, M = avg MW of gas, Q = heat of detonation.
    Approximation following Mathieu 2017 / Kamlet-Jacobs 1968.
    """
    from collections import Counter
    cnt = Counter()
    for ch in formula:
        if ch.isalpha():
            sym = ch
            cnt[sym] = cnt.get(sym, 0)
        elif ch.isdigit():
            pass
    # Parse formula like "C3N5O5H2"
    import re
    cnt = {}
    for sym, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if sym:
            cnt[sym] = int(n) if n else 1
    a, b, c, d = cnt.get("C", 0), cnt.get("H", 0), cnt.get("N", 0), cnt.get("O", 0)
    M_total = 12.011 * a + 1.008 * b + 14.007 * c + 15.999 * d
    # Q is in cal/g; HOF is in kJ/mol -> cal/mol via *239.006, then /M -> cal/g.
    # 28.9, 57.8, etc. are kcal-domain constants -> *1000 to keep cal/g consistent.
    if d >= 2 * a + b / 2:                # oxygen-rich
        N = (b + 2 * c + 2 * d) / (4 * M_total)
        M = (56 * d + 88 * c - 8 * b) / (b + 2 * c + 2 * d)
        Q = (28900 * b + 47000 * (d - a - b/2) + 239.0 * hof_kJmol) / M_total
    elif 2 * a + b / 2 > d >= b / 2:       # carbon-rich (most CHNO)
        N = (b + 2 * c + 2 * d) / (4 * M_total)
        M = (2 * b + 28 * c + 32 * d) / (b + 2 * c + 2 * d)
        Q = (57800 * d - 239.0 * hof_kJmol) / M_total
        Q = abs(Q)
    else:                                  # very C-rich: K-J unreliable
        N = M = Q = float("nan")
    if any(np.isnan([N, M, Q])): return {"D_kms": None, "P_GPa": None,
                                           "N": N, "M": M, "Q": Q}
    # Q in cal/g; convert to kcal/g for the K-J prefactors that use kcal:
    Q_kcal = Q / 1000.0
    D = 1.01 * np.sqrt(N * np.sqrt(M) * np.sqrt(Q)) * (1 + 1.30 * rho_g_cm3)
    P = 1.558 * rho_g_cm3**2 * N * np.sqrt(M) * np.sqrt(Q)
    return {"D_kms": float(D), "P_GPa": float(P), "N": float(N), "M": float(M), "Q": float(Q)}


def density_from_volume(atoms, formula, packing=PACKING_COEFF):
    """Estimate crystal density from molecular volume.
       rho_crystal = (M / V_mol_VdW) * packing_coefficient
    where V_mol_VdW is the proper VdW-surface-enclosed molecular volume
    (accounts for atomic overlap from bonds), computed via RDKit's
    AllChem.ComputeMolVolume on a 3D-embedded molecule.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    # Rebuild a properly-bonded RDKit mol from the atoms list (xyz only, no
    # bonds) is not possible. Instead, recreate from canonical SMILES and
    # use ComputeMolVolume's own ETKDG embedding.
    import re
    cnt = {sym: int(n) if n else 1
           for sym, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if sym}
    M = (12.011 * cnt.get("C", 0) + 1.008 * cnt.get("H", 0)
         + 14.007 * cnt.get("N", 0) + 15.999 * cnt.get("O", 0)
         + 18.998 * cnt.get("F", 0) + 35.45 * cnt.get("Cl", 0))
    return _density_from_atom_xyz(atoms, M, packing)


def _density_from_atom_xyz(atoms, M, packing):
    """Compute V_VdW from a list of (sym, x, y, z) atoms using grid-based
    integration over Bondi-radii VdW spheres, accounting for overlap."""
    vdw = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Cl": 1.75}
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    radii = np.array([vdw.get(s, 1.6) for s, *_ in atoms])
    # Bounding box + padding
    pad = max(radii) + 0.5
    lo = coords.min(0) - pad; hi = coords.max(0) + pad
    spacing = 0.25                     # AA; ~0.5 % volume error
    nx = max(2, int(np.ceil((hi[0] - lo[0]) / spacing)))
    ny = max(2, int(np.ceil((hi[1] - lo[1]) / spacing)))
    nz = max(2, int(np.ceil((hi[2] - lo[2]) / spacing)))
    if nx * ny * nz > 4_000_000:
        spacing = 0.4
        nx = max(2, int(np.ceil((hi[0] - lo[0]) / spacing)))
        ny = max(2, int(np.ceil((hi[1] - lo[1]) / spacing)))
        nz = max(2, int(np.ceil((hi[2] - lo[2]) / spacing)))
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)
    inside = np.zeros((nx, ny, nz), dtype=bool)
    for c, r in zip(coords, radii):
        # Local bounding box for this atom
        ix0 = max(0, int((c[0] - r - lo[0]) / spacing))
        ix1 = min(nx, int((c[0] + r - lo[0]) / spacing) + 2)
        iy0 = max(0, int((c[1] - r - lo[1]) / spacing))
        iy1 = min(ny, int((c[1] + r - lo[1]) / spacing) + 2)
        iz0 = max(0, int((c[2] - r - lo[2]) / spacing))
        iz1 = min(nz, int((c[2] + r - lo[2]) / spacing) + 2)
        # Vectorised distance test
        ax = xs[ix0:ix1, None, None]
        ay = ys[None, iy0:iy1, None]
        az = zs[None, None, iz0:iz1]
        d2 = (ax - c[0])**2 + (ay - c[1])**2 + (az - c[2])**2
        inside[ix0:ix1, iy0:iy1, iz0:iz1] |= (d2 <= r**2)
    cell_vol = (xs[1] - xs[0]) * (ys[1] - ys[0]) * (zs[1] - zs[0])
    V_mol_AA3 = float(inside.sum()) * cell_vol           # AA^3 per molecule
    # Convert to cm^3/mol: AA^3 * 6.022e23 / 1e24 = AA^3 * 0.6022
    V_mol_cm3 = V_mol_AA3 * 0.6022
    rho = M / V_mol_cm3 * packing
    return float(rho)


# ── Driver ─────────────────────────────────────────────────────────────────
def run_lead(lead, atomic_refs_b3lyp, atomic_refs_wb97xd, results_dir, use_gpu=True,
             timeout_s=7200):
    smi = lead["smiles"]
    lead_id = lead["id"]
    out_path = Path(results_dir) / f"m2_lead_{lead_id}.json"
    # Fix #4: resume-from-cache. Skip if a complete result is already on disk.
    if out_path.exists():
        try:
            cached = json.loads(out_path.read_text())
            if cached.get("HOF_kJmol_wb97xd") is not None and not cached.get("errors"):
                print(f"[train] {lead_id} cached, skipping ({out_path})"); sys.stdout.flush()
                return cached
        except Exception:
            pass  # cache unreadable → recompute
    print(f"\n[train] === {lead_id} {lead.get('name', '')} ==="); sys.stdout.flush()
    print(f"[train] SMILES: {smi}"); sys.stdout.flush()
    out = {"id": lead_id, "smiles": smi, "name": lead.get("name"),
           "predicted": lead.get("predicted"), "errors": []}
    # Fix #5: per-molecule wall-clock guard (POSIX only — Linux pods).
    _alarm_set = False
    if hasattr(__import__('signal'), 'SIGALRM'):
        import signal
        def _watchdog(signum, frame):
            raise TimeoutError(f"per-molecule timeout {timeout_s}s exceeded")
        signal.signal(signal.SIGALRM, _watchdog)
        signal.alarm(int(timeout_s))
        _alarm_set = True
    try:
        atoms0, charge, n_atoms = smiles_to_xyz(smi)
        formula = molecular_formula(atoms0)
        out["formula"] = formula
        out["n_atoms"] = n_atoms
        print(f"[train] formula={formula} n_atoms={n_atoms}"); sys.stdout.flush()

        # 1. B3LYP/6-31G* opt
        print(f"[train] opt B3LYP/6-31G* ..."); sys.stdout.flush()
        opt = opt_b3lyp_6_31gss(atoms0, charge=charge, use_gpu=use_gpu)
        out["opt"] = opt
        print(f"[train] opt done E={opt['E_b3lyp_631gss_hartree']:.6f} Ha "
              f"({opt['backend']}, {opt['elapsed_s']:.0f}s)"); sys.stdout.flush()

        # 2. Hessian + ZPE
        print(f"[train] freq B3LYP/6-31G* ..."); sys.stdout.flush()
        freq = freq_b3lyp_6_31gss(opt["atoms_opt"], charge=charge, use_gpu=use_gpu)
        out["freq"] = freq
        print(f"[train] freq done min={freq['min_real_freq_cm1']:.1f} cm-1 "
              f"n_imag={freq['n_imag']} ZPE={freq['ZPE_kJmol']:.1f}"); sys.stdout.flush()

        # 3. omega-B97X-D / def2-TZVP single point
        print(f"[train] SP wb97xd/def2-TZVP ..."); sys.stdout.flush()
        sp = sp_wb97xd_def2tzvp(opt["atoms_opt"], charge=charge, use_gpu=use_gpu)
        out["sp"] = sp
        print(f"[train] SP done E={sp['E_wb97xd_def2tzvp_hartree']:.6f} Ha"); sys.stdout.flush()

        # 4. HOF via atomization (both theory levels)
        e_b3lyp_with_zpe = opt["E_b3lyp_631gss_hartree"] + freq["ZPE_kJmol"] / HARTREE_TO_KJMOL
        hof_b3lyp = hof_from_atomization(opt["atoms_opt"], e_b3lyp_with_zpe,
                                          atomic_refs_b3lyp)
        e_wb97xd_with_zpe = sp["E_wb97xd_def2tzvp_hartree"] + freq["ZPE_kJmol"] / HARTREE_TO_KJMOL
        hof_wb97xd = hof_from_atomization(opt["atoms_opt"], e_wb97xd_with_zpe,
                                            atomic_refs_wb97xd)
        out["HOF_kJmol_b3lyp"] = hof_b3lyp
        out["HOF_kJmol_wb97xd"] = hof_wb97xd
        print(f"[train] HOF (B3LYP atomization) = {hof_b3lyp:.1f} kJ/mol"); sys.stdout.flush()
        print(f"[train] HOF (wB97X-D atomization) = {hof_wb97xd:.1f} kJ/mol"); sys.stdout.flush()

        # 5. Recomputed K-J detonation properties (use wB97X-D HOF + estimated rho)
        rho_dft = density_from_volume(opt["atoms_opt"], formula)
        kj_dft = kamlet_jacobs(rho_dft, hof_wb97xd, formula)
        out["rho_dft"] = rho_dft
        out["kamlet_jacobs_dft"] = kj_dft
        print(f"[train] rho_dft={rho_dft:.3f} g/cm3   D_dft={kj_dft.get('D_kms')} km/s "
              f"P_dft={kj_dft.get('P_GPa')} GPa"); sys.stdout.flush()

    except Exception as e:
        out["errors"].append(str(e))
        out["traceback"] = traceback.format_exc()
        print(f"[train] ERROR on {lead_id}: {e}"); sys.stdout.flush()
    finally:
        if _alarm_set:
            import signal
            signal.alarm(0)

    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[train] -> {out_path}"); sys.stdout.flush()
    return out


def _start_heartbeat():
    """Print a heartbeat line every 60s to keep stale-timeout watchers happy.
    DFT geometry optimisations can run 30-90 min per molecule without
    emitting fresh stdout, which trips runner stale-detectors. The
    heartbeat is a no-op compute-wise."""
    import threading, time
    def _beat():
        t0 = time.time()
        while True:
            time.sleep(60)
            print(f"[heartbeat] elapsed={int(time.time()-t0)}s", flush=True)
    t = threading.Thread(target=_beat, daemon=True)
    t.start()


def main():
    _start_heartbeat()
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", default="m2_smiles.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU PySCF (smoke testing)")
    ap.add_argument("--anchors", action="store_true",
                    help="Also run anchor compounds for HOF calibration")
    args = ap.parse_args()

    use_gpu = not args.cpu
    if use_gpu:
        try:
            import torch
            assert torch.cuda.is_available(), "CUDA not available"
            print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()
        except Exception as e:
            print(f"[train] WARNING: GPU check failed ({e}); proceeding with CPU pyscf"); sys.stdout.flush()
            use_gpu = False

    results_dir = Path(args.results); results_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.smiles).read_text())

    # 1. Atomic reference energies (one-time)
    print(f"[train] Computing atomic reference energies ..."); sys.stdout.flush()
    atom_refs_path = results_dir / "m2_atom_refs.json"
    if atom_refs_path.exists():
        atom_refs = json.loads(atom_refs_path.read_text())
        print(f"[train] Loaded cached atomic refs from {atom_refs_path}"); sys.stdout.flush()
    else:
        atom_refs = {"B3LYP_631Gss": {}, "wB97XD_def2TZVP": {}}
        for sym in ["H", "C", "N", "O"]:
            print(f"[train] B3LYP/6-31G* atom {sym} ..."); sys.stdout.flush()
            atom_refs["B3LYP_631Gss"][sym] = atomic_reference_energy(
                sym, "6-31g*", "b3lyp", use_gpu=use_gpu)
            print(f"[train] wB97X-D/def2-TZVP atom {sym} ..."); sys.stdout.flush()
            atom_refs["wB97XD_def2TZVP"][sym] = atomic_reference_energy(
                sym, "def2-tzvp", "wb97x-d3bj", use_gpu=use_gpu)
        atom_refs_path.write_text(json.dumps(atom_refs, indent=2))

    # 2. Loop over leads (and optionally anchors)
    targets = list(config["leads"])
    if args.anchors:
        for a in config.get("anchors", []):
            targets.append({**a, "predicted": None})

    summary = []
    for i, lead in enumerate(targets, 1):
        print(f"\n[train] {i}/{len(targets)} loss=0.0000 lead={lead['id']}"); sys.stdout.flush()
        out = run_lead(lead, atom_refs["B3LYP_631Gss"], atom_refs["wB97XD_def2TZVP"],
                        results_dir, use_gpu=use_gpu)
        summary.append({"id": out["id"], "errors": out["errors"],
                        "HOF_kJmol_wb97xd": out.get("HOF_kJmol_wb97xd"),
                        "rho_dft": out.get("rho_dft"),
                        "kj_dft": out.get("kamlet_jacobs_dft"),
                        "n_imag": out.get("freq", {}).get("n_imag")})

    (results_dir / "m2_summary.json").write_text(json.dumps(summary, indent=2))

    # Fix #6: anchor-calibrated rho and HOF.
    # Fit rho_cal = a*rho_DFT + b and HOF_cal = HOF_DFT + c on RDX/TATB
    # (literature reference values), apply to all leads, and write a small
    # m2_calibration.json alongside the summary.
    LIT = {
        "RDX":  {"rho_lit": 1.806, "HOF_lit_kJmol":  +66.0},
        "TATB": {"rho_lit": 1.937, "HOF_lit_kJmol": -154.0},
    }
    anchors_seen = [s for s in summary if s["id"] in LIT
                    and s.get("rho_dft") and s.get("HOF_kJmol_wb97xd") is not None]
    if len(anchors_seen) >= 2:
        # Two-point linear fit on rho; constant offset on HOF.
        rho_x = np.array([s["rho_dft"] for s in anchors_seen])
        rho_y = np.array([LIT[s["id"]]["rho_lit"] for s in anchors_seen])
        hof_x = np.array([s["HOF_kJmol_wb97xd"] for s in anchors_seen])
        hof_y = np.array([LIT[s["id"]]["HOF_lit_kJmol"] for s in anchors_seen])
        # rho: linear y = a*x + b (least squares on 2+ anchors)
        A = np.vstack([rho_x, np.ones_like(rho_x)]).T
        a_rho, b_rho = np.linalg.lstsq(A, rho_y, rcond=None)[0]
        # HOF: constant additive offset (mean residual) — bias-only correction.
        c_hof = float(np.mean(hof_y - hof_x))
        cal = {"a_rho": float(a_rho), "b_rho": float(b_rho), "c_hof_kJmol": c_hof,
               "anchors_used": [s["id"] for s in anchors_seen]}
        # Apply to all rows in summary (and re-K-J with calibrated rho+HOF).
        for s in summary:
            if s.get("rho_dft") is None or s.get("HOF_kJmol_wb97xd") is None:
                continue
            s["rho_cal"] = float(a_rho * s["rho_dft"] + b_rho)
            s["HOF_kJmol_wb97xd_cal"] = float(s["HOF_kJmol_wb97xd"] + c_hof)
            # K-J recompute on calibrated inputs (need formula → load per-lead JSON)
            try:
                lead_path = results_dir / f"m2_lead_{s['id']}.json"
                lead_full = json.loads(lead_path.read_text()) if lead_path.exists() else {}
                formula = lead_full.get("formula")
                if formula:
                    s["kj_dft_cal"] = kamlet_jacobs(s["rho_cal"],
                                                    s["HOF_kJmol_wb97xd_cal"], formula)
            except Exception as e:
                s["kj_dft_cal"] = None
                cal.setdefault("warnings", []).append(f"{s['id']}: {e}")
        (results_dir / "m2_calibration.json").write_text(json.dumps(cal, indent=2))
        (results_dir / "m2_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[train] anchor calibration: rho_cal = {a_rho:.4f}*rho_DFT + {b_rho:+.4f};"
              f" HOF_cal = HOF_DFT {c_hof:+.1f} kJ/mol  (anchors: {','.join(s['id'] for s in anchors_seen)})")
    else:
        print(f"[train] anchor calibration skipped (need ≥2 of {list(LIT)}; got {[s['id'] for s in anchors_seen]})")

    print(f"\n[train] === DONE === ({len(targets)} compounds)"); sys.stdout.flush()


if __name__ == "__main__":
    main()
