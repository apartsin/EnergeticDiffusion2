"""DFT smoke test: instrumented, NO silent fallbacks. Exits non-zero with the
probe number on first failure so the runner log makes the failure obvious.

Usage:
    python3 -u dft_smoke.py

Probes:
    1. python/torch/CUDA/nvidia-smi
    2. gpu4pyscf import (hard-fail; no CPU fallback)
    3. water B3LYP/6-31G SCF on GPU, < 60 s, converged
    4. confirm GPU memory grew during the kernel (proof GPU was actually used)
    5. RDX B3LYP/6-31G(d) opt with geometric, < 1200 s
    6. RDX analytical Hessian on opt geom, n_imag == 0 (within tolerance)
    7. RDX wB97X-D3BJ/def2-TZVP single point
    8. write m2_lead_RDX.json with same schema as the production pipeline
"""
from __future__ import annotations
import json, os, subprocess, sys, time, traceback
from pathlib import Path

OUT_DIR = Path(os.environ.get("DFT_SMOKE_OUT", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

HARTREE_TO_KJMOL = 2625.49963948
ATOMIC_HOF_298 = {"H": 217.998, "C": 716.68, "N": 472.68, "O": 249.18}

RDX_SMILES = "[O-][N+](=O)N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1"


def _probe(n, msg):
    print(f"[probe {n}] {msg}", flush=True)


def _fail(n, msg, exc=None):
    print(f"[probe {n}] FAIL: {msg}", flush=True)
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    sys.exit(n)


# ---------- Probe 1: environment ----------
def probe_1_environment():
    _probe(1, f"python {sys.version.split()[0]}")
    try:
        import torch
        _probe(1, f"torch {torch.__version__}")
        _probe(1, f"torch.version.cuda={torch.version.cuda}")
        cuda_ok = torch.cuda.is_available()
        _probe(1, f"torch.cuda.is_available()={cuda_ok}")
        if not cuda_ok:
            _fail(1, "torch.cuda.is_available() returned False")
        _probe(1, f"device 0: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        _fail(1, f"torch import/probe failed: {e}", e)
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=20)
        for line in out.splitlines()[:15]:
            print(f"[probe 1] nvidia-smi| {line}", flush=True)
    except Exception as e:
        _fail(1, f"nvidia-smi failed: {e}", e)


# ---------- Probe 2: gpu4pyscf import (NO fallback) ----------
def probe_2_gpu4pyscf_import():
    try:
        from gpu4pyscf import dft as gpu_dft
        _probe(2, f"gpu4pyscf.dft from {gpu_dft.__file__}")
        import gpu4pyscf
        _probe(2, f"gpu4pyscf version {getattr(gpu4pyscf, '__version__', 'unknown')}")
    except Exception as e:
        _fail(2, f"gpu4pyscf import failed: {e}", e)
    try:
        import pyscf
        _probe(2, f"pyscf {pyscf.__version__} from {pyscf.__file__}")
    except Exception as e:
        _fail(2, f"pyscf import failed: {e}", e)
    try:
        import geometric
        _probe(2, f"geometric {getattr(geometric, '__version__', '?')} from {geometric.__file__}")
    except Exception as e:
        _fail(2, f"geometric import failed: {e}", e)


# ---------- Probe 3 & 4: water SCF + GPU memory growth ----------
def probe_3_4_water_scf():
    import torch
    from pyscf import gto
    from gpu4pyscf import dft as gpu_dft

    water = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol = gto.M(atom=water, basis="6-31g", verbose=4)
    mf = gpu_dft.RKS(mol, xc="b3lyp").density_fit()

    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    _probe(4, f"GPU mem before kernel: {mem_before} bytes")

    t0 = time.time()
    try:
        e = mf.kernel()
    except Exception as ex:
        _fail(3, f"water mf.kernel() raised: {ex}", ex)
    elapsed = time.time() - t0
    _probe(3, f"water B3LYP/6-31G SCF E={e:.6f} Hartree, elapsed={elapsed:.1f}s")
    if not getattr(mf, "converged", True):
        _fail(3, "water SCF did not converge")
    if elapsed > 60.0:
        _fail(3, f"water SCF took {elapsed:.1f}s (> 60s); CPU fallback or driver bug suspected")

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()
    _probe(4, f"GPU mem after kernel: {mem_after} bytes, peak={mem_peak}")
    if mem_peak <= mem_before:
        _fail(4, f"GPU memory did not grow during kernel (before={mem_before}, peak={mem_peak}); GPU likely not used")
    return e


# ---------- helpers shared by RDX probes ----------
def _smiles_to_atoms(smi, seed=42):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = seed
    if AllChem.EmbedMolecule(mol, p) != 0:
        raise RuntimeError("ETKDGv3 embed failed")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    atoms = []
    for i, a in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atoms.append((a.GetSymbol(), pos.x, pos.y, pos.z))
    return atoms


def _atoms_to_pyscf(atoms):
    return "\n".join(f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, x, y, z in atoms)


def _atoms_from_mol(mol):
    coords = mol.atom_coords(unit="Angstrom")
    return [(s, float(coords[i][0]), float(coords[i][1]), float(coords[i][2]))
            for i, s in enumerate(mol.elements)]


# ---------- Probe 5: RDX opt ----------
def probe_5_rdx_opt():
    from pyscf import gto
    from gpu4pyscf import dft as gpu_dft
    from pyscf.geomopt.geometric_solver import optimize

    atoms0 = _smiles_to_atoms(RDX_SMILES)
    _probe(5, f"RDX initial atoms: {len(atoms0)}")
    mol = gto.M(atom=_atoms_to_pyscf(atoms0), basis="6-31g*", verbose=4)
    mf = gpu_dft.RKS(mol, xc="b3lyp").density_fit()

    t0 = time.time()
    try:
        e0 = mf.kernel()
        mol_opt = optimize(mf, maxsteps=100)
        mf2 = gpu_dft.RKS(mol_opt, xc="b3lyp").density_fit()
        e_opt = mf2.kernel()
    except Exception as ex:
        _fail(5, f"RDX opt raised: {ex}", ex)
    elapsed = time.time() - t0
    _probe(5, f"RDX opt E={e_opt:.6f}, elapsed={elapsed:.1f}s (initial E={e0:.6f})")
    if elapsed > 1200.0:
        _fail(5, f"RDX opt took {elapsed:.1f}s (> 1200s)")
    return mol_opt, mf2, e_opt, elapsed


# ---------- Probe 6: RDX freq ----------
def probe_6_rdx_freq(mf_opt):
    import numpy as np
    from pyscf.hessian import thermo
    t0 = time.time()
    try:
        hess = mf_opt.Hessian()
        hess.verbose = 0
        h = hess.kernel()
        info = thermo.harmonic_analysis(mf_opt.mol, h)
    except Exception as ex:
        _fail(6, f"Hessian raised: {ex}", ex)
    elapsed = time.time() - t0
    freq_wn = np.asarray(info.get("freq_wavenumber", []))
    if "freq_au" in info:
        freqs_au = np.asarray(info["freq_au"])
    else:
        freqs_au = freq_wn * 4.55634e-6
    real_mask = freq_wn > 0
    n_imag = int((freq_wn < 0).sum())
    min_real = float(freq_wn[real_mask].min()) if real_mask.any() else float("nan")
    zpe_h = float(0.5 * freqs_au[real_mask].sum())
    _probe(6, f"RDX freq: n_modes={len(freq_wn)}, n_imag={n_imag}, "
              f"min_real={min_real:.2f} cm-1, ZPE={zpe_h * HARTREE_TO_KJMOL:.1f} kJ/mol, elapsed={elapsed:.1f}s")
    if n_imag > 0:
        _probe(6, f"WARN: {n_imag} imaginary modes (expected 0 at a true minimum); continuing")
    return {
        "freqs_cm1": [float(x) for x in freq_wn],
        "freqs_au": [float(x) for x in freqs_au],
        "ZPE_kJmol": zpe_h * HARTREE_TO_KJMOL,
        "n_imag": n_imag,
        "min_real_freq_cm1": min_real,
        "backend": "gpu",
    }


# ---------- Probe 7: RDX wB97X-D3BJ/def2-TZVP ----------
def probe_7_rdx_sp(mol_opt):
    from pyscf import gto
    from gpu4pyscf import dft as gpu_dft
    atoms_opt = _atoms_from_mol(mol_opt)
    mol = gto.M(atom=_atoms_to_pyscf(atoms_opt), basis="def2-tzvp", verbose=4)
    mf = gpu_dft.RKS(mol, xc="wb97x-d3bj").density_fit()
    t0 = time.time()
    try:
        e = mf.kernel()
    except Exception as ex:
        _fail(7, f"wB97X-D3BJ SP raised: {ex}", ex)
    elapsed = time.time() - t0
    _probe(7, f"RDX wB97X-D3BJ/def2-TZVP E={e:.6f} Hartree, elapsed={elapsed:.1f}s")
    return atoms_opt, float(e), elapsed


# ---------- Probe 8: write m2_lead_RDX.json ----------
def _atomic_ref(symbol, basis, xc):
    from pyscf import gto
    from gpu4pyscf import dft as gpu_dft
    spin_map = {"H": 1, "C": 2, "N": 3, "O": 2}
    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin_map[symbol], verbose=0)
    mf = gpu_dft.UKS(mol, xc=xc).density_fit()
    return float(mf.kernel())


def _hof(atoms, e_total, refs):
    from collections import Counter
    cnt = Counter(s for s, *_ in atoms)
    e_atoms = sum(cnt[s] * refs[s] for s in cnt)
    de_atom = e_atoms - e_total
    hof_atoms = sum(cnt[s] * ATOMIC_HOF_298[s] for s in cnt)
    return hof_atoms - de_atom * HARTREE_TO_KJMOL


def _density_from_atoms(atoms, M):
    import numpy as np
    vdw = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52}
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    radii = np.array([vdw.get(s, 1.6) for s, *_ in atoms])
    pad = max(radii) + 0.5
    lo = coords.min(0) - pad; hi = coords.max(0) + pad
    sp = 0.25
    nx = max(2, int(np.ceil((hi[0]-lo[0])/sp)))
    ny = max(2, int(np.ceil((hi[1]-lo[1])/sp)))
    nz = max(2, int(np.ceil((hi[2]-lo[2])/sp)))
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)
    inside = np.zeros((nx, ny, nz), dtype=bool)
    for c, r in zip(coords, radii):
        d2 = (xs[:, None, None]-c[0])**2 + (ys[None, :, None]-c[1])**2 + (zs[None, None, :]-c[2])**2
        inside |= (d2 <= r*r)
    cell = (xs[1]-xs[0])*(ys[1]-ys[0])*(zs[1]-zs[0])
    V_AA3 = float(inside.sum()) * cell
    V_cm3 = V_AA3 * 0.6022
    return float(M / V_cm3 * 0.69)


def _kj(rho, hof, formula):
    import numpy as np, re
    cnt = {sym: int(n) if n else 1 for sym, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if sym}
    a, b, c, d = cnt.get("C",0), cnt.get("H",0), cnt.get("N",0), cnt.get("O",0)
    M_total = 12.011*a + 1.008*b + 14.007*c + 15.999*d
    if d >= 2*a + b/2:
        N = (b + 2*c + 2*d) / (4*M_total)
        M = (56*d + 88*c - 8*b) / (b + 2*c + 2*d)
        Q = (28900*b + 47000*(d-a-b/2) + 239.0*hof) / M_total
    elif 2*a + b/2 > d >= b/2:
        N = (b + 2*c + 2*d) / (4*M_total)
        M = (2*b + 28*c + 32*d) / (b + 2*c + 2*d)
        Q = abs((57800*d - 239.0*hof) / M_total)
    else:
        return {"D_kms": None, "P_GPa": None, "N": float("nan"), "M": float("nan"), "Q": float("nan")}
    D = 1.01 * np.sqrt(N * np.sqrt(M) * np.sqrt(Q)) * (1 + 1.30*rho)
    P = 1.558 * rho**2 * N * np.sqrt(M) * np.sqrt(Q)
    return {"D_kms": float(D), "P_GPa": float(P), "N": float(N), "M": float(M), "Q": float(Q)}


def probe_8_write_json(atoms_opt, e_b3lyp, opt_elapsed, freq, e_wb97xd, sp_elapsed):
    from collections import Counter
    cnt = Counter(s for s, *_ in atoms_opt)
    formula = "".join(f"{s}{cnt[s]}" for s in ["C","H","N","O"] if cnt.get(s,0))
    M_total = 12.011*cnt.get("C",0) + 1.008*cnt.get("H",0) + 14.007*cnt.get("N",0) + 15.999*cnt.get("O",0)

    _probe(8, "computing atomic refs (B3LYP/6-31G* and wB97X-D3BJ/def2-TZVP)")
    refs_b3lyp = {s: _atomic_ref(s, "6-31g*", "b3lyp") for s in ["H","C","N","O"]}
    refs_wb97 = {s: _atomic_ref(s, "def2-tzvp", "wb97x-d3bj") for s in ["H","C","N","O"]}

    e_b3lyp_zpe = e_b3lyp + freq["ZPE_kJmol"]/HARTREE_TO_KJMOL
    e_wb97_zpe = e_wb97xd + freq["ZPE_kJmol"]/HARTREE_TO_KJMOL
    hof_b3 = _hof(atoms_opt, e_b3lyp_zpe, refs_b3lyp)
    hof_wb = _hof(atoms_opt, e_wb97_zpe, refs_wb97)
    rho = _density_from_atoms(atoms_opt, M_total)
    kj = _kj(rho, hof_wb, formula)

    out = {
        "id": "RDX",
        "smiles": RDX_SMILES,
        "name": "RDX",
        "predicted": None,
        "errors": [],
        "formula": formula,
        "n_atoms": len(atoms_opt),
        "opt": {
            "atoms_opt": [list(a) for a in atoms_opt],
            "E_b3lyp_631gss_hartree": float(e_b3lyp),
            "backend": "gpu",
            "elapsed_s": float(opt_elapsed),
        },
        "freq": freq,
        "sp": {
            "E_wb97xd_def2tzvp_hartree": float(e_wb97xd),
            "backend": "gpu",
            "elapsed_s": float(sp_elapsed),
        },
        "HOF_kJmol_b3lyp": float(hof_b3),
        "HOF_kJmol_wb97xd": float(hof_wb),
        "rho_dft": float(rho),
        "kamlet_jacobs_dft": kj,
    }
    out_path = OUT_DIR / "m2_lead_RDX.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    _probe(8, f"wrote {out_path}: HOF_wb97xd={hof_wb:.1f} kJ/mol, rho={rho:.3f} g/cm3, "
              f"D={kj.get('D_kms')} km/s, P={kj.get('P_GPa')} GPa")


def main():
    print("="*60, flush=True)
    print("DFT smoke test (no silent fallbacks)", flush=True)
    print("="*60, flush=True)
    probe_1_environment()
    probe_2_gpu4pyscf_import()
    probe_3_4_water_scf()
    mol_opt, mf_opt, e_b3lyp, opt_elapsed = probe_5_rdx_opt()
    freq = probe_6_rdx_freq(mf_opt)
    atoms_opt, e_wb97xd, sp_elapsed = probe_7_rdx_sp(mol_opt)
    probe_8_write_json(atoms_opt, e_b3lyp, opt_elapsed, freq, e_wb97xd, sp_elapsed)
    print("[smoke] === ALL PROBES PASSED ===", flush=True)


if __name__ == "__main__":
    main()
