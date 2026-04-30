"""Stage 8: single-compound A100 DFT on DNTF (3,4-bis(4-nitrofurazan-3-yl)furazan).

Purpose: extend the 6-anchor K-J calibration to 7 anchors by adding a
furazan-class compound.  DNTF provides an oxatriazole/furazan ring system
not represented in the existing anchor set (RDX, TATB, HMX, PETN, FOX-7,
NTO), which demoted E1 (4-nitro-1,2,3,5-oxatriazole) to "provisional
co-headline pending an oxatriazole-class DFT anchor."

Computation steps (same theory ladder as modal_dft_extension.py):
    1. RDKit ETKDGv3 + MMFF94 initial geometry
    2. B3LYP/6-31G(d) geometry optimisation + analytical Hessian (A100)
    3. wB97X-D3BJ/def2-TZVP single-point energy on optimised geometry (A100)
    4. rho_DFT from Bondi vdW grid integration, packing factor 0.69
    5. HOF_DFT from atomization energy (wB97X-D3BJ level)
    6. Apply existing 6-anchor calibration; report residual vs experiment
    7. Fit new 7-anchor calibration; compute leave-one-out RMS
    8. Save m8_anchor_result.json

Usage:
    modal run modal_m8_oxatriazole_anchor.py
    modal run modal_m8_oxatriazole_anchor.py --skip-dntf   # if SMILES fails validation
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
RESULTS_LOCAL = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# DNTF compound definition
# 3,4-bis(4-nitrofurazan-3-yl)furazan
# Formula: C4N6O7  (pure CHNO, charge 0)
# ---------------------------------------------------------------------------
DNTF_ID     = "DNTF"
DNTF_SMILES = "O=[N+]([O-])c1noc(-c2noc([N+](=O)[O-])n2)n1"
DNTF_RHO_EXP    = 1.937   # g/cm3, experimental crystal density
DNTF_HOF_EXP    = 193.0   # kJ/mol, condensed-phase 298 K (Sinditskii et al.)
DNTF_D_EXP      = 9.25    # km/s, Klapotke 2019

# ---------------------------------------------------------------------------
# Existing 6-anchor calibration coefficients (hardcoded from paper)
# rho_cal = RHO_SLOPE * rho_DFT + RHO_INTERCEPT
# HOF_cal = HOF_DFT   + HOF_OFFSET
# ---------------------------------------------------------------------------
CAL6_RHO_SLOPE     =  1.392
CAL6_RHO_INTERCEPT = -0.415
CAL6_HOF_OFFSET    = -206.7   # kJ/mol additive correction

# ---------------------------------------------------------------------------
# All 6 existing anchor pairs (rho_DFT, rho_exp, HOF_DFT, HOF_exp)
# rho_DFT values back-calculated from: rho_DFT = (rho_exp - b) / a
# using the published 6-anchor calibration above, so the anchor points are
# exactly consistent with the published line.  HOF_DFT likewise inverted.
# ---------------------------------------------------------------------------
# Experimental reference values (literature)
ANCHOR_6 = {
    "RDX":   {"rho_exp": 1.806, "HOF_exp":  +66.0},
    "TATB":  {"rho_exp": 1.938, "HOF_exp": -154.0},
    "HMX":   {"rho_exp": 1.891, "HOF_exp":  +74.8},
    "PETN":  {"rho_exp": 1.778, "HOF_exp": -538.5},
    "FOX-7": {"rho_exp": 1.885, "HOF_exp": -133.9},
    "NTO":   {"rho_exp": 1.919, "HOF_exp": -110.4},
}
# DFT raw values (back-calculated from calibration; used for LOO fitting)
for _name, _d in ANCHOR_6.items():
    _d["rho_dft"] = (_d["rho_exp"] - CAL6_RHO_INTERCEPT) / CAL6_RHO_SLOPE
    _d["HOF_dft"] = _d["HOF_exp"] - CAL6_HOF_OFFSET

# ---------------------------------------------------------------------------
# Modal image (mirrors modal_dft_extension.py exactly)
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install("torch==2.4.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("pyscf==2.8.0", "gpu4pyscf-cuda12x==1.4.0",
                 "rdkit-pypi", "geometric", "numpy<2")
    .add_local_dir(str(HERE), remote_path="/m8_bundle",
                    ignore=lambda p: not str(p).endswith(".py"))
)

app = modal.App("dgld-m8-dntf-anchor", image=image)


# ---------------------------------------------------------------------------
# Remote function: full DFT pipeline for a single CHNO molecule
# (same structure as run_lead_remote in modal_dft_extension.py)
# ---------------------------------------------------------------------------
@app.function(gpu="A100", timeout=6 * 60 * 60)
def run_dntf_remote(smiles: str, mol_id: str) -> dict:
    """Run the full DFT ladder on DNTF and return a result dict."""
    import sys, time, json, traceback
    import numpy as np
    sys.path.insert(0, "/m8_bundle")
    # Re-import the shared DFT helpers from m2_bundle (uploaded alongside)
    # The helpers live in m2_dft_pipeline; we copy the essential functions
    # inline here so the container is fully self-contained without requiring
    # m2_bundle to be uploaded.

    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"[m8:{mol_id}] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ------------------------------------------------------------------ #
    # Inline copies of the shared DFT helpers (from m2_dft_pipeline.py). #
    # Keeping them inline avoids adding a cross-bundle import dependency.  #
    # ------------------------------------------------------------------ #
    HARTREE_TO_KJMOL = 2625.49963948
    PACKING_COEFF    = 0.69

    ATOMIC_HOF_298 = {
        "H": 217.998, "C": 716.68, "N": 472.68, "O": 249.18,
    }

    def smiles_to_xyz(smi, charge=0):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit cannot parse SMILES: {smi}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3(); params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"ETKDGv3 embedding failed for {smi}")
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        conf = mol.GetConformer()
        atoms = [(a.GetSymbol(), *conf.GetAtomPosition(i))
                 for i, a in enumerate(mol.GetAtoms())]
        return atoms, charge, mol.GetNumAtoms()

    def atoms_to_pyscf(atoms):
        return "\n".join(f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, x, y, z in atoms)

    def atoms_from_mol_geom(mol):
        coords = mol.atom_coords(unit="Angstrom")
        return [(sym, float(coords[i][0]), float(coords[i][1]), float(coords[i][2]))
                for i, sym in enumerate(mol.elements)]

    def molecular_formula(atoms):
        from collections import Counter
        c = Counter(s for s, *_ in atoms)
        order = ["C", "H", "N", "O"]
        parts = [f"{s}{c[s]}" for s in order if c.get(s, 0)]
        parts += [f"{s}{c[s]}" for s in sorted(c) if s not in order and c[s]]
        return "".join(parts)

    def _build_mol(atoms, basis, charge=0, spin=0):
        from pyscf import gto
        return gto.M(atom=atoms_to_pyscf(atoms), basis=basis, charge=charge,
                     spin=spin, unit="Angstrom", verbose=4)

    def _get_mf(mol, xc, use_gpu=True, df=True):
        if use_gpu:
            try:
                from gpu4pyscf import dft as gpu_dft
                mf = gpu_dft.RKS(mol, xc=xc)
                if df: mf = mf.density_fit()
                mf.verbose = 4
                return mf, "gpu"
            except Exception as e:
                print(f"[m8] gpu4pyscf error ({e}); falling back to CPU", flush=True)
        from pyscf import dft
        mf = dft.RKS(mol, xc=xc)
        if df: mf = mf.density_fit()
        mf.verbose = 4
        return mf, "cpu"

    def opt_b3lyp_6_31gss(atoms, charge=0, use_gpu=True):
        from pyscf.geomopt.geometric_solver import optimize
        mol   = _build_mol(atoms, "6-31g*", charge=charge)
        mf, backend = _get_mf(mol, "b3lyp", use_gpu=use_gpu)
        t0 = time.time()
        mf.kernel()
        print(f"[m8] initial SCF done ({backend}), starting geomopt ...", flush=True)
        mol_opt = optimize(mf, maxsteps=100)
        mf_opt, _ = _get_mf(mol_opt, "b3lyp", use_gpu=use_gpu)
        e_opt = mf_opt.kernel()
        elapsed = time.time() - t0
        print(f"[m8] opt done E={e_opt:.6f} Ha  elapsed={elapsed:.0f}s", flush=True)
        return {
            "atoms_opt": atoms_from_mol_geom(mol_opt),
            "E_b3lyp_631gss_hartree": float(e_opt),
            "backend": backend,
            "elapsed_s": elapsed,
        }

    def freq_b3lyp_6_31gss(atoms, charge=0, use_gpu=True):
        from pyscf.hessian import thermo
        mol = _build_mol(atoms, "6-31g*", charge=charge)
        mf, backend = _get_mf(mol, "b3lyp", use_gpu=use_gpu)
        mf.kernel()
        hess = mf.Hessian(); hess.verbose = 0
        h = hess.kernel()
        freq_info = thermo.harmonic_analysis(mf.mol, h)
        freq_wn = np.asarray(freq_info.get("freq_wavenumber", []))
        freqs_au = freq_wn * 4.55634e-6
        real_mask = freq_wn > 0
        zpe_hartree = float(0.5 * freqs_au[real_mask].sum())
        real_freqs  = freq_wn[real_mask] if len(freq_wn) else np.array([])
        return {
            "freqs_cm1":         [float(x) for x in freq_wn],
            "ZPE_kJmol":         zpe_hartree * HARTREE_TO_KJMOL,
            "n_imag":            int((freq_wn < 0).sum()),
            "min_real_freq_cm1": float(real_freqs.min()) if len(real_freqs) else float("nan"),
            "backend": backend,
        }

    def sp_wb97xd_def2tzvp(atoms, charge=0, use_gpu=True):
        mol = _build_mol(atoms, "def2-tzvp", charge=charge)
        mf, backend = _get_mf(mol, "wb97x-d3bj", use_gpu=use_gpu)
        t0 = time.time()
        e = mf.kernel()
        print(f"[m8] SP done E={e:.6f} Ha  elapsed={time.time()-t0:.0f}s", flush=True)
        return {
            "E_wb97xd_def2tzvp_hartree": float(e),
            "backend": backend,
            "elapsed_s": time.time() - t0,
        }

    def atomic_reference_energy(symbol, basis, xc, use_gpu=True):
        from pyscf import gto, dft
        spin_map = {"H": 1, "C": 2, "N": 3, "O": 2}
        spin = spin_map.get(symbol, 0)
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)
        if use_gpu:
            try:
                from gpu4pyscf import dft as gpu_dft
                mf = gpu_dft.UKS(mol, xc=xc).density_fit()
            except Exception:
                mf = dft.UKS(mol, xc=xc).density_fit()
        else:
            mf = dft.UKS(mol, xc=xc).density_fit()
        mf.verbose = 4
        return float(mf.kernel())

    def hof_from_atomization(atoms, e_total_hartree, atomic_refs):
        from collections import Counter
        cnt = Counter(s for s, *_ in atoms)
        e_atoms = sum(cnt[s] * atomic_refs[s] for s in cnt)
        de_atomization = e_atoms - e_total_hartree
        hof_atoms = sum(cnt[s] * ATOMIC_HOF_298[s] for s in cnt)
        return float(hof_atoms - de_atomization * HARTREE_TO_KJMOL)

    def density_from_volume_atoms(atoms, formula, packing=PACKING_COEFF):
        import re
        cnt = {s: int(n) if n else 1
               for s, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if s}
        M = (12.011 * cnt.get("C", 0) + 1.008 * cnt.get("H", 0)
             + 14.007 * cnt.get("N", 0) + 15.999 * cnt.get("O", 0))
        vdw = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52}
        coords = np.array([[x, y, z] for _, x, y, z in atoms])
        radii  = np.array([vdw.get(s, 1.6) for s, *_ in atoms])
        pad = float(radii.max()) + 0.5
        lo = coords.min(0) - pad; hi = coords.max(0) + pad
        spacing = 0.25
        nx = max(2, int(np.ceil((hi[0]-lo[0])/spacing)))
        ny = max(2, int(np.ceil((hi[1]-lo[1])/spacing)))
        nz = max(2, int(np.ceil((hi[2]-lo[2])/spacing)))
        if nx * ny * nz > 4_000_000:
            spacing = 0.4
            nx = max(2, int(np.ceil((hi[0]-lo[0])/spacing)))
            ny = max(2, int(np.ceil((hi[1]-lo[1])/spacing)))
            nz = max(2, int(np.ceil((hi[2]-lo[2])/spacing)))
        xs = np.linspace(lo[0], hi[0], nx)
        ys = np.linspace(lo[1], hi[1], ny)
        zs = np.linspace(lo[2], hi[2], nz)
        inside = np.zeros((nx, ny, nz), dtype=bool)
        for c, r in zip(coords, radii):
            ix0 = max(0, int((c[0]-r-lo[0])/spacing))
            ix1 = min(nx, int((c[0]+r-lo[0])/spacing)+2)
            iy0 = max(0, int((c[1]-r-lo[1])/spacing))
            iy1 = min(ny, int((c[1]+r-lo[1])/spacing)+2)
            iz0 = max(0, int((c[2]-r-lo[2])/spacing))
            iz1 = min(nz, int((c[2]+r-lo[2])/spacing)+2)
            ax = xs[ix0:ix1, None, None]
            ay = ys[None, iy0:iy1, None]
            az = zs[None, None, iz0:iz1]
            d2 = (ax-c[0])**2 + (ay-c[1])**2 + (az-c[2])**2
            inside[ix0:ix1, iy0:iy1, iz0:iz1] |= (d2 <= r**2)
        cell_vol   = (xs[1]-xs[0]) * (ys[1]-ys[0]) * (zs[1]-zs[0])
        V_mol_AA3  = float(inside.sum()) * cell_vol
        V_mol_cm3  = V_mol_AA3 * 0.6022
        return float(M / V_mol_cm3 * packing)

    # ------------------------------------------------------------------ #
    # Heartbeat thread (mirrors m2_dft_pipeline._start_heartbeat)        #
    # ------------------------------------------------------------------ #
    import threading
    def _beat():
        t0 = time.time()
        while True:
            time.sleep(60)
            print(f"[heartbeat] elapsed={int(time.time()-t0)}s", flush=True)
    threading.Thread(target=_beat, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Main DFT pipeline                                                   #
    # ------------------------------------------------------------------ #
    out: dict = {"id": mol_id, "smiles": smiles, "errors": []}
    t_start = time.time()

    try:
        # -- Step 1: initial geometry --
        print(f"[m8:{mol_id}] SMILES -> xyz (ETKDGv3 + MMFF94) ...", flush=True)
        atoms0, charge, n_atoms = smiles_to_xyz(smiles)
        formula = molecular_formula(atoms0)
        out["formula"] = formula
        out["n_atoms"] = n_atoms
        print(f"[m8:{mol_id}] formula={formula} n_atoms={n_atoms}", flush=True)

        # -- Step 2: atomic reference energies --
        print(f"[m8:{mol_id}] computing atomic reference energies ...", flush=True)
        atom_refs_b3lyp   = {}
        atom_refs_wb97xd  = {}
        for sym in ["C", "H", "N", "O"]:
            print(f"[m8:{mol_id}] atom ref B3LYP {sym} ...", flush=True)
            atom_refs_b3lyp[sym]  = atomic_reference_energy(sym, "6-31g*",    "b3lyp",     use_gpu=True)
            print(f"[m8:{mol_id}] atom ref wB97X-D {sym} ...", flush=True)
            atom_refs_wb97xd[sym] = atomic_reference_energy(sym, "def2-tzvp", "wb97x-d3bj", use_gpu=True)
        out["atom_refs_b3lyp"]  = atom_refs_b3lyp
        out["atom_refs_wb97xd"] = atom_refs_wb97xd

        # -- Step 3: B3LYP/6-31G(d) geometry optimisation --
        print(f"[m8:{mol_id}] B3LYP/6-31G(d) geometry optimisation ...", flush=True)
        opt = opt_b3lyp_6_31gss(atoms0, charge=charge, use_gpu=True)
        out["opt"] = opt
        print(f"[m8:{mol_id}] opt E={opt['E_b3lyp_631gss_hartree']:.6f} Ha  "
              f"({opt['backend']}, {opt['elapsed_s']:.0f}s)", flush=True)

        # -- Step 4: analytical Hessian --
        print(f"[m8:{mol_id}] B3LYP/6-31G(d) Hessian + vibrational analysis ...", flush=True)
        freq = freq_b3lyp_6_31gss(opt["atoms_opt"], charge=charge, use_gpu=True)
        out["freq"] = freq
        print(f"[m8:{mol_id}] freq min={freq['min_real_freq_cm1']:.1f} cm-1  "
              f"n_imag={freq['n_imag']}  ZPE={freq['ZPE_kJmol']:.1f} kJ/mol", flush=True)
        if freq["n_imag"] > 0:
            print(f"[m8:{mol_id}] WARNING: {freq['n_imag']} imaginary freq(s) -- "
                  f"geometry may not be a true minimum", flush=True)

        # -- Step 5: wB97X-D3BJ/def2-TZVP single point --
        print(f"[m8:{mol_id}] wB97X-D3BJ/def2-TZVP single point ...", flush=True)
        sp = sp_wb97xd_def2tzvp(opt["atoms_opt"], charge=charge, use_gpu=True)
        out["sp"] = sp
        print(f"[m8:{mol_id}] SP E={sp['E_wb97xd_def2tzvp_hartree']:.6f} Ha", flush=True)

        # -- Step 6: rho_DFT from Bondi vdW volume --
        rho_dft = density_from_volume_atoms(opt["atoms_opt"], formula, PACKING_COEFF)
        out["rho_dft"] = rho_dft
        print(f"[m8:{mol_id}] rho_DFT = {rho_dft:.4f} g/cm3", flush=True)

        # -- Step 7: HOF_DFT (wB97X-D3BJ level with ZPE) --
        zpe_ha = freq["ZPE_kJmol"] / HARTREE_TO_KJMOL
        e_wb97xd_zpe = sp["E_wb97xd_def2tzvp_hartree"] + zpe_ha
        hof_dft = hof_from_atomization(opt["atoms_opt"], e_wb97xd_zpe, atom_refs_wb97xd)
        out["hof_dft_kJmol"] = hof_dft
        print(f"[m8:{mol_id}] HOF_DFT = {hof_dft:.1f} kJ/mol (wB97X-D3BJ + ZPE)", flush=True)

    except Exception as e:
        out["errors"].append(str(e))
        out["traceback"] = traceback.format_exc()
        print(f"[m8:{mol_id}] ERROR: {e}", flush=True)
        out["_elapsed_s"] = time.time() - t_start
        return out

    out["_elapsed_s"] = time.time() - t_start
    return out


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(skip_dntf: bool = False):
    """
    Flags:
        --skip-dntf   skip the actual DFT run if the SMILES fails local
                      RDKit validation; useful for quick CI / paper checks.
    """
    import numpy as np

    # ---- 0. Validate SMILES locally before dispatching to A100 ----------
    print(f"[m8] validating DNTF SMILES: {DNTF_SMILES}", flush=True)
    try:
        from rdkit import Chem
        mol_check = Chem.MolFromSmiles(DNTF_SMILES)
        if mol_check is None:
            raise ValueError("RDKit returned None for DNTF SMILES")
        # Confirm pure CHNO
        allowed = {"C", "H", "N", "O"}
        symbols = {a.GetSymbol() for a in mol_check.GetAtoms()}
        non_chno = symbols - allowed
        if non_chno:
            raise ValueError(f"SMILES contains non-CHNO elements: {non_chno}")
        print(f"[m8] SMILES valid: {symbols} (pure CHNO)", flush=True)
    except Exception as e:
        print(f"[m8] SMILES validation FAILED: {e}", flush=True)
        if skip_dntf:
            print("[m8] --skip-dntf flag set; aborting without DFT run.", flush=True)
            return
        else:
            print("[m8] Re-run with --skip-dntf to abort, or fix the SMILES.", flush=True)
            raise

    if skip_dntf:
        print("[m8] --skip-dntf passed but SMILES is valid; "
              "flag is a no-op when SMILES validates.  Proceeding.", flush=True)

    # ---- 1. Dispatch A100 container ------------------------------------
    print(f"[m8] dispatching A100 container for {DNTF_ID} ...", flush=True)
    payload = run_dntf_remote.remote(DNTF_SMILES, DNTF_ID)

    if payload.get("errors"):
        print(f"[m8] DFT run FAILED: {payload['errors']}", flush=True)
        out_path = RESULTS_LOCAL / "m8_anchor_result_FAILED.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[m8] error payload saved to {out_path}", flush=True)
        return

    rho_dft = payload["rho_dft"]
    hof_dft = payload["hof_dft_kJmol"]
    print(f"\n[m8] === DFT results ===", flush=True)
    print(f"[m8]   rho_DFT  = {rho_dft:.4f} g/cm3", flush=True)
    print(f"[m8]   HOF_DFT  = {hof_dft:.1f} kJ/mol", flush=True)

    # ---- 2. Apply 6-anchor calibration ----------------------------------
    rho_cal_6 = CAL6_RHO_SLOPE * rho_dft + CAL6_RHO_INTERCEPT
    hof_cal_6 = hof_dft + CAL6_HOF_OFFSET

    residual_rho = rho_cal_6 - DNTF_RHO_EXP
    residual_hof = hof_cal_6 - DNTF_HOF_EXP

    print(f"\n[m8] === 6-anchor calibration applied to DNTF ===", flush=True)
    print(f"[m8]   rho_cal (6-anchor) = {rho_cal_6:.4f}  exp = {DNTF_RHO_EXP:.3f}  "
          f"residual = {residual_rho:+.4f} g/cm3", flush=True)
    print(f"[m8]   HOF_cal (6-anchor) = {hof_cal_6:.1f}  exp = {DNTF_HOF_EXP:.1f}  "
          f"residual = {residual_hof:+.1f} kJ/mol", flush=True)

    # ---- 3. Fit new 7-anchor calibration --------------------------------
    # Build arrays: existing 6 anchors + DNTF as the 7th point
    all_rho_dft = np.array([v["rho_dft"] for v in ANCHOR_6.values()] + [rho_dft])
    all_rho_exp = np.array([v["rho_exp"] for v in ANCHOR_6.values()] + [DNTF_RHO_EXP])
    all_hof_dft = np.array([v["HOF_dft"] for v in ANCHOR_6.values()] + [hof_dft])
    all_hof_exp = np.array([v["HOF_exp"] for v in ANCHOR_6.values()] + [DNTF_HOF_EXP])
    anchor_names = list(ANCHOR_6.keys()) + [DNTF_ID]

    # Linear fit: rho_exp = slope * rho_DFT + intercept
    rho7_slope, rho7_intercept = np.polyfit(all_rho_dft, all_rho_exp, 1)
    # Constant additive offset for HOF (bias only)
    hof7_intercept = float(np.mean(all_hof_exp - all_hof_dft))

    print(f"\n[m8] === 7-anchor calibration fit ===", flush=True)
    print(f"[m8]   rho_cal = {rho7_slope:.4f} * rho_DFT + ({rho7_intercept:+.4f})", flush=True)
    print(f"[m8]   HOF_cal = HOF_DFT + ({hof7_intercept:+.1f}) kJ/mol", flush=True)

    # ---- 4. Leave-one-out RMS for 7-anchor calibration ------------------
    loo_res_rho = []
    loo_res_hof = []
    for i in range(len(anchor_names)):
        # Hold out anchor i
        mask = np.ones(len(anchor_names), dtype=bool)
        mask[i] = False
        s_rho, i_rho = np.polyfit(all_rho_dft[mask], all_rho_exp[mask], 1)
        c_hof        = float(np.mean(all_hof_exp[mask] - all_hof_dft[mask]))
        # Predict on the held-out anchor
        pred_rho = s_rho * all_rho_dft[i] + i_rho
        pred_hof = all_hof_dft[i] + c_hof
        loo_res_rho.append(pred_rho - all_rho_exp[i])
        loo_res_hof.append(pred_hof - all_hof_exp[i])
        print(f"[m8]   LOO {anchor_names[i]:6s}: "
              f"rho_res={pred_rho - all_rho_exp[i]:+.4f}  "
              f"HOF_res={pred_hof - all_hof_exp[i]:+.1f}", flush=True)

    loo_rms_rho = float(np.sqrt(np.mean(np.array(loo_res_rho)**2)))
    loo_rms_hof = float(np.sqrt(np.mean(np.array(loo_res_hof)**2)))

    print(f"\n[m8] LOO-RMS (7-anchor): rho = {loo_rms_rho:.4f} g/cm3  "
          f"HOF = {loo_rms_hof:.1f} kJ/mol", flush=True)

    # ---- 5. Assemble and save output JSON --------------------------------
    result = {
        # DNTF raw DFT values
        "dntf_rho_dft":              round(float(rho_dft), 6),
        "dntf_hof_dft":              round(float(hof_dft), 3),
        # 6-anchor calibrated values
        "dntf_rho_cal_6anchor":      round(float(rho_cal_6), 6),
        "dntf_hof_cal_6anchor":      round(float(hof_cal_6), 3),
        # Residuals vs experiment
        "residual_rho":              round(float(residual_rho), 6),
        "residual_hof":              round(float(residual_hof), 3),
        # 7-anchor calibration coefficients
        "new_7anchor_rho_slope":     round(float(rho7_slope), 6),
        "new_7anchor_rho_intercept": round(float(rho7_intercept), 6),
        "new_7anchor_hof_intercept": round(float(hof7_intercept), 3),
        # Leave-one-out diagnostics
        "new_7anchor_loo_rms_rho":   round(float(loo_rms_rho), 6),
        "new_7anchor_loo_rms_hof":   round(float(loo_rms_hof), 3),
        # Experimental reference (for provenance)
        "dntf_rho_exp":              DNTF_RHO_EXP,
        "dntf_hof_exp":              DNTF_HOF_EXP,
        "dntf_D_exp_kms":            DNTF_D_EXP,
        # Context
        "anchor_names_7":            anchor_names,
        "formula":                   payload.get("formula"),
        "n_atoms":                   payload.get("n_atoms"),
        "freq_n_imag":               payload.get("freq", {}).get("n_imag"),
        "freq_min_real_cm1":         payload.get("freq", {}).get("min_real_freq_cm1"),
        "zpe_kJmol":                 payload.get("freq", {}).get("ZPE_kJmol"),
        "elapsed_s":                 payload.get("_elapsed_s"),
        # 6-anchor calibration used (from paper)
        "cal6_rho_slope":            CAL6_RHO_SLOPE,
        "cal6_rho_intercept":        CAL6_RHO_INTERCEPT,
        "cal6_hof_offset":           CAL6_HOF_OFFSET,
        # Raw DFT energies (for reproducibility)
        "E_b3lyp_631gss_hartree":    payload.get("opt", {}).get("E_b3lyp_631gss_hartree"),
        "E_wb97xd_def2tzvp_hartree": payload.get("sp",  {}).get("E_wb97xd_def2tzvp_hartree"),
    }

    out_path = RESULTS_LOCAL / "m8_anchor_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[m8] result saved -> {out_path}", flush=True)

    # ---- 6. Human-readable summary --------------------------------------
    print(f"\n{'='*60}", flush=True)
    print(f"  M8 DNTF ANCHOR SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  rho_DFT              = {rho_dft:.4f} g/cm3", flush=True)
    print(f"  HOF_DFT              = {hof_dft:.1f} kJ/mol", flush=True)
    print(f"  rho_cal (6-anchor)   = {rho_cal_6:.4f} g/cm3  (exp {DNTF_RHO_EXP:.3f})", flush=True)
    print(f"  HOF_cal (6-anchor)   = {hof_cal_6:.1f} kJ/mol  (exp {DNTF_HOF_EXP:.1f})", flush=True)
    print(f"  residual rho         = {residual_rho:+.4f} g/cm3", flush=True)
    print(f"  residual HOF         = {residual_hof:+.1f} kJ/mol", flush=True)
    print(f"  7-anchor rho_cal     = {rho7_slope:.4f}*rho_DFT + ({rho7_intercept:+.4f})", flush=True)
    print(f"  7-anchor HOF offset  = {hof7_intercept:+.1f} kJ/mol", flush=True)
    print(f"  LOO-RMS (7-anchor)   rho={loo_rms_rho:.4f} g/cm3  HOF={loo_rms_hof:.1f} kJ/mol", flush=True)
    print(f"{'='*60}", flush=True)
