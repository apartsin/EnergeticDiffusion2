"""T2: independent ML density cross-check on L1 and E1.

Pre-flight (PRE_FLIGHT.md): Chem. Mater. 2024 model weights are NOT
publicly released, nor is the Casey 2020 3D-CNN.  Fall-back chosen
per EXPERIMENTATION_PLAN.md:

    Bondi-vdW grid integration with packing factor bracketed at
    pk in {0.65, 0.69, 0.72}.

We deliberately use the same Bondi-vdW machinery that the production
6-anchor calibration uses, but vary the packing factor over the
loose-to-tight band reported in the energetic-materials literature
(see Politzer & Murray 2014 for typical 0.65 - 0.72 envelope).

Inputs:
    L1: O=[N+]([O-])c1noc([N+](=O)[O-])c1[N+](=O)[O-]   3,4,5-trinitro-1,2-isoxazole
    E1: O=[N+]([O-])c1nnon1                              4-nitro-1,2,3,5-oxatriazole

Reference values (in-paper 6-anchor calibrated):
    L1 rho_cal = 2.09 g/cm3
    E1 rho_cal = 2.04 g/cm3

Output:
    t2_density_bundle/results/t2_density_crosscheck.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
RESULTS_LOCAL = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

COMPOUNDS = {
    "L1": {
        "smiles":  "O=[N+]([O-])c1noc([N+](=O)[O-])c1[N+](=O)[O-]",
        "rho_paper": 2.09,
    },
    "E1": {
        "smiles":  "O=[N+]([O-])c1nnon1",
        "rho_paper": 2.04,
    },
}

PACKING_FACTORS = [0.65, 0.69, 0.72]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("rdkit==2024.3.5", "numpy")
)

app = modal.App("dgld-t2-density-crosscheck", image=image)


@app.function(cpu=2.0, memory=4096, timeout=30 * 60)
def run_density_remote(compound_id: str, smiles: str, rho_paper: float) -> dict:
    import re
    from collections import Counter

    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out: dict = {"id": compound_id, "smiles": smiles, "rho_paper": rho_paper}
    t0 = time.time()

    # ---- 1. RDKit ETKDGv3 + MMFF94 geometry --------------------------------
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"RDKit cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.maxAttempts = 200
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"ETKDGv3 embed failed for {smiles}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    atoms = [(a.GetSymbol(),
              float(conf.GetAtomPosition(i).x),
              float(conf.GetAtomPosition(i).y),
              float(conf.GetAtomPosition(i).z))
             for i, a in enumerate(mol.GetAtoms())]
    cnt = Counter(s for s, *_ in atoms)
    formula = "".join(f"{s}{cnt[s]}" for s in ["C", "H", "N", "O"] if cnt.get(s))
    out["formula"] = formula
    out["n_atoms"] = mol.GetNumAtoms()

    # ---- 2. Molecular weight ----------------------------------------------
    M = (12.011 * cnt.get("C", 0) + 1.008 * cnt.get("H", 0)
         + 14.007 * cnt.get("N", 0) + 15.999 * cnt.get("O", 0))
    out["M_amu"] = round(float(M), 4)

    # ---- 3. Bondi vdW grid integration ------------------------------------
    vdw = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52}
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    radii  = np.array([vdw.get(s, 1.6) for s, *_ in atoms])
    pad = float(radii.max()) + 0.5
    lo = coords.min(0) - pad; hi = coords.max(0) + pad
    spacing = 0.20  # finer than m8 default (0.25), smaller molecule
    nx = max(2, int(np.ceil((hi[0]-lo[0])/spacing)))
    ny = max(2, int(np.ceil((hi[1]-lo[1])/spacing)))
    nz = max(2, int(np.ceil((hi[2]-lo[2])/spacing)))
    if nx * ny * nz > 8_000_000:
        spacing = 0.30
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
    cell_vol  = (xs[1]-xs[0]) * (ys[1]-ys[0]) * (zs[1]-zs[0])
    V_mol_AA3 = float(inside.sum()) * cell_vol
    V_mol_cm3 = V_mol_AA3 * 0.6022   # Å^3 -> cm3/mol via N_A * 1e-24
    out["V_mol_AA3"] = round(float(V_mol_AA3), 3)
    out["V_mol_cm3_per_mol"] = round(float(V_mol_cm3), 4)

    # ---- 4. Packing-factor bracket ----------------------------------------
    bracket = {}
    for pk in PACKING_FACTORS:
        rho = float(M / V_mol_cm3 * pk)
        bracket[f"pk_{pk:.2f}"] = round(rho, 4)
    out["bracket_g_per_cm3"] = bracket

    # ---- 5. Delta vs in-paper 6-anchor calibrated value ------------------
    rho_prod_pk = bracket["pk_0.69"]
    delta_abs = round(rho_prod_pk - rho_paper, 4)
    delta_pct = round(100.0 * (rho_prod_pk - rho_paper) / rho_paper, 2)
    out["delta_pk069_vs_paper_abs"] = delta_abs
    out["delta_pk069_vs_paper_pct"] = delta_pct

    out["_elapsed_s"] = round(time.time() - t0, 2)
    print(f"[t2:{compound_id}] formula={formula} V_mol={V_mol_cm3:.3f}",
          flush=True)
    print(f"[t2:{compound_id}] bracket = {bracket}", flush=True)
    print(f"[t2:{compound_id}] delta @ pk=0.69 vs paper: "
          f"{delta_abs:+.4f} g/cm3 ({delta_pct:+.2f}%)", flush=True)
    return out


@app.local_entrypoint()
def main():
    payload = {}
    for cid, info in COMPOUNDS.items():
        print(f"[t2] dispatching {cid} -> {info['smiles']}", flush=True)
        result = run_density_remote.remote(cid, info["smiles"], info["rho_paper"])
        payload[cid] = result

    out_path = RESULTS_LOCAL / "t2_density_crosscheck.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[t2] saved -> {out_path}", flush=True)

    print(f"\n{'='*60}")
    print(f"  T2 DENSITY CROSS-CHECK SUMMARY")
    print(f"{'='*60}")
    for cid, res in payload.items():
        b = res["bracket_g_per_cm3"]
        print(f"  {cid}: rho [{b['pk_0.65']:.3f}, {b['pk_0.69']:.3f}, "
              f"{b['pk_0.72']:.3f}] g/cm3  vs paper {res['rho_paper']:.3f}; "
              f"delta@0.69 = {res['delta_pk069_vs_paper_pct']:+.2f}%")
    print(f"{'='*60}")
