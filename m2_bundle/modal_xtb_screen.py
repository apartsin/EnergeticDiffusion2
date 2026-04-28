"""Stage 2: Modal-CPU xTB pre-screen of E-set 500.

For each SMILES:
  - generate 3D conformer with RDKit (ETKDG + UFF cleanup)
  - xTB --opt vtight to converge
  - parse: converged?, HOMO-LUMO gap (eV), graph-unchanged?
  - return {smiles, converged, gap_eV, graph_unchanged, opt_xyz}

Cheap: ~64 parallel CPU containers, <$1, <10 min.

Usage:
    python -m modal run modal_xtb_screen.py
"""
from __future__ import annotations
import json, os, subprocess, tempfile, traceback
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
E500_PATH = ROOT / "results" / "extension_set" / "e_set_500_smiles.json"
OUT_PATH = ROOT / "results" / "extension_set" / "e_set_xtb_screen.json"

image = (
    modal.Image.from_registry("python:3.11-slim")
    .apt_install("xtb", "build-essential")
    .pip_install("rdkit", "numpy<2")
)

app = modal.App("dgld-eset-xtb", image=image)


@app.function(cpu=4.0, timeout=20 * 60, max_containers=64)
def screen_smiles_remote(smiles: str) -> dict:
    res = {"smiles": smiles, "converged": False, "gap_eV": None,
           "graph_unchanged": None, "opt_xyz": None, "error": None}
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            res["error"] = "rdkit_parse_failed"; return res
        m_h = Chem.AddHs(m)
        rc = AllChem.EmbedMolecule(m_h, randomSeed=42, useRandomCoords=True,
                                     maxAttempts=200)
        if rc != 0:
            # fallback
            rc = AllChem.EmbedMolecule(m_h, randomSeed=7,
                                         useRandomCoords=True, maxAttempts=400)
            if rc != 0:
                res["error"] = "embed_failed"; return res
        try:
            AllChem.UFFOptimizeMolecule(m_h, maxIters=400)
        except Exception:
            pass
        ref_smi = Chem.MolToSmiles(Chem.RemoveHs(m_h), canonical=True)

        with tempfile.TemporaryDirectory() as tdir:
            xyz = Path(tdir) / "in.xyz"
            atoms = m_h.GetAtoms()
            conf = m_h.GetConformer()
            lines = [f"{m_h.GetNumAtoms()}", "smoke"]
            for i, a in enumerate(atoms):
                p = conf.GetAtomPosition(i)
                lines.append(f"{a.GetSymbol():<2}  {p.x:.6f}  {p.y:.6f}  {p.z:.6f}")
            xyz.write_text("\n".join(lines))
            charge = Chem.GetFormalCharge(m_h)
            cmd = ["xtb", str(xyz), "--opt", "tight", "--chrg", str(charge),
                   "--gfn", "2", "--norestart"]
            try:
                p = subprocess.run(cmd, cwd=tdir, capture_output=True,
                                     text=True, timeout=900)
            except subprocess.TimeoutExpired:
                res["error"] = "xtb_timeout"; return res
            stdout = (p.stdout or "") + "\n" + (p.stderr or "")
            # parse
            converged = "GEOMETRY OPTIMIZATION CONVERGED" in stdout or \
                        "*** GEOMETRY OPTIMIZATION CONVERGED ***" in stdout or \
                        " converged geometry " in stdout.lower()
            res["converged"] = bool(converged)
            # HOMO-LUMO gap (eV) — xtb prints "HL-Gap            X.XXXXXX Eh   Y.YYYY eV"
            for line in stdout.splitlines():
                if "HL-Gap" in line and "eV" in line:
                    parts = line.split()
                    for j, tok in enumerate(parts):
                        if tok == "eV" and j > 0:
                            try:
                                res["gap_eV"] = float(parts[j - 1]); break
                            except Exception:
                                pass
                    if res["gap_eV"] is not None: break
            # graph-unchanged: build mol from xtbopt.xyz and compare canonical SMILES
            opt_xyz_path = Path(tdir) / "xtbopt.xyz"
            if opt_xyz_path.exists():
                opt_xyz = opt_xyz_path.read_text()
                res["opt_xyz"] = opt_xyz
                # No xyz->smiles in RDKit core; use openbabel-like via rdkit DetermineBonds (3.x)
                try:
                    from rdkit.Chem import rdDetermineBonds
                    raw = Chem.MolFromXYZBlock(opt_xyz)
                    if raw is not None:
                        try:
                            rdDetermineBonds.DetermineConnectivity(raw)
                            rdDetermineBonds.DetermineBondOrders(raw, charge=charge)
                            new_smi = Chem.MolToSmiles(Chem.RemoveHs(raw),
                                                        canonical=True)
                            res["graph_unchanged"] = (new_smi == ref_smi)
                        except Exception:
                            res["graph_unchanged"] = None
                except Exception:
                    res["graph_unchanged"] = None
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    return res


@app.local_entrypoint()
def main():
    payload = json.loads(E500_PATH.read_text())
    smis = [it["smiles"] for it in payload["items"]]
    print(f"[xtb] screening {len(smis)} SMILES on Modal CPU")
    results = []
    for r in screen_smiles_remote.map(smis):
        results.append(r)
        if len(results) % 25 == 0:
            print(f"[xtb] {len(results)}/{len(smis)} done")
    n_conv = sum(1 for r in results if r.get("converged"))
    n_gap = sum(1 for r in results if r.get("gap_eV") and r["gap_eV"] >= 1.5)
    n_graph = sum(1 for r in results if r.get("graph_unchanged"))
    print(f"[xtb] converged: {n_conv}/{len(results)}, "
          f"gap>=1.5eV: {n_gap}, graph_unchanged: {n_graph}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"items": results,
                                       "summary": {"n": len(results),
                                                   "converged": n_conv,
                                                   "gap_ge_1p5_eV": n_gap,
                                                   "graph_unchanged": n_graph}},
                                      indent=2))
    print(f"[xtb] -> {OUT_PATH}")
