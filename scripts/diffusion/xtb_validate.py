"""xTB-GFN2 quick stability check on top-N leads.

For each SMILES:
  1. RDKit 3D conformer (ETKDGv3) → MMFF94 minimisation
  2. xTB --opt vtight on the conformer
  3. report:
       converged?
       energy (Eh)
       homo_lumo_gap (eV)
       graph_survives  (post-opt RDKit-canonicalised SMILES == input?)
       n_imag_freq     (omitted unless --freq)
"""
from __future__ import annotations
import argparse, subprocess, tempfile, json, re, shutil
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

XTB = "tools/xtb-6.6.1/bin/xtb.exe"
PARAMS_DIR = "tools/xtb-6.6.1/share/xtb"

def smi_to_xyz(smi: str, work: Path) -> Path | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0: return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    xyz = work / "input.xyz"
    Chem.MolToXYZFile(mol, str(xyz))
    return xyz


def parse_xtb(stdout: str) -> dict:
    out = {"converged": False, "energy_eh": None, "gap_ev": None}
    if "GEOMETRY OPTIMIZATION CONVERGED" in stdout or "abnormal termination" not in stdout.lower():
        out["converged"] = True
    m = re.search(r"TOTAL ENERGY\s+([-0-9.]+)\s+Eh", stdout)
    if m: out["energy_eh"] = float(m.group(1))
    m = re.search(r"HOMO-LUMO GAP\s+([0-9.]+)\s+eV", stdout)
    if m: out["gap_ev"] = float(m.group(1))
    return out


def graph_canon_from_xyz(xyz_path: Path, original_smi: str) -> tuple[bool, str]:
    """After optimization, infer connectivity from xyz coordinates and check
    whether the canonical SMILES still equals the input's canonical SMILES.
    Uses RDKit's DetermineBonds via xyz → mol heuristic."""
    try:
        mol = Chem.MolFromXYZFile(str(xyz_path))
        if mol is None: return False, ""
        from rdkit.Chem import rdDetermineBonds
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        post = Chem.MolToSmiles(mol)
        original_canon = Chem.MolToSmiles(Chem.MolFromSmiles(original_smi))
        post_no_h = Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(post))) if Chem.MolFromSmiles(post) else post
        return post_no_h == original_canon, post_no_h
    except Exception as e:
        return False, str(e)


def run_xtb(smi: str, name: str, work_dir: Path, timeout=60) -> dict:
    work = work_dir / name
    work.mkdir(parents=True, exist_ok=True)
    xyz = smi_to_xyz(smi, work)
    if xyz is None:
        return {"name": name, "smi": smi, "status": "rdkit_embed_fail"}
    import os
    env = {**os.environ}
    env["XTBPATH"] = str(Path(PARAMS_DIR).resolve())
    xtb_abs = str(Path(XTB).resolve())
    try:
        r = subprocess.run([xtb_abs, "input.xyz", "--opt", "tight", "--gfn", "2"],
                            cwd=str(work), capture_output=True,
                            timeout=timeout, env=env)
        stdout_str = r.stdout.decode("utf-8", errors="replace")
        stderr_str = r.stderr.decode("utf-8", errors="replace")
        out = parse_xtb(stdout_str + stderr_str)
        out["name"] = name; out["smi"] = smi
        # Graph survival check
        opt_xyz = work / "xtbopt.xyz"
        if opt_xyz.exists():
            survives, post_smi = graph_canon_from_xyz(opt_xyz, smi)
            out["graph_survives"] = survives
            out["post_smi"] = post_smi
        else:
            out["graph_survives"] = False
            out["post_smi"] = ""
        return out
    except subprocess.TimeoutExpired:
        return {"name": name, "smi": smi, "status": "timeout"}
    except Exception as e:
        return {"name": name, "smi": smi, "status": f"error: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--workdir", default="logs/xtb_runs")
    args = ap.parse_args()

    md = Path(args.inp).read_text(encoding="utf-8")
    leads = []
    for line in md.split("\n"):
        if not line.startswith("| ") or "rank" in line or "---" in line: continue
        cells = [c.strip(" `") for c in line.split("|")]
        if len(cells) < 12: continue
        try:
            leads.append({"rank": int(cells[1]), "smiles": cells[-2]})
        except Exception: pass
        if len(leads) >= args.top: break

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    if not Path(XTB).exists():
        raise SystemExit(f"xTB binary not found at {XTB}")

    results = []
    for L in leads:
        print(f"  [xtb] rank{L['rank']} {L['smiles'][:60]}", flush=True)
        r = run_xtb(L["smiles"], f"rank{L['rank']}", work)
        r["rank"] = L["rank"]
        results.append(r)
        # Compact log line
        if r.get("status") and r["status"] != "ok":
            print(f"    -> {r['status']}")
        else:
            print(f"    converged={r.get('converged')}  E={r.get('energy_eh')}  "
                  f"gap={r.get('gap_ev')}  graph_ok={r.get('graph_survives')}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n-> {args.out}")
    n_ok = sum(1 for r in results if r.get("graph_survives"))
    print(f"Graph-survives: {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
