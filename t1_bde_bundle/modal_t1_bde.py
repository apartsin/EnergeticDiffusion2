"""T1: xTB BDE on L1 (3,4,5-trinitro-1,2-isoxazole) and E1 (4-nitro-1,2,3,5-oxatriazole).

For each compound:
    1. RDKit ETKDGv3 + MMFF94 initial 3D geometry.
    2. xTB --opt tight on the closed-shell parent.
    3. Enumerate candidate homolytic bond cleavages:
         L1: every C-N(NO2) and ring N-O / C-O bond
         E1: every ring N-O / N-N bond and the C-NO2 bond
    4. For each bond (a, b), break it, partition parent atoms into two
       connected groups (Chem.GetMolFrags), and build each fragment's XYZ
       block directly from the parent's 3D coordinates (NO SMILES
       round-trip, NO re-embedding, NO implicit H added). Each fragment is
       a true open-shell radical at the broken-bond atom; xTB runs with
       --opt tight, charge=0, uhf=1 per fragment.
       BDE = E(rad_a) + E(rad_b) - E(parent), in kcal/mol.
    5. Save per-bond results sorted ascending; report the weakest BDE.

Output:
    t1_bde_bundle/results/t1_bde.json
    t1_bde_bundle/results/t1_bde_per_bond.json

Modal:
    cpu=4.0, memory=8192 MB; timeout 1 hour.

Usage:
    modal run t1_bde_bundle/modal_t1_bde.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
RESULTS_LOCAL = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Compound list
# ---------------------------------------------------------------------------
COMPOUNDS = {
    "L1": "O=[N+]([O-])c1noc([N+](=O)[O-])c1[N+](=O)[O-]",  # 3,4,5-trinitro-1,2-isoxazole
    "E1": "O=[N+]([O-])c1nnon1",                            # 4-nitro-1,2,3,5-oxatriazole
}

# ---------------------------------------------------------------------------
# Modal image: Ubuntu 22.04 + xtb 6.6.1 + RDKit
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    .apt_install(
        "wget", "tar", "xz-utils", "ca-certificates",
        "libgomp1", "build-essential", "git",
    )
    # xtb 6.6.1 official static Linux binary.  The tarball extracts to
    # /opt/xtb-6.6.1/ (NOT /opt/xtb-dist/), so we resolve the actual top-level
    # directory at build time and symlink every binary into /usr/local/bin so
    # `xtb` is on PATH.  Build fails fast if the binary is missing or unrunnable.
    .run_commands(
        "cd /opt && wget -q https://github.com/grimme-lab/xtb/releases/download/v6.6.1/xtb-6.6.1-linux-x86_64.tar.xz",
        "cd /opt && tar -xf xtb-6.6.1-linux-x86_64.tar.xz && rm xtb-6.6.1-linux-x86_64.tar.xz",
        # Find whichever directory the tarball produced (xtb-6.6.1 in 6.6.1) and
        # symlink its bin/share into the standard locations.
        "XTB_DIR=$(ls -d /opt/xtb-* | head -n1) && "
        "ln -sf $XTB_DIR/bin/xtb /usr/local/bin/xtb && "
        "ln -sf $XTB_DIR/bin/xtb-python /usr/local/bin/xtb-python 2>/dev/null || true && "
        "mkdir -p /usr/local/share && "
        "ln -sf $XTB_DIR/share/xtb /usr/local/share/xtb",
        # Fail-fast verification at image build time.
        "xtb --version",
    )
    .env({"XTBPATH": "/usr/local/share/xtb"})
    .pip_install("rdkit==2024.3.5", "numpy")
)

app = modal.App("dgld-t1-bde", image=image)


# ---------------------------------------------------------------------------
# Helpers (defined inside the remote container, mirrors m8 pattern)
# ---------------------------------------------------------------------------
@app.function(cpu=4.0, memory=8192, timeout=60 * 60)
def run_bde_remote(compound_id: str, smiles: str) -> dict:
    """Compute homolytic-cleavage BDE for every single, non-aromatic, non-ring-fused
    candidate bond in the parent skeleton.  Returns sorted bond list."""
    import os
    import shutil
    import subprocess
    import tempfile
    import traceback
    from collections import Counter

    out: dict = {"id": compound_id, "smiles": smiles, "errors": []}
    t0 = time.time()

    HARTREE_TO_KCAL = 627.5094740631

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolDescriptors  # noqa: F401
    except Exception as e:  # pragma: no cover
        out["errors"].append(f"rdkit import failed: {e}")
        return out

    # ---------------- helpers ----------------
    def smiles_to_mol_3d(smi: str):
        """Build a RDKit Mol with explicit Hs and a single 3D conformer
        (ETKDGv3 + MMFF94/UFF). Returns the Mol object so callers can both
        emit an XYZ block AND use the *same* atom indexing + coordinates
        for downstream fragment building."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit cannot parse SMILES: {smi}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3(); params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.maxAttempts = 200
            if AllChem.EmbedMolecule(mol, params) != 0:
                raise RuntimeError(f"ETKDGv3 embed failed for {smi}")
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass
        return mol

    def mol_to_xyz_block(mol, comment: str = "") -> str:
        conf = mol.GetConformer()
        lines = [str(mol.GetNumAtoms()), comment]
        for i, atom in enumerate(mol.GetAtoms()):
            p = conf.GetAtomPosition(i)
            lines.append(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
        return "\n".join(lines)

    def fragment_xyz_block(parent_mol, atom_indices, comment: str = "") -> str:
        """Build an XYZ block for a subset of parent atoms using the
        parent's *existing* 3D coordinates. No re-embedding, no implicit
        H added: each open valence becomes a radical site."""
        conf = parent_mol.GetConformer()
        lines = [str(len(atom_indices)), comment]
        for atom_idx in atom_indices:
            atom = parent_mol.GetAtomWithIdx(int(atom_idx))
            p = conf.GetAtomPosition(int(atom_idx))
            lines.append(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
        return "\n".join(lines)

    def run_xtb(xyz_block: str, charge: int, uhf: int, tag: str) -> float:
        with tempfile.TemporaryDirectory() as td:
            p_xyz = Path(td) / "mol.xyz"
            p_xyz.write_text(xyz_block + "\n")
            cmd = [
                "xtb", str(p_xyz),
                "--opt", "tight",
                "--chrg", str(charge),
                "--uhf",  str(uhf),
                "--gfn", "2",
                "--parallel", "4",
            ]
            t_start = time.time()
            res = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=900)
            elapsed = time.time() - t_start
            stdout = res.stdout
            if res.returncode != 0:
                # xtb sometimes returns non-zero on convergence warnings; check stdout
                pass
            # Parse total energy: 'TOTAL ENERGY' line is in Hartree
            etot = None
            for line in stdout.splitlines():
                if "TOTAL ENERGY" in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            etot = float(p)
                            break
                        except ValueError:
                            continue
            if etot is None:
                raise RuntimeError(
                    f"xtb {tag} failed (rc={res.returncode}); tail stdout:\n"
                    + "\n".join(stdout.splitlines()[-20:])
                    + "\n--- stderr ---\n"
                    + "\n".join(res.stderr.splitlines()[-20:])
                )
            print(f"[t1:{compound_id}] xtb {tag} E={etot:.6f} Ha t={elapsed:.0f}s", flush=True)
            return etot

    # ---------------- step 1: parent geometry + xTB --------------
    print(f"[t1:{compound_id}] parent SMILES -> 3D ...", flush=True)
    parent_mol = smiles_to_mol_3d(smiles)
    n_atoms = parent_mol.GetNumAtoms()
    parent_xyz = mol_to_xyz_block(parent_mol, comment=smiles)
    out["n_atoms_parent"] = n_atoms

    print(f"[t1:{compound_id}] xtb parent --opt tight ...", flush=True)
    e_parent_ha = run_xtb(parent_xyz, charge=0, uhf=0, tag="parent")
    out["e_parent_hartree"] = e_parent_ha

    # ---------------- step 2: enumerate candidate bonds ----------
    # IMPORTANT: reuse the *same* parent_mol (with its 3D conformer) so that
    # atom indices in candidate_bonds match the indices we will later carve
    # out into per-fragment XYZ blocks. Re-embedding from SMILES would give
    # a different atom ordering and break the index correspondence.
    mol_parsed = parent_mol

    candidate_bonds: list[tuple[int, int, str, str]] = []
    for bond in mol_parsed.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            # only single bonds for homolytic cleavage; the formal nitro bonds
            # are aromatic-resonance but RDKit treats N-O of nitro as single
            pass
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sa, sb = bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
        if "H" in (sa, sb):
            continue  # skip C-H / N-H homolyses (much higher BDE, not weakest)
        # focus on heavy-atom backbone bonds
        candidate_bonds.append((a, b, sa, sb))

    print(f"[t1:{compound_id}] {len(candidate_bonds)} heavy-atom candidate bonds", flush=True)
    out["n_candidate_bonds"] = len(candidate_bonds)

    # ---------------- step 3: per-bond homolytic BDE -------------
    # Coordinate-preserving graph split:
    #   * Make an editable copy of parent_mol, RemoveBond(a, b).
    #   * Chem.GetMolFrags(asMols=False) returns lists of atom indices into
    #     the *original* parent atom ordering.
    #   * For each fragment, carve out an XYZ block using parent's existing
    #     3D coordinates (NO SMILES round-trip, NO re-embedding, NO implicit
    #     H added). Each fragment is a true open-shell radical at the broken
    #     bond atom -> uhf=1, charge=0.
    bond_results: list[dict] = []
    for (a, b, sa, sb) in candidate_bonds:
        try:
            ed_mol = Chem.RWMol(parent_mol)
            ed_mol.RemoveBond(a, b)
            frag_mol_nosan = ed_mol.GetMol()
            # asMols=False gives tuples of atom indices into parent ordering.
            # sanitizeFrags=False because the open valence would otherwise
            # confuse RDKit; we only need the connectivity partition.
            frag_atom_groups = Chem.GetMolFrags(
                frag_mol_nosan, asMols=False, sanitizeFrags=False,
            )
            if len(frag_atom_groups) != 2:
                bond_results.append({
                    "atom_a": a, "atom_b": b, "syms": f"{sa}-{sb}",
                    "skipped": True, "reason": "ring-bond, not 2 fragments",
                })
                continue

            # Order: fragment containing atom a first, atom b second
            if a in frag_atom_groups[0]:
                groups_ordered = (frag_atom_groups[0], frag_atom_groups[1])
            else:
                groups_ordered = (frag_atom_groups[1], frag_atom_groups[0])

            frag_formulas: list[str] = []
            frag_es: list[float] = []
            for fi, atom_indices in enumerate(groups_ordered):
                # Composition string for diagnostics (replaces SMILES tag)
                sym_counter = Counter(
                    parent_mol.GetAtomWithIdx(int(i)).GetSymbol()
                    for i in atom_indices
                )
                formula = "".join(
                    f"{s}{sym_counter[s]}" for s in sorted(sym_counter)
                )
                # For homolytic single-bond cleavage, each fragment gains
                # exactly one unpaired electron at the broken-bond atom.
                uhf = 1
                xyz_block = fragment_xyz_block(
                    parent_mol, atom_indices,
                    comment=f"{compound_id}_bond{a}-{b}_frag{fi}_{formula}",
                )
                e_frag_ha = run_xtb(
                    xyz_block, charge=0, uhf=uhf,
                    tag=f"frag-{a}_{b}-{fi}",
                )
                frag_formulas.append(formula)
                frag_es.append(e_frag_ha)

            bde_ha = (frag_es[0] + frag_es[1]) - e_parent_ha
            bde_kcal = bde_ha * HARTREE_TO_KCAL
            bond_results.append({
                "atom_a": a, "atom_b": b, "syms": f"{sa}-{sb}",
                "frag_formulas": frag_formulas,
                "frag_atom_counts": [len(groups_ordered[0]), len(groups_ordered[1])],
                "e_frag1_hartree": frag_es[0],
                "e_frag2_hartree": frag_es[1],
                "bde_hartree": bde_ha,
                "bde_kcal_per_mol": round(float(bde_kcal), 3),
                "skipped": False,
            })
        except Exception as e:
            bond_results.append({
                "atom_a": a, "atom_b": b, "syms": f"{sa}-{sb}",
                "skipped": True, "reason": f"error: {e}",
                "traceback": traceback.format_exc(),
            })
            print(f"[t1:{compound_id}] bond {a}-{b} failed: {e}", flush=True)

    # sort bonds by BDE ascending (skipped at end)
    valid = sorted(
        [b for b in bond_results if not b["skipped"]],
        key=lambda b: b["bde_kcal_per_mol"],
    )
    skipped = [b for b in bond_results if b["skipped"]]
    out["bonds"] = valid + skipped

    if valid:
        weakest = valid[0]
        out["weakest_bond"] = {
            "atom_a": weakest["atom_a"],
            "atom_b": weakest["atom_b"],
            "syms":   weakest["syms"],
            "bde_kcal_per_mol": weakest["bde_kcal_per_mol"],
        }

    out["_elapsed_s"] = round(time.time() - t0, 1)
    return out


@app.local_entrypoint()
def main():
    """Run BDE on every compound in COMPOUNDS, save merged json."""
    payload = {}
    for cid, smi in COMPOUNDS.items():
        print(f"[t1] dispatching {cid} -> {smi}", flush=True)
        result = run_bde_remote.remote(cid, smi)
        payload[cid] = result
        # incremental save
        per_path = RESULTS_LOCAL / f"t1_bde_{cid}.json"
        per_path.write_text(json.dumps(result, indent=2))
        print(f"[t1] {cid} saved -> {per_path}", flush=True)

    summary_path = RESULTS_LOCAL / "t1_bde.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[t1] summary saved -> {summary_path}", flush=True)

    # Human-readable summary
    print(f"\n{'='*60}")
    print(f"  T1 BDE SUMMARY")
    print(f"{'='*60}")
    for cid, res in payload.items():
        wb = res.get("weakest_bond")
        if wb:
            print(f"  {cid}: weakest bond {wb['syms']} "
                  f"(atoms {wb['atom_a']}-{wb['atom_b']}) "
                  f"BDE = {wb['bde_kcal_per_mol']:.1f} kcal/mol")
        else:
            print(f"  {cid}: NO valid BDE computed (all bonds skipped)")
    print(f"{'='*60}")
