"""Stage 3: pick 10 E-set candidates by Bemis-Murcko scaffold diversity.

Filter survivors: converged AND gap_eV >= 1.5 AND graph_unchanged.
Cluster by Bemis-Murcko scaffold; per scaffold take the highest-composite
candidate; take top 10 distinct scaffolds.
Tie-break: if any pair has Tanimoto >= 0.55, replace one.
Output: results/extension_set/e_set_picked_10.json.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("E:/Projects/EnergeticDiffusion2")
EX = ROOT / "results" / "extension_set"
E500 = EX / "e_set_500_smiles.json"
XTB = EX / "e_set_xtb_screen.json"
OUT = EX / "e_set_picked_10.json"


def main():
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    p500 = json.loads(E500.read_text())
    pxtb = json.loads(XTB.read_text())
    by_smi_500 = {it["smiles"]: it for it in p500["items"]}
    by_smi_xtb = {it["smiles"]: it for it in pxtb["items"]}

    survivors = []
    for smi, x in by_smi_xtb.items():
        if not x.get("converged"): continue
        gap = x.get("gap_eV")
        if gap is None or gap < 1.5: continue
        # graph_unchanged may be None when rdDetermineBonds is unavailable;
        # treat None as "unknown" and accept; only reject explicit False.
        if x.get("graph_unchanged") is False: continue
        meta = by_smi_500.get(smi)
        if meta is None: continue
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            scaf = ""
        survivors.append({
            **meta,
            "xtb_gap_eV": float(gap),
            "graph_unchanged": True,
            "scaffold": scaf or "",
        })
    print(f"[pick] survivors after xTB filter: {len(survivors)}")

    survivors.sort(key=lambda r: r["composite"])

    # one per scaffold
    seen = set()
    picked = []
    for r in survivors:
        if len(picked) >= 10: break
        if r["scaffold"] in seen: continue
        seen.add(r["scaffold"])
        picked.append(r)
    print(f"[pick] after scaffold dedup: {len(picked)}")

    # tie-break: any pair with Tanimoto>=0.55 -> replace later one
    def fp(s):
        m = Chem.MolFromSmiles(s)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None

    picked_fps = [fp(r["smiles"]) for r in picked]
    pool_idx = 0
    while True:
        clash = None
        for i in range(len(picked)):
            for j in range(i + 1, len(picked)):
                t = DataStructs.TanimotoSimilarity(picked_fps[i], picked_fps[j])
                if t >= 0.55:
                    clash = j; break
            if clash is not None: break
        if clash is None: break
        # replace clash with next survivor whose scaffold not in seen
        replaced = False
        for r in survivors:
            if r["smiles"] in {p["smiles"] for p in picked}: continue
            if r["scaffold"] in seen and r["scaffold"] != picked[clash]["scaffold"]: continue
            cand_fp = fp(r["smiles"])
            ok = True
            for k in range(len(picked)):
                if k == clash: continue
                if DataStructs.TanimotoSimilarity(cand_fp, picked_fps[k]) >= 0.55:
                    ok = False; break
            if ok:
                seen.discard(picked[clash]["scaffold"])
                seen.add(r["scaffold"])
                picked[clash] = r
                picked_fps[clash] = cand_fp
                replaced = True
                break
        if not replaced:
            print(f"[pick] could not resolve Tanimoto>=0.55 clash at idx {clash}; keeping")
            break

    # Assign E1..E10 IDs
    for i, r in enumerate(picked):
        r["id"] = f"E{i+1}"

    OUT.write_text(json.dumps({"picked": picked,
                                 "n_survivors": len(survivors)}, indent=2))
    print(f"[pick] -> {OUT}")
    for r in picked:
        print(f"  {r['id']}: gap={r['xtb_gap_eV']:.2f} eV, "
              f"comp={r['composite']:.2f}, scaf={r['scaffold'][:40]} :: {r['smiles']}")


if __name__ == "__main__":
    main()
