"""M4a: MOSES-style distribution-learning metrics on the M1''' headline pool.

For the four guidance conditions (C0/C1/C2/C3, ~80k SMILES each), report:
  - Validity (RDKit parses)
  - Uniqueness @ 10k
  - Novelty vs labelled-master (65k canonical SMILES)
  - Internal Diversity (IntDiv1; mean pairwise Tanimoto on 1000 sample)
  - Filters (PAINS + chem_redflags pass rate)
  - Bemis-Murcko scaffold count
  - Avg Tanimoto similarity to nearest labelled-master neighbour (SNN)

Output: results/m4a_metrics.json + results/m4a_summary.md

Bundle: 4 SMILES files (1.7-1.8 MB each) + labelled_master.csv + chem_redflags.py
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="Directory containing the 4 condition SMILES files")
    ap.add_argument("--labelled_master", required=True,
                    help="Path to labelled_master.csv")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--limit", type=int, default=10000,
                    help="Cap per file for fast metrics")
    ap.add_argument("--div_sample", type=int, default=1000,
                    help="Sample size for internal diversity")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Loading labelled master from {args.labelled_master}")
    sys.stdout.flush()
    lm = pd.read_csv(args.labelled_master, usecols=["smiles"])
    lm_canon = set()
    lm_fps = []
    for smi in lm["smiles"]:
        c = canon(smi)
        if c:
            lm_canon.add(c)
            m = Chem.MolFromSmiles(c)
            if m: lm_fps.append(fp(m))
    print(f"[train] labelled-master: {len(lm_canon)} canonical")
    sys.stdout.flush()

    # PAINS filter catalog
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_cat = FilterCatalog(params)

    # Optional chem_redflags
    try:
        sys.path.insert(0, ".")
        from chem_redflags import screen
        has_redflags = True
        print("[train] chem_redflags available")
    except Exception as e:
        has_redflags = False
        print(f"[train] chem_redflags not loaded ({e}); using PAINS only")
    sys.stdout.flush()

    files = sorted(Path(args.results_dir).glob("*.txt"))
    print(f"[train] Found {len(files)} files in {args.results_dir}")
    sys.stdout.flush()

    summary = {"per_condition": [], "args": vars(args)}
    for f_idx, f in enumerate(files, 1):
        cond = f.stem
        print(f"\n[train] {f_idx}/{len(files)} loss=0.0000 cond={cond}")
        sys.stdout.flush()
        smis_raw = [s.strip() for s in f.read_text().splitlines() if s.strip()]
        smis_raw = smis_raw[:args.limit]

        # Validity
        canon_smis = []
        for s in smis_raw:
            c = canon(s)
            if c: canon_smis.append(c)
        validity = len(canon_smis) / max(len(smis_raw), 1)
        # Uniqueness
        unique_canon = set(canon_smis)
        uniqueness = len(unique_canon) / max(len(canon_smis), 1)
        # Novelty vs labelled-master
        novel = unique_canon - lm_canon
        novelty = len(novel) / max(len(unique_canon), 1)
        unique_list = list(unique_canon)
        print(f"  validity={validity:.3f}  uniqueness={uniqueness:.3f}  novelty(vs LM)={novelty:.3f}")
        sys.stdout.flush()

        # Internal diversity (1 - mean pairwise Tanimoto on a sample)
        sample = unique_list[:args.div_sample] if len(unique_list) > args.div_sample else unique_list
        mols = [Chem.MolFromSmiles(s) for s in sample]
        mols = [m for m in mols if m]
        fps = [fp(m) for m in mols]
        if len(fps) >= 2:
            sims = []
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            int_div = float(1 - np.mean(sims)) if sims else None
        else:
            int_div = None
        print(f"  IntDiv1 (n={len(fps)}): {int_div:.3f}" if int_div else "  IntDiv1: N/A")
        sys.stdout.flush()

        # Scaffold count
        scafs = set()
        for m in mols:
            try:
                sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
                if sc: scafs.add(sc)
            except: pass
        n_scaf = len(scafs)
        print(f"  scaffolds: {n_scaf}")
        sys.stdout.flush()

        # Filters
        n_pains_pass = 0; n_redflag_pass = 0; n_filter_total = 0
        for s in unique_list[:5000]:
            n_filter_total += 1
            m = Chem.MolFromSmiles(s)
            if m is None: continue
            if not pains_cat.HasMatch(m):
                n_pains_pass += 1
            if has_redflags:
                try:
                    res = screen(s)
                    if res["status"] == "ok":
                        n_redflag_pass += 1
                except: pass
        pains_rate = n_pains_pass / max(n_filter_total, 1)
        redflag_rate = (n_redflag_pass / max(n_filter_total, 1)) if has_redflags else None
        print(f"  PAINS-pass: {pains_rate:.3f} (sample n={n_filter_total})")
        if redflag_rate is not None:
            print(f"  chem_redflags pass: {redflag_rate:.3f}")
        sys.stdout.flush()

        # SNN: avg max Tanimoto to labelled master (sample 500)
        snn_sample = sample[:500]
        nn_tans = []
        for s in snn_sample:
            m = Chem.MolFromSmiles(s)
            if m is None: continue
            q = fp(m)
            tans = DataStructs.BulkTanimotoSimilarity(q, lm_fps)
            nn_tans.append(max(tans))
        snn_mean = float(np.mean(nn_tans)) if nn_tans else None
        print(f"  SNN (avg max Tani to LM, n={len(nn_tans)}): {snn_mean:.3f}")
        sys.stdout.flush()

        summary["per_condition"].append({
            "condition": cond,
            "n_raw": len(smis_raw),
            "n_canon_valid": len(canon_smis),
            "n_unique": len(unique_canon),
            "n_novel_vs_LM": len(novel),
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty_vs_LM": novelty,
            "int_div1": int_div,
            "scaffolds_in_sample": n_scaf,
            "pains_pass_rate": pains_rate,
            "redflags_pass_rate": redflag_rate,
            "snn_to_LM": snn_mean,
        })

    Path(out_dir / "m4a_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[train] -> {out_dir / 'm4a_metrics.json'}")

    # Markdown summary
    md = ["# M4a MOSES-style metrics on M1''' headline pool",
           "", "| Condition | validity | uniqueness | novelty(LM) | IntDiv1 | scaffolds | PAINS-pass | redflags-pass | SNN(LM) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in summary["per_condition"]:
        md.append(
            f"| {r['condition']} | {r['validity']:.3f} | {r['uniqueness']:.3f} | "
            f"{r['novelty_vs_LM']:.3f} | "
            f"{r['int_div1']:.3f} | {r['scaffolds_in_sample']} | "
            f"{r['pains_pass_rate']:.3f} | "
            f"{r['redflags_pass_rate']:.3f} | "
            f"{r['snn_to_LM']:.3f} |"
        )
    Path(out_dir / "m4a_summary.md").write_text("\n".join(md))
    print(f"[train] -> {out_dir / 'm4a_summary.md'}")
    print("[train] === DONE ===")


if __name__ == "__main__":
    main()
