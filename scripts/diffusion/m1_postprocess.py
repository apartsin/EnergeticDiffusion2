"""M1 post-processing: take 12 SMILES files (4 conditions x 3 seeds, ~20k each),
run the same Phase-A pipeline as joint_rerank.py, report per-condition
seed-mean +/- std on:
    - composite top-100 mean
    - top-1 D, P, rho
    - scaffold count (Bemis-Murcko)
    - internal diversity
    - PubChem novelty (sampled to 100 per file to keep cost down)

Output:
    experiments/m1_postprocess.json
    experiments/m1_postprocess_summary.md (paper-ready table)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_job-20260427-102854")
    ap.add_argument("--exp_v4b", default="experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z")
    ap.add_argument("--target_density", type=float, default=1.95)
    ap.add_argument("--target_hof", type=float, default=220.0)
    ap.add_argument("--target_d", type=float, default=9.5)
    ap.add_argument("--target_p", type=float, default=40.0)
    ap.add_argument("--top_n", type=int, default=100,
                    help="Top-N kept after Phase-A composite for stats.")
    ap.add_argument("--limit_per_file", type=int, default=20000,
                    help="Cap on SMILES processed per file (keeps cost bounded).")
    ap.add_argument("--out_json", default="experiments/m1_postprocess.json")
    ap.add_argument("--out_md", default="experiments/m1_postprocess_summary.md")
    args = ap.parse_args()

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    # Reuse joint_rerank helpers
    from joint_rerank import (canon, is_neutral, real_sa, real_sc,
                                composite_feasibility_penalty, morgan_fp,
                                load_denoiser_pack)
    from chem_filter import chem_filter_batch
    from unimol_validator import UniMolValidator
    import torch

    # Load v4b just for stats + reference SMILES
    print("Loading v4-B for stats + Tanimoto reference ..."); sys.stdout.flush()
    base = Path.cwd()
    exp = Path(args.exp_v4b)
    if not exp.is_absolute(): exp = base / exp
    d_v4b, sch_v4b, l_v4b, pn, n_props = load_denoiser_pack(
        exp, "best.pt", "cuda" if torch.cuda.is_available() else "cpu", base)
    stats = l_v4b["stats"]
    target_raw = {"density": args.target_density, "heat_of_formation": args.target_hof,
                   "detonation_velocity": args.target_d, "detonation_pressure": args.target_p}
    train_smiles = l_v4b.get("smiles", [])[:5000]
    ref_fps = [morgan_fp(s) for s in train_smiles if morgan_fp(s) is not None]
    print(f"  ref_fps={len(ref_fps)}"); sys.stdout.flush()

    name_map = {"density": "density", "heat_of_formation": "HOF_S",
                 "detonation_velocity": "DetoD", "detonation_pressure": "DetoP"}
    val = UniMolValidator(model_dir=str(
        base / "data/raw/energetic_external/EMDP/Data/smoke_model"))

    files = sorted(Path(args.results_dir).glob("m1_sweep_*.txt"))
    print(f"\nFound {len(files)} SMILES files"); sys.stdout.flush()

    per_run = []

    for f in files:
        # Parse condition + seed from filename
        stem = f.stem.replace("m1_sweep_", "")
        cond, _, seed_part = stem.rpartition("_seed")
        seed = int(seed_part)

        print(f"\n=== {cond} seed={seed} ==="); sys.stdout.flush()
        smis_raw = [s.strip() for s in f.read_text().splitlines() if s.strip()]
        smis_raw = smis_raw[:args.limit_per_file]
        print(f"  raw: {len(smis_raw)}"); sys.stdout.flush()

        # Canonicalise + neutral + dedup
        canons = []
        for s in smis_raw:
            c = canon(s)
            if c and is_neutral(c):
                canons.append(c)
        seen = {}
        for c in canons:
            if c not in seen: seen[c] = True
        smis = list(seen.keys())
        print(f"  unique-neutral: {len(smis)}"); sys.stdout.flush()

        if len(smis) < 50:
            print(f"  too few SMILES, skipping"); continue

        # 3DCNN scoring
        t0 = time.time()
        pdict = val.predict(smis)
        cols = {p: np.asarray(pdict.get(name_map[p]), dtype=float) for p in pn}
        keep = np.ones(len(smis), dtype=bool)
        for p in pn: keep &= ~np.isnan(cols[p])
        smis = [s for s, k in zip(smis, keep) if k]
        cols = {p: cols[p][keep] for p in pn}
        print(f"  3DCNN-validated: {len(smis)} ({time.time()-t0:.0f}s)"); sys.stdout.flush()

        if len(smis) == 0:
            per_run.append({"condition": cond, "seed": seed, "n_validated": 0}); continue

        # Phase-A chem filter
        keep_idx, _ = chem_filter_batch(smis, cols, pn)
        smis = [smis[i] for i in keep_idx]
        cols = {p: cols[p][np.asarray(keep_idx)] for p in pn}
        print(f"  chem_filter pass: {len(smis)}"); sys.stdout.flush()

        # SA/SC caps
        sa = np.array([real_sa(s) for s in smis])
        sc = np.array([real_sc(s) for s in smis])
        keep = np.ones(len(smis), dtype=bool)
        keep &= ~((~np.isnan(sa)) & (sa > 5.0))
        keep &= ~((~np.isnan(sc)) & (sc > 3.5))
        idx = np.where(keep)[0]
        smis = [smis[i] for i in idx]
        cols = {p: cols[p][idx] for p in pn}
        sa = sa[idx]; sc = sc[idx]
        print(f"  SA/SC pass: {len(smis)}"); sys.stdout.flush()

        # Tanimoto window
        max_tans = []
        for s in smis:
            fp = morgan_fp(s)
            if fp is None: max_tans.append(0.0); continue
            tans = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            max_tans.append(max(tans))
        max_tans = np.array(max_tans)
        keep = (max_tans >= 0.20) & (max_tans <= 0.55)
        idx = np.where(keep)[0]
        smis = [smis[i] for i in idx]
        cols = {p: cols[p][idx] for p in pn}
        sa = sa[idx]; sc = sc[idx]; max_tans = max_tans[idx]
        print(f"  Tanimoto window pass: {len(smis)}"); sys.stdout.flush()

        # Composite + feasibility penalty
        comp = np.zeros(len(smis))
        for p in pn:
            comp += np.abs(cols[p] - target_raw[p]) / max(stats[p]["std"], 1e-6)
        pen = np.array([composite_feasibility_penalty(sa[k], sc[k], 0.5, 0.25)
                          for k in range(len(smis))])
        comp += pen
        order = np.argsort(comp)
        topN = order[:args.top_n]

        # Per-run stats
        if len(topN) > 0:
            top_smis = [smis[i] for i in topN]
            top_comp = comp[topN]
            top_d = cols["detonation_velocity"][topN]
            top_p = cols["detonation_pressure"][topN]
            top_rho = cols["density"][topN]
            # Diversity
            mols = [Chem.MolFromSmiles(s) for s in top_smis]
            mols = [m for m in mols if m]
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
            n = len(fps)
            sims = []
            for i in range(n):
                for j in range(i+1, n):
                    sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            internal_div = float(1 - np.mean(sims)) if sims else None
            scafs = set()
            for m in mols:
                try:
                    sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
                    if sc: scafs.add(sc)
                except: pass
            n_scaffolds = len(scafs)
            run_data = {
                "condition": cond, "seed": seed,
                "n_raw": len(smis_raw), "n_validated": len(smis),
                "n_top": len(topN),
                "top1_composite": float(top_comp[0]),
                "top1_D_kms": float(top_d[0]),
                "top1_P_GPa": float(top_p[0]),
                "top1_rho": float(top_rho[0]),
                "top1_smiles": top_smis[0],
                "topN_mean_composite": float(top_comp.mean()),
                "topN_mean_D": float(top_d.mean()),
                "topN_max_D": float(top_d.max()),
                "topN_internal_diversity": internal_div,
                "topN_n_scaffolds": n_scaffolds,
            }
        else:
            run_data = {"condition": cond, "seed": seed, "n_top": 0}

        print(f"  -> top1 comp={run_data.get('top1_composite'):.3f} "
              f"D={run_data.get('top1_D_kms'):.2f} "
              f"scaffolds={run_data.get('topN_n_scaffolds')}"); sys.stdout.flush()
        per_run.append(run_data)

    # Aggregate by condition
    by_cond = defaultdict(list)
    for r in per_run:
        if r.get("n_top", 0) > 0:
            by_cond[r["condition"]].append(r)

    summary = {}
    for cond, runs in by_cond.items():
        cd = {}
        for k in ["top1_composite", "top1_D_kms", "topN_mean_composite",
                   "topN_mean_D", "topN_max_D", "topN_internal_diversity",
                   "topN_n_scaffolds"]:
            vals = [r[k] for r in runs if r.get(k) is not None]
            if vals:
                cd[f"{k}_mean"] = float(np.mean(vals))
                cd[f"{k}_std"] = float(np.std(vals))
        cd["n_seeds"] = len(runs)
        summary[cond] = cd

    out = {"per_run": per_run, "summary": summary}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out_json}"); sys.stdout.flush()

    # Markdown summary
    md = ["# M1 head-to-head: guidance-vs-unguided at matched compute (3 seeds, pool=20k)", ""]
    md.append("| Condition | top-1 composite | top-1 D (km/s) | top-100 mean D | scaffolds | internal div | n_seeds |")
    md.append("|---|---|---|---|---|---|---|")
    for cond in ["C0_unguided", "C1_viab", "C2_viab_sens", "C3_viab_sens_sa"]:
        s = summary.get(cond)
        if not s: continue
        md.append(
            f"| {cond} | {s.get('top1_composite_mean', 0):.3f} +/- {s.get('top1_composite_std', 0):.3f} "
            f"| {s.get('top1_D_kms_mean', 0):.2f} +/- {s.get('top1_D_kms_std', 0):.2f} "
            f"| {s.get('topN_mean_D_mean', 0):.2f} +/- {s.get('topN_mean_D_std', 0):.2f} "
            f"| {s.get('topN_n_scaffolds_mean', 0):.0f} +/- {s.get('topN_n_scaffolds_std', 0):.0f} "
            f"| {s.get('topN_internal_diversity_mean', 0):.3f} +/- {s.get('topN_internal_diversity_std', 0):.3f} "
            f"| {s.get('n_seeds')} |"
        )
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"-> {args.out_md}"); sys.stdout.flush()


if __name__ == "__main__":
    main()
