"""Slim runpod postprocess for M6 3-seed sampling.

Reads stats from meta.json (no heavy denoiser ckpt) + ref_smiles from
labelled_master.csv. Runs 3DCNN + chem_filter + SA/SC + Tanimoto + composite
+ scaffolds for each SMILES file. Outputs json + md.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--labelled_master", required=True)
    ap.add_argument("--smoke_model", required=True,
                    help="Directory with the 3DCNN smoke model")
    ap.add_argument("--target_density", type=float, default=1.95)
    ap.add_argument("--target_hof", type=float, default=220.0)
    ap.add_argument("--target_d", type=float, default=9.5)
    ap.add_argument("--target_p", type=float, default=40.0)
    ap.add_argument("--top_n", type=int, default=100)
    ap.add_argument("--limit_per_file", type=int, default=20000)
    ap.add_argument("--out_json", default="results/m6_post.json")
    ap.add_argument("--out_md", default="results/m6_post.md")
    args = ap.parse_args()

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    from chem_filter import chem_filter_batch
    from feasibility_utils import (real_sa, real_sc, SA_DROP_ABOVE, SC_DROP_ABOVE,
                                     composite_feasibility_penalty)
    from unimol_validator import UniMolValidator

    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else None

    def is_neutral(s):
        m = Chem.MolFromSmiles(s)
        if m is None: return False
        if Chem.GetFormalCharge(m) != 0: return False
        return not any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms())

    def morgan_fp(smi):
        m = Chem.MolFromSmiles(smi)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None

    print("[train] Loading stats from meta.json"); sys.stdout.flush()
    meta = json.loads(Path(args.meta_json).read_text())
    stats = meta["stats"]
    pn = meta["property_names"]

    target_raw = {"density": args.target_density, "heat_of_formation": args.target_hof,
                   "detonation_velocity": args.target_d, "detonation_pressure": args.target_p}

    print(f"[train] Loading ref SMILES from {args.labelled_master}"); sys.stdout.flush()
    lm = pd.read_csv(args.labelled_master, usecols=["smiles"], nrows=5000)
    train_smiles = lm["smiles"].tolist()
    ref_fps = [morgan_fp(s) for s in train_smiles if morgan_fp(s) is not None]
    print(f"[train] ref_fps={len(ref_fps)}"); sys.stdout.flush()

    print(f"[train] Loading 3DCNN from {args.smoke_model}"); sys.stdout.flush()
    val = UniMolValidator(model_dir=args.smoke_model)
    name_map = {"density": "density", "heat_of_formation": "HOF_S",
                 "detonation_velocity": "DetoD", "detonation_pressure": "DetoP"}

    files = sorted(Path(args.results_dir).glob("*.txt"))
    print(f"[train] Found {len(files)} SMILES files"); sys.stdout.flush()

    per_run = []
    for f_idx, f in enumerate(files, 1):
        # Parse condition + seed from filename like m1_3seed_C0_unguided_seed0
        stem = f.stem
        for prefix in ("m1_3seed_", "m1_sweep_", "m1ss_", "m1pp_"):
            stem = stem.replace(prefix, "")
        cond = stem
        seed = 0
        if "_seed" in stem:
            cond, _, sp = stem.rpartition("_seed")
            try: seed = int(sp)
            except: pass

        print(f"\n[train] {f_idx}/{len(files)} loss=0.0000 cond={cond} seed={seed}")
        sys.stdout.flush()
        smis_raw = [s.strip() for s in f.read_text().splitlines() if s.strip()]
        smis_raw = smis_raw[:args.limit_per_file]
        canons = []
        for s in smis_raw:
            c = canon(s)
            if c and is_neutral(c): canons.append(c)
        seen = {}
        for c in canons:
            if c not in seen: seen[c] = True
        smis = list(seen.keys())
        print(f"  unique-neutral: {len(smis)}"); sys.stdout.flush()

        if len(smis) < 50:
            per_run.append({"condition": cond, "seed": seed, "n_top": 0}); continue

        # 3DCNN scoring
        t0 = time.time()
        pdict = val.predict(smis)
        cols = {p: np.asarray(pdict.get(name_map[p]), dtype=float) for p in pn}
        keep = np.ones(len(smis), dtype=bool)
        for p in pn: keep &= ~np.isnan(cols[p])
        smis = [s for s, k in zip(smis, keep) if k]
        cols = {p: cols[p][keep] for p in pn}
        print(f"  3DCNN: {len(smis)} ({time.time()-t0:.0f}s)"); sys.stdout.flush()

        # chem_filter
        keep_idx, _ = chem_filter_batch(smis, cols, pn)
        smis = [smis[i] for i in keep_idx]
        cols = {p: cols[p][np.asarray(keep_idx)] for p in pn}
        # SA/SC
        sa = np.array([real_sa(s) for s in smis])
        sc = np.array([real_sc(s) for s in smis])
        keep = np.ones(len(smis), dtype=bool)
        keep &= ~((~np.isnan(sa)) & (sa > 5.0))
        keep &= ~((~np.isnan(sc)) & (sc > 3.5))
        idx = np.where(keep)[0]
        smis = [smis[i] for i in idx]
        cols = {p: cols[p][idx] for p in pn}
        sa = sa[idx]; sc = sc[idx]
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
        print(f"  Phase-A: {len(smis)}"); sys.stdout.flush()

        # Composite
        comp = np.zeros(len(smis))
        for p in pn:
            comp += np.abs(cols[p] - target_raw[p]) / max(stats[p]["std"], 1e-6)
        pen = np.array([composite_feasibility_penalty(sa[k], sc[k], 0.5, 0.25)
                          for k in range(len(smis))])
        comp += pen
        order = np.argsort(comp)
        topN = order[:args.top_n]

        if len(topN) > 0:
            top_smis = [smis[i] for i in topN]
            top_comp = comp[topN]
            top_d = cols["detonation_velocity"][topN]
            top_p = cols["detonation_pressure"][topN]
            top_rho = cols["density"][topN]
            mols = [Chem.MolFromSmiles(s) for s in top_smis]
            mols = [m for m in mols if m]
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
            sims = []
            for i in range(len(fps)):
                for j in range(i+1, len(fps)):
                    sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            internal_div = float(1 - np.mean(sims)) if sims else None
            scafs = set()
            for m in mols:
                try:
                    sc_str = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
                    if sc_str: scafs.add(sc_str)
                except: pass
            run_data = {
                "condition": cond, "seed": seed,
                "n_validated": len(smis),
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
                "topN_n_scaffolds": len(scafs),
            }
        else:
            run_data = {"condition": cond, "seed": seed, "n_top": 0}
        per_run.append(run_data)
        print(f"  -> top1 D={run_data.get('top1_D_kms')} comp={run_data.get('top1_composite')} scafs={run_data.get('topN_n_scaffolds')}")
        sys.stdout.flush()

    # Aggregate
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

    md = ["# M6 3-seed postprocess summary", "",
           "| Condition | top-1 composite | top-1 D | top-100 mean D | scaffolds | int_div | n_seeds |",
           "|---|---|---|---|---|---|---|"]
    for cond in ["C0_unguided", "C1_viab_sens", "C2_viab_sens_hazard", "C3_hazard_only"]:
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
    print(f"\n[train] -> {args.out_json}")
    print(f"[train] -> {args.out_md}")
    print("[train] === DONE ===")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
