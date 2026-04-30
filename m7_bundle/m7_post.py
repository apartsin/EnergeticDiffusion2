"""M7 postprocessor: fuse 5 lanes, widen novelty window, rank top-200.

Key differences from m6_post.py:
  - Tanimoto novelty window: [0.15, 0.65]  (was [0.20, 0.55])
  - Fuses all lanes before ranking (not per-lane top-100)
  - Top-200 output to capture broader lead space
  - Cross-lane dedup via canonical SMILES before scoring
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SCRIPTS_DIFF = HERE.parent / "scripts" / "diffusion"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS_DIFF))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--labelled_master", required=True)
    ap.add_argument("--smoke_model", required=True)
    ap.add_argument("--target_density", type=float, default=1.95)
    ap.add_argument("--target_hof", type=float, default=220.0)
    ap.add_argument("--target_d", type=float, default=9.5)
    ap.add_argument("--target_p", type=float, default=40.0)
    ap.add_argument("--tanimoto_min", type=float, default=0.15)
    ap.add_argument("--tanimoto_max", type=float, default=0.65)
    ap.add_argument("--top_n", type=int, default=200)
    ap.add_argument("--out_json", default="results/m7_post.json")
    ap.add_argument("--out_md", default="results/m7_post.md")
    args = ap.parse_args()

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    from chem_filter import chem_filter
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

    print("[m7pp] Loading meta"); sys.stdout.flush()
    meta = json.loads(Path(args.meta_json).read_text())
    stats = meta["stats"]; pn = meta["property_names"]

    target_raw = {"density": args.target_density, "heat_of_formation": args.target_hof,
                   "detonation_velocity": args.target_d, "detonation_pressure": args.target_p}

    print(f"[m7pp] Loading ref SMILES from {args.labelled_master}"); sys.stdout.flush()
    lm = pd.read_csv(args.labelled_master, usecols=["smiles"], nrows=5000)
    train_smiles = lm["smiles"].tolist()
    ref_fps = [morgan_fp(s) for s in train_smiles if morgan_fp(s) is not None]
    print(f"[m7pp] ref_fps={len(ref_fps)}"); sys.stdout.flush()

    print(f"[m7pp] Loading 3DCNN from {args.smoke_model}"); sys.stdout.flush()
    val = UniMolValidator(model_dir=args.smoke_model)
    name_map = {"density": "density", "heat_of_formation": "HOF_S",
                 "detonation_velocity": "DetoD", "detonation_pressure": "DetoP"}

    files = sorted(Path(args.results_dir).glob("m7_*.txt"))
    print(f"[m7pp] Found {len(files)} lane files"); sys.stdout.flush()

    # Collect + dedup all raw SMILES across lanes
    all_smis_raw = []
    lane_counts = {}
    for f in files:
        lane = f.stem
        lines = [l.strip() for l in f.read_text(encoding="utf-8").split("\n") if l.strip()]
        lane_counts[lane] = len(lines)
        all_smis_raw.extend(lines)
    print(f"[m7pp] Total raw across all lanes: {len(all_smis_raw)}"); sys.stdout.flush()

    # Canonicalize + dedup
    seen = set(); dedup = []
    for s in all_smis_raw:
        c = canon(s)
        if c and c not in seen:
            seen.add(c); dedup.append(c)
    print(f"[m7pp] After canon+dedup: {len(dedup)}"); sys.stdout.flush()

    # Chem filter (chemistry-only, no property bounds at this stage)
    keep_chem = [chem_filter(s, props=None)[0] for s in dedup]
    smis = [s for s, ok in zip(dedup, keep_chem) if ok]
    print(f"[m7pp] After chem filter: {len(smis)}"); sys.stdout.flush()

    # Neutral filter
    smis = [s for s in smis if is_neutral(s)]
    print(f"[m7pp] After neutral filter: {len(smis)}"); sys.stdout.flush()

    # SA/SC caps
    sa_arr = np.array([real_sa(s) for s in smis])
    sc_arr = np.array([real_sc(s) for s in smis])
    keep_feasible = (sa_arr <= SA_DROP_ABOVE) & (sc_arr <= SC_DROP_ABOVE)
    smis = [s for s, ok in zip(smis, keep_feasible) if ok]
    print(f"[m7pp] After SA/SC caps: {len(smis)}"); sys.stdout.flush()

    # Tanimoto novelty window (wider than m6)
    max_tans = []
    for s in smis:
        fp = morgan_fp(s)
        if fp is None: max_tans.append(0.0); continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        max_tans.append(max(sims))
    max_tans = np.array(max_tans)
    keep_novel = (max_tans >= args.tanimoto_min) & (max_tans <= args.tanimoto_max)
    smis = [s for s, ok in zip(smis, keep_novel) if ok]
    print(f"[m7pp] After Tanimoto [{args.tanimoto_min},{args.tanimoto_max}]: {len(smis)}"); sys.stdout.flush()

    if not smis:
        print("[m7pp] No candidates survived filtering!"); return

    # 3DCNN scoring
    print(f"[m7pp] Running 3DCNN on {len(smis)} candidates..."); sys.stdout.flush()
    t0 = time.time()
    preds = val.predict(smis)
    print(f"[m7pp] 3DCNN done in {time.time()-t0:.1f}s"); sys.stdout.flush()

    # Build rows
    rows = []
    sa_final = [real_sa(s) for s in smis]
    sc_final = [real_sc(s) for s in smis]
    for i, s in enumerate(smis):
        row = {"smiles": s, "sa": sa_final[i], "sc": sc_final[i], "maxtan": max_tans[keep_novel][i]}
        for k in pn:
            if k not in preds:
                continue
            raw_val = float(preds[k][i])
            import math
            if math.isnan(raw_val):
                continue
            mapped = name_map.get(k, k)
            row[mapped] = raw_val  # validator already returns physical units
        rows.append(row)

    # Sort by composite (ascending = better):
    # sum of normalized property distances + SA/SC penalty
    _mapped_to_orig = {"density": "density", "HOF_S": "heat_of_formation",
                       "DetoD": "detonation_velocity", "DetoP": "detonation_pressure"}
    for r in rows:
        prop_dist = sum(
            abs(r.get(mk, 0.0) - target_raw[ok]) / max(stats[ok]["std"], 1e-6)
            for mk, ok in _mapped_to_orig.items()
        )
        r["composite"] = prop_dist + composite_feasibility_penalty(r["sa"], r["sc"], 0.5, 0.25)
    rows.sort(key=lambda r: r["composite"])
    top = rows[:args.top_n]

    # Scaffold diversity
    def murcko(s):
        m = Chem.MolFromSmiles(s)
        if m is None: return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    scaffolds = set(murcko(r["smiles"]) for r in top if murcko(r["smiles"]))
    int_div = 1.0
    if len(top) > 1:
        top_fps = [morgan_fp(r["smiles"]) for r in top if morgan_fp(r["smiles"])]
        pairs = [1.0 - DataStructs.TanimotoSimilarity(a, b)
                 for i, a in enumerate(top_fps) for b in top_fps[i+1:]]
        int_div = float(np.mean(pairs)) if pairs else 1.0

    result = {
        "n_raw_total": len(all_smis_raw),
        "n_dedup": len(dedup),
        "n_validated": len(smis),
        "n_top": len(top),
        "tanimoto_window": [args.tanimoto_min, args.tanimoto_max],
        "lane_counts": lane_counts,
        "topN_internal_diversity": int_div,
        "topN_n_scaffolds": len(scaffolds),
        "top1_composite": top[0]["composite"] if top else None,
        "top1_D_kms": top[0].get("DetoD") if top else None,
        "top1_smiles": top[0]["smiles"] if top else None,
        "topN_mean_D": float(np.mean([r["DetoD"] for r in top if r.get("DetoD")])) if top else None,
        "topN_max_D": float(max(r["DetoD"] for r in top if r.get("DetoD"))) if top else None,
        "top_candidates": top,
    }

    Path(args.out_json).parent.mkdir(exist_ok=True, parents=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[m7pp] JSON -> {args.out_json}"); sys.stdout.flush()

    md = [f"# M7 100k-pool results", "",
          f"Lanes: {len(files)} | Raw: {result['n_raw_total']:,} | "
          f"Valid: {result['n_validated']:,} | Top-{args.top_n}: {result['n_top']}",
          f"Tanimoto window: {args.tanimoto_min}–{args.tanimoto_max}  "
          f"IntDiv: {int_div:.3f}  Scaffolds: {len(scaffolds)}", "",
          "## Top-10 candidates", "",
          "| Rank | SMILES | D (km/s) | rho | P (GPa) | composite | maxtan |",
          "|------|--------|----------|-----|---------|-----------|--------|"]
    for i, r in enumerate(top[:10], 1):
        md.append(f"| {i} | `{r['smiles']}` | {r.get('DetoD', 'n/a'):.2f} | "
                  f"{r.get('density', 'n/a'):.3f} | {r.get('DetoP', 'n/a'):.1f} | "
                  f"{r['composite']:.4f} | {r['maxtan']:.3f} |")

    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"[m7pp] MD  -> {args.out_md}")
    print(f"\n[m7pp] Summary:")
    print(f"  top1 D (km/s): {result['top1_D_kms']}")
    print(f"  topN max D:    {result['topN_max_D']}")
    print(f"  topN mean D:   {result['topN_mean_D']}")
    print(f"  n_validated:   {result['n_validated']}")
    print(f"  n_scaffolds:   {result['topN_n_scaffolds']}")


if __name__ == "__main__":
    main()
