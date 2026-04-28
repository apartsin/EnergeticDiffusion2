"""Stage 1: Mine the E-set 500 candidates from 4 sampling pools.

Pools (per brief):
  pool=40k unguided       -> m1_bundle/results/m1pp_C0_unguided.txt
  pool=80k unguided       -> m1_bundle/results/m1_anneal_clamp_anneal0_clamp5.txt
  pool=40k guided viab+sens -> m1_bundle/results/m1pp_C1_viab_sens.txt
  pool=20k guided SA      -> m1_bundle/results/m1grid_sv1_sh2.txt  (sv=SA-guidance, sh=hazard)

Filters:
  - canonical, neutral, no radicals
  - exclude L1-L20 / R-rejects (lead set)
  - SA <= 5.0, SC <= 3.5
  - Tanimoto-NN window [0.20, 0.55] vs labelled master (Phase-A novelty band)
  - Tanimoto-NN <= 0.55 vs L1-L20 (force scaffold novelty)
Composite reranker (scaffold-aware Phase-A): weighted distance to (rho=1.95,
  HOF=220, D=9.5, P=40) + SA/SC penalty + scaffold-bucket diversity bonus.
Top 500 -> results/extension_set/e_set_500_smiles.json.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("E:/Projects/EnergeticDiffusion2")
OUT_DIR = ROOT / "results" / "extension_set"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_500 = OUT_DIR / "e_set_500_smiles.json"

POOLS = [
    ("pool_40k_unguided",   ROOT / "m1_bundle/results/m1pp_C0_unguided.txt"),
    ("pool_80k_unguided",   ROOT / "m1_bundle/results/m1_anneal_clamp_anneal0_clamp5.txt"),
    ("pool_40k_viab_sens",  ROOT / "m1_bundle/results/m1pp_C1_viab_sens.txt"),
    ("pool_20k_sa",         ROOT / "m1_bundle/results/m1grid_sv1_sh2.txt"),
]

# Targets used for composite (matches m6_post.py / paper §5)
TARGETS = {"density": 1.95, "heat_of_formation": 220.0,
           "detonation_velocity": 9.5, "detonation_pressure": 40.0}

# m6_postprocess_bundle on path for chem_filter / feasibility / 3DCNN
sys.path.insert(0, str(ROOT / "m6_postprocess_bundle"))
sys.path.insert(0, str(ROOT))


def main():
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    import pandas as pd

    from chem_filter import chem_filter_batch
    from feasibility_utils import (real_sa, real_sc,
                                     composite_feasibility_penalty)
    from unimol_validator import UniMolValidator

    def canon(s):
        try:
            m = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(m, canonical=True) if m else None
        except Exception:
            return None

    def is_neutral(s):
        m = Chem.MolFromSmiles(s)
        if m is None: return False
        if Chem.GetFormalCharge(m) != 0: return False
        return not any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms())

    def morgan_fp(s):
        m = Chem.MolFromSmiles(s)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None

    # ---- L1..L20 / R rejects exclusion set ---------------------------------
    leads_dir = ROOT / "m2_bundle" / "results"
    excl_smiles = set()
    L_ids = ["L1","L2","L3","L4","L5","L9","L11","L13","L16","L18","L19","L20"]
    R_ids = ["R2","R3","R14"]
    L_canon_list = []
    for lid in L_ids + R_ids:
        p = leads_dir / f"m2_lead_{lid}.json"
        if p.exists():
            d = json.loads(p.read_text())
            smi = d.get("smiles") or d.get("name", "")
            c = canon(smi)
            if c:
                excl_smiles.add(c)
                if lid in L_ids:
                    L_canon_list.append(c)
    print(f"[mine] excl L+R smiles: {len(excl_smiles)} (L_only={len(L_canon_list)})")
    L_fps = [morgan_fp(s) for s in L_canon_list if morgan_fp(s) is not None]

    # ---- load union pool ---------------------------------------------------
    union = {}
    pool_counts = {}
    for name, path in POOLS:
        if not path.exists():
            print(f"[mine] WARN missing pool {path}")
            pool_counts[name] = 0
            continue
        n_raw = 0
        n_canon = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s: continue
            n_raw += 1
            c = canon(s)
            if not c or not is_neutral(c): continue
            n_canon += 1
            if c in excl_smiles: continue
            if c not in union:
                union[c] = name  # remember first-seen pool
        pool_counts[name] = {"raw": n_raw, "canon_neutral": n_canon}
        print(f"[mine] {name}: raw={n_raw}, canon_neutral={n_canon}, union_size_so_far={len(union)}")

    smis = list(union.keys())
    print(f"[mine] dedup union size: {len(smis)}")

    # ---- 3DCNN scoring -----------------------------------------------------
    # The dataset stats live in m6_postprocess_bundle/meta.json
    meta = json.loads((ROOT / "m6_postprocess_bundle" / "meta.json").read_text())
    stats = meta["stats"]
    pn = meta["property_names"]
    smoke_model = ROOT / "m6_postprocess_bundle"
    # find smoke_model dir (chem_filter expects full path with model files)
    # smoke_model is .tar.gz; extract if needed
    sm_dir = ROOT / "m6_postprocess_bundle" / "_smoke_model"
    if not sm_dir.exists():
        import tarfile
        tarball = ROOT / "m6_postprocess_bundle" / "smoke_model.tar.gz"
        if tarball.exists():
            sm_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tarball, "r:gz") as t:
                t.extractall(sm_dir)
            print(f"[mine] extracted smoke_model -> {sm_dir}")

    # Find actual model dir (smoke_model/.../config.yaml)
    model_dir = None
    for cand in sm_dir.rglob("config.yaml"):
        model_dir = cand.parent; break
    if model_dir is None:
        # Try alternative: just point to sm_dir
        model_dir = sm_dir
    print(f"[mine] 3DCNN model dir: {model_dir}")

    # 3DCNN can be slow for 100k+; sub-sample to 30k if pool larger
    MAX_SCORE = 30000
    if len(smis) > MAX_SCORE:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(smis), size=MAX_SCORE, replace=False)
        smis = [smis[i] for i in sorted(idx)]
        print(f"[mine] subsampled to {len(smis)} for 3DCNN scoring")

    val = UniMolValidator(model_dir=str(model_dir))
    name_map = {"density": "density", "heat_of_formation": "HOF_S",
                 "detonation_velocity": "DetoD", "detonation_pressure": "DetoP"}

    # batch predict
    t0 = time.time()
    pdict = val.predict(smis)
    cols = {p: np.asarray(pdict.get(name_map[p]), dtype=float) for p in pn}
    keep = np.ones(len(smis), dtype=bool)
    for p in pn: keep &= ~np.isnan(cols[p])
    smis = [s for s, k in zip(smis, keep) if k]
    cols = {p: cols[p][keep] for p in pn}
    print(f"[mine] 3DCNN: {len(smis)} ({time.time()-t0:.0f}s)")

    # chem_filter (Phase-A red-flags)
    keep_idx, _ = chem_filter_batch(smis, cols, pn)
    smis = [smis[i] for i in keep_idx]
    cols = {p: cols[p][np.asarray(keep_idx)] for p in pn}
    print(f"[mine] chem_filter: {len(smis)}")

    # SA / SC
    sa = np.array([real_sa(s) for s in smis])
    sc = np.array([real_sc(s) for s in smis])
    keep = np.ones(len(smis), dtype=bool)
    keep &= ~((~np.isnan(sa)) & (sa > 5.0))
    keep &= ~((~np.isnan(sc)) & (sc > 3.5))
    idx = np.where(keep)[0]
    smis = [smis[i] for i in idx]
    cols = {p: cols[p][idx] for p in pn}
    sa = sa[idx]; sc = sc[idx]
    print(f"[mine] SA/SC: {len(smis)}")

    # Tanimoto window vs labelled master
    print("[mine] computing Tanimoto vs labelled_master")
    lm_csv = ROOT / "m6_postprocess_bundle" / "labelled_master.csv"
    lm = pd.read_csv(lm_csv, usecols=["smiles"], nrows=5000)
    ref_fps = [morgan_fp(s) for s in lm["smiles"].tolist() if morgan_fp(s) is not None]
    print(f"[mine] ref_fps={len(ref_fps)}")

    fps = [morgan_fp(s) for s in smis]
    max_tan = []
    for fp in fps:
        if fp is None: max_tan.append(0.0); continue
        tans = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        max_tan.append(max(tans) if tans else 0.0)
    max_tan = np.array(max_tan)
    keep = (max_tan >= 0.20) & (max_tan <= 0.55)
    idx = np.where(keep)[0]
    smis = [smis[i] for i in idx]
    cols = {p: cols[p][idx] for p in pn}
    sa = sa[idx]; sc = sc[idx]
    fps = [fps[i] for i in idx]
    print(f"[mine] Tanimoto-window: {len(smis)}")

    # Tanimoto vs L set <= 0.55 (force novelty against validated leads)
    if L_fps:
        max_tan_L = []
        for fp in fps:
            if fp is None: max_tan_L.append(0.0); continue
            tans = DataStructs.BulkTanimotoSimilarity(fp, L_fps)
            max_tan_L.append(max(tans) if tans else 0.0)
        max_tan_L = np.array(max_tan_L)
        keep = max_tan_L <= 0.55
        idx = np.where(keep)[0]
        smis = [smis[i] for i in idx]
        cols = {p: cols[p][idx] for p in pn}
        sa = sa[idx]; sc = sc[idx]; fps = [fps[i] for i in idx]
        print(f"[mine] vs-L novelty <=0.55: {len(smis)}")

    if len(smis) < 200:
        print(f"[mine] FATAL: only {len(smis)} candidates, pool too sparse, stopping")
        sys.exit(2)

    # ---- composite reranker -----------------------------------------------
    comp = np.zeros(len(smis))
    for p in pn:
        comp += np.abs(cols[p] - TARGETS[p]) / max(stats[p]["std"], 1e-6)
    pen = np.array([composite_feasibility_penalty(sa[k], sc[k], 0.5, 0.25)
                      for k in range(len(smis))])
    comp += pen

    # scaffold bucket -> small bonus per first-of-scaffold
    scafs = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            scafs.append(""); continue
        try:
            sc_str = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            sc_str = ""
        scafs.append(sc_str or "")

    order = np.argsort(comp)
    seen_scaf = set()
    picked = []
    # First, take 1 per scaffold up to 200, then fill by composite
    for i in order:
        if len(picked) >= 200: break
        if scafs[i] in seen_scaf: continue
        seen_scaf.add(scafs[i])
        picked.append(int(i))
    # Fill remainder by composite ranking (excluding already picked)
    pset = set(picked)
    for i in order:
        if len(picked) >= 500: break
        if int(i) in pset: continue
        picked.append(int(i))
        pset.add(int(i))

    out = []
    for rank, i in enumerate(picked):
        out.append({
            "rank": rank,
            "smiles": smis[i],
            "scaffold": scafs[i],
            "composite": float(comp[i]),
            "rho_3dcnn": float(cols["density"][i]),
            "HOF_3dcnn": float(cols["heat_of_formation"][i]),
            "D_3dcnn": float(cols["detonation_velocity"][i]),
            "P_3dcnn": float(cols["detonation_pressure"][i]),
            "SA": float(sa[i]) if not np.isnan(sa[i]) else None,
            "SC": float(sc[i]) if not np.isnan(sc[i]) else None,
        })

    OUT_500.write_text(json.dumps({
        "n_total_pool_union_canon_neutral": len(union),
        "n_after_3dcnn_chem_SA_SC_tan": len(smis),
        "n_picked": len(out),
        "pool_counts": pool_counts,
        "L_R_excluded": sorted(excl_smiles),
        "items": out,
    }, indent=2))
    print(f"[mine] -> {OUT_500} ({len(out)} entries)")


if __name__ == "__main__":
    main()
