"""M10 postprocessor: filter, UniMol-score, and rank the 200k M10 pool.

Reads per-lane raw SMILES from m10_bundle/results/m10_L*.txt,
applies the standard pipeline:
  1. Canonicalize + dedup across lanes
  2. Chem filter (valence, element set, ring constraints)
  3. Neutral-only
  4. SA <= 4.5, SC <= 5 caps
  5. Tanimoto novelty window [0.10, 0.70] vs training set
  6. UniMol 3D-CNN scoring (D, rho, P, HOF)
  7. Rank by composite distance to target; emit top-200

Outputs:
  m10_bundle/results/m10_post.json
  m10_bundle/results/m10_post.md

Usage (after m10 sampling completes):
    python -m modal run m10_bundle/modal_m10_post.py
"""
from __future__ import annotations
import json, time
from pathlib import Path
import modal

HERE         = Path(__file__).parent.resolve()
PROJECT_ROOT = HERE.parent
COMBO        = PROJECT_ROOT / "combo_bundle"
M6           = PROJECT_ROOT / "m6_postprocess_bundle"
SMOKE_MODEL  = M6 / "_smoke_model" / "smoke_model"
RESULTS      = HERE / "results"
DATA_MASTER  = PROJECT_ROOT / "data" / "training" / "master" / "labeled_master.csv"

# ---------------------------------------------------------------------------
# Collect lane SMILES files locally (written by modal_m10_200k.py)
# ---------------------------------------------------------------------------
LANE_FILES = sorted(RESULTS.glob("m10_L*.txt"))

# ---------------------------------------------------------------------------
# Modal image: same CUDA+UniMol stack as reinvent_bundle scorer
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "libxrender1", "libxext6")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "rdkit-pypi", "numpy", "pandas", "scipy", "scikit-learn",
        "huggingface_hub", "selfies==2.1.1",
    )
    .pip_install("unimol_tools==0.1.5")
    .add_local_dir(str(SMOKE_MODEL), remote_path="/smoke_model")
    .add_local_file(str(M6 / "unimol_validator.py"), remote_path="/app/unimol_validator.py")
    .add_local_file(str(COMBO / "meta.json"),        remote_path="/app/meta.json")
)

app = modal.App("dgld-m10-post", image=image)


@app.function(gpu="A10G", timeout=2 * 60 * 60, memory=16_384)
def score_and_rank(
    smiles_list: list[str],
    ref_smiles: list[str],
    target_d: float = 9.5,
    target_p: float = 40.0,
    target_density: float = 1.95,
    tanimoto_min: float = 0.10,
    tanimoto_max: float = 0.70,
    sa_cap: float = 4.5,
    sc_cap: float = 5.0,
    top_n: int = 200,
) -> dict:
    """Filter + score + rank on remote GPU. Returns top_n candidates."""
    import sys, math
    import numpy as np

    assert __import__("torch").cuda.is_available(), "CUDA not available"
    print(f"[m10post] GPU: {__import__('torch').cuda.get_device_name(0)}", flush=True)

    sys.path.insert(0, "/app")
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog("rdApp.*")
    from unimol_validator import UniMolValidator

    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else None

    def morgan_fp(smi):
        m = Chem.MolFromSmiles(smi)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None

    ALLOWED_ATOMS = {"C", "H", "N", "O", "F"}

    def chem_ok(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return False
        if m.GetNumHeavyAtoms() < 8:
            return False
        syms = {a.GetSymbol() for a in m.GetAtoms()}
        if not syms.issubset(ALLOWED_ATOMS):
            return False
        if Chem.GetFormalCharge(m) != 0:
            return False
        if any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()):
            return False
        return True

    def sa_score(smi):
        try:
            from rdkit.Chem import RDConfig
            import os
            sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
            import sascorer
            m = Chem.MolFromSmiles(smi)
            return sascorer.calculateScore(m) if m else 9.0
        except Exception:
            return 9.0  # conservative if sascorer unavailable

    def sc_score(smi):
        try:
            from rdkit.Chem import QED
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return 9.0
            mw = Descriptors.MolWt(m)
            rings = m.GetRingInfo().NumRings()
            return 1.0 + 0.5 * rings + mw / 400.0
        except Exception:
            return 9.0

    print(f"[m10post] Input: {len(smiles_list)} SMILES, {len(ref_smiles)} ref SMILES", flush=True)

    # 1. Canonicalize + dedup
    seen = set(); dedup = []
    for s in smiles_list:
        c = canon(s)
        if c and c not in seen:
            seen.add(c); dedup.append(c)
    print(f"[m10post] After dedup: {len(dedup)}", flush=True)

    # 2. Chem filter
    smis = [s for s in dedup if chem_ok(s)]
    print(f"[m10post] After chem filter: {len(smis)}", flush=True)

    # 3. SA/SC caps
    sa_arr = [sa_score(s) for s in smis]
    smis = [s for s, sa in zip(smis, sa_arr) if sa <= sa_cap]
    print(f"[m10post] After SA cap ({sa_cap}): {len(smis)}", flush=True)

    # 4. Tanimoto novelty window vs ref
    ref_fps = [fp for fp in (morgan_fp(s) for s in ref_smiles[:3000]) if fp]
    max_tans = []
    for s in smis:
        fp = morgan_fp(s)
        if fp is None:
            max_tans.append(0.0); continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        max_tans.append(max(sims) if sims else 0.0)
    max_tans_arr = np.array(max_tans)
    keep = (max_tans_arr >= tanimoto_min) & (max_tans_arr <= tanimoto_max)
    smis = [s for s, ok in zip(smis, keep) if ok]
    max_tans_final = max_tans_arr[keep].tolist()
    print(f"[m10post] After Tanimoto [{tanimoto_min},{tanimoto_max}]: {len(smis)}", flush=True)

    if not smis:
        return {"error": "no candidates survived filtering", "n_survived": 0}

    # 5. UniMol 3D-CNN scoring
    print(f"[m10post] Scoring {len(smis)} with UniMol...", flush=True)
    val = UniMolValidator(model_dir="/smoke_model")
    t0 = time.time()
    preds = val.predict(smis)
    print(f"[m10post] UniMol done in {time.time()-t0:.1f}s", flush=True)

    # 6. Build rows and rank
    rows = []
    for i, s in enumerate(smis):
        d_val  = float(preds["DetoD"][i])   if "DetoD"   in preds else float("nan")
        rho    = float(preds["density"][i])  if "density" in preds else float("nan")
        p_val  = float(preds["DetoP"][i])    if "DetoP"   in preds else float("nan")
        hof    = float(preds["HOF_S"][i])    if "HOF_S"   in preds else float("nan")
        if math.isnan(d_val):
            continue
        dist = (abs(d_val - target_d) / 0.5 +
                abs(rho - target_density) / 0.1 +
                abs(p_val - target_p) / 3.0)
        rows.append({
            "smiles":   s,
            "DetoD":    round(d_val, 4),
            "density":  round(rho, 4),
            "DetoP":    round(p_val, 4),
            "HOF_S":    round(hof, 2) if not math.isnan(hof) else None,
            "maxtan":   round(max_tans_final[i], 4),
            "composite": round(dist, 4),
        })

    rows.sort(key=lambda r: r["composite"])
    top = rows[:top_n]

    # summary
    ds = [r["DetoD"] for r in top]
    summary = {
        "n_input":      len(smiles_list),
        "n_dedup":      len(dedup),
        "n_survived":   len(rows),
        "n_top":        len(top),
        "top1_D_kms":   top[0]["DetoD"] if top else None,
        "top1_smiles":  top[0]["smiles"] if top else None,
        "top1_rho":     top[0]["density"] if top else None,
        "top1_P_GPa":   top[0]["DetoP"] if top else None,
        "topN_max_D":   round(max(ds), 4) if ds else None,
        "topN_mean_D":  round(sum(ds)/len(ds), 4) if ds else None,
    }
    print(f"[m10post] top1 D={summary['top1_D_kms']} km/s  max D={summary['topN_max_D']}", flush=True)
    print("=== DONE ===", flush=True)
    return {"summary": summary, "top_candidates": top}


@app.local_entrypoint()
def main():
    lane_files = sorted(RESULTS.glob("m10_L*.txt"))
    if not lane_files:
        raise FileNotFoundError(
            f"No m10_L*.txt files found in {RESULTS}. "
            "Run modal_m10_200k.py first."
        )

    print(f"[local] Loading {len(lane_files)} lane files...", flush=True)
    all_smiles: list[str] = []
    for f in lane_files:
        lines = [l.strip() for l in f.read_text(encoding="utf-8").split("\n") if l.strip()]
        all_smiles.extend(lines)
        print(f"[local]   {f.name}: {len(lines)} SMILES", flush=True)
    print(f"[local] Total raw: {len(all_smiles):,}", flush=True)

    if not DATA_MASTER.exists():
        raise FileNotFoundError(f"Training master not found: {DATA_MASTER}")
    import pandas as pd
    ref_df = pd.read_csv(DATA_MASTER, usecols=["smiles"], nrows=5000)
    ref_smiles = ref_df["smiles"].dropna().tolist()
    print(f"[local] Ref SMILES: {len(ref_smiles)}", flush=True)

    t0 = time.time()
    result = score_and_rank.remote(all_smiles, ref_smiles)
    elapsed = time.time() - t0
    print(f"[local] Remote returned in {elapsed:.0f}s", flush=True)

    if "error" in result:
        print(f"[local] ERROR: {result['error']}")
        return

    summary = result["summary"]
    top = result["top_candidates"]

    out_json = RESULTS / "m10_post.json"
    out_json.write_text(json.dumps({**summary, "top_candidates": top}, indent=2), encoding="utf-8")
    print(f"[local] JSON -> {out_json}", flush=True)

    md = [
        "# M10 200k-pool postprocessing results",
        "",
        f"Raw: {summary['n_input']:,} | Dedup: {summary['n_dedup']:,} | "
        f"Survived filter: {summary['n_survived']:,} | Top-{summary['n_top']}: {summary['n_top']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| top-1 D (km/s) | {summary['top1_D_kms']} |",
        f"| top-1 rho (g/cm3) | {summary['top1_rho']} |",
        f"| top-1 P (GPa) | {summary['top1_P_GPa']} |",
        f"| top-N max D | {summary['topN_max_D']} |",
        f"| top-N mean D | {summary['topN_mean_D']} |",
        "",
        "## Top-10 by composite score",
        "",
        "| Rank | SMILES | D (km/s) | rho | P (GPa) | maxtan | composite |",
        "|------|--------|----------|-----|---------|--------|-----------|",
    ]
    for i, r in enumerate(top[:10], 1):
        md.append(
            f"| {i} | `{r['smiles']}` | {r['DetoD']} | "
            f"{r['density']} | {r['DetoP']} | {r['maxtan']} | {r['composite']} |"
        )

    out_md = RESULTS / "m10_post.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[local] MD  -> {out_md}", flush=True)

    print("\n[local] === M10 POST RESULTS ===")
    print(f"  top-1 D   : {summary['top1_D_kms']} km/s")
    print(f"  top-1 rho : {summary['top1_rho']} g/cm3")
    print(f"  top-1 P   : {summary['top1_P_GPa']} GPa")
    print(f"  top-N max D: {summary['topN_max_D']} km/s")
    print(f"  n survived : {summary['n_survived']}")
    print(f"  top-1 SMILES: {summary['top1_smiles']}")
