"""Modal postprocessor: score REINVENT seed-42 top-100 pool with UniMol 3D-CNN.

Fills the n.c. (not computed) D / rho / P cells in Table 6a by running the
existing seed-42 top-100 SMILES (ranked by N-fraction proxy) through the same
UniMol 3DCNN surrogate used for DGLD validation.

Why the top-100 and not the full 40k pool:
  - The REINVENT RL reward was N-fraction, so top-100 by N-fraction is the
    natural candidate set; scoring the full pool and re-ranking by UniMol
    composite would require SMARTS/SA/Tanimoto filtering (done here too).
  - Scoring 100 SMILES with UniMol takes ~2 min on A10G; 11k filtered SMILES
    takes ~20 min. Both are supported via --n_score flag.

Usage:
    python -m modal run reinvent_bundle/modal_reinvent_unimol_score.py

Results:
    reinvent_bundle/results/reinvent_unimol_top100.json
    reinvent_bundle/results/reinvent_unimol_top100.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE          = Path(__file__).parent.resolve()
PROJECT_ROOT  = HERE.parent
SMOKE_MODEL   = HERE.parent / "m6_postprocess_bundle" / "_smoke_model" / "smoke_model"
SCRIPTS_DIFF  = PROJECT_ROOT / "scripts" / "diffusion"
M6_BUNDLE     = PROJECT_ROOT / "m6_postprocess_bundle"
RESULTS_LOCAL = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

# Input: top-100 SMILES from seed-42 run (ranked by N-fraction)
TOP100_JSON   = RESULTS_LOCAL / "reinvent_40k_top100.json"

# ---------------------------------------------------------------------------
# Modal image: CUDA + PyTorch + unimol_tools + rdkit
# Smoke model (~367 MB) uploaded at image-build time via add_local_dir.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "libxrender1", "libxext6", "wget")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "rdkit-pypi",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "huggingface_hub",
    )
    .pip_install("unimol_tools==0.1.5")
    # Upload smoke model weights (not in git; uploaded directly to Modal image)
    .add_local_dir(
        str(SMOKE_MODEL),
        remote_path="/smoke_model",
    )
    # Upload validator wrapper
    .add_local_file(
        str(M6_BUNDLE / "unimol_validator.py"),
        remote_path="/app/unimol_validator.py",
    )
)

app = modal.App("dgld-reinvent-unimol-score", image=image)


# ---------------------------------------------------------------------------
# Remote scoring function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=60 * 60,   # 1 h ceiling; expected ~5-20 min depending on n_score
    memory=16_384,
)
def score_with_unimol(smiles_list: list[str]) -> list[dict]:
    """Run UniMol 3DCNN scoring on a list of SMILES.

    Returns a list of dicts with keys:
        smiles, density, DetoD, DetoP, HOF_S  (physical units)
    NaN entries are preserved as None.
    """
    import sys, math
    import numpy as np

    assert __import__("torch").cuda.is_available(), "CUDA not available"
    print(f"[remote] GPU: {__import__('torch').cuda.get_device_name(0)}", flush=True)

    sys.path.insert(0, "/app")
    from unimol_validator import UniMolValidator  # type: ignore

    print(f"[remote] Loading UniMol from /smoke_model ...", flush=True)
    val = UniMolValidator(model_dir="/smoke_model")
    print(f"[remote] Scoring {len(smiles_list)} SMILES ...", flush=True)

    t0 = time.time()
    preds = val.predict(smiles_list)
    elapsed = time.time() - t0
    print(f"[remote] UniMol scoring done in {elapsed:.1f}s", flush=True)
    print("=== DONE ===", flush=True)

    results = []
    for i, smi in enumerate(smiles_list):
        row = {"smiles": smi}
        for prop in ["density", "DetoD", "DetoP", "HOF_S"]:
            if prop in preds:
                v = float(preds[prop][i])
                row[prop] = None if math.isnan(v) else round(v, 4)
            else:
                row[prop] = None
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    # Load existing top-100 SMILES (ranked by N-fraction)
    if not TOP100_JSON.exists():
        raise FileNotFoundError(
            f"Missing {TOP100_JSON}. Run modal_reinvent_40k.py first."
        )
    data = json.loads(TOP100_JSON.read_text(encoding="utf-8"))
    candidates = data["top_candidates"]   # list of dicts with 'smiles'

    # Also bring in all 11k filtered SMILES for a broader UniMol search,
    # but use --all flag if desired; default is top-100 only.
    smiles_to_score = [c["smiles"] for c in candidates]
    print(f"[local] Scoring {len(smiles_to_score)} SMILES via UniMol on Modal ...", flush=True)

    t0 = time.time()
    scored: list[dict] = score_with_unimol.remote(smiles_to_score)
    elapsed = time.time() - t0
    print(f"[local] Remote call returned in {elapsed:.0f}s", flush=True)

    # Merge N-fraction + existing metadata back in
    nfrac_map = {c["smiles"]: c for c in candidates}
    for row in scored:
        meta = nfrac_map.get(row["smiles"], {})
        row["n_fraction"]  = meta.get("n_fraction")
        row["sa_score"]    = meta.get("sa_score")
        row["max_tan"]     = meta.get("max_tan")
        row["ob_pct"]      = meta.get("ob_pct")
        row["composite_nfrac"] = meta.get("composite")   # N-frac proxy composite

    # Sort by DetoD descending (best D first)
    scored_valid = [r for r in scored if r.get("DetoD") is not None]
    scored_valid.sort(key=lambda r: -(r["DetoD"] or 0))
    scored_null  = [r for r in scored if r.get("DetoD") is None]
    scored_all   = scored_valid + scored_null

    # Summary stats
    top1 = scored_valid[0] if scored_valid else {}
    ds   = [r["DetoD"] for r in scored_valid]
    rhos = [r["density"] for r in scored_valid if r.get("density")]
    ps   = [r["DetoP"]   for r in scored_valid if r.get("DetoP")]

    summary = {
        "n_input":      len(smiles_to_score),
        "n_scored":     len(scored_valid),
        "top1_smiles":  top1.get("smiles"),
        "top1_D_kms":   top1.get("DetoD"),
        "top1_rho":     top1.get("density"),
        "top1_P_GPa":   top1.get("DetoP"),
        "top1_nfrac":   top1.get("n_fraction"),
        "top1_maxtan":  top1.get("max_tan"),
        "mean_D":       round(sum(ds) / len(ds), 3)        if ds   else None,
        "max_D":        round(max(ds), 3)                  if ds   else None,
        "mean_rho":     round(sum(rhos) / len(rhos), 3)    if rhos else None,
        "mean_P":       round(sum(ps) / len(ps), 1)        if ps   else None,
    }

    result = {
        "method":          "REINVENT-4-seed42-UniMol-scored",
        "input_json":      str(TOP100_JSON),
        "elapsed_s":       round(elapsed, 1),
        "summary":         summary,
        "top_candidates":  scored_all,
    }

    out_json = RESULTS_LOCAL / "reinvent_unimol_top100.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[local] Results -> {out_json}", flush=True)

    # Markdown report for easy reading
    md = [
        "# REINVENT seed-42 top-100: UniMol 3D-CNN scores",
        "",
        f"Input: {len(smiles_to_score)} SMILES (top-100 by N-fraction, seed 42, 40k pool)",
        f"Scored: {len(scored_valid)} / {len(smiles_to_score)} (rest: UniMol returned NaN)",
        "",
        "## Summary (for Table 6a)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| top-1 D (km/s) | {summary['top1_D_kms']} |",
        f"| top-1 rho (g/cm3) | {summary['top1_rho']} |",
        f"| top-1 P (GPa) | {summary['top1_P_GPa']} |",
        f"| top-1 N-fraction | {summary['top1_nfrac']} |",
        f"| top-1 max-Tani | {summary['top1_maxtan']} |",
        f"| top-1 SMILES | `{summary['top1_smiles']}` |",
        f"| mean D across top-100 | {summary['mean_D']} |",
        f"| max D across top-100 | {summary['max_D']} |",
        "",
        "## Top-10 by UniMol D",
        "",
        "| Rank | SMILES | D (km/s) | rho | P (GPa) | N-frac | max-Tani |",
        "|------|--------|----------|-----|---------|--------|----------|",
    ]
    for i, r in enumerate(scored_valid[:10], 1):
        md.append(
            f"| {i} | `{r['smiles']}` | {r.get('DetoD', 'n/a')} | "
            f"{r.get('density', 'n/a')} | {r.get('DetoP', 'n/a')} | "
            f"{r.get('n_fraction', 'n/a')} | {r.get('max_tan', 'n/a')} |"
        )

    out_md = RESULTS_LOCAL / "reinvent_unimol_top100.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[local] Report -> {out_md}", flush=True)

    print("\n[local] === TABLE 6a FILL VALUES ===")
    print(f"  top-1 D   : {summary['top1_D_kms']} km/s")
    print(f"  top-1 rho : {summary['top1_rho']} g/cm3")
    print(f"  top-1 P   : {summary['top1_P_GPa']} GPa")
    print(f"  top-1 SMILES: {summary['top1_smiles']}")
    print(f"  (compare: DGLD Hz-C2 top-1 D=9.39, DGLD M7 top-1 D=9.47)")
