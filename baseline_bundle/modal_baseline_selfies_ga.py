"""Modal wrapper — SELFIES-GA baseline at scale.

Runs baseline_selfies_ga.py on a T4 GPU with:
    pool = 2000 molecules
    30 generations
    top-100 output

The UniMol (3DCNN) predictor uses GPU when available; GA mutations and
composite scoring are CPU-only, so T4 is sufficient and cheaper than A100.

Usage:
    python -m modal run baseline_bundle/modal_baseline_selfies_ga.py

Results land in:
    baseline_bundle/results/selfies_ga_top100.json
    baseline_bundle/results/selfies_ga_top100.csv   (convenience CSV)

Design notes:
- Image is based on the same cuda:12.4.1 base used in modal_smoke.py.
- unimol_tools is installed via pip; the smoke_model weights are uploaded
  as a Modal volume mount from the local directory.
- The GA logic (baseline_selfies_ga.py) and the scoring utilities
  (unimol_validator.py) are mounted from local source so edits do not
  require an image rebuild.
- Corpus is also mounted from local baseline_bundle/corpus.csv.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Local paths (resolved at submission time)
# ---------------------------------------------------------------------------
HERE             = Path(__file__).parent.resolve()            # baseline_bundle/
PROJECT_ROOT     = HERE.parent                                 # EnergeticDiffusion2/
RESULTS_LOCAL    = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

SMOKE_MODEL_LOCAL = (PROJECT_ROOT
                     / "data/raw/energetic_external/EMDP/Data/smoke_model")
SCRIPTS_DIFF_LOCAL = PROJECT_ROOT / "scripts/diffusion"


# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git", "build-essential",
        # X11 rendering libs needed by rdkit >= 2024 in headless containers
        "libxrender1", "libxext6",
    )
    # PyTorch first so unimol_tools links against the correct CUDA ABI
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # unimol_tools manages rdkit, numpy, etc.; selfies and pyyaml are extras
    # huggingface_hub is required by unimol_tools 0.1.x to download backbone weights
    .pip_install(
        "unimol_tools==0.1.5",
        "selfies==2.1.1",
        "huggingface_hub",
    )
    # Mount local Python source files (no image rebuild on edit)
    .add_local_dir(
        str(HERE),
        remote_path="/baseline_bundle",
        # Only mount the Python files we need; skip large assets and results
        ignore=lambda p: (
            p.suffix not in (".py", ".csv", ".json")
            or "results" in str(p)
        ),
    )
    .add_local_dir(
        str(SCRIPTS_DIFF_LOCAL),
        remote_path="/scripts_diff",
        ignore=lambda p: p.suffix != ".py",
    )
    .add_local_dir(
        str(SMOKE_MODEL_LOCAL),
        remote_path="/smoke_model",
        # Include all files: model_0.pth, model_1.pth, config.yaml, etc.
    )
)

app = modal.App("dgld-selfies-ga-baseline", image=image)


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=4 * 60 * 60,    # 4 h; typical run ~40–90 min for 30 gen * 2k pool
    memory=16_384,           # 16 GB RAM for unimol_tools conformer generation
)
def run_selfies_ga_remote(
    n_pool: int = 2000,
    n_gen: int = 30,
    n_top: int = 100,
    elite_frac: float = 0.50,
    new_frac: float = 0.20,
    max_mut: int = 3,
    seed: int = 42,
    with_novelty: bool = False,
) -> dict:
    """Run the SELFIES-GA and return the results dict (same structure as
    the JSON written locally by baseline_selfies_ga.py main())."""
    import sys
    import time

    # Make local mounts importable
    sys.path.insert(0, "/baseline_bundle")
    sys.path.insert(0, "/scripts_diff")

    from pathlib import Path
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import torch

    print(f"[remote] CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[remote] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Import local modules
    from baseline_selfies_ga import (  # type: ignore[import]
        load_corpus_smiles, run_ga,
    )
    from unimol_validator import UniMolValidator  # type: ignore[import]

    corpus_path = Path("/baseline_bundle/corpus.csv")
    model_dir   = Path("/smoke_model")

    print(f"[remote] Loading corpus from {corpus_path} ...", flush=True)
    corpus_smiles = load_corpus_smiles(corpus_path, max_n=50_000)
    print(f"[remote]   {len(corpus_smiles)} valid corpus entries", flush=True)

    print(f"[remote] Loading UniMol validator from {model_dir} ...", flush=True)
    validator = UniMolValidator(model_dir)

    # Optional novelty reference fps
    ref_fps = None
    if with_novelty:
        print("[remote] Building reference fingerprints ...", flush=True)
        import random
        rng = random.Random(seed)
        ref_sample = rng.sample(corpus_smiles, min(5000, len(corpus_smiles)))
        ref_fps = []
        for smi in ref_sample:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                ref_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        print(f"[remote]   {len(ref_fps)} reference fps", flush=True)

    t0 = time.time()
    top_candidates = run_ga(
        corpus_smiles=corpus_smiles,
        validator=validator,
        n_pool=n_pool,
        n_gen=n_gen,
        n_top=n_top,
        elite_frac=elite_frac,
        new_frac=new_frac,
        max_mutations=max_mut,
        seed=seed,
        ref_fps=ref_fps,
    )
    elapsed = time.time() - t0

    payload = {
        "method":         "SELFIES-GA",
        "n_pool":         n_pool,
        "n_gen":          n_gen,
        "n_top":          len(top_candidates),
        "seed":           seed,
        "elapsed_s":      round(elapsed, 1),
        "top_candidates": top_candidates,
        "summary": {
            "top1_composite":      top_candidates[0]["composite"] if top_candidates else None,
            "top1_D_kms":          top_candidates[0]["d"]         if top_candidates else None,
            "top1_P_GPa":          top_candidates[0]["p"]         if top_candidates else None,
            "top1_rho":            top_candidates[0]["rho"]        if top_candidates else None,
            "topN_mean_composite": float(np.mean([r["composite"] for r in top_candidates]))
                                   if top_candidates else None,
            "topN_mean_D":         float(np.mean([r["d"] for r in top_candidates
                                                   if r["d"] is not None]))
                                   if top_candidates else None,
            "topN_max_D":          float(max(r["d"] for r in top_candidates
                                              if r["d"] is not None))
                                   if top_candidates else None,
        },
    }
    print(f"\n[remote] Done. {len(top_candidates)} candidates in {elapsed:.0f}s",
          flush=True)
    if top_candidates:
        r0 = top_candidates[0]
        print(f"  #1: comp={r0['composite']:.4f}  rho={r0['rho']:.3f}  "
              f"D={r0['d']:.2f}  P={r0['p']:.2f}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    import csv

    print("[local] Submitting SELFIES-GA to Modal T4 ...", flush=True)
    t0 = time.time()

    results = run_selfies_ga_remote.remote(
        n_pool=2000,
        n_gen=30,
        n_top=100,
        elite_frac=0.50,
        new_frac=0.20,
        max_mut=3,
        seed=42,
        with_novelty=False,    # flip to True to enable Tanimoto novelty scoring
    )

    elapsed_local = time.time() - t0
    print(f"[local] Remote call returned in {elapsed_local:.0f}s total", flush=True)

    # Save JSON
    out_json = RESULTS_LOCAL / "selfies_ga_top100.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[local] JSON -> {out_json}", flush=True)

    # Save CSV for easy inspection
    candidates = results.get("top_candidates", [])
    out_csv = RESULTS_LOCAL / "selfies_ga_top100.csv"
    if candidates:
        fieldnames = ["rank", "composite", "rho", "hof", "d", "p", "maxtan", "smiles"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for i, row in enumerate(candidates, 1):
                writer.writerow({"rank": i, **row})
        print(f"[local] CSV  -> {out_csv}", flush=True)

    # Print summary
    s = results.get("summary", {})
    print(f"\n[local] Summary:")
    print(f"  top1 composite : {s.get('top1_composite')}")
    print(f"  top1 D (km/s)  : {s.get('top1_D_kms')}")
    print(f"  top1 P (GPa)   : {s.get('top1_P_GPa')}")
    print(f"  top1 rho       : {s.get('top1_rho')}")
    print(f"  topN mean D    : {s.get('topN_mean_D')}")
    print(f"  topN max D     : {s.get('topN_max_D')}")
    print(f"  elapsed (remote): {results.get('elapsed_s')}s")

    if s.get("top1_composite") is None:
        raise SystemExit("[local] FAILED: no valid candidates returned")
    print("[local] PASSED")
