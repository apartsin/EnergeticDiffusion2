"""Modal wrapper for M7 100k-pool diverse-lane sampling.

First run uploads model weights (~1.8 GB total) to a named Modal Volume.
Subsequent runs reuse the volume and skip the upload automatically.

Usage:
    # Run (uploads models on first call, skips on subsequent):
    python -m modal run m7_bundle/modal_m7_100k.py

    # Skip model upload explicitly (if you know they are already in the volume):
    python -m modal run m7_bundle/modal_m7_100k.py::main --skip-upload

    # Download results after completion:
    modal volume get dgld-m7-results m7 ./m7_bundle/results/m7_modal/

    # Detached launch (keeps job alive after local client disconnects):
    python -m modal run --detach m7_bundle/modal_m7_100k.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE  = Path(__file__).parent.resolve()
PROJ  = HERE.parent
COMBO = PROJ / "combo_bundle"
M1    = PROJ / "m1_bundle"

# ---------------------------------------------------------------------------
# Source files embedded into the image (small; rebuilt when content changes)
# ---------------------------------------------------------------------------
_SRC_FILES = [
    (COMBO / "model.py",                  "/app/model.py"),
    (COMBO / "limo_model.py",             "/app/limo_model.py"),
    (COMBO / "guided_v2_sampler.py",      "/app/guided_v2_sampler.py"),
    (COMBO / "train_multihead_latent.py", "/app/train_multihead_latent.py"),
    (COMBO / "vocab.json",                "/app/vocab.json"),
    (COMBO / "meta.json",                 "/app/meta.json"),
    (HERE  / "m7_100k.py",               "/app/m7_100k.py"),
]

# ---------------------------------------------------------------------------
# Model checkpoints stored in a named Volume (too large for image layers)
# ---------------------------------------------------------------------------
_CKPTS = [
    (COMBO / "limo_best.pt",       "limo_best.pt"),
    (COMBO / "v4b_best.pt",        "v4b_best.pt"),
    (COMBO / "v3_best.pt",         "v3_best.pt"),
    (M1    / "score_model_v3f.pt", "score_model_v3f.pt"),
]


# ---------------------------------------------------------------------------
# Image: PyTorch 2.4 + CUDA 12.4 + rdkit + selfies
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"
    )
    .pip_install(
        "rdkit==2024.3.5",
        "selfies>=2.1.1",
        "numpy",
        "pandas",
    )
)
for local_p, remote_p in _SRC_FILES:
    image = image.add_local_file(str(local_p), remote_p)

app = modal.App("dgld-m7-100k", image=image)

model_vol   = modal.Volume.from_name("dgld-m7-models",  create_if_missing=True)
results_vol = modal.Volume.from_name("dgld-m7-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100",
    timeout=2 * 60 * 60,
    volumes={
        "/models":  model_vol,
        "/results": results_vol,
    },
)
def run_m7_remote(pool_per_run: int = 10_000) -> dict:
    import subprocess
    import sys
    from pathlib import Path

    import torch
    assert torch.cuda.is_available(), "No CUDA GPU available"
    print(f"[remote] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    out_dir = Path("/results/m7")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "/app/m7_100k.py",
        "--v4b_ckpt",     "/models/v4b_best.pt",
        "--v3_ckpt",      "/models/v3_best.pt",
        "--limo_ckpt",    "/models/limo_best.pt",
        "--score_model",  "/models/score_model_v3f.pt",
        "--meta_json",    "/app/meta.json",
        "--vocab_json",   "/app/vocab.json",
        "--pool_per_run", str(pool_per_run),
        "--results_dir",  "/results/m7",
    ]
    print(f"[remote] {' '.join(cmd)}", flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    print(f"[remote] Finished in {elapsed:.0f}s rc={proc.returncode}", flush=True)
    results_vol.commit()
    print("[remote] Volume committed.", flush=True)
    print("=== DONE ===", flush=True)

    summary_path = Path("/results/m7/m7_summary.json")
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {"rc": proc.returncode, "elapsed_s": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(skip_upload: bool = False, pool_per_run: int = 10_000):
    # --- Model upload (idempotent: check all checkpoints already present) ---
    if not skip_upload:
        existing_names: set[str] = set()
        try:
            for entry in model_vol.listdir("/"):
                existing_names.add(Path(entry.path).name)
        except Exception as exc:
            print(f"[local] Could not list volume ({exc}); will upload all.",
                  flush=True)

        missing = [(lp, rp) for lp, rp in _CKPTS if rp not in existing_names]
        if not missing:
            print("[local] All model checkpoints already in volume. Skipping upload.",
                  flush=True)
        else:
            print(f"[local] Uploading {len(missing)} checkpoint(s) (~1.8 GB total) ...",
                  flush=True)
            with model_vol.batch_upload() as batch:
                for local_path, remote_name in missing:
                    size_mb = local_path.stat().st_size / 1024 / 1024
                    print(f"  {local_path.name}  ({size_mb:.0f} MB)", flush=True)
                    batch.put_file(str(local_path), remote_name)
            print("[local] Upload complete.", flush=True)

    # --- Launch remote job ---
    print(f"[local] Launching M7 100k on Modal A100 (pool_per_run={pool_per_run}) ...",
          flush=True)
    t0 = time.time()
    summary = run_m7_remote.remote(pool_per_run=pool_per_run)
    elapsed = time.time() - t0

    print(f"[local] Job finished in {elapsed:.0f}s", flush=True)
    print(f"[local] Summary:\n{json.dumps(summary, indent=2)}", flush=True)
    print(
        "\n[local] Download results with:\n"
        "  modal volume get dgld-m7-results m7 ./m7_bundle/results/m7_modal/",
        flush=True,
    )
