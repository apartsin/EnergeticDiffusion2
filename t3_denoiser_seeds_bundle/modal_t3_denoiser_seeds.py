"""T3: multi-seed denoiser retraining (production v4b architecture).

Closes Reviewer 1's "seed variance on the diffusion model itself is not
reported" finding by adding two more seeds (1 and 2) of the production
v4b architecture; seed=42 is the existing checkpoint already in the paper.

Pipeline per seed:
    1. Mount latents_trustcond.pt (1.6 GB) from a Modal Volume.
    2. Mount scripts/diffusion source from the image layer.
    3. Write a per-seed YAML config (v4b recipe + seed override).
    4. Run scripts/diffusion/train.py --config <yaml>.
    5. Copy checkpoints/best.pt out to /results/denoiser_v4b_seed{N}.pt.

The first invocation uploads the latents file once; subsequent runs
short-circuit on `modal.Volume.listdir`.

Outputs (after `modal volume get` step shown in the local-entrypoint footer):
    t3_denoiser_seeds_bundle/results/denoiser_v4b_seed1.pt
    t3_denoiser_seeds_bundle/results/denoiser_v4b_seed2.pt

Usage:
    python -m modal run --detach t3_denoiser_seeds_bundle/modal_t3_denoiser_seeds.py
    python -m modal run t3_denoiser_seeds_bundle/modal_t3_denoiser_seeds.py::main --seeds 1
    modal volume get dgld-t3-results / ./t3_denoiser_seeds_bundle/results/

The training-time budget is the v4b YAML wall-clock cap (`total_time_minutes`,
default 90); we can extend per the EXPERIMENTATION_PLAN ~6 hr by overriding
`--total-time-minutes` in the local entrypoint (default 360 here so
each seed gets the same compute budget the plan allocated).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
PROJ = HERE.parent
RESULTS_LOCAL = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

# Source files we need to embed in the image
SCRIPTS_DIFF = PROJ / "scripts" / "diffusion"
CONFIG_V4B   = PROJ / "configs" / "diffusion_expanded_v4b.yaml"

# Latents (large; goes in a Volume, NOT image)
LATENTS_LOCAL = PROJ / "data" / "training" / "diffusion" / "latents_trustcond.pt"
LATENTS_NAME  = "latents_trustcond.pt"

# Image layer files
_SRC_FILES = [
    (SCRIPTS_DIFF / "train.py",    "/app/scripts/diffusion/train.py"),
    (SCRIPTS_DIFF / "model.py",    "/app/scripts/diffusion/model.py"),
    (CONFIG_V4B,                   "/app/configs/diffusion_expanded_v4b.yaml"),
]

# ---------------------------------------------------------------------------
# Modal image: PyTorch 2.4 + CUDA 12.4 + numpy + pyyaml
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"
    )
    .pip_install(
        "numpy<2",
        "pyyaml>=6.0",
    )
)
for local_p, remote_p in _SRC_FILES:
    image = image.add_local_file(str(local_p), remote_p)

app = modal.App("dgld-t3-denoiser-seeds", image=image)

# Volumes
data_vol    = modal.Volume.from_name("dgld-t3-data",    create_if_missing=True)
results_vol = modal.Volume.from_name("dgld-t3-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Remote function: train one seed
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100",
    timeout=8 * 60 * 60,   # 8h hard cap
    volumes={
        "/data":    data_vol,
        "/results": results_vol,
    },
)
def train_seed_remote(seed: int, total_time_minutes: int = 360) -> dict:
    """Train v4b denoiser at the requested seed; copy best.pt to /results."""
    import os
    import shutil
    import subprocess
    import sys

    import torch
    import yaml

    assert torch.cuda.is_available(), "No CUDA GPU available"
    print(f"[t3:seed={seed}] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- 1. Build per-seed config ------------------------------------------
    cfg_in = "/app/configs/diffusion_expanded_v4b.yaml"
    with open(cfg_in) as f:
        cfg = yaml.safe_load(f)

    # Override paths to live inside the container's /workspace
    cfg["run"]["seed"] = int(seed)
    cfg["run"]["name"] = f"diffusion_subset_cond_expanded_v4b_seed{seed}"
    cfg["run"]["notes"] = (
        f"T3 multi-seed denoiser retraining: seed={seed}, total_time={total_time_minutes}min"
    )
    cfg["paths"]["base"]            = "/workspace"
    cfg["paths"]["experiments_dir"] = "experiments"
    cfg["paths"]["latents_pt"]      = "data/latents_trustcond.pt"
    cfg["training"]["total_time_minutes"] = int(total_time_minutes)

    workspace = Path("/workspace")
    (workspace / "configs").mkdir(parents=True, exist_ok=True)
    (workspace / "experiments").mkdir(parents=True, exist_ok=True)
    (workspace / "data").mkdir(parents=True, exist_ok=True)

    # Symlink latents from /data Volume into /workspace/data/
    src_latents = Path("/data") / LATENTS_NAME
    dst_latents = workspace / "data" / "latents_trustcond.pt"
    if not src_latents.exists():
        raise FileNotFoundError(f"Latents file not found in volume: {src_latents}")
    if not dst_latents.exists():
        os.symlink(src_latents, dst_latents)

    cfg_path = workspace / "configs" / f"diffusion_v4b_seed{seed}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[t3:seed={seed}] config -> {cfg_path}", flush=True)

    # ---- 2. Run the trainer ------------------------------------------------
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app:" + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "/app/scripts/diffusion/train.py",
        "--config", str(cfg_path),
    ]
    print(f"[t3:seed={seed}] launching: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, check=False)
    elapsed = time.time() - t0
    print(f"[t3:seed={seed}] training rc={proc.returncode} elapsed={elapsed:.0f}s",
          flush=True)

    # ---- 3. Locate experiment dir + copy best.pt --------------------------
    exp_root = workspace / "experiments"
    candidate_dirs = sorted(
        [d for d in exp_root.iterdir()
         if d.is_dir() and d.name.startswith(f"diffusion_subset_cond_expanded_v4b_seed{seed}_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not candidate_dirs:
        return {
            "seed": seed,
            "rc":   proc.returncode,
            "elapsed_s": round(elapsed, 1),
            "error": "no experiment dir created",
        }
    exp_dir = candidate_dirs[-1]
    best = exp_dir / "checkpoints" / "best.pt"
    if not best.exists():
        return {
            "seed": seed, "rc": proc.returncode, "elapsed_s": round(elapsed, 1),
            "error": f"best.pt missing under {exp_dir}",
        }

    out_path = Path("/results") / f"denoiser_v4b_seed{seed}.pt"
    shutil.copy2(best, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[t3:seed={seed}] best.pt copied -> {out_path}  ({size_mb:.1f} MB)",
          flush=True)

    # also save the train.jsonl log for audit
    log_src = exp_dir / "train.jsonl"
    if log_src.exists():
        shutil.copy2(log_src, Path("/results") / f"train_v4b_seed{seed}.jsonl")

    results_vol.commit()
    print(f"[t3:seed={seed}] Volume committed.", flush=True)

    return {
        "seed":      seed,
        "rc":        proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "exp_dir":   str(exp_dir),
        "size_mb":   round(size_mb, 1),
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    seeds: str = "1,2",
    total_time_minutes: int = 360,
    skip_upload: bool = False,
):
    """Run the v4b training at each comma-separated seed.

    Args:
        seeds: comma-separated seed list, e.g. "1,2" (default).
        total_time_minutes: per-seed wall-clock budget (default 360 = 6 hr).
        skip_upload: skip latents upload check (assumes it's already in volume).
    """
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    print(f"[t3] seeds = {seed_list}  total_time_minutes = {total_time_minutes}",
          flush=True)

    # ---- 1. Ensure latents file is uploaded to the data volume -----------
    if not skip_upload:
        existing: set[str] = set()
        try:
            for entry in data_vol.listdir("/"):
                existing.add(Path(entry.path).name)
        except Exception as exc:
            print(f"[t3] could not list data_vol ({exc}); will upload.", flush=True)
        if LATENTS_NAME in existing:
            print(f"[t3] latents already in volume; skipping upload.", flush=True)
        else:
            if not LATENTS_LOCAL.exists():
                raise FileNotFoundError(
                    f"Local latents file missing: {LATENTS_LOCAL}"
                )
            size_gb = LATENTS_LOCAL.stat().st_size / 1024 / 1024 / 1024
            print(f"[t3] uploading latents ({size_gb:.2f} GB) ...", flush=True)
            with data_vol.batch_upload() as batch:
                batch.put_file(str(LATENTS_LOCAL), LATENTS_NAME)
            print(f"[t3] upload complete.", flush=True)

    # ---- 2. Dispatch each seed sequentially ------------------------------
    summaries = []
    for s in seed_list:
        print(f"\n[t3] dispatching seed={s}", flush=True)
        try:
            r = train_seed_remote.remote(seed=s, total_time_minutes=total_time_minutes)
        except Exception as exc:
            r = {"seed": s, "error": str(exc)}
        summaries.append(r)
        out = RESULTS_LOCAL / f"t3_seed{s}_summary.json"
        out.write_text(json.dumps(r, indent=2))
        print(f"[t3] seed={s} summary -> {out}", flush=True)

    full = RESULTS_LOCAL / "t3_summary.json"
    full.write_text(json.dumps(summaries, indent=2))
    print(f"\n[t3] full summary -> {full}", flush=True)

    print(
        "\n[t3] download checkpoints with:\n"
        "  modal volume get dgld-t3-results / ./t3_denoiser_seeds_bundle/results/",
        flush=True,
    )
