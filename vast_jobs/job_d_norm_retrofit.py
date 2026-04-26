"""Vast.ai job D — latent normalization retrofit on LIMO v1.

Standardizes latents per-dim → retrains denoiser. Tests whether the L5
norm-mismatch finding is the actual bottleneck (independent of MolMIM swap).

Required input files:
  - latents_expanded.pt
  - latents_trustcond.pt
  - denoiser_train.py
  - model.py
  - diffusion_expanded_v4b_norm.yaml
"""
import sys, os, time, shutil, subprocess
import torch
import numpy as np
assert torch.cuda.is_available(), "CUDA not available."
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs")
writer.add_text("phase", "model_download: starting norm retrofit", 0)
writer.flush()

print("[train] Loading latents_trustcond.pt …"); sys.stdout.flush()
blob = torch.load("latents_trustcond.pt", weights_only=False)
z = blob["z_mu"].float()
mu_global  = z.mean(dim=0)
std_global = z.std(dim=0).clamp(min=1e-6)
print(f"[train] Pre-norm: mean ‖z‖ = {z.norm(dim=-1).mean().item():.2f}"); sys.stdout.flush()
z_norm = (z - mu_global) / std_global
print(f"[train] Post-norm: mean ‖z‖ = {z_norm.norm(dim=-1).mean().item():.2f}"); sys.stdout.flush()

blob["z_mu"]    = z_norm
blob["norm_stats"] = {"mu": mu_global, "std": std_global}
torch.save(blob, "latents_trustcond_norm.pt")
print("[train] Saved latents_trustcond_norm.pt"); sys.stdout.flush()
writer.add_text("phase", f"data_loaded: latents normalised (norm 8 → 32)", 0)

# ── Train denoiser on normalized latents ─────────────────────────────────
print("[train] Training denoiser on normalised latents …"); sys.stdout.flush()
writer.add_text("phase", "training_start", 0)
writer.flush()

# Replace the latents path in the config to point at our normalised file.
config = open("diffusion_expanded_v4b_norm.yaml").read()
config = config.replace("latents_trustcond.pt", "latents_trustcond_norm.pt")
with open("config_run.yaml", "w") as f:
    f.write(config)

ret = subprocess.run([sys.executable, "denoiser_train.py",
                       "--config", "config_run.yaml"])
if ret.returncode != 0:
    print(f"[train] denoiser train failed"); sys.exit(ret.returncode)

# Save best
exps = sorted([d for d in os.listdir(".") if d.startswith("diffusion_subset_cond_expanded_v4b_norm")])
if exps:
    best = os.path.join(exps[-1], "checkpoints/best.pt")
    if os.path.exists(best):
        shutil.copy(best, "results/denoiser_v4b_norm_best.pt")
        torch.save({"mu": mu_global, "std": std_global},
                   "results/norm_stats.pt")
        print("[train] Saved denoiser_v4b_norm_best.pt + norm_stats.pt"); sys.stdout.flush()

print("[train] === DONE ==="); sys.stdout.flush()
writer.add_text("phase", "done: norm retrofit complete", 0)
writer.flush()
shutil.copytree("runs", "results/tb_runs", dirs_exist_ok=True)
