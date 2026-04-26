"""Vast.ai job C — massive joint v3 + v4-B rerank (pool=20 000).

Generates 20k samples from each denoiser, dedupes, validates, applies
chem_filter + SA/SC + Tanimoto window, ranks, returns top-200 leads.

Required input files:
  - denoiser_v3_best.pt      (700 MB)
  - denoiser_v4b_best.pt     (700 MB)
  - latents_expanded.pt      (1.6 GB) — for stats only; could slim
  - limo_v1_best.pt          (140 MB)
  - smoke_model_dir.tar      (3DCNN ensemble, ~700 MB)
  - rerank_code.tar.gz       (our rerank scripts + LIMO model + chem_filter)
"""
import sys, os, time, shutil, subprocess
import torch
assert torch.cuda.is_available(), "CUDA not available."
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs")
writer.add_text("phase", "model_download: starting massive rerank", 0)
writer.flush()

print("[train] Installing rdkit-pypi unimol_tools …"); sys.stdout.flush()
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "rdkit-pypi", "unimol_tools"], check=True)

print("[train] Extracting rerank code + smoke model …"); sys.stdout.flush()
subprocess.run(["tar", "-xzf", "rerank_code.tar.gz"], check=True)
subprocess.run(["tar", "-xf", "smoke_model_dir.tar"], check=True)

# Symlink ckpts into expected paths
os.makedirs("experiments/v4b/checkpoints", exist_ok=True)
os.makedirs("experiments/v3/checkpoints", exist_ok=True)
shutil.copy("denoiser_v4b_best.pt", "experiments/v4b/checkpoints/best.pt")
shutil.copy("denoiser_v3_best.pt",  "experiments/v3/checkpoints/best.pt")

writer.add_text("phase", "data_loaded: ckpts + code in place", 0)

print("[train] Running massive joint rerank (pool=20000) …"); sys.stdout.flush()
writer.add_text("phase", "training_start: pool=20000", 0)
writer.flush()
ret = subprocess.run([sys.executable, "scripts/diffusion/joint_rerank.py",
                       "--exp_v4b", "experiments/v4b",
                       "--exp_v3",  "experiments/v3",
                       "--cfg", "7", "--n_pool_each", "20000", "--n_keep", "200",
                       "--target_density", "1.95", "--target_d", "9.5",
                       "--target_p", "40", "--target_hof", "220",
                       "--hard_sa", "5.0", "--hard_sc", "3.5",
                       "--tanimoto_min", "0.20", "--tanimoto_max", "0.55",
                       "--require_neutral", "--with_chem_filter",
                       "--with_feasibility",
                       "--out", "results/joint_rerank_massive.md"])
if ret.returncode != 0:
    print(f"[train] joint_rerank failed (exit {ret.returncode})"); sys.exit(ret.returncode)

print(f"[train] === DONE ==="); sys.stdout.flush()
writer.add_text("phase", "done: massive rerank complete", 0)
writer.flush()
shutil.copytree("runs", "results/tb_runs", dirs_exist_ok=True)
