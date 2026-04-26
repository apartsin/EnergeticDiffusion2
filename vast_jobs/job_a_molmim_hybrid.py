"""Vast.ai job A — MolMIM hybrid: encode 382k + retrain denoiser at 512-d.

Phase 1: Install NeMo (or assume it's in the image), load molmim_70m_24_3.nemo,
encode all SMILES → latents_molmim.pt (382k × 512).

Phase 2: Build expanded conditioning blob from the slim bundle. Train a
denoiser at 512-d (smaller than our 1024-d v4-B since latent dim halves).

Outputs to ./results/:
  - latents_molmim.pt
  - denoiser_v9_molmim_best.pt
  - train.log
  - tb_runs/

Required input files (uploaded via R2):
  - molmim_70m_24_3.nemo
  - smiles_cond_bundle.pt
  - denoiser_train.py (our scripts/diffusion/train.py)
  - model.py (our scripts/diffusion/model.py)
  - diffusion_expanded_v9_512d.yaml
"""
import sys, os, time, json, shutil, subprocess
import torch
assert torch.cuda.is_available(), "CUDA not available."
device = torch.device("cuda")
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs")
writer.add_text("phase", "model_download: starting MolMIM hybrid job", 0)
writer.flush()

# ── Phase 0: install NeMo if needed ─────────────────────────────────────
print("[train] Checking NeMo …"); sys.stdout.flush()
try:
    import nemo
    print(f"[train] NeMo {nemo.__version__} present"); sys.stdout.flush()
except ImportError:
    print("[train] Installing NeMo (this takes ~5 min) …"); sys.stdout.flush()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "nemo-toolkit[nlp]==1.23.0"], check=True)
    import nemo

writer.add_text("phase", "nemo_ready", 0)

# ── Phase 1: encode SMILES with MolMIM ──────────────────────────────────
print("[train] Loading MolMIM .nemo …"); sys.stdout.flush()
from nemo.collections.nlp.models.language_modeling.megatron.molmim_model import MolMIMModel
import torch
model = MolMIMModel.restore_from("molmim_70m_24_3.nemo", map_location=device)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"[train] Model loaded: MolMIM ({n_params:,} params) on {device}"); sys.stdout.flush()
writer.add_text("phase", f"model_loaded: MolMIM 70M on {device}", 0)

bundle = torch.load("smiles_cond_bundle.pt", weights_only=False)
smiles = bundle["smiles"]
N = len(smiles)
print(f"[train] Loaded {N:,} SMILES"); sys.stdout.flush()
writer.add_text("phase", f"data_loaded: {N} SMILES tokenised", 0)

# Batched MolMIM encode → 512-d latents
print(f"[train] Encoding ({N} SMILES, batch=256) …"); sys.stdout.flush()
writer.add_text("phase", f"encode_start: {N} samples", 0)
batch_size = 256
latents = torch.zeros(N, 512, dtype=torch.float32)
t_start = time.time()
total_steps = (N + batch_size - 1) // batch_size
for step, i in enumerate(range(0, N, batch_size), start=1):
    chunk = smiles[i:i+batch_size]
    with torch.no_grad():
        z = model.encode(chunk)             # (B, 512) — actual API may differ slightly per NeMo version
    latents[i:i+len(chunk)] = z.cpu()
    if step % 50 == 0 or step == total_steps:
        elapsed = time.time() - t_start
        rate = (i + len(chunk)) / max(elapsed, 1)
        eta = (N - i - len(chunk)) / max(rate, 1)
        print(f"  {step}/{total_steps} loss=N/A epoch=encode "
              f"rate={rate:.0f}/s  eta={eta/60:.1f}m"); sys.stdout.flush()
        writer.add_scalar("encode/rate", rate, step)
torch.save({
    "z_mu":           latents,
    "smiles":         smiles,
    "values_raw":     bundle["values_raw"],
    "values_norm":    bundle["values_norm"],
    "cond_valid":     bundle["cond_valid"],
    "cond_weight":    bundle["cond_weight"],
    "tiers":          bundle["tiers"],
    "stats":          bundle["stats"],
    "property_names": bundle["property_names"],
    "meta":           {"encoder": "MolMIM 70M v1.3",
                          "latent_dim": 512,
                          "encoded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
}, "results/latents_molmim.pt")
print(f"[train] Saved latents_molmim.pt ({latents.shape})"); sys.stdout.flush()
writer.add_text("phase", f"encode_complete: {N} samples", 0)

# ── Phase 2: retrain denoiser at 512-d ──────────────────────────────────
print("[train] Phase 2: train denoiser at 512-d …"); sys.stdout.flush()
writer.add_text("phase", "denoiser_train_start: 512-d", 0)
# Dispatch to existing trainer (uploaded as denoiser_train.py)
ret = subprocess.run([sys.executable, "denoiser_train.py",
                       "--config", "diffusion_expanded_v9_512d.yaml"],
                      env={**os.environ, "LATENTS_OVERRIDE": "results/latents_molmim.pt"})
if ret.returncode != 0:
    print(f"[train] denoiser train failed (exit {ret.returncode})"); sys.stdout.flush()
    sys.exit(ret.returncode)

# Move best checkpoint to results
exp_dirs = sorted([d for d in os.listdir(".")
                   if d.startswith("diffusion_subset_cond_expanded_v9_")])
if exp_dirs:
    best = os.path.join(exp_dirs[-1], "checkpoints/best.pt")
    if os.path.exists(best):
        shutil.copy(best, "results/denoiser_v9_molmim_best.pt")
        print(f"[train] Saved denoiser_v9_molmim_best.pt"); sys.stdout.flush()

writer.add_text("phase", "done: MolMIM hybrid complete", 0)
writer.flush()
shutil.copytree("runs", "results/tb_runs", dirs_exist_ok=True)
print("[train] === DONE ==="); sys.stdout.flush()
