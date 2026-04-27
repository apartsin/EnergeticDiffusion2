"""
Train subset-conditional denoiser on cached LIMO latents.

Same requirements as the VAE trainer:
  - Unattended (wall-clock budget)
  - Resumable from checkpoint
  - Versioned experiment directory
  - YAML config + external overrides
  - Rich streaming JSONL + text log
  - fp16 mixed precision
  - EMA-tracked weights for stable sampling
  - NaN guard

Usage:
    python scripts/diffusion/train.py --config configs/diffusion.yaml
    python scripts/diffusion/train.py --config configs/diffusion.yaml --smoke
    python scripts/diffusion/train.py --config configs/diffusion.yaml --resume <exp>
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from model import (
    ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample,
)


# ── runtime state ────────────────────────────────────────────────────────────
@dataclass
class RuntimeState:
    global_step:       int = 0
    epoch:             int = 0
    best_val:          float = float("inf")
    t_start:           float = 0.0
    nan_streak:        int = 0
    no_improve_evals:  int = 0


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── dataset ──────────────────────────────────────────────────────────────────
class LatentDataset(Dataset):
    """Wraps the encode_latents.py output. Returns (z, values_norm, cond_valid,
    cond_weight) per row. Subset sampling is done in the training loop, NOT
    here, so the mask is recomputed every epoch."""
    def __init__(self, blob: dict):
        self.z          = blob["z_mu"].float()            # (N, 1024)
        self.values     = blob["values_norm"].float()     # (N, 4) — NaN where missing
        self.cond_valid = blob["cond_valid"].bool()       # (N, 4) — A/B and present
        if "cond_weight" in blob:
            self.cond_weight = blob["cond_weight"].float()  # (N, 4) tier-aware
        else:
            self.cond_weight = self.cond_valid.float()
        # sanitize NaNs -> 0 (they will always be masked anyway)
        self.values_san = torch.where(torch.isnan(self.values),
                                       torch.zeros_like(self.values),
                                       self.values)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        return self.z[i], self.values_san[i], self.cond_valid[i]


def sample_training_mask(cond_valid: torch.Tensor, cfg: dict,
                          generator: torch.Generator,
                          cond_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Per-example random subset. Only properties with cond_valid=True are
    eligible. Returns (B, n_props) float {0,1}.

    Distribution:  size 0 / 1 / 2 / 3 / 4 with probabilities from config.

    If cond_weight is provided AND cfg.training.weighted_mask is true,
    properties with higher cond_weight are preferred when picking the subset
    (Tier-A/B beats Tier-D).

    If cfg.training.property_dropout > 0, each chosen entry is then independently
    zeroed with that probability. Helps the model handle arbitrary-subset
    conditioning at sample time.
    """
    B, P = cond_valid.shape
    probs = cfg["training"]["subset_size_probs"]
    sizes = torch.tensor(list(probs.keys()), dtype=torch.long)
    size_weights = torch.tensor(list(probs.values()), dtype=torch.float)
    desired = sizes[torch.multinomial(size_weights, B, replacement=True,
                                        generator=generator)]
    weighted = bool(cfg["training"].get("weighted_mask", False)) and cond_weight is not None
    mask = torch.zeros_like(cond_valid, dtype=torch.float)
    for i in range(B):
        elig = torch.where(cond_valid[i])[0]
        k = min(desired[i].item(), len(elig))
        if k <= 0:
            continue
        if weighted:
            w = cond_weight[i, elig].clamp(min=1e-6)
            chosen = elig[torch.multinomial(w, k, replacement=False, generator=generator)]
        else:
            chosen = elig[torch.randperm(len(elig), generator=generator)[:k]]
        mask[i, chosen] = 1.0
    pdrop = float(cfg["training"].get("property_dropout", 0.0))
    if pdrop > 0:
        keep = (torch.rand(mask.shape, generator=generator) >= pdrop).float()
        mask = mask * keep
    return mask


# ── logger ───────────────────────────────────────────────────────────────────
class RunLogger:
    def __init__(self, exp_dir: Path, jsonl_name: str, text_name: str):
        self.jsonl = open(exp_dir / jsonl_name, "a", buffering=1, encoding="utf-8")
        self.text  = open(exp_dir / text_name,  "a", buffering=1, encoding="utf-8")
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def info(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] INFO  {msg}"
        print(line)
        self.text.write(line + "\n")

    def warn(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] WARN  {msg}"
        print(line)
        self.text.write(line + "\n")

    def event(self, kind: str, **fields):
        rec = {"t": time.time(), "kind": kind, **fields}
        self.jsonl.write(json.dumps(rec) + "\n")

    def close(self):
        self.jsonl.close()
        self.text.close()


# ── checkpoint ───────────────────────────────────────────────────────────────
def save_checkpoint(path: Path, model, optim, scaler, scheduler, ema,
                     state: RuntimeState, cfg: dict, extra: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model_state":     model.state_dict(),
        "ema_state":       ema.state_dict() if ema else None,
        "optim_state":     optim.state_dict(),
        "scaler_state":    scaler.state_dict() if scaler else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "runtime": {
            "global_step":      state.global_step,
            "epoch":            state.epoch,
            "best_val":         state.best_val,
            "t_start":          state.t_start,
            "no_improve_evals": state.no_improve_evals,
        },
        "config":   cfg,
        "extra":    extra or {},
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)


def rotate_checkpoints(ckpt_dir: Path, pattern: str, keep: int):
    files = sorted(ckpt_dir.glob(pattern))
    for f in files[:-keep]:
        try: f.unlink()
        except OSError: pass


def try_resume(exp_dir: Path, model, optim, scaler, scheduler, ema,
               state: RuntimeState) -> bool:
    ckpt = exp_dir / "checkpoints/last.pt"
    if not ckpt.exists():
        return False
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state"])
    optim.load_state_dict(blob["optim_state"])
    if scaler and blob.get("scaler_state"):       scaler.load_state_dict(blob["scaler_state"])
    if scheduler and blob.get("scheduler_state"): scheduler.load_state_dict(blob["scheduler_state"])
    if ema and blob.get("ema_state"):             ema.load_state_dict(blob["ema_state"])
    rt = blob["runtime"]
    state.global_step = rt["global_step"]; state.epoch = rt["epoch"]
    state.best_val    = rt["best_val"];    state.t_start = rt.get("t_start", time.time())
    state.no_improve_evals = rt.get("no_improve_evals", 0)
    return True


def setup_experiment_dir(base: Path, cfg: dict, resume: str | None) -> Path:
    if resume:
        p = Path(resume)
        if not p.is_absolute(): p = base / p
        if not p.exists(): raise FileNotFoundError(p)
        return p
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"{cfg['run']['name']}_{ts}"
    exp = base / cfg["paths"]["experiments_dir"] / name
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "checkpoints").mkdir(exist_ok=True)
    with open(exp / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return exp


# ── training loop ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke",  action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = Path(cfg["paths"]["base"])

    if args.smoke:
        cfg["training"]["total_time_minutes"] = 5
        cfg["training"]["epochs"] = 1

    seed = cfg["run"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    exp_dir = setup_experiment_dir(base, cfg, args.resume)
    log = RunLogger(exp_dir, cfg["log"]["jsonl_filename"], cfg["log"]["text_filename"])
    log.info(f"Experiment dir: {exp_dir}")
    log.info(f"Device: {args.device}")
    log.info(f"Config: {args.config}")

    # ── data ──────────────────────────────────────────────────────────────────
    latents_path = base / cfg["paths"]["latents_pt"]
    log.info(f"Loading latents: {latents_path}")
    blob = torch.load(latents_path, weights_only=False)
    ds = LatentDataset(blob)
    N = len(ds)
    log.info(f"  N={N:,}  latent_dim={ds.z.shape[1]}  n_props={ds.values.shape[1]}")
    log.info(f"  conditioning inventory:")
    for j, p in enumerate(blob["property_names"]):
        log.info(f"    {p:28s} {int(ds.cond_valid[:, j].sum()):>7,} valid")

    # split
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    n_val = int(cfg["data"]["val_frac"] * N)
    val_idx = order[:n_val]; tr_idx = order[n_val:]
    log.info(f"  split: train={len(tr_idx):,}  val={len(val_idx):,}")

    # smoke: subsample
    if args.smoke:
        tr_idx = tr_idx[:5000]
        val_idx = val_idx[:500]
        log.warn(f"SMOKE: train={len(tr_idx)}  val={len(val_idx)}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = ConditionalDenoiser(
        latent_dim   = ds.z.shape[1],
        hidden       = cfg["model"]["hidden"],
        n_blocks     = cfg["model"]["n_blocks"],
        time_dim     = cfg["model"]["time_dim"],
        prop_emb_dim = cfg["model"]["prop_emb_dim"],
        n_props      = ds.values.shape[1],
        dropout      = cfg["model"].get("dropout", 0.0),
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {n_params/1e6:.2f}M")

    schedule = NoiseSchedule(T=cfg["training"]["T"], device=args.device)

    ema = EMA(model, decay=cfg["training"]["ema_decay"])

    # ── optimizer + scheduler ─────────────────────────────────────────────────
    optim_obj = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        betas=tuple(cfg["training"]["betas"]),
        weight_decay=cfg["training"]["weight_decay"],
    )

    steps_per_epoch = (len(tr_idx) + cfg["training"]["batch_size"] - 1) // cfg["training"]["batch_size"]
    total_steps = steps_per_epoch * cfg["training"]["epochs"]
    warmup = cfg["training"]["warmup_steps"]
    min_lr_ratio = cfg["training"]["min_lr_ratio"]

    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1 - min_lr_ratio) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim_obj, lr_lambda)

    use_amp = cfg["training"]["precision"] == "fp16" and args.device == "cuda"
    scaler = GradScaler() if use_amp else None
    log.info(f"Mixed precision: {'fp16' if use_amp else 'OFF'}")

    state = RuntimeState(t_start=time.time())
    resumed = try_resume(exp_dir, model, optim_obj, scaler, scheduler, ema, state)
    if resumed:
        log.info(f"Resumed step={state.global_step}  best_val={state.best_val:.4f}")

    # signal handling
    stop_flag = {"stop": False}
    def on_sig(sig, frame):
        log.warn(f"Received signal {sig}. Finishing current step then stopping.")
        stop_flag["stop"] = True
    try:
        signal.signal(signal.SIGINT, on_sig)
        signal.signal(signal.SIGTERM, on_sig)
    except Exception:
        pass

    # ── subset-sampling generator ────────────────────────────────────────────
    gen = torch.Generator(); gen.manual_seed(seed)

    # ── move data to device if small; otherwise keep on CPU and transfer each batch
    latents_on_device = False
    total_mem_gb = ds.z.numel() * 4 / 1e9
    if total_mem_gb < 2.5 and args.device == "cuda":
        ds.z = ds.z.to(args.device)
        ds.values_san = ds.values_san.to(args.device)
        ds.cond_valid = ds.cond_valid.to(args.device)
        ds.cond_weight = ds.cond_weight.to(args.device)
        latents_on_device = True
        log.info(f"Latents ({total_mem_gb:.2f} GB) moved to GPU")

    # ── extremes oversampling weights (for training row sampling) ────────────
    osample_factor = float(cfg["training"].get("oversample_extremes_factor", 1.0))
    osample_q = float(cfg["training"].get("oversample_extremes_quantile", 0.10))
    osample_mode = str(cfg["training"].get("oversample_mode", "both"))   # both|high|low
    osample_min_weight = float(cfg["training"].get("oversample_min_cond_weight", 0.0))
    if osample_factor > 1.0:
        vn_cpu = ds.values_san.detach().cpu().numpy()
        cv_cpu = ds.cond_valid.detach().cpu().numpy().astype(bool)
        cw_cpu = ds.cond_weight.detach().cpu().numpy()
        row_w = np.ones(N, dtype=np.float64)
        n_props = vn_cpu.shape[1]
        for j in range(n_props):
            col = vn_cpu[:, j]
            valid = cv_cpu[:, j]
            # restrict the *quantile bookkeeping* to high-trust rows so that
            # smoke-model lows don't define what counts as extreme.
            if osample_min_weight > 0:
                trust = valid & (cw_cpu[:, j] >= osample_min_weight)
            else:
                trust = valid
            v = col[trust]
            if len(v) < 50:        # not enough trusted values
                v = col[valid]
            if len(v) == 0: continue
            lo = np.quantile(v, osample_q)
            hi = np.quantile(v, 1.0 - osample_q)
            if osample_mode == "high":
                extremes = trust & (col >= hi)
            elif osample_mode == "low":
                extremes = trust & (col <= lo)
            else:  # both
                extremes = trust & ((col <= lo) | (col >= hi))
            row_w[extremes] *= osample_factor
        # restrict to train rows
        train_w = row_w[tr_idx]
        train_w = train_w / train_w.sum()
        n_boost = int((row_w[tr_idx] > 1.0).sum())
        log.info(f"Oversampling: mode={osample_mode} factor={osample_factor} "
                 f"q={osample_q} min_cond_weight={osample_min_weight}  "
                 f"boosted={n_boost:,}/{len(tr_idx):,} rows ({100*n_boost/len(tr_idx):.1f}%)")
    else:
        train_w = None

    bs       = cfg["training"]["batch_size"]
    ga       = cfg["training"]["grad_accum"]
    log_every = cfg["log"]["log_every_steps"]
    val_every = cfg["log"]["val_every_steps"]
    ckpt_every = cfg["log"]["ckpt_every_steps"]
    max_minutes = cfg["training"]["total_time_minutes"]

    def elapsed_min(): return (time.time() - state.t_start) / 60
    def over_budget(): return elapsed_min() >= max_minutes

    log.info("=" * 72); log.info("TRAINING STARTED"); log.info("=" * 72)
    log.event("training_start", n_train=len(tr_idx), n_val=len(val_idx),
              n_params=int(n_params), total_steps=int(total_steps), config=cfg)

    # ── main loop ─────────────────────────────────────────────────────────────
    for epoch in range(state.epoch, cfg["training"]["epochs"]):
        state.epoch = epoch
        log.info(f"--- Epoch {epoch+1}/{cfg['training']['epochs']} ---")
        # shuffle train indices (with extremes oversampling if enabled)
        if train_w is not None:
            perm = rng.choice(tr_idx, size=len(tr_idx), replace=True, p=train_w)
        else:
            perm = rng.permutation(tr_idx)
        ep_loss = 0.0; ep_n = 0
        optim_obj.zero_grad()
        batches = [perm[i:i+bs] for i in range(0, len(perm), bs)]

        for batch_idx, idx in enumerate(batches):
            idx_t = torch.from_numpy(idx).long()
            z0    = ds.z[idx_t].to(args.device, non_blocking=True) if not latents_on_device else ds.z[idx_t]
            v_n   = ds.values_san[idx_t].to(args.device, non_blocking=True) if not latents_on_device else ds.values_san[idx_t]
            c_v   = ds.cond_valid[idx_t].to(args.device, non_blocking=True) if not latents_on_device else ds.cond_valid[idx_t]
            c_w   = ds.cond_weight[idx_t].to(args.device, non_blocking=True) if not latents_on_device else ds.cond_weight[idx_t]

            B = z0.shape[0]

            # sample random mask for this batch (cond_weight biases selection)
            mask = sample_training_mask(c_v.cpu(), cfg, gen, c_w.cpu()).to(args.device)
            # classifier-free guidance dropout: extra rate for fully-unconditional
            if random.random() < cfg["training"].get("cfg_dropout_rate", 0.0):
                mask = torch.zeros_like(mask)

            # timestep
            t = torch.randint(0, schedule.T, (B,), device=args.device, dtype=torch.long)
            with autocast(enabled=use_amp):
                z_t, noise = schedule.q_sample(z0, t)
                eps_pred = model(z_t, t, v_n, mask)
                # Per-row loss weighting (T2): rows whose conditioning is more
                # trustworthy contribute more gradient. Weight = α + (1-α) *
                # mean(cond_weight * mask). When mask=0 (uncond), weight=α.
                lw_alpha = float(cfg["training"].get("loss_weight_alpha", 1.0))
                # Min-SNR loss reweighting (Hang et al. 2023). γ=5 typical.
                msnr_gamma = float(cfg["training"].get("min_snr_gamma", 0.0))
                per_row = ((eps_pred - noise) ** 2).mean(dim=-1)
                if msnr_gamma > 0.0:
                    ab_t = schedule.alpha_bar[t]                           # (B,)
                    snr  = ab_t / (1.0 - ab_t).clamp(min=1e-8)
                    msnr = torch.minimum(snr, snr.new_full(snr.shape, msnr_gamma)) / snr
                    per_row = per_row * msnr
                if lw_alpha < 1.0:
                    rw = (c_w * mask).mean(dim=-1)
                    w = lw_alpha + (1.0 - lw_alpha) * rw
                    w = w / (w.mean() + 1e-9)
                    loss = (w * per_row).mean()
                elif msnr_gamma > 0.0:
                    loss = per_row.mean()
                else:
                    loss = F.mse_loss(eps_pred, noise)
                loss = loss / ga

            if torch.isnan(loss) or torch.isinf(loss):
                state.nan_streak += 1
                log.warn(f"NaN loss at step {state.global_step}  streak={state.nan_streak}")
                if state.nan_streak >= cfg["guards"]["nan_guard"]["max_consecutive"]:
                    log.warn("NaN guard tripped. Abort."); break
                optim_obj.zero_grad()
                continue
            state.nan_streak = 0

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % ga == 0:
                if use_amp:
                    scaler.unscale_(optim_obj)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                     cfg["training"]["grad_clip"])
                    scaler.step(optim_obj); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                     cfg["training"]["grad_clip"])
                    optim_obj.step()
                optim_obj.zero_grad()
                scheduler.step()
                ema.update(model)
                state.global_step += 1

                ep_loss += loss.item() * ga; ep_n += 1

                if state.global_step % log_every == 0:
                    lr_cur = optim_obj.param_groups[0]["lr"]
                    log.info(f"step {state.global_step:6d}  "
                              f"loss={ep_loss/ep_n:.4f}  "
                              f"lr={lr_cur:.2e}  t={elapsed_min():.1f}m")
                    log.event("train_step", step=state.global_step,
                               loss=ep_loss/ep_n, lr=lr_cur,
                               elapsed_min=elapsed_min())

                if state.global_step % val_every == 0:
                    val_loss = validate(model, ds, val_idx, schedule, cfg,
                                         args.device, use_amp, gen, log)
                    log.info(f"[VAL] step={state.global_step}  val_loss={val_loss:.4f}")
                    log.event("val", step=state.global_step, val_loss=val_loss)
                    if val_loss < state.best_val - cfg["early_stop"]["min_delta"]:
                        state.best_val = val_loss
                        state.no_improve_evals = 0
                        save_checkpoint(exp_dir / "checkpoints/best.pt", model, optim_obj,
                                         scaler, scheduler, ema, state, cfg,
                                         {"val_loss": val_loss})
                        log.info(f"  ✓ new best val_loss={val_loss:.4f} — saved best.pt")
                    else:
                        state.no_improve_evals += 1
                    model.train()

                if state.global_step % ckpt_every == 0:
                    save_checkpoint(exp_dir / f"checkpoints/step_{state.global_step:08d}.pt",
                                     model, optim_obj, scaler, scheduler, ema, state, cfg)
                    save_checkpoint(exp_dir / "checkpoints/last.pt",
                                     model, optim_obj, scaler, scheduler, ema, state, cfg)
                    rotate_checkpoints(exp_dir / "checkpoints", "step_*.pt",
                                        cfg["log"]["keep_last_checkpoints"])

                if (cfg["early_stop"]["enabled"]
                        and state.no_improve_evals >= cfg["early_stop"]["patience_evals"]):
                    log.warn(f"Early stop: {state.no_improve_evals} val evals no improvement")
                    stop_flag["stop"] = True
                if over_budget():
                    log.warn(f"Wall-clock budget {max_minutes}m exceeded")
                    stop_flag["stop"] = True
                if stop_flag["stop"]: break
            if stop_flag["stop"]: break

        log.info(f"Epoch {epoch+1} done: avg loss={ep_loss/max(ep_n,1):.4f}")
        if stop_flag["stop"]: break

    # final save
    save_checkpoint(exp_dir / "checkpoints/last.pt",
                     model, optim_obj, scaler, scheduler, ema, state, cfg)
    log.info("=" * 72)
    log.info(f"FINISHED  steps={state.global_step}  elapsed={elapsed_min():.1f}m  best_val={state.best_val:.4f}")
    log.info("=" * 72)
    log.event("training_end", total_steps=state.global_step,
              elapsed_min=elapsed_min(), best_val=state.best_val)
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump({
            "run_name":     cfg["run"]["name"],
            "start_time":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(state.t_start)),
            "end_time":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_steps":  state.global_step,
            "total_minutes": elapsed_min(),
            "best_val":     state.best_val,
            "device":       args.device,
            "resumed":      bool(args.resume),
        }, f, indent=2)
    log.close()


@torch.no_grad()
def validate(model, ds, val_idx, schedule, cfg, device, use_amp, gen, log):
    model.eval()
    bs = cfg["training"]["batch_size"] * 2
    total = 0.0; n = 0
    latents_on_device = ds.z.device.type == "cuda"
    for i in range(0, len(val_idx), bs):
        idx = val_idx[i:i+bs]
        idx_t = torch.from_numpy(idx).long()
        z0    = ds.z[idx_t].to(device) if not latents_on_device else ds.z[idx_t]
        v_n   = ds.values_san[idx_t].to(device) if not latents_on_device else ds.values_san[idx_t]
        c_v   = ds.cond_valid[idx_t].to(device) if not latents_on_device else ds.cond_valid[idx_t]
        c_w   = ds.cond_weight[idx_t].to(device) if not latents_on_device else ds.cond_weight[idx_t]
        B = z0.shape[0]
        mask = sample_training_mask(c_v.cpu(), cfg, gen, c_w.cpu()).to(device)
        t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)
        with autocast(enabled=use_amp):
            z_t, noise = schedule.q_sample(z0, t)
            eps_pred = model(z_t, t, v_n, mask)
            loss = F.mse_loss(eps_pred, noise)
        total += loss.item(); n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    main()
