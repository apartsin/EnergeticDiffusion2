"""
Train SA and SC score predictors on LIMO latents.

Fast: two MLPs × 382k latents × ~5 min each on RTX 2060.

Outputs to experiments/guidance_<ts>/:
    config_snapshot.yaml
    train.jsonl  train.log
    checkpoints/sa_best.pt  sa_last.pt
    checkpoints/sc_best.pt  sc_last.pt
    metadata.json
    report.html

Usage:
    python scripts/guidance/train.py --config configs/guidance.yaml
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent))
from model import ScorePredictor


# ── helpers ──────────────────────────────────────────────────────────────────
def load_config(path):
    with open(path) as f: return yaml.safe_load(f)


def setup_exp(base: Path, cfg: dict) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"{cfg['run']['name']}_{ts}"
    exp = base / cfg["paths"]["experiments_dir"] / name
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "checkpoints").mkdir(exist_ok=True)
    with open(exp / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return exp


class Log:
    def __init__(self, exp: Path):
        self.txt   = open(exp / "train.log",   "a", buffering=1, encoding="utf-8")
        self.jsonl = open(exp / "train.jsonl", "a", buffering=1, encoding="utf-8")
        try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

    def info(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line); self.txt.write(line + "\n")

    def event(self, kind, **fields):
        self.jsonl.write(json.dumps({"t": time.time(), "kind": kind, **fields}) + "\n")

    def close(self): self.txt.close(); self.jsonl.close()


# ── training per score ───────────────────────────────────────────────────────
def train_predictor(name: str, z: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                     stats: dict, cfg: dict, exp_dir: Path, device: str,
                     log: Log) -> dict:
    """Train one ScorePredictor for a given score.
    z:    (N, 1024) latents
    y:    (N,) raw scores (may contain NaN)
    mask: (N,) bool, True where score is valid
    stats: {"mean", "std"} used to standardize training target
    """
    log.info(f"\n=== Training {name} predictor ===")
    # filter to valid rows
    z_v = z[mask]
    y_v = y[mask]
    log.info(f"  valid samples: {len(z_v):,}")

    # standardize
    mu, sd = float(stats["mean"]), float(stats["std"])
    sd = max(sd, 1e-6)
    y_std = (y_v - mu) / sd

    # split
    N = len(z_v)
    rng = np.random.default_rng(cfg["run"]["seed"])
    order = rng.permutation(N)
    n_val = int(cfg["data"]["val_frac"] * N)
    va, tr = order[:n_val], order[n_val:]

    model = ScorePredictor(
        in_dim=z.shape[1],
        hidden=cfg["model"]["hidden"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  params: {n_params/1e6:.2f}M  hidden={cfg['model']['hidden']}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"])

    bs = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    total_steps = max(1, (len(tr) // bs) * epochs)
    warmup = cfg["training"]["warmup_steps"]

    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # move to GPU (small, fits)
    z_gpu = z_v.to(device)
    y_gpu = y_std.to(device)
    use_amp = cfg["training"]["precision"] == "fp16" and device == "cuda"
    scaler = GradScaler() if use_amp else None

    best_val = float("inf")
    best_path = exp_dir / f"checkpoints/{name}_best.pt"
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        perm = rng.permutation(tr)
        model.train()
        ep_loss = 0.0; n = 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]
            zb = z_gpu[idx]; yb = y_gpu[idx]
            opt.zero_grad()
            with autocast(enabled=use_amp):
                pred = model(zb)
                loss = F.mse_loss(pred, yb)
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step()
            step += 1
            ep_loss += loss.item(); n += 1
            if step % cfg["log"]["log_every_steps"] == 0:
                log.info(f"  [{name}] step {step:5d}  loss={ep_loss/n:.4f}  lr={opt.param_groups[0]['lr']:.2e}")
                log.event(f"{name}_train", step=step, loss=ep_loss/n,
                          lr=opt.param_groups[0]["lr"])

        # validation per epoch
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            for i in range(0, len(va), bs * 4):
                idx = va[i:i+bs*4]
                with autocast(enabled=use_amp):
                    p = model(z_gpu[idx])
                val_preds.append(p.cpu().numpy())
                val_targets.append(y_gpu[idx].cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_mse = float(((val_preds - val_targets) ** 2).mean())
            # R² computed on standardized targets (same ratio regardless)
            ss_res = float(((val_targets - val_preds) ** 2).sum())
            ss_tot = float(((val_targets - val_targets.mean()) ** 2).sum())
            val_r2  = 1 - ss_res / max(ss_tot, 1e-9)
            val_mae = float(np.abs(val_preds - val_targets).mean())

        log.info(f"  [{name}] epoch {epoch+1}/{epochs}  val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  R²={val_r2:.4f}")
        log.event(f"{name}_val", epoch=epoch, val_mse=val_mse,
                  val_mae=val_mae, r2=val_r2)

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "model_state":  model.state_dict(),
                "score_name":   name,
                "stats":        {"mean": mu, "std": sd},
                "config":       cfg,
                "val_mse":      val_mse,
                "val_r2":       val_r2,
                "val_mae":      val_mae,
                "epoch":        epoch,
            }, best_path)
            log.info(f"  [{name}]  ✓ new best ({val_mse:.4f}) → {best_path.name}")

    # final 'last' save
    torch.save({
        "model_state": model.state_dict(),
        "score_name":  name,
        "stats":       {"mean": mu, "std": sd},
        "config":      cfg,
        "final_val_mse": val_mse,
        "final_val_r2":  val_r2,
    }, exp_dir / f"checkpoints/{name}_last.pt")

    log.info(f"  [{name}] training done. best val_mse={best_val:.4f}  elapsed={(time.time()-t0):.0f}s")
    return {"best_val_mse": best_val, "best_val_r2": val_r2, "best_val_mae": val_mae,
            "n_train": int(len(tr)), "n_val": int(len(va)),
            "elapsed_sec": float(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = Path(cfg["paths"]["base"])

    seed = cfg["run"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    exp = setup_exp(base, cfg)
    log = Log(exp)
    log.info(f"Experiment: {exp}")
    log.info(f"Device: {args.device}")

    # load data
    path = base / cfg["paths"]["latents_with_scores"]
    log.info(f"Loading {path}")
    blob = torch.load(path, weights_only=False)
    z = blob["z_mu"].float()
    sa = blob["sa_score"].float()
    sc = blob["sc_score"].float()
    sa_mask = ~torch.isnan(sa)
    sc_mask = ~torch.isnan(sc)
    stats = blob["score_stats"]
    log.info(f"  N={len(z):,}  latent_dim={z.shape[1]}")
    log.info(f"  SA valid: {int(sa_mask.sum()):,}  mean={stats['sa']['mean']:.3f}  std={stats['sa']['std']:.3f}")
    log.info(f"  SC valid: {int(sc_mask.sum()):,}  mean={stats['sc']['mean']:.3f}  std={stats['sc']['std']:.3f}")

    # train SA
    sa_res = train_predictor("sa", z, sa, sa_mask, stats["sa"], cfg, exp, args.device, log)
    # train SC
    sc_res = train_predictor("sc", z, sc, sc_mask, stats["sc"], cfg, exp, args.device, log)

    # metadata
    meta = {
        "run_name":   cfg["run"]["name"],
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sa_results": sa_res,
        "sc_results": sc_res,
        "device":     args.device,
    }
    with open(exp / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    log.info("=" * 72)
    log.info(f"DONE. SA: R²={sa_res['best_val_r2']:.4f}  MAE={sa_res['best_val_mae']:.4f}")
    log.info(f"      SC: R²={sc_res['best_val_r2']:.4f}  MAE={sc_res['best_val_mae']:.4f}")
    log.info("=" * 72)
    log.close()


if __name__ == "__main__":
    main()
