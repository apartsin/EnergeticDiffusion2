"""Train LIMO v3 = frozen v1 encoder + transformer decoder.

Reuses the same data + tokenizer + loss + checkpoint plumbing as
limo_finetune.py. Only difference: the model is `LIMOVAEv3`, encoder is
frozen, only the new decoder trains.

Usage:
    python scripts/vae/limo_v3_train.py --config configs/vae_limo_v3.yaml
"""
from __future__ import annotations
import argparse, json, math, os, random, signal, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import selfies as sf

sys.path.insert(0, str(Path(__file__).parent))
from limo_model import (LIMOVAE, SELFIESTokenizer, LIMO_VOCAB_SIZE,
                          LIMO_MAX_LEN, LIMO_PAD_IDX,
                          build_limo_vocab, save_vocab, load_vocab,
                          find_limo_repo)
from limo_v3_model import LIMOVAEv3
# Reuse the bulky data utils from limo_finetune
from limo_finetune import (build_energetic_subset, encode_dataset,
                              prepare_or_load_cache, split_train_val_test,
                              BucketSampler, RuntimeState, RunLogger,
                              save_checkpoint, rotate_checkpoints,
                              try_resume, setup_experiment_dir,
                              compute_loss, validate)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke",  action="store_true")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    cfg = load_config(args.config)
    base = Path(cfg["paths"]["base"])
    if args.smoke:
        cfg["training"]["total_time_minutes"] = 5
        cfg["training"]["epochs"] = 1

    seed = cfg["run"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    exp = setup_experiment_dir(base, cfg, args.resume)
    log = RunLogger(exp, cfg["log"]["jsonl_filename"], cfg["log"]["text_filename"])
    log.info(f"v3 experiment dir: {exp}")
    log.info(f"Device: {args.device}")

    # vocab + tokenizer
    limo_dir = find_limo_repo(base)
    vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    log.info(f"vocab={tok.vocab_size}  max_len={tok.max_len}")

    # ── load v1 weights ───────────────────────────────────────────────────
    init_from = cfg["paths"]["init_from"]
    init_path = base / init_from if not Path(init_from).is_absolute() else Path(init_from)
    log.info(f"init encoder from: {init_path}")
    v1_blob = torch.load(init_path, map_location="cpu", weights_only=False)
    v1_state = v1_blob.get("model_state") or v1_blob.get("state_dict") or v1_blob

    # ── build v3 model ────────────────────────────────────────────────────
    arch = cfg.get("arch", {})
    model = LIMOVAEv3(
        v1_state_dict=v1_state,
        max_len=LIMO_MAX_LEN, vocab_len=LIMO_VOCAB_SIZE,
        latent_dim=cfg["arch"].get("latent_dim", 1024),
        embedding_dim=cfg["arch"].get("embedding_dim", 64),
        d_model=arch.get("d_model", 384),
        n_heads=arch.get("n_heads", 6),
        n_layers=arch.get("n_layers", 4),
        ff_mult=arch.get("ff_mult", 4),
        dropout=arch.get("dropout", 0.1),
        n_memory=arch.get("n_memory", 16),
        freeze_encoder=cfg["arch"].get("freeze_encoder", True),
    ).to(args.device)
    log.info(f"model: {model}")

    # ── data ──────────────────────────────────────────────────────────────
    cache_dir = base / cfg["paths"]["cache_dir"]
    blob = prepare_or_load_cache(base, cache_dir, tok, cfg, log)
    seqs, lens = blob["seqs"], blob["lens"]
    log.info(f"Dataset: N={len(seqs):,}  max_len={blob['max_len']}  vocab={blob['vocab_size']}")
    if args.smoke and len(seqs) > 2000:
        seqs = seqs[:2000]; lens = lens[:2000]
        log.warn("SMOKE MODE: subsetting to 2000 rows")

    (tr_seqs, tr_lens), (va_seqs, va_lens), (te_seqs, te_lens) = split_train_val_test(
        seqs, lens, cfg["data"]["train_frac"], cfg["data"]["val_frac"], seed)
    log.info(f"Split: train={len(tr_seqs):,}  val={len(va_seqs):,}  test={len(te_seqs):,}")

    # ── optimizer + scheduler ────────────────────────────────────────────
    train_params = [p for p in model.parameters() if p.requires_grad]
    log.info(f"trainable params: {sum(p.numel() for p in train_params)/1e6:.2f} M")
    optim = torch.optim.AdamW(train_params,
                                lr=cfg["training"]["lr"],
                                betas=tuple(cfg["training"]["betas"]),
                                weight_decay=cfg["training"]["weight_decay"])
    bs = cfg["training"]["batch_size"]
    steps_per_epoch = (len(tr_seqs) + bs - 1) // bs
    total_steps = steps_per_epoch * cfg["training"]["epochs"]
    warmup = cfg["training"]["warmup_steps"]
    min_lr_ratio = cfg["training"]["min_lr_ratio"]
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return min_lr_ratio + (1 - min_lr_ratio) * cos
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    use_amp = (cfg["training"]["precision"] == "fp16" and args.device == "cuda")
    scaler = GradScaler() if use_amp else None
    log.info(f"Mixed precision: {'fp16' if use_amp else 'OFF'}")

    state = RuntimeState(t_start=time.time())
    resumed = try_resume(exp, model, optim, scaler, None, scheduler, state, log)
    if resumed:
        log.info(f"Resumed step={state.global_step} best_val={state.best_val:.4f}")

    stop = {"x": False}
    def on_sig(s, f):
        log.warn(f"signal {s} → soft stop"); stop["x"] = True
    try: signal.signal(signal.SIGINT, on_sig); signal.signal(signal.SIGTERM, on_sig)
    except Exception: pass

    log.info("=" * 60); log.info("TRAINING (v3 transformer decoder)"); log.info("=" * 60)

    log_every  = cfg["log"]["log_every_steps"]
    val_every  = cfg["log"]["val_every_steps"]
    ckpt_every = cfg["log"]["ckpt_every_steps"]
    max_min    = cfg["training"]["total_time_minutes"]
    elapsed_min = lambda: (time.time() - state.t_start) / 60
    over_budget = lambda: elapsed_min() >= max_min

    rng = np.random.default_rng(seed)
    for epoch in range(state.epoch, cfg["training"]["epochs"]):
        state.epoch = epoch
        log.info(f"--- Epoch {epoch+1}/{cfg['training']['epochs']} ---")
        sampler = BucketSampler(tr_lens, batch_size=bs, shuffle=True,
                                  seed=seed + epoch)
        ep_loss = ep_nll = ep_kl = ep_acc = 0.0; ep_n = 0
        for batch_idx, idx in enumerate(sampler):
            x = tr_seqs[idx].to(args.device)
            with autocast(enabled=use_amp):
                loss, nll, kl, acc = compute_loss(
                    model, x, beta=cfg["training"]["kl"]["beta"],
                    free_bits=cfg["training"]["kl"].get("free_bits", 0.0))
            if torch.isnan(loss) or torch.isinf(loss):
                state.nan_streak += 1
                log.warn(f"NaN at step {state.global_step}  streak={state.nan_streak}")
                if state.nan_streak >= cfg["guards"]["nan_guard"]["max_consecutive"]:
                    log.warn("NaN guard tripped"); stop["x"] = True; break
                optim.zero_grad(); continue
            state.nan_streak = 0
            optim.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(train_params, cfg["training"]["grad_clip"])
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, cfg["training"]["grad_clip"])
                optim.step()
            scheduler.step(); state.global_step += 1
            ep_loss += loss.item(); ep_nll += nll.item(); ep_kl += kl.item()
            ep_acc += float(acc); ep_n += 1
            if state.global_step % log_every == 0:
                lr_cur = optim.param_groups[0]["lr"]
                log.info(f"step {state.global_step:6d}  "
                          f"loss={ep_loss/ep_n:.4f}  nll={ep_nll/ep_n:.4f}  "
                          f"kl={ep_kl/ep_n:.3f}  acc={100*ep_acc/ep_n:.2f}%  "
                          f"lr={lr_cur:.2e}  t={elapsed_min():.1f}m")
                log.event("train", step=state.global_step,
                            loss=ep_loss/ep_n, nll=ep_nll/ep_n, kl=ep_kl/ep_n,
                            acc=ep_acc/ep_n, lr=lr_cur)
            if state.global_step % val_every == 0:
                vloss, vnll, vacc = validate(model, va_seqs, va_lens,
                                                cfg["training"]["batch_size"]*2,
                                                args.device, use_amp,
                                                cfg["training"]["kl"]["beta"],
                                                cfg["training"]["kl"].get("free_bits", 0.0))
                log.info(f"[VAL] step={state.global_step}  "
                          f"val_loss={vloss:.4f}  val_nll={vnll:.4f}  val_acc={100*vacc:.2f}%")
                log.event("val", step=state.global_step, val_loss=vloss,
                            val_nll=vnll, val_acc=vacc)
                if vloss < state.best_val - cfg["early_stop"]["min_delta"]:
                    state.best_val = vloss; state.no_improve_evals = 0
                    save_checkpoint(exp / "checkpoints/best.pt", model, optim,
                                      scaler, scheduler, None, state, cfg)
                    log.info(f"  ✓ new best val_loss={vloss:.4f} → best.pt")
                else:
                    state.no_improve_evals += 1
                model.train()
            if state.global_step % ckpt_every == 0:
                save_checkpoint(exp / f"checkpoints/step_{state.global_step:08d}.pt",
                                  model, optim, scaler, scheduler, None, state, cfg)
                save_checkpoint(exp / "checkpoints/last.pt",
                                  model, optim, scaler, scheduler, None, state, cfg)
                rotate_checkpoints(exp / "checkpoints", "step_*.pt",
                                     cfg["log"]["keep_last_checkpoints"])
            if cfg["early_stop"]["enabled"] and \
                    state.no_improve_evals >= cfg["early_stop"]["patience_evals"]:
                log.warn(f"Early stop ({state.no_improve_evals} no-improve)")
                stop["x"] = True
            if over_budget():
                log.warn(f"Wall-clock {max_min}m budget exceeded")
                stop["x"] = True
            if stop["x"]: break
        log.info(f"Epoch {epoch+1} done: avg loss={ep_loss/max(ep_n,1):.4f}")
        if stop["x"]: break

    save_checkpoint(exp / "checkpoints/last.pt", model, optim, scaler, scheduler,
                      None, state, cfg)
    log.info("=" * 60)
    log.info(f"TRAINING FINISHED. total_steps={state.global_step}  "
              f"elapsed={elapsed_min():.1f}m  best_val={state.best_val:.4f}")
    log.info("=" * 60)
    log.event("end", total_steps=state.global_step, elapsed_min=elapsed_min(),
                best_val=state.best_val)
    log.close()


if __name__ == "__main__":
    sys.exit(main())
