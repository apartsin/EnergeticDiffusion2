"""Train LIMO v3 = frozen v1 encoder + transformer decoder.

Self-contained trainer (doesn't depend on missing helpers).
"""
from __future__ import annotations
import argparse, json, math, os, random, signal, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).parent))
from limo_model import (LIMOVAE, SELFIESTokenizer, LIMO_VOCAB_SIZE,
                          LIMO_MAX_LEN, LIMO_PAD_IDX,
                          build_limo_vocab, save_vocab, load_vocab,
                          find_limo_repo)
from limo_v3_model import LIMOVAEv3
from limo_finetune import (build_energetic_subset, prepare_or_load_cache,
                              compute_loss)


@dataclass
class RT:
    step: int = 0
    epoch: int = 0
    best_val: float = float("inf")
    no_improve: int = 0


def setup_exp(base, cfg):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    exp = base / cfg["paths"]["experiments_dir"] / f"{cfg['run']['name']}_{ts}"
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "checkpoints").mkdir(exist_ok=True)
    with open(exp / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    cfg = yaml.safe_load(open(args.config))
    base = Path(cfg["paths"]["base"])
    if args.smoke:
        cfg["training"]["epochs"] = 1
        cfg["training"]["total_time_minutes"] = 5

    seed = cfg["run"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    exp = setup_exp(base, cfg)
    log_f = open(exp / "train.log", "a", buffering=1, encoding="utf-8")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line); log_f.write(line + "\n")
    log(f"v3 exp: {exp}  device: {args.device}")

    # ── tokenizer ────────────────────────────────────────────────────────
    limo_dir = find_limo_repo(base)
    vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    log(f"vocab={tok.vocab_size}  max_len={tok.max_len}")

    # ── data (uses prepare_or_load_cache; respects augmented_source) ─────
    cache_dir = base / cfg["paths"]["cache_dir"]
    class _LoggerShim:
        def info(self, msg): log(msg)
        def warn(self, msg): log("WARN: " + msg)
        def warning(self, msg): log("WARN: " + msg)
    blob = prepare_or_load_cache(base, cache_dir, tok, cfg, _LoggerShim())
    seqs, lens = blob["seqs"], blob["lens"]
    log(f"Dataset N={len(seqs):,}")

    if args.smoke and len(seqs) > 4000:
        seqs = seqs[:4000]; lens = lens[:4000]
        log("SMOKE: subsetting to 4000")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(seqs))
    n_val = max(int(0.025 * len(seqs)), 200)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    log(f"split: train={len(tr_idx):,}  val={len(val_idx):,}")

    # ── load v1 encoder weights ──────────────────────────────────────────
    v1_path = base / cfg["paths"]["init_from"]
    log(f"v1 ckpt: {v1_path}")
    v1_blob = torch.load(v1_path, map_location="cpu", weights_only=False)
    v1_state = v1_blob.get("model_state") or v1_blob.get("state_dict") or v1_blob

    arch = cfg["arch"]
    model = LIMOVAEv3(
        v1_state_dict=v1_state,
        max_len=LIMO_MAX_LEN, vocab_len=LIMO_VOCAB_SIZE,
        latent_dim=arch.get("latent_dim", 1024),
        embedding_dim=arch.get("embedding_dim", 64),
        d_model=arch.get("d_model", 384),
        n_heads=arch.get("n_heads", 6),
        n_layers=arch.get("n_layers", 4),
        ff_mult=arch.get("ff_mult", 4),
        dropout=arch.get("dropout", 0.1),
        n_memory=arch.get("n_memory", 16),
        freeze_encoder=arch.get("freeze_encoder", True),
    ).to(args.device)
    log(repr(model))

    # ── optimizer ────────────────────────────────────────────────────────
    train_params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable: {sum(p.numel() for p in train_params)/1e6:.2f}M")
    opt = torch.optim.AdamW(train_params,
                              lr=cfg["training"]["lr"],
                              betas=tuple(cfg["training"]["betas"]),
                              weight_decay=cfg["training"]["weight_decay"])
    bs = cfg["training"]["batch_size"]
    total_steps = ((len(tr_idx) + bs - 1) // bs) * cfg["training"]["epochs"]
    warmup = cfg["training"]["warmup_steps"]
    min_lr_ratio = cfg["training"]["min_lr_ratio"]
    def lr_lambda(s):
        if s < warmup: return s / max(1, warmup)
        prog = (s - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return min_lr_ratio + (1 - min_lr_ratio) * cos
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = (cfg["training"]["precision"] == "fp16" and args.device == "cuda")
    scaler = GradScaler() if use_amp else None
    log(f"amp: {'fp16' if use_amp else 'OFF'}")

    rt = RT()
    t0 = time.time()
    elapsed_min = lambda: (time.time() - t0) / 60
    over_budget = lambda: elapsed_min() >= cfg["training"]["total_time_minutes"]

    pad = LIMO_PAD_IDX
    beta = cfg["training"]["kl"]["beta"]
    free_bits = cfg["training"]["kl"].get("free_bits", 0.0)
    log_every = cfg["log"]["log_every_steps"]
    val_every = cfg["log"]["val_every_steps"]
    ckpt_every = cfg["log"]["ckpt_every_steps"]
    patience = cfg["early_stop"]["patience_evals"]
    min_delta = cfg["early_stop"]["min_delta"]

    @torch.no_grad()
    def run_val():
        model.eval()
        total_loss = total_nll = total_kl = total_acc = 0.0; n = 0
        for i in range(0, len(val_idx), bs * 2):
            ids = val_idx[i:i + bs * 2]
            x = seqs[ids].to(args.device)
            with autocast(enabled=use_amp):
                loss, nll, kl, acc = compute_loss(model, x, beta, free_bits, pad)
            total_loss += loss.item(); total_nll += nll.item()
            total_kl += kl.item(); total_acc += acc; n += 1
        model.train()
        return total_loss / n, total_nll / n, total_acc / n

    log("=" * 50); log("TRAINING (v3 transformer decoder)")
    stop = {"x": False}
    def on_sig(s, f):
        log(f"signal {s} → soft stop"); stop["x"] = True
    try:
        signal.signal(signal.SIGINT, on_sig); signal.signal(signal.SIGTERM, on_sig)
    except Exception: pass

    model.train()
    nan_streak = 0
    for epoch in range(cfg["training"]["epochs"]):
        rt.epoch = epoch
        log(f"--- Epoch {epoch+1}/{cfg['training']['epochs']} ---")
        order = rng.permutation(tr_idx)
        ep_loss = ep_nll = ep_kl = ep_acc = 0.0; ep_n = 0
        for i in range(0, len(order), bs):
            ids = order[i:i + bs]
            x = seqs[ids].to(args.device)
            with autocast(enabled=use_amp):
                loss, nll, kl, acc = compute_loss(model, x, beta, free_bits, pad)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_streak += 1
                log(f"NaN at step {rt.step}  streak={nan_streak}")
                if nan_streak >= 3:
                    log("NaN guard tripped"); stop["x"] = True; break
                opt.zero_grad(); continue
            nan_streak = 0
            opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(train_params, cfg["training"]["grad_clip"])
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, cfg["training"]["grad_clip"])
                opt.step()
            scheduler.step(); rt.step += 1
            ep_loss += loss.item(); ep_nll += nll.item(); ep_kl += kl.item()
            ep_acc += acc; ep_n += 1
            if rt.step % log_every == 0:
                lr_cur = opt.param_groups[0]["lr"]
                log(f"step {rt.step:6d}  loss={ep_loss/ep_n:.4f}  "
                    f"nll={ep_nll/ep_n:.4f}  kl={ep_kl/ep_n:.3f}  "
                    f"acc={100*ep_acc/ep_n:.2f}%  lr={lr_cur:.2e}  t={elapsed_min():.1f}m")
            if rt.step % val_every == 0:
                vl, vn, va = run_val()
                log(f"[VAL] step={rt.step}  val_loss={vl:.4f}  val_nll={vn:.4f}  val_acc={100*va:.2f}%")
                if vl < rt.best_val - min_delta:
                    rt.best_val = vl; rt.no_improve = 0
                    torch.save({"model_state": model.state_dict(),
                                  "config": cfg, "step": rt.step,
                                  "best_val": vl},
                                 exp / "checkpoints/best.pt")
                    log(f"  ✓ best.pt @ step {rt.step}  val={vl:.4f}")
                else:
                    rt.no_improve += 1
                    if rt.no_improve >= patience:
                        log(f"early stop ({rt.no_improve} no-improve)"); stop["x"] = True
            if rt.step % ckpt_every == 0:
                torch.save({"model_state": model.state_dict(),
                              "config": cfg, "step": rt.step},
                             exp / "checkpoints/last.pt")
            if over_budget():
                log(f"wall-clock {cfg['training']['total_time_minutes']}m exceeded"); stop["x"] = True
            if stop["x"]: break
        log(f"epoch {epoch+1} avg loss={ep_loss/max(ep_n,1):.4f}")
        if stop["x"]: break

    torch.save({"model_state": model.state_dict(), "config": cfg, "step": rt.step},
                 exp / "checkpoints/last.pt")
    log(f"DONE total_steps={rt.step}  elapsed={elapsed_min():.1f}m  best_val={rt.best_val:.4f}")
    log_f.close()


if __name__ == "__main__":
    sys.exit(main())
