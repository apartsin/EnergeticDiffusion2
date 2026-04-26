"""
Fine-tune the pretrained LIMO VAE on EnergeticDiffusion2's energetic-biased
subset. Features:

  - Unattended operation (wall-clock time budget + graceful shutdown)
  - Resumable from checkpoint (--resume <path>)
  - Versioned output directory (timestamp-based)
  - YAML config file
  - Mixed-precision (fp16) training
  - Rich structured logging: JSONL (train.jsonl) + human (train.log)
  - Automatic cached pre-tokenization (only re-runs when SMILES set changes)
  - Bucket-sorted batches (reduce padding waste)
  - Best / last / rotating checkpoints
  - NaN guard (abort on repeated loss NaNs)
  - Validation loss tracking, early stop

Usage:
    python scripts/vae/limo_finetune.py --config configs/vae_limo.yaml
    python scripts/vae/limo_finetune.py --config configs/vae_limo.yaml --resume experiments/limo_ft_energetic_20260424T180000Z

The experiment directory contains:
    config_snapshot.yaml
    train.jsonl       (one JSON per log line, parseable)
    train.log         (human-readable)
    checkpoints/
        last.pt       (always most recent)
        best.pt       (lowest val NLL so far)
        step_00001000.pt  (rotating, keep_last_checkpoints recent)
    metadata.json     (run info: start/end times, final metrics)
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.cuda.amp import GradScaler, autocast

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import selfies as sf

sys.path.insert(0, str(Path(__file__).parent))
from limo_model import (
    LIMOVAE, SELFIESTokenizer,
    LIMO_VOCAB_SIZE, LIMO_MAX_LEN, LIMO_PAD_IDX,
    build_limo_vocab, save_vocab, load_vocab, find_limo_repo,
)


# ── config ───────────────────────────────────────────────────────────────────
@dataclass
class RuntimeState:
    global_step: int = 0
    epoch:       int = 0
    best_val:    float = float("inf")
    t_start:     float = 0.0
    nan_streak:  int = 0
    no_improve_evals: int = 0


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── data preparation ─────────────────────────────────────────────────────────
def build_energetic_subset(base: Path, cfg: dict | None = None) -> pd.DataFrame:
    """Return DataFrame of canonical SMILES = labeled ∪ energetic-biased unlabeled.

    If `cfg["data"]["augmented_source"]` is set, loads ONLY that file with
    NO deduplication (so motif-oversampled duplicates are preserved).
    """
    aug = (cfg or {}).get("data", {}).get("augmented_source")
    if not aug:
        # Look at paths.labeled_master/unlabeled_master too — if either points
        # at a single augmented file, treat it as augmented.
        paths = (cfg or {}).get("paths", {})
        lm_path = paths.get("labeled_master", "data/training/master/labeled_master.csv")
        um_path = paths.get("unlabeled_master", "data/training/master/unlabeled_master.csv")
        if lm_path == um_path and "motif" in str(lm_path).lower():
            aug = lm_path
    if aug:
        path = base / aug if not Path(aug).is_absolute() else Path(aug)
        df = pd.read_csv(path, low_memory=False)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        out = df[[col]].rename(columns={col: "smiles"}).dropna()
        # NO drop_duplicates: oversampled rows are intentional
        return out

    lm = pd.read_csv(base / "data/training/master/labeled_master.csv",
                     low_memory=False, usecols=["smiles"])
    um = pd.read_csv(base / "data/training/master/unlabeled_master.csv",
                     low_memory=False,
                     usecols=["smiles", "source_dataset", "has_nitro",
                              "has_azide", "energetic_proxy_score"])
    um["has_nitro"] = um["has_nitro"].astype(str).str.lower().isin(["true", "1"])
    um["has_azide"] = um["has_azide"].astype(str).str.lower().isin(["true", "1"])
    mask = (
        (um["source_dataset"] == "rnnmgm_ds9") |
        (um["energetic_proxy_score"] >= 6) |
        (um["has_nitro"]) |
        (um["has_azide"])
    )
    energetic_um = um.loc[mask, ["smiles"]]
    combined = pd.concat([lm[["smiles"]], energetic_um], ignore_index=True)
    combined = combined.dropna().drop_duplicates("smiles")
    return combined


def encode_dataset(smiles: list[str], tok: SELFIESTokenizer,
                    drop_oov: bool, drop_too_long: bool,
                    max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-encode SMILES to padded integer tensors and lengths.

    Returns (seq_tensor [N, max_len], length_tensor [N]).
    """
    pad_idx = tok.pad_idx
    seqs, lens = [], []
    n_sfs_fail = n_oov = n_too_long = n_ok = 0
    for smi in smiles:
        try:
            sfs = sf.encoder(smi)
        except Exception:
            n_sfs_fail += 1
            continue
        if not sfs:
            n_sfs_fail += 1
            continue
        toks = list(sf.split_selfies(sfs))
        oov = any(t not in tok.sym_to_idx for t in toks)
        too_long = len(toks) > max_len
        if oov and drop_oov:
            n_oov += 1
            continue
        if too_long and drop_too_long:
            n_too_long += 1
            continue
        ids = [tok.sym_to_idx.get(t, pad_idx) for t in toks][:max_len]
        true_len = len(ids)
        ids = ids + [pad_idx] * (max_len - len(ids))
        seqs.append(ids)
        lens.append(true_len)
        n_ok += 1
    print(f"    encoded: {n_ok:,} (dropped: sfs_fail={n_sfs_fail}, oov={n_oov}, too_long={n_too_long})")
    return torch.tensor(seqs, dtype=torch.long), torch.tensor(lens, dtype=torch.long)


def prepare_or_load_cache(base: Path, cache_dir: Path, tok: SELFIESTokenizer,
                          cfg: dict, logger) -> dict:
    """Tokenize-once to cache; reuse on subsequent runs.

    Cache key = hash of (SMILES set, vocab, max_len, drop flags). If any changed,
    we re-tokenize. Else we load saved tensors.
    """
    import hashlib

    logger.info("Building energetic-biased SMILES set …")
    subset_df = build_energetic_subset(base, cfg)
    smiles = subset_df["smiles"].tolist()    # NOT sorted, NOT deduped — preserves oversampling
    logger.info(f"  {len(smiles):,} SMILES rows (incl. duplicates from oversampling)")

    # cache key
    h = hashlib.md5()
    h.update(str(len(smiles)).encode())
    h.update(str(smiles[:100]).encode())
    h.update(str(smiles[-100:]).encode())
    h.update(str(tok.vocab_size).encode())
    h.update(str(tok.max_len).encode())
    h.update(str(cfg["data"]["drop_oov"]).encode())
    h.update(str(cfg["data"]["drop_too_long"]).encode())
    key = h.hexdigest()[:12]
    tensor_path = cache_dir / f"energetic_ft_{key}.pt"

    if tensor_path.exists():
        logger.info(f"Cache hit → {tensor_path.name}")
        blob = torch.load(tensor_path, weights_only=False)
        return blob

    logger.info(f"Cache miss → re-tokenizing (will save to {tensor_path.name})")
    cache_dir.mkdir(parents=True, exist_ok=True)
    seqs, lens = encode_dataset(smiles, tok,
                                drop_oov=cfg["data"]["drop_oov"],
                                drop_too_long=cfg["data"]["drop_too_long"],
                                max_len=tok.max_len)
    blob = {
        "seqs":   seqs,
        "lens":   lens,
        "smiles": [s for s, seq in zip(smiles, [None]*len(seqs))][:len(seqs)],
        "vocab_size": tok.vocab_size,
        "max_len":    tok.max_len,
        "cache_key":  key,
    }
    # smiles list after filtering: we lost molecules to sfs/oov/too_long
    # better: recompute
    # For now, we just save seqs/lens. Training doesn't need SMILES list.
    torch.save({"seqs": seqs, "lens": lens,
                "vocab_size": tok.vocab_size, "max_len": tok.max_len,
                "cache_key": key}, tensor_path)
    return torch.load(tensor_path, weights_only=False)


# ── data loader ──────────────────────────────────────────────────────────────
class BucketedLoader:
    """Minimal length-bucketed sampler to reduce padding waste."""
    def __init__(self, seqs: torch.Tensor, lens: torch.Tensor,
                 batch_size: int, shuffle: bool, seed: int):
        self.seqs = seqs; self.lens = lens
        self.batch_size = batch_size; self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self):
        # sort by length, then group into batches, then shuffle batch order
        order = np.argsort(self.lens.numpy())
        if self.shuffle:
            # noise on the sort so same-length items aren't always adjacent
            order = order[np.argsort(np.arange(len(order))
                                     + self.rng.uniform(-2, 2))]
        batches = [order[i:i+self.batch_size]
                   for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)
        for b in batches:
            yield self.seqs[b], self.lens[b]

    def __len__(self):
        return (len(self.seqs) + self.batch_size - 1) // self.batch_size


def split_train_val_test(seqs: torch.Tensor, lens: torch.Tensor,
                          train_frac: float, val_frac: float, seed: int
                        ) -> tuple[tuple, tuple, tuple]:
    n = len(seqs)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_tr = int(train_frac * n)
    n_va = int(val_frac * n)
    tr, va, te = order[:n_tr], order[n_tr:n_tr+n_va], order[n_tr+n_va:]
    return ((seqs[tr], lens[tr]), (seqs[va], lens[va]), (seqs[te], lens[te]))


# ── loss ─────────────────────────────────────────────────────────────────────
def compute_loss(model: LIMOVAE, x: torch.Tensor, beta: float,
                  free_bits: float = 0.0, pad_idx: int = 0):
    """Forward + loss. Returns (loss, nll, kl, recon_accuracy)."""
    log_probs, z, mu, log_var = model(x)
    B, T, V = log_probs.shape

    # NLL over non-pad positions
    nll = F.nll_loss(log_probs.reshape(-1, V), x.reshape(-1),
                     ignore_index=pad_idx, reduction="mean")

    # KL with optional free-bits floor
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kld = kl_per_dim.sum(dim=1).mean()
    loss = nll + beta * kld

    # reconstruction accuracy (token-level, excluding pads)
    with torch.no_grad():
        preds = log_probs.argmax(dim=2)
        non_pad = x != pad_idx
        total = non_pad.sum().item()
        correct = ((preds == x) & non_pad).sum().item()
        recon_acc = correct / max(total, 1)

    return loss, nll, kld, recon_acc


# ── logging utilities ────────────────────────────────────────────────────────
class RunLogger:
    def __init__(self, exp_dir: Path, jsonl_filename: str, text_filename: str):
        self.exp_dir = exp_dir
        self.jsonl = open(exp_dir / jsonl_filename, "a", buffering=1, encoding="utf-8")
        self.text  = open(exp_dir / text_filename,  "a", buffering=1, encoding="utf-8")
        # ensure stdout can also handle unicode on Windows
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
        """Structured JSONL event."""
        rec = {"t": time.time(), "kind": kind, **fields}
        self.jsonl.write(json.dumps(rec) + "\n")

    def close(self):
        self.jsonl.close()
        self.text.close()


# ── checkpointing ────────────────────────────────────────────────────────────
def save_checkpoint(path: Path, model, optim, scaler, scheduler,
                     state: RuntimeState, cfg: dict, extra: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model_state":     model.state_dict(),
        "optim_state":     optim.state_dict(),
        "scaler_state":    scaler.state_dict() if scaler is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "runtime":         {
            "global_step": state.global_step,
            "epoch":       state.epoch,
            "best_val":    state.best_val,
            "t_start":     state.t_start,
            "no_improve_evals": state.no_improve_evals,
        },
        "config":   cfg,
        "extra":    extra or {},
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # write atomically
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)


def rotate_checkpoints(ckpt_dir: Path, pattern: str, keep: int):
    files = sorted(ckpt_dir.glob(pattern))
    for f in files[:-keep]:
        try:
            f.unlink()
        except OSError:
            pass


def try_resume(exp_dir: Path, model, optim, scaler, scheduler, state: RuntimeState
               ) -> bool:
    """If checkpoints/last.pt exists, restore state. Return True if resumed."""
    ckpt = exp_dir / "checkpoints/last.pt"
    if not ckpt.exists():
        return False
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state"])
    optim.load_state_dict(blob["optim_state"])
    if scaler and blob.get("scaler_state"):
        scaler.load_state_dict(blob["scaler_state"])
    if scheduler and blob.get("scheduler_state"):
        scheduler.load_state_dict(blob["scheduler_state"])
    rt = blob["runtime"]
    state.global_step = rt["global_step"]
    state.epoch       = rt["epoch"]
    state.best_val    = rt["best_val"]
    state.t_start     = rt.get("t_start", time.time())
    state.no_improve_evals = rt.get("no_improve_evals", 0)
    return True


# ── experiment directory ─────────────────────────────────────────────────────
def setup_experiment_dir(base: Path, cfg: dict, resume: str | None) -> Path:
    if resume:
        exp_dir = Path(resume)
        if not exp_dir.is_absolute():
            exp_dir = base / exp_dir
        if not exp_dir.exists():
            raise FileNotFoundError(f"Resume path not found: {exp_dir}")
        return exp_dir

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"{cfg['run']['name']}_{ts}"
    exp_dir = base / cfg["paths"]["experiments_dir"] / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    # snapshot config
    with open(exp_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return exp_dir


# ── main training loop ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None,
                    help="Resume from an existing experiment directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke",  action="store_true",
                    help="Override epochs=1 and subset=1k for a sanity run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = Path(cfg["paths"]["base"])

    if args.smoke:
        cfg["training"]["epochs"] = 1
        cfg["training"]["total_time_minutes"] = 10

    # seeds
    seed = cfg["run"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # experiment dir
    exp_dir = setup_experiment_dir(base, cfg, args.resume)
    log = RunLogger(exp_dir,
                    cfg["log"]["jsonl_filename"],
                    cfg["log"]["text_filename"])
    log.info(f"Experiment dir: {exp_dir}")
    log.info(f"Device: {args.device}")
    log.info(f"Config: {args.config}")
    if args.resume:
        log.info(f"RESUMING from: {args.resume}")

    # ── load LIMO tokenizer + model ──────────────────────────────────────────
    log.info("Loading LIMO tokenizer and pretrained checkpoint …")
    limo_dir = base / cfg["paths"]["limo_dir"]
    vocab_cache = limo_dir / "vocab_cache.json"
    if vocab_cache.exists():
        alphabet = load_vocab(vocab_cache)
    else:
        alphabet = build_limo_vocab(limo_dir / "zinc250k.smi")
        save_vocab(alphabet, vocab_cache)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    log.info(f"  vocab={tok.vocab_size}  max_len={tok.max_len}")

    model = LIMOVAE()
    init_from = cfg.get("paths", {}).get("init_from")
    if init_from and not args.resume:
        init_path = base / init_from if not Path(init_from).is_absolute() else Path(init_from)
        log.info(f"  init_from: {init_path}")
        prior = torch.load(init_path, map_location="cpu", weights_only=False)
        ms = prior.get("model_state") or prior.get("state_dict") or prior
        miss, unx = model.load_state_dict(ms, strict=False)
        log.info(f"  v1 weights loaded: missing={len(miss)} unexpected={len(unx)}")
    else:
        missing, unexpected = model.load_limo_weights(limo_dir / "vae.pt", strict=True)
        log.info(f"  weights loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(args.device)

    # ── data ──────────────────────────────────────────────────────────────────
    cache_dir = base / cfg["paths"]["cache_dir"]
    blob = prepare_or_load_cache(base, cache_dir, tok, cfg, log)
    seqs, lens = blob["seqs"], blob["lens"]
    log.info(f"Dataset: N={len(seqs):,}  max_len={blob['max_len']}  vocab={blob['vocab_size']}")
    if args.smoke and len(seqs) > 2000:
        seqs = seqs[:2000]; lens = lens[:2000]
        log.warn("SMOKE MODE: subsetting to 2000 rows")

    (tr_seqs, tr_lens), (va_seqs, va_lens), (te_seqs, te_lens) = \
        split_train_val_test(seqs, lens,
                              cfg["data"]["train_frac"],
                              cfg["data"]["val_frac"],
                              seed)
    log.info(f"Split: train={len(tr_seqs):,}  val={len(va_seqs):,}  test={len(te_seqs):,}")

    bs  = cfg["training"]["batch_size"]
    ga  = cfg["training"]["grad_accum"]
    train_loader = BucketedLoader(tr_seqs, tr_lens, batch_size=bs, shuffle=True,
                                    seed=seed)
    val_loader   = BucketedLoader(va_seqs, va_lens, batch_size=bs, shuffle=False,
                                    seed=seed)

    # ── optimizer + scheduler ─────────────────────────────────────────────────
    optim_obj = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        betas=tuple(cfg["training"]["betas"]),
        weight_decay=cfg["training"]["weight_decay"])

    steps_per_epoch = len(train_loader) // ga
    total_steps = steps_per_epoch * cfg["training"]["epochs"]
    warmup = cfg["training"]["warmup_steps"]
    min_lr_ratio = cfg["training"]["min_lr_ratio"]

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1 - min_lr_ratio) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim_obj, lr_lambda)

    # ── mixed precision ──────────────────────────────────────────────────────
    use_amp = cfg["training"]["precision"] == "fp16" and args.device == "cuda"
    scaler = GradScaler() if use_amp else None
    log.info(f"Mixed precision: {'fp16' if use_amp else 'OFF'}")

    # ── runtime state + resume ───────────────────────────────────────────────
    state = RuntimeState(t_start=time.time())
    resumed = try_resume(exp_dir, model, optim_obj, scaler, scheduler, state)
    if resumed:
        log.info(f"Resumed at step={state.global_step}  best_val={state.best_val:.4f}")

    # ── signal handling for graceful shutdown ────────────────────────────────
    stop_requested = {"flag": False}
    def handle_sig(signum, frame):
        log.warn(f"Received signal {signum}. Will stop after current step.")
        stop_requested["flag"] = True
    try:
        signal.signal(signal.SIGINT, handle_sig)
        signal.signal(signal.SIGTERM, handle_sig)
    except (ValueError, AttributeError):
        pass  # main thread only

    # ── training loop ────────────────────────────────────────────────────────
    log.info("=" * 72)
    log.info("TRAINING STARTED")
    log.info("=" * 72)
    log.event("training_start", n_train=len(tr_seqs), n_val=len(va_seqs),
              total_steps=total_steps, config=cfg)

    beta      = cfg["training"]["kl"]["beta"]
    free_bits = cfg["training"]["kl"].get("free_bits", 0.0)
    pad_idx   = tok.pad_idx
    log_every = cfg["log"]["log_every_steps"]
    val_every = cfg["log"]["val_every_steps"]
    ckpt_every = cfg["log"]["ckpt_every_steps"]
    max_minutes = cfg["training"].get("total_time_minutes", 9999)

    def elapsed_min():
        return (time.time() - state.t_start) / 60

    def time_budget_exceeded():
        return elapsed_min() >= max_minutes

    model.train()
    for epoch in range(state.epoch, cfg["training"]["epochs"]):
        state.epoch = epoch
        log.info(f"--- Epoch {epoch+1}/{cfg['training']['epochs']} ---")
        log.event("epoch_start", epoch=epoch)

        ep_loss = ep_nll = ep_kld = ep_acc = 0.0
        ep_n = 0
        optim_obj.zero_grad()

        for batch_idx, (x, lns) in enumerate(train_loader):
            x = x.to(args.device, non_blocking=True)

            with autocast(enabled=use_amp):
                loss, nll, kld, acc = compute_loss(model, x, beta, free_bits, pad_idx)
                loss = loss / ga

            if torch.isnan(loss) or torch.isinf(loss):
                state.nan_streak += 1
                log.warn(f"NaN/inf loss at step {state.global_step} (streak={state.nan_streak})")
                if (cfg["guards"]["nan_guard"]["enabled"]
                        and state.nan_streak >= cfg["guards"]["nan_guard"]["max_consecutive"]):
                    log.warn("NaN guard tripped. Aborting.")
                    break
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
                    scaler.step(optim_obj)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    cfg["training"]["grad_clip"])
                    optim_obj.step()
                optim_obj.zero_grad()
                scheduler.step()
                state.global_step += 1

                ep_loss += loss.item() * ga; ep_nll += nll.item()
                ep_kld += kld.item(); ep_acc += acc; ep_n += 1

                if state.global_step % log_every == 0:
                    lr_cur = optim_obj.param_groups[0]["lr"]
                    log.info(f"step {state.global_step:6d}  "
                             f"loss={ep_loss/ep_n:.4f}  nll={ep_nll/ep_n:.4f}  "
                             f"kl={ep_kld/ep_n:.3f}  acc={ep_acc/ep_n*100:.2f}%  "
                             f"lr={lr_cur:.2e}  t={elapsed_min():.1f}m")
                    log.event("train_step", step=state.global_step,
                              loss=ep_loss/ep_n, nll=ep_nll/ep_n,
                              kl=ep_kld/ep_n, acc=ep_acc/ep_n, lr=lr_cur,
                              elapsed_min=elapsed_min())

                # ── validation ──
                if state.global_step % val_every == 0:
                    val_loss, val_nll, val_acc = validate(model, val_loader,
                                                           args.device, beta,
                                                           free_bits, pad_idx,
                                                           use_amp)
                    log.info(f"[VAL] step={state.global_step}  "
                             f"val_loss={val_loss:.4f}  val_nll={val_nll:.4f}  "
                             f"val_acc={val_acc*100:.2f}%")
                    log.event("val", step=state.global_step,
                              val_loss=val_loss, val_nll=val_nll, val_acc=val_acc)
                    if val_nll < state.best_val - cfg["early_stop"]["min_delta"]:
                        state.best_val = val_nll
                        state.no_improve_evals = 0
                        save_checkpoint(exp_dir / "checkpoints/best.pt", model, optim_obj,
                                         scaler, scheduler, state, cfg,
                                         {"val_loss": val_loss, "val_acc": val_acc})
                        log.info(f"  ✓ new best val_nll={val_nll:.4f} — saved best.pt")
                    else:
                        state.no_improve_evals += 1
                    model.train()

                # ── checkpoint ──
                if state.global_step % ckpt_every == 0:
                    ckpt_path = exp_dir / f"checkpoints/step_{state.global_step:08d}.pt"
                    save_checkpoint(ckpt_path, model, optim_obj, scaler, scheduler,
                                     state, cfg)
                    save_checkpoint(exp_dir / "checkpoints/last.pt", model, optim_obj,
                                     scaler, scheduler, state, cfg)
                    rotate_checkpoints(exp_dir / "checkpoints", "step_*.pt",
                                        cfg["log"]["keep_last_checkpoints"])

                # ── early stop / wall clock ──
                if (cfg["early_stop"]["enabled"]
                        and state.no_improve_evals >= cfg["early_stop"]["patience_evals"]):
                    log.warn(f"Early stop: no improvement for {state.no_improve_evals} val evals")
                    stop_requested["flag"] = True
                if time_budget_exceeded():
                    log.warn(f"Time budget {max_minutes}m exceeded — stopping")
                    stop_requested["flag"] = True
                if stop_requested["flag"]:
                    break
            if stop_requested["flag"]:
                break

        log.info(f"Epoch {epoch+1} done: avg loss={ep_loss/max(ep_n,1):.4f}")
        log.event("epoch_end", epoch=epoch,
                  avg_loss=ep_loss/max(ep_n,1),
                  avg_acc=ep_acc/max(ep_n,1))
        if stop_requested["flag"]:
            break

    # ── final checkpoint + metadata ──
    save_checkpoint(exp_dir / "checkpoints/last.pt", model, optim_obj, scaler, scheduler,
                     state, cfg)
    log.info("=" * 72)
    log.info(f"TRAINING FINISHED. total_steps={state.global_step}  elapsed={elapsed_min():.1f}m  best_val={state.best_val:.4f}")
    log.info("=" * 72)
    log.event("training_end", total_steps=state.global_step,
              elapsed_min=elapsed_min(), best_val=state.best_val)

    metadata = {
        "run_name":        cfg["run"]["name"],
        "start_time":      time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime(state.t_start)),
        "end_time":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_steps":     state.global_step,
        "total_minutes":   elapsed_min(),
        "best_val_nll":    state.best_val,
        "device":          args.device,
        "resumed":         bool(args.resume),
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.close()


@torch.no_grad()
def validate(model, loader, device, beta, free_bits, pad_idx, use_amp):
    model.eval()
    tot_loss = tot_nll = tot_acc = 0.0
    n = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            loss, nll, kld, acc = compute_loss(model, x, beta, free_bits, pad_idx)
        tot_loss += loss.item(); tot_nll += nll.item(); tot_acc += acc
        n += 1
    return tot_loss/max(n,1), tot_nll/max(n,1), tot_acc/max(n,1)


if __name__ == "__main__":
    main()
