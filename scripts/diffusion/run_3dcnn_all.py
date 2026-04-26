"""
One-shot: run 3DCNN (Uni-Mol smoke_model) on all SMILES in latents.pt and
save 8-target predictions.

Output adds to the existing latents blob:
    predictions_3dcnn: (N, 8) float32  [density, DetoD, DetoP, DetoQ,
                                          DetoT, DetoV, HOF_S, BDE]
    predictions_3dcnn_valid: (N,) bool  (True where prediction succeeded)

Usage:
    python scripts/diffusion/run_3dcnn_all.py \\
        --latents data/training/diffusion/latents.pt \\
        --out     data/training/diffusion/latents_with_3dcnn.pt \\
        --batch 256

Resume-safe: if the output file exists with partial predictions it will
continue from where it stopped.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path("E:/Projects/EnergeticDiffusion2")


PROP_NAMES = ["density", "DetoD", "DetoP", "DetoQ", "DetoT", "DetoV",
              "HOF_S", "BDE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="data/training/diffusion/latents.pt")
    ap.add_argument("--out",     default="data/training/diffusion/latents_with_3dcnn.pt")
    ap.add_argument("--preds_only", default="data/training/diffusion/preds_3dcnn.pt",
                    help="Small intermediate file with just the predictions array (fast incremental saves)")
    ap.add_argument("--model",   default="data/raw/energetic_external/EMDP/Data/smoke_model")
    ap.add_argument("--batch",   type=int, default=256)
    ap.add_argument("--ckpt_every_batches", type=int, default=20)
    ap.add_argument("--reset_every_batches", type=int, default=80,
                    help="Re-instantiate MolPredict and force GC every N batches "
                         "to clear unimol_tools' internal caches (prevents the "
                         "rate from collapsing to ~7 mol/s after long runs)")
    ap.add_argument("--limit",   type=int, default=None, help="smoke test")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    latents_path = BASE / args.latents
    out_path     = BASE / args.out
    preds_path   = BASE / args.preds_only

    # lazy import to surface errors clearly
    from unimol_tools import MolPredict

    print(f"Loading {latents_path} …")
    blob = torch.load(latents_path, weights_only=False)
    smiles = blob["smiles"]
    N = len(smiles)
    if args.limit:
        smiles = smiles[:args.limit]
        N = len(smiles)
    print(f"  {N:,} molecules")

    # resume from the small preds file (fast load)
    resume_start = 0
    predictions = np.full((N, len(PROP_NAMES)), np.nan, dtype=np.float32)
    valid_mask  = np.zeros(N, dtype=bool)
    if preds_path.exists():
        prev = torch.load(preds_path, weights_only=False)
        if prev.get("n") == N:
            predictions[:] = prev["predictions"].numpy()
            valid_mask[:]  = prev["valid"].numpy()
            resume_start = int(valid_mask.argmin()) if not valid_mask.all() else N
            print(f"Resuming from row {resume_start:,} (already have {int(valid_mask.sum()):,} valid)")
        else:
            print(f"preds_only shape mismatch (n={prev.get('n')} vs {N}); starting fresh")

    if resume_start >= N:
        print("All predictions already present. Nothing to do.")
        return 0

    # Load model
    print(f"Loading 3DCNN model: {args.model}")
    model_dir = BASE / args.model

    def make_predictor():
        return MolPredict(load_model=str(model_dir))

    predictor = make_predictor()
    print("  model loaded.")

    import gc
    try:
        import torch as _torch
        cuda_clear = lambda: _torch.cuda.empty_cache() if _torch.cuda.is_available() else None
    except Exception:
        cuda_clear = lambda: None

    # Batched inference
    bs = args.batch
    t0 = time.time()
    last_reset_batch = 0
    print(f"\nRunning inference on rows {resume_start:,} .. {N-1:,} (batch {bs}) …", flush=True)
    n_processed = 0
    for i in range(resume_start, N, bs):
        batch_idx_zero_based = (i - resume_start) // bs
        if (batch_idx_zero_based > 0
                and batch_idx_zero_based - last_reset_batch >= args.reset_every_batches):
            print(f"  [reset] re-instantiating MolPredict + GC (batch {batch_idx_zero_based}) …", flush=True)
            del predictor
            gc.collect()
            cuda_clear()
            predictor = make_predictor()
            last_reset_batch = batch_idx_zero_based
            print(f"  [reset] resumed.", flush=True)
        chunk_smi = smiles[i:i+bs]
        chunk_df  = pd.DataFrame({"smiles": chunk_smi})
        try:
            out = predictor.predict(chunk_df)
            arr = np.asarray(out)
            if arr.ndim == 1:
                arr = arr[:, None]
            # clip to 8 targets (some variants may return fewer)
            ntargets = min(arr.shape[1], len(PROP_NAMES))
            predictions[i:i+len(chunk_smi), :ntargets] = arr[:, :ntargets]
            valid_mask[i:i+len(chunk_smi)] = True
        except Exception as e:
            print(f"  batch @ {i} failed: {e}")
            # mark this batch as failed (leave NaN); continue
            valid_mask[i:i+len(chunk_smi)] = False

        n_processed += len(chunk_smi)
        batch_num = (i // bs) + 1
        if batch_num % 2 == 0 or (i + bs) >= N:
            rate = n_processed / max(time.time() - t0, 1)
            eta  = (N - i - len(chunk_smi)) / max(rate, 1)
            print(f"  {i+len(chunk_smi):,}/{N:,}  rate={rate:.0f}/s  elapsed={(time.time()-t0)/60:.1f}m  eta={eta/60:.1f}m", flush=True)
        # checkpoint: fast tiny save every --ckpt_every_batches
        if batch_num % args.ckpt_every_batches == 0 or (i + bs) >= N:
            _save_preds_only(preds_path, predictions, valid_mask, N)

    print(f"\nDone. Elapsed: {(time.time()-t0)/60:.1f} min  "
          f"({N/max(time.time()-t0,1):.0f} mol/s)")

    # Merge into full blob ONCE at the end
    _save_preds_only(preds_path, predictions, valid_mask, N)
    print(f"Merging predictions into {out_path} …")
    _save_partial(out_path, blob, predictions, valid_mask)
    n_valid = int(valid_mask.sum())
    print(f"\nSaved {n_valid:,}/{N:,} predictions → {out_path}")
    print("Per-property stats (valid rows only):")
    for j, p in enumerate(PROP_NAMES):
        v = predictions[valid_mask, j]
        v = v[~np.isnan(v)]
        if len(v):
            print(f"  {p:10s}  n={len(v):,}  mean={v.mean():.3f}  median={np.median(v):.3f}  "
                  f"p5={np.percentile(v,5):.3f}  p95={np.percentile(v,95):.3f}")


def _save_preds_only(preds_path, predictions, valid_mask, N):
    """Small (~15MB) incremental checkpoint. Much faster than re-saving 1.6GB blob."""
    blob = {
        "predictions": torch.from_numpy(predictions),
        "valid":       torch.from_numpy(valid_mask),
        "property_names": PROP_NAMES,
        "n":           N,
        "n_valid":     int(valid_mask.sum()),
        "updated_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = str(preds_path) + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, preds_path)


def _save_partial(out_path, original_blob, predictions, valid_mask):
    """One-shot final save: merges predictions into original 1.6GB blob."""
    out = dict(original_blob)
    out["predictions_3dcnn"]       = torch.from_numpy(predictions)
    out["predictions_3dcnn_valid"] = torch.from_numpy(valid_mask)
    out["predictions_3dcnn_meta"] = {
        "property_names": PROP_NAMES,
        "n_valid":        int(valid_mask.sum()),
        "updated_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = str(out_path) + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    sys.exit(main())
