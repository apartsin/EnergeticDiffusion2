"""
Compute SA (Ertl 2009) and SC (Coley 2018) scores for every SMILES in the
cached LIMO latents file, and append scores to a new blob.

Runs entirely on CPU. Highly parallel via multiprocessing.

Usage:
    python scripts/guidance/compute_sa_sc.py \\
        --latents data/training/diffusion/latents.pt \\
        --out     data/training/diffusion/latents_with_scores.pt \\
        --workers 8
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


BASE = Path("E:/Projects/EnergeticDiffusion2")


# ── SA score setup (from LIMO's Contrib sascorer) ───────────────────────────
def _init_sa():
    """Worker-side import (so pickling works)."""
    import sys as _sys
    _sys.path.insert(0, str(BASE / "external" / "LIMO"))
    global _sascorer
    import sascorer as _sascorer
    return _sascorer


# ── SC score setup (Coley standalone numpy model) ──────────────────────────
def _init_sc():
    import sys as _sys
    _sys.path.insert(0, str(BASE / "external" / "scscore"))
    from scscore.standalone_model_numpy import SCScorer as _SCScorer
    m = _SCScorer()
    m.restore(str(BASE / "external/scscore/models/full_reaxys_model_1024bool"
                       "/model.ckpt-10654.as_numpy.json.gz"))
    return m


# worker-process-global holders (initialised once per process)
_WORKER_STATE = {"sa": None, "sc": None}


def _worker_init():
    _WORKER_STATE["sa"] = _init_sa()
    _WORKER_STATE["sc"] = _init_sc()


def _score_one(smiles: str) -> tuple[float, float]:
    """Compute (SA, SC) for a single SMILES."""
    if not isinstance(smiles, str) or not smiles:
        return (float("nan"), float("nan"))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (float("nan"), float("nan"))
    try:
        sa = float(_WORKER_STATE["sa"].calculateScore(mol))
    except Exception:
        sa = float("nan")
    try:
        _, sc = _WORKER_STATE["sc"].get_score_from_smi(smiles)
        sc = float(sc)
    except Exception:
        sc = float("nan")
    return (sa, sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--chunksize", type=int, default=64)
    ap.add_argument("--limit",  type=int, default=None, help="For smoke test")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    latents_path = Path(args.latents)
    if not latents_path.is_absolute(): latents_path = BASE / latents_path
    out_path = Path(args.out)
    if not out_path.is_absolute(): out_path = BASE / out_path

    print(f"Loading {latents_path} …")
    blob = torch.load(latents_path, weights_only=False)
    smiles = blob["smiles"]
    if args.limit:
        smiles = smiles[: args.limit]
    N = len(smiles)
    print(f"  {N:,} molecules")
    print(f"Workers: {args.workers}  chunksize: {args.chunksize}")

    # smoke check workers
    _worker_init()
    test = _score_one(smiles[0])
    print(f"  smoke (first molecule): SA={test[0]:.3f}  SC={test[1]:.3f}")

    # parallel compute
    t0 = time.time()
    sa_arr = np.full(N, np.nan, dtype=np.float32)
    sc_arr = np.full(N, np.nan, dtype=np.float32)

    # Windows: use spawn context to avoid fork issues
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_worker_init) as pool:
        done = 0
        for i, (sa, sc) in enumerate(
                pool.imap(_score_one, smiles, chunksize=args.chunksize)):
            sa_arr[i] = sa
            sc_arr[i] = sc
            done += 1
            if done % 5000 == 0 or done == N:
                rate = done / max(time.time() - t0, 1)
                eta = (N - done) / max(rate, 1)
                print(f"  {done:,}/{N:,}  rate={rate:.0f}/s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone. Elapsed: {elapsed/60:.1f} min  ({N/max(elapsed,1):.0f} mol/s)")

    # stats
    valid_sa = ~np.isnan(sa_arr); valid_sc = ~np.isnan(sc_arr)
    print(f"\nSA valid: {valid_sa.sum():,}  mean={sa_arr[valid_sa].mean():.3f}  "
          f"median={np.median(sa_arr[valid_sa]):.3f}  "
          f"p25={np.percentile(sa_arr[valid_sa], 25):.3f}  "
          f"p75={np.percentile(sa_arr[valid_sa], 75):.3f}")
    print(f"SC valid: {valid_sc.sum():,}  mean={sc_arr[valid_sc].mean():.3f}  "
          f"median={np.median(sc_arr[valid_sc]):.3f}  "
          f"p25={np.percentile(sc_arr[valid_sc], 25):.3f}  "
          f"p75={np.percentile(sc_arr[valid_sc], 75):.3f}")

    # save
    out_blob = dict(blob)
    out_blob["sa_score"] = torch.from_numpy(sa_arr)
    out_blob["sc_score"] = torch.from_numpy(sc_arr)
    out_blob["score_stats"] = {
        "sa": {"mean": float(sa_arr[valid_sa].mean()), "std": float(sa_arr[valid_sa].std())},
        "sc": {"mean": float(sc_arr[valid_sc].mean()), "std": float(sc_arr[valid_sc].std())},
    }
    out_blob["score_meta"] = {
        "sa_source":   "Ertl 2009 via RDKit Contrib (LIMO/sascorer.py)",
        "sc_source":   "Coley 2018 standalone numpy model, full_reaxys_model_1024bool",
        "n_valid_sa":  int(valid_sa.sum()),
        "n_valid_sc":  int(valid_sc.sum()),
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_blob, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
