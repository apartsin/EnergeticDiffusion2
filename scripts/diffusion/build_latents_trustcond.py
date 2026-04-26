"""Build a Tier-A/B-only-conditioning variant of latents_expanded.pt.

The idea: the diffusion *prior* still learns from all 382k rows, but the
*conditioning signal* only fires on rows whose values are Tier-A/B trusted.
Mechanism: zero out cond_valid for any (row, prop) where cond_weight < 0.99.
With weighted_mask=True, those entries become ineligible for the random
subset sampling, so they only contribute to unconditional pretraining.

Output: latents_trustcond.pt with same shapes, only cond_valid changed.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp",  default="data/training/diffusion/latents_expanded.pt")
    ap.add_argument("--out",  default="data/training/diffusion/latents_trustcond.pt")
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--threshold", type=float, default=0.99,
                    help="Min cond_weight to keep (1.0 == Tier-A/B; 0.7 == Tier-D)")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    inp = base / args.inp
    out = base / args.out
    print(f"Loading {inp}")
    blob = torch.load(inp, weights_only=False)
    cw = blob["cond_weight"]
    cv_old = blob["cond_valid"]
    cv_new = cv_old & (cw >= args.threshold)
    print("\nPer-property valid counts:")
    print(f"  {'prop':25s}  {'old':>8s}  {'new':>8s}  {'kept_pct':>8s}")
    for j, p in enumerate(blob["property_names"]):
        a = int(cv_old[:, j].sum())
        b = int(cv_new[:, j].sum())
        print(f"  {p:25s}  {a:>8,d}  {b:>8,d}  {100*b/max(a,1):>7.1f}%")
    blob["cond_valid"] = cv_new

    # Recompute stats from these trusted rows (likely identical to before but
    # explicit for safety)
    raw = blob["values_raw"].numpy() if hasattr(blob["values_raw"], "numpy") else blob["values_raw"]
    import numpy as np
    cv_np = cv_new.numpy().astype(bool)
    new_stats = {}
    new_norm = np.zeros_like(raw, dtype=np.float32)
    for j, p in enumerate(blob["property_names"]):
        v = raw[cv_np[:, j], j]
        if len(v) < 5:
            print(f"  WARN: only {len(v)} trusted rows for {p}")
            mu = float(blob["stats"][p]["mean"])
            sd = float(blob["stats"][p]["std"])
        else:
            mu = float(np.mean(v))
            sd = float(np.std(v))
        new_stats[p] = {"mean": mu, "std": sd, "count": int(len(v))}
        new_norm[:, j] = (raw[:, j] - mu) / (sd + 1e-9)
    new_norm = np.where(np.isnan(new_norm), 0.0, new_norm)
    blob["stats"] = new_stats
    blob["values_norm"] = torch.from_numpy(new_norm)
    blob["trustcond_meta"] = {
        "source": str(inp),
        "threshold": args.threshold,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"\nSaving {out}")
    tmp = str(out) + ".tmp"
    torch.save(blob, tmp)
    import os; os.replace(tmp, out)
    print("Done.")

if __name__ == "__main__":
    sys.exit(main())
