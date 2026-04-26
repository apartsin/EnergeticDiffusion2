"""Filter latents_expanded.pt to rows whose SMILES carry energetic motifs.

Keeps the same dict structure but with N' rows (N' < N). Recomputes property
z-score stats from the filtered set's *Tier-A/B* rows (so the conditioning
target scale is grounded in real energetic chemistry, not the broader pool).

Usage:
    python scripts/diffusion/filter_energetic.py \
        --in  data/training/diffusion/latents_expanded.pt \
        --out data/training/diffusion/latents_v4_filtered.pt
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# SMARTS that mark an energetic-material candidate. ANY match keeps the row.
ENERGETIC_SMARTS = [
    "[N+](=O)[O-]",         # nitro / nitrate ester / nitramine all share this
    "[N-]=[N+]=N",          # azide (anionic representation)
    "N=[N+]=[N-]",           # azide (alt charge)
    "N(=O)=O",              # nitro (neutral form occasionally seen)
    "c1nnnn1",              # tetrazole (aromatic)
    "C1=NN=NN1",            # tetrazole (kekulized)
    "c1nonc1",              # furazan / furoxan (1,2,5-oxadiazole)
    "c1nnoc1",              # 1,2,5-oxadiazole isomer
    "c1ncnc1",              # imidazole / pyrimidine ring (proxy for N-rich)
    "c1ncncn1",             # triazine
    "[#7]-[#7]-[#7]",       # N-N-N chain (azide-like)
    "C(=O)O[N+](=O)[O-]",   # nitrate ester carbonate
    "[NX3]([N+](=O)[O-])",  # nitramine N-NO2
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",
                    default="data/training/diffusion/latents_expanded.pt")
    ap.add_argument("--out", default="data/training/diffusion/latents_v4_filtered.pt")
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--require_motif", action="store_true", default=True)
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    inp = base / args.inp
    out = base / args.out

    print(f"Loading {inp} …")
    blob = torch.load(inp, weights_only=False)
    smiles = blob["smiles"]
    N = len(smiles)
    print(f"  N = {N:,}")

    # Compile SMARTS once
    patterns = []
    for smarts in ENERGETIC_SMARTS:
        p = Chem.MolFromSmarts(smarts)
        if p is None:
            print(f"  WARN: failed to parse SMARTS {smarts!r}")
        else:
            patterns.append(p)

    print(f"Scanning {N:,} SMILES against {len(patterns)} SMARTS patterns …")
    keep = np.zeros(N, dtype=bool)
    t0 = time.time()
    fail = 0
    for i, smi in enumerate(smiles):
        if i and i % 50000 == 0:
            rate = i / (time.time() - t0)
            print(f"  {i:>7,}/{N:,}  rate={rate:.0f}/s  kept_so_far={int(keep[:i].sum()):,}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fail += 1
            continue
        for p in patterns:
            if mol.HasSubstructMatch(p):
                keep[i] = True
                break
    print(f"  done in {(time.time()-t0)/60:.1f} min")
    print(f"  parse failures: {fail:,}")
    print(f"  kept: {int(keep.sum()):,} / {N:,} ({100*keep.sum()/N:.1f} %)")

    # Always keep all Tier-A/B rows even if SMARTS missed (some experimental
    # energetics are exotic, e.g. all-nitrogen frames without explicit NO2).
    cw = blob["cond_weight"].numpy()
    cv = blob["cond_valid"].numpy().astype(bool)
    trusted_any = ((cw >= 0.99) & cv).any(axis=1)
    rescued = int((trusted_any & ~keep).sum())
    keep |= trusted_any
    print(f"  rescued {rescued:,} Tier-A/B rows lacking SMARTS hit")
    print(f"  final kept: {int(keep.sum()):,} ({100*keep.sum()/N:.1f} %)")

    idx = np.where(keep)[0]

    # Subset every aligned tensor
    new_blob = {}
    for k, v in blob.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == N:
            new_blob[k] = v[idx].contiguous()
        elif isinstance(v, list) and len(v) == N:
            new_blob[k] = [v[i] for i in idx.tolist()]
        elif isinstance(v, np.ndarray) and v.shape and v.shape[0] == N:
            new_blob[k] = v[idx]
        else:
            new_blob[k] = v   # scalars / dicts / metadata

    # Recompute z-score stats from the FILTERED Tier-A/B rows so target z-scores
    # at sample time refer to the real energetic distribution, not the broader pool.
    raw = new_blob["values_raw"].numpy()
    cv_new = new_blob["cond_valid"].numpy().astype(bool)
    cw_new = new_blob["cond_weight"].numpy()
    trusted = (cw_new >= 0.99) & cv_new   # Tier-A/B only
    new_stats = {}
    new_norm = np.zeros_like(raw, dtype=np.float32)
    for j, p in enumerate(new_blob["property_names"]):
        v = raw[trusted[:, j], j]
        if len(v) < 10:
            print(f"  WARN: only {len(v)} trusted rows for {p}, falling back to all valid")
            v = raw[cv_new[:, j], j]
        mu = float(np.mean(v))
        sd = float(np.std(v))
        new_stats[p] = {"mean": mu, "std": sd, "count": int(len(v))}
        col = raw[:, j]
        new_norm[:, j] = (col - mu) / (sd + 1e-9)
        print(f"  stats[{p}]: mean={mu:.4f} std={sd:.4f} n={len(v)}")
    new_blob["stats"] = new_stats
    new_blob["values_norm"] = torch.from_numpy(new_norm)

    # Sanitize NaN→0 in values_norm so the dataset wrapper doesn't have to
    new_blob["values_norm"] = torch.where(
        torch.isnan(new_blob["values_norm"]),
        torch.zeros_like(new_blob["values_norm"]),
        new_blob["values_norm"]
    )

    new_blob["filter_meta"] = {
        "source":         str(inp),
        "n_before":       N,
        "n_after":        int(keep.sum()),
        "smarts":         ENERGETIC_SMARTS,
        "filtered_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stats_recomputed_from": "filtered Tier-A/B rows only",
    }

    print(f"\nSaving {out} …")
    tmp = str(out) + ".tmp"
    torch.save(new_blob, tmp)
    import os; os.replace(tmp, out)
    print(f"Done. New blob has {int(keep.sum()):,} rows.")


if __name__ == "__main__":
    sys.exit(main())
