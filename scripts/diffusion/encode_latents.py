"""
One-shot: encode the 326k energetic-biased SMILES with the frozen fine-tuned
LIMO VAE and save latents + property values + tier info + conditioning mask.

Outputs a single .pt cache with everything the diffusion training loop needs:

    latents.pt:
        z_mu         (N, 1024)  float32   - deterministic latent mean from LIMO
        values_raw   (N, 4)     float32   - raw property values (NaN where missing)
        values_norm  (N, 4)     float32   - standardized values (NaN where missing)
        tiers        (N, 4)     int8      - tier codes 0=A 1=B 2=C 3=D 4=missing
        cond_valid   (N, 4)     bool      - True if tier in {A, B} (usable for conditioning)
        stats        dict                 - per-property {mean, std, count}
        property_names list               - ['density','heat_of_formation','detonation_velocity','detonation_pressure']
        smiles       list                 - canonical SMILES per row
        meta         dict                 - checkpoint path, timestamp, etc.

Usage:
    python scripts/diffusion/encode_latents.py \
        --ckpt experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt \
        --out data/training/diffusion/latents.pt
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import selfies as sf
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).parent.parent / "vae"))
from limo_model import (
    LIMOVAE, SELFIESTokenizer, load_vocab, build_limo_vocab, save_vocab,
    LIMO_MAX_LEN, find_limo_repo,
)


PROPS = ["density", "heat_of_formation", "detonation_velocity", "detonation_pressure"]
TIER_CODE = {"A": 0, "B": 1, "C": 2, "D": 3}   # 4 = missing


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def build_energetic_rows(base: Path) -> pd.DataFrame:
    """Return one row per unique canonical SMILES from the energetic-biased
    subset, with merged property values and per-property tier from labeled_master.
    Unlabeled-only rows get NaN properties and tier=4 (missing).
    """
    lm = pd.read_csv(base / "data/training/master/labeled_master.csv",
                     low_memory=False,
                     usecols=["smiles"] + PROPS + [f"{p}_tier" for p in PROPS])
    um = pd.read_csv(base / "data/training/master/unlabeled_master.csv",
                     low_memory=False,
                     usecols=["smiles", "source_dataset", "has_nitro",
                              "has_azide", "energetic_proxy_score"])
    um["has_nitro"] = um["has_nitro"].astype(str).str.lower().isin(["true", "1"])
    um["has_azide"] = um["has_azide"].astype(str).str.lower().isin(["true", "1"])

    # energetic-biased unlabeled mask
    mask = (
        (um["source_dataset"] == "rnnmgm_ds9") |
        (um["energetic_proxy_score"] >= 6) |
        (um["has_nitro"]) |
        (um["has_azide"])
    )
    um_sub = um.loc[mask, ["smiles"]].copy()
    for p in PROPS:
        um_sub[p] = np.nan
        um_sub[f"{p}_tier"] = None

    combined = pd.concat([lm, um_sub], ignore_index=True)
    # drop duplicates — prefer labeled (first occurrence wins since lm is first)
    combined = combined.drop_duplicates("smiles", keep="first")
    combined = combined.dropna(subset=["smiles"])
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True, help="LIMO fine-tuned best.pt path")
    ap.add_argument("--out",    required=True, help="Output .pt cache path")
    ap.add_argument("--base",   default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch",  type=int, default=256)
    ap.add_argument("--max_rows", type=int, default=None,
                    help="For smoke testing; process only this many rows")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    base = Path(args.base)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = base / ckpt_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device:    {args.device}")
    print(f"Checkpoint:{ckpt_path}")
    print(f"Output:    {out_path}")

    # ── load model ───────────────────────────────────────────────────────────
    print("Loading LIMO tokenizer …")
    limo_dir = find_limo_repo(base)
    vocab_cache = limo_dir / "vocab_cache.json"
    if vocab_cache.exists():
        alphabet = load_vocab(vocab_cache)
    else:
        alphabet = build_limo_vocab(limo_dir / "zinc250k.smi")
        save_vocab(alphabet, vocab_cache)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)

    print(f"Loading fine-tuned checkpoint …")
    blob = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model = LIMOVAE()
    model.load_state_dict(blob["model_state"], strict=True)
    model.to(args.device).eval()
    print(f"  step={blob['runtime']['global_step']}  best_val={blob['runtime']['best_val']:.4f}")

    # ── build dataset ────────────────────────────────────────────────────────
    print("\nBuilding energetic-biased dataset …")
    df = build_energetic_rows(base)
    print(f"  {len(df):,} unique SMILES before canonicalization")

    print("Canonicalizing SMILES …")
    df["canon"] = df["smiles"].apply(canon)
    df = df.dropna(subset=["canon"]).drop_duplicates("canon")
    print(f"  {len(df):,} unique canonical")

    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"  SMOKE: limited to {len(df):,} rows")

    # ── compute per-property standardization stats (from Tier A+B only) ──────
    stats = {}
    for p in PROPS:
        tc = f"{p}_tier"
        clean = df[df[tc].isin(["A", "B"])][p].dropna()
        if len(clean) >= 10:
            mu, sd = float(clean.mean()), float(clean.std())
            if sd < 1e-6: sd = 1.0
        else:
            mu, sd = 0.0, 1.0
        stats[p] = {"mean": mu, "std": sd, "count": int(len(clean))}
        print(f"  {p}: n={len(clean):,}  mean={mu:.3f}  std={sd:.3f}")

    # ── encode latents ───────────────────────────────────────────────────────
    print("\nTokenizing + encoding …")
    t0 = time.time()
    all_z = []
    all_vals_raw  = []
    all_vals_norm = []
    all_tiers     = []
    all_valid     = []
    all_smi       = []

    smiles_list = df["canon"].tolist()
    N = len(smiles_list)

    # pre-tokenize in one pass
    tokens = []
    keep_idx = []
    for i, smi in enumerate(smiles_list):
        t = tok.smiles_to_tensor(smi)
        if t is None:
            continue
        tokens.append(t[0])
        keep_idx.append(i)
    df_k = df.iloc[keep_idx].reset_index(drop=True)
    X = torch.stack(tokens)
    print(f"  tokenized {len(X):,}  ({time.time()-t0:.1f}s)")

    # encode in batches with mixed precision
    t1 = time.time()
    z_chunks = []
    from torch.cuda.amp import autocast
    use_amp = args.device == "cuda"
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            batch = X[i:i+args.batch].to(args.device)
            if len(batch) == 1:  # BN requires >=2 in eval path? generally ok but guard
                batch = torch.cat([batch, batch])
                with autocast(enabled=use_amp):
                    _z, mu, _lv = model.encode(batch)
                z_chunks.append(mu[:1].cpu().float())
            else:
                with autocast(enabled=use_amp):
                    _z, mu, _lv = model.encode(batch)
                z_chunks.append(mu.cpu().float())
            if (i // args.batch) % 50 == 0:
                pct = 100 * i / len(X)
                print(f"    encoding: {i:,}/{len(X):,}  ({pct:.1f}%)  "
                      f"elapsed={time.time()-t1:.0f}s")
    z_mu = torch.cat(z_chunks, dim=0).numpy()
    print(f"  encoded  {len(z_mu):,}  ({time.time()-t1:.1f}s  → "
          f"{len(z_mu)/max(time.time()-t1,1):.0f} mol/sec)")

    # ── collect property arrays ──────────────────────────────────────────────
    vals_raw  = np.full((len(df_k), 4), np.nan, dtype=np.float32)
    vals_norm = np.full((len(df_k), 4), np.nan, dtype=np.float32)
    tiers     = np.full((len(df_k), 4), 4, dtype=np.int8)  # 4 = missing
    valid     = np.zeros((len(df_k), 4), dtype=bool)
    for j, p in enumerate(PROPS):
        tc = f"{p}_tier"
        vals = df_k[p].to_numpy()
        tier_col = df_k[tc].astype(object).tolist()
        for i in range(len(df_k)):
            if not np.isnan(vals[i]):
                vals_raw[i, j] = vals[i]
                vals_norm[i, j] = (vals[i] - stats[p]["mean"]) / stats[p]["std"]
            t_str = tier_col[i]
            if isinstance(t_str, str) and t_str in TIER_CODE:
                tiers[i, j] = TIER_CODE[t_str]
                # conditioning validity: Tier A or B only
                if t_str in ("A", "B") and not np.isnan(vals[i]):
                    valid[i, j] = True

    # ── save ─────────────────────────────────────────────────────────────────
    blob_out = {
        "z_mu":           torch.from_numpy(z_mu),                 # (N, 1024)
        "values_raw":     torch.from_numpy(vals_raw),             # (N, 4)
        "values_norm":    torch.from_numpy(vals_norm),            # (N, 4)
        "tiers":          torch.from_numpy(tiers),                # (N, 4)
        "cond_valid":     torch.from_numpy(valid),                # (N, 4)
        "stats":          stats,
        "property_names": PROPS,
        "smiles":         df_k["canon"].tolist(),
        "meta": {
            "checkpoint":   str(ckpt_path),
            "ckpt_step":    int(blob["runtime"]["global_step"]),
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_molecules":  int(len(z_mu)),
            "latent_dim":   int(z_mu.shape[1]),
        },
    }
    torch.save(blob_out, out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")

    # summary print
    print("\nConditioning inventory (valid = Tier A or B + value present):")
    for j, p in enumerate(PROPS):
        n_valid = int(valid[:, j].sum())
        print(f"  {p:28s} {n_valid:>7,} / {len(z_mu):,} valid")


if __name__ == "__main__":
    main()
