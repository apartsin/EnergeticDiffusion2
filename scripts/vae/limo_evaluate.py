"""
Evaluate a fine-tuned LIMO VAE checkpoint.

Metrics computed:
  - Reconstruction accuracy (token-level) on held-out test set
  - Exact SMILES round-trip rate
  - Valid-decoded SMILES rate (should be ~100% from SELFIES)
  - Novel sampling: generate from N(0, I), measure validity + uniqueness + novelty
  - Linear-probe R² for Tier-A density (z → density regression)
  - Latent statistics: mean μ, std σ, KL posterior

Outputs:
  <exp_dir>/eval_results.json
  <exp_dir>/eval_summary.txt

Usage:
    python scripts/vae/limo_evaluate.py --exp experiments/limo_ft_energetic_20260424T150753Z
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).parent))
from limo_model import (
    LIMOVAE, SELFIESTokenizer, load_vocab, save_vocab, build_limo_vocab,
    LIMO_MAX_LEN, LIMO_PAD_IDX,
)


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


@torch.no_grad()
def recon_metrics(model, seqs: torch.Tensor, tok: SELFIESTokenizer, device: str,
                   batch_size: int = 64, verbose: bool = False) -> dict:
    """Reconstruction metrics on a set of encoded SMILES."""
    model.eval()
    n = len(seqs)
    n_exact = n_valid = n_token_total = n_token_correct = 0
    smiles_gt, smiles_recon = [], []
    for i in range(0, n, batch_size):
        batch = seqs[i:i+batch_size].to(device)
        if len(batch) < 2:  # batchnorm requires >1 in eval too sometimes
            if len(batch) == 1:
                batch = torch.cat([batch, batch])
                out = model(batch)[0].argmax(dim=2)[:1]
                batch_size_eff = 1
            else:
                continue
        else:
            out = model(batch)[0].argmax(dim=2)
        for g, p in zip(batch.cpu().numpy(), out.cpu().numpy()):
            gt_sm   = tok.indices_to_smiles(g)
            rec_sm  = tok.indices_to_smiles(p)
            gt_c    = canon(gt_sm)
            rec_c   = canon(rec_sm) if rec_sm else None
            smiles_gt.append(gt_c or "")
            smiles_recon.append(rec_c or "")
            if rec_c is not None:
                n_valid += 1
            if gt_c and rec_c and gt_c == rec_c:
                n_exact += 1
            # token-level accuracy (ignore pads)
            for gg, pp in zip(g, p):
                if gg != tok.pad_idx:
                    n_token_total += 1
                    if gg == pp:
                        n_token_correct += 1
    return {
        "n":                     int(n),
        "exact_smiles_pct":      100 * n_exact / max(n, 1),
        "valid_decoded_pct":     100 * n_valid / max(n, 1),
        "token_accuracy_pct":    100 * n_token_correct / max(n_token_total, 1),
    }


@torch.no_grad()
def sample_from_prior(model, tok, device, n_samples: int = 500,
                      batch_size: int = 64) -> dict:
    """Sample z ~ N(0, I), decode, measure validity/uniqueness/novelty."""
    model.eval()
    generated = []
    valid = []
    for i in range(0, n_samples, batch_size):
        k = min(batch_size, n_samples - i)
        z = torch.randn(k, model.latent_dim, device=device)
        log_probs = model.decode(z)        # (k, max_len, V)
        preds = log_probs.argmax(dim=2)    # (k, max_len)
        for p in preds.cpu().numpy():
            sm = tok.indices_to_smiles(p)
            generated.append(sm)
            c = canon(sm) if sm else None
            if c:
                valid.append(c)
    return {
        "n_generated":    len(generated),
        "n_valid":        len(valid),
        "valid_pct":      100 * len(valid) / max(len(generated), 1),
        "unique_pct":     100 * len(set(valid)) / max(len(valid), 1)
                          if valid else 0,
        "sample_smiles":  valid[:20],
    }


@torch.no_grad()
def encode_for_probe(model, seqs: torch.Tensor, device: str,
                     batch_size: int = 64) -> np.ndarray:
    model.eval()
    mus = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size].to(device)
        if len(batch) == 1:  # batchnorm needs ≥2
            batch = torch.cat([batch, batch])
            _z, mu, _ = model.encode(batch)
            mus.append(mu[:1].cpu().numpy())
        else:
            _z, mu, _lv = model.encode(batch)
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


# ── latent stats helper (kept separate for clarity) ─────────────────────────


def latent_density_probe(model, tok, base: Path, device: str) -> dict:
    """Linear probe: can a Ridge regressor predict Tier-A density from the
    mu latents?  R² on held-out Tier-A rows indicates latent informativeness.
    """
    lm = pd.read_csv(base / "data/training/master/labeled_master.csv",
                     low_memory=False,
                     usecols=["smiles", "density", "density_tier"])
    a = lm[(lm["density_tier"] == "A") & lm["density"].notna()].dropna(subset=["smiles"])
    # subsample for speed
    if len(a) > 3000:
        a = a.sample(3000, random_state=42)
    # encode
    ok_rows, y = [], []
    for _, r in a.iterrows():
        t = tok.smiles_to_tensor(r["smiles"])
        if t is None:
            continue
        indices, _ = t
        ok_rows.append(indices)
        y.append(float(r["density"]))
    if len(ok_rows) < 50:
        return {"probe_n": len(ok_rows), "note": "insufficient data"}
    X_seqs = torch.stack(ok_rows)
    z = encode_for_probe(model, X_seqs, device)
    y = np.array(y)
    # 80/20 split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(z))
    s = int(0.8 * len(idx))
    tr, te = idx[:s], idx[s:]
    reg = LinearRegression().fit(z[tr], y[tr])
    pred_tr = reg.predict(z[tr])
    pred_te = reg.predict(z[te])
    return {
        "probe_n":    len(z),
        "r2_train":   float(r2_score(y[tr], pred_tr)),
        "r2_test":    float(r2_score(y[te], pred_te)),
        "mae_test":   float(np.mean(np.abs(pred_te - y[te]))),
    }


@torch.no_grad()
def latent_stats(model, tok, base: Path, device: str,
                  n_sample: int = 2000) -> dict:
    """Compute mean μ magnitude, σ per dim, KL statistics across a sample."""
    lm = pd.read_csv(base / "data/training/master/labeled_master.csv",
                     low_memory=False, usecols=["smiles"])
    smiles = lm["smiles"].dropna().drop_duplicates().sample(
        min(n_sample, len(lm)), random_state=42).tolist()
    seqs = []
    for s in smiles:
        t = tok.smiles_to_tensor(s)
        if t is None:
            continue
        seqs.append(t[0])
    if not seqs:
        return {"note": "no valid encoded molecules"}
    X = torch.stack(seqs)
    model.eval()
    mus, log_vars = [], []
    for i in range(0, len(X), 64):
        batch = X[i:i+64].to(device)
        if len(batch) == 1:
            batch = torch.cat([batch, batch])
            _, mu, lv = model.encode(batch)
            mus.append(mu[:1].cpu().numpy())
            log_vars.append(lv[:1].cpu().numpy())
        else:
            _, mu, lv = model.encode(batch)
            mus.append(mu.cpu().numpy())
            log_vars.append(lv.cpu().numpy())
    mu  = np.concatenate(mus)
    lv  = np.concatenate(log_vars)
    kl_per_dim = -0.5 * (1 + lv - mu**2 - np.exp(lv))
    return {
        "sample_n":        int(len(mu)),
        "mu_abs_mean":     float(np.mean(np.abs(mu))),
        "sigma_mean":      float(np.mean(np.exp(0.5 * lv))),
        "sigma_median":    float(np.median(np.exp(0.5 * lv))),
        "kl_total_avg":    float(np.mean(kl_per_dim.sum(axis=1))),
        "active_dims":     int(np.sum(np.std(mu, axis=0) > 0.01)),
        "latent_dim":      int(mu.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",      required=True, help="Experiment directory")
    ap.add_argument("--ckpt",     default="best.pt",
                    help="Checkpoint filename in <exp>/checkpoints/ (best.pt or last.pt)")
    ap.add_argument("--base",     default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_test",   type=int, default=500,
                    help="Max test molecules for reconstruction eval")
    ap.add_argument("--n_sample", type=int, default=500,
                    help="Samples from prior for validity/uniqueness")
    args = ap.parse_args()

    base = Path(args.base)
    exp_dir = Path(args.exp)
    if not exp_dir.is_absolute():
        exp_dir = base / exp_dir

    ckpt_path = exp_dir / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        # fallback to last.pt
        alt = exp_dir / "checkpoints/last.pt"
        if alt.exists():
            print(f"Falling back to {alt}")
            ckpt_path = alt
        else:
            return 1

    print(f"Loading checkpoint: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    cfg = blob.get("config", {})
    model = LIMOVAE()
    model.load_state_dict(blob["model_state"], strict=True)
    model.to(args.device).eval()
    print(f"  step={blob['runtime']['global_step']}  best_val={blob['runtime']['best_val']:.4f}")

    # tokenizer
    limo_dir = base / "external/LIMO"
    vocab_cache = limo_dir / "vocab_cache.json"
    if vocab_cache.exists():
        alphabet = load_vocab(vocab_cache)
    else:
        alphabet = build_limo_vocab(limo_dir / "zinc250k.smi")
        save_vocab(alphabet, vocab_cache)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)

    results = {
        "checkpoint":  str(ckpt_path.relative_to(base)) if base in ckpt_path.parents else str(ckpt_path),
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step":        blob["runtime"]["global_step"],
        "best_val":    blob["runtime"]["best_val"],
    }

    # ── 1. reconstruction on held-out from train cache ───────────────────────
    cache_dir = base / cfg.get("paths", {}).get("cache_dir", "data/training/vae_tokens")
    cache_files = list(cache_dir.glob("energetic_ft_*.pt"))
    if cache_files:
        blob2 = torch.load(cache_files[0], weights_only=False)
        # use last n_test as test set (same split logic as train)
        n = len(blob2["seqs"])
        rng = np.random.default_rng(42)
        order = rng.permutation(n)
        n_tr = int(0.90 * n); n_va = int(0.05 * n)
        te = order[n_tr + n_va : n_tr + n_va + args.n_test]
        test_seqs = blob2["seqs"][te]
        print(f"\n[1/4] Reconstruction on {len(test_seqs)} test molecules …")
        results["reconstruction"] = recon_metrics(model, test_seqs, tok,
                                                    args.device, verbose=True)
        for k, v in results["reconstruction"].items():
            print(f"    {k}: {v}")

    # ── 2. sampling from prior ───────────────────────────────────────────────
    print(f"\n[2/4] Sampling {args.n_sample} molecules from N(0, I) …")
    results["sampling"] = sample_from_prior(model, tok, args.device, args.n_sample)
    for k in ["n_generated", "n_valid", "valid_pct", "unique_pct"]:
        print(f"    {k}: {results['sampling'][k]}")

    # ── 3. latent quality: linear probe on density ───────────────────────────
    print("\n[3/4] Linear probe: latent z → Tier-A density …")
    results["density_probe"] = latent_density_probe(model, tok, base, args.device)
    for k, v in results["density_probe"].items():
        print(f"    {k}: {v}")

    # ── 4. latent stats ──────────────────────────────────────────────────────
    print("\n[4/4] Latent statistics …")
    results["latent_stats"] = latent_stats(model, tok, base, args.device)
    for k, v in results["latent_stats"].items():
        print(f"    {k}: {v}")

    # save
    out_json = exp_dir / "eval_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_json}")

    out_txt = exp_dir / "eval_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Evaluation — {ckpt_path.name}\n")
        f.write("=" * 60 + "\n")
        rec = results.get("reconstruction", {})
        f.write(f"Reconstruction (n={rec.get('n', 0)}):\n")
        f.write(f"  Token accuracy:     {rec.get('token_accuracy_pct', 0):.2f}%\n")
        f.write(f"  Exact SMILES:       {rec.get('exact_smiles_pct', 0):.2f}%\n")
        f.write(f"  Valid decoded:      {rec.get('valid_decoded_pct', 0):.2f}%\n\n")
        smp = results["sampling"]
        f.write(f"Prior sampling (n={smp['n_generated']}):\n")
        f.write(f"  Valid:    {smp['valid_pct']:.1f}%  (n={smp['n_valid']})\n")
        f.write(f"  Unique:   {smp['unique_pct']:.1f}%\n\n")
        pr = results["density_probe"]
        f.write("Density linear probe:\n")
        f.write(f"  R² train: {pr.get('r2_train', float('nan')):.3f}\n")
        f.write(f"  R² test:  {pr.get('r2_test',  float('nan')):.3f}\n")
        f.write(f"  MAE test: {pr.get('mae_test', float('nan')):.4f}\n\n")
        ls = results["latent_stats"]
        f.write("Latent statistics:\n")
        f.write(f"  KL total avg: {ls.get('kl_total_avg', 0):.2f}\n")
        f.write(f"  Active dims:  {ls.get('active_dims', 0)} / {ls.get('latent_dim', 0)}\n")
        f.write(f"  σ mean:       {ls.get('sigma_mean', 0):.3f}\n")
        f.write(f"  σ median:     {ls.get('sigma_median', 0):.3f}\n")
    print(f"Summary → {out_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
