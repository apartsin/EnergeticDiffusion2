"""Train a time-conditional viability classifier directly in latent space.

Inputs:
    latents_trustcond.pt  - (N, 1024) z_mu values for all 382k training rows
    a SMILES list aligned with z_mu (from the latents.pt 'smiles' field)
    a trained smiles-space viability classifier (from train_viability.py)

Procedure:
    1. Score every training SMILES with the smiles-space RF -> P(viable | x)
       Use these as soft labels for the latent-space model.
    2. For multiple noise levels sigma in a curriculum, sample
            z_t = z_0 + sigma * eps,   eps ~ N(0, I)
       and train a small MLP   (z_t, sigma) -> P(viable | z_t, sigma)
       to match the SMILES-space label.

Output:
    experiments/viability_latent_v1/model.pt - state dict of the MLP
    experiments/viability_latent_v1/summary.json

Usage:
    python scripts/viability/train_viability_latent.py \
        --latents data/training/diffusion/latents_trustcond.pt \
        --smiles_rf experiments/viability_rf_v1/model.joblib \
        --out experiments/viability_latent_v1
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentViability(nn.Module):
    """MLP head over latent + scalar sigma."""
    def __init__(self, latent_dim: int = 1024, hidden: int = 256, n_blocks: int = 3,
                 sigma_emb_dim: int = 64):
        super().__init__()
        self.sig_emb = nn.Sequential(
            nn.Linear(1, sigma_emb_dim), nn.SiLU(),
            nn.Linear(sigma_emb_dim, sigma_emb_dim))
        layers = [nn.Linear(latent_dim + sigma_emb_dim, hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_blocks - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, z, sigma):
        s = self.sig_emb(sigma.view(-1, 1).float())
        h = self.trunk(torch.cat([z, s], dim=-1))
        return self.head(h)  # logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--smiles_rf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_sigma", type=int, default=10)
    ap.add_argument("--sigma_max", type=float, default=2.0)
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"Device: {args.device}")
    print(f"Loading latents from {args.latents} ...")
    blob = torch.load(args.latents, weights_only=False, map_location="cpu")
    z = blob["z_mu"].float()
    smiles = blob.get("smiles") or blob.get("smiles_list")
    if smiles is None:
        raise SystemExit("latents file missing 'smiles' field")
    print(f"  z.shape={tuple(z.shape)}  smiles={len(smiles)}")
    if args.max_rows:
        z = z[:args.max_rows]
        smiles = smiles[:args.max_rows]
        print(f"  truncated to {args.max_rows}")

    # 1) Soft labels via the SMILES-space RF
    print("Scoring training SMILES with SMILES-space RF (this takes ~5 min for 382k) ...")
    import sys; sys.path.insert(0, "scripts/viability")
    from train_viability import featurize
    import joblib
    rf = joblib.load(args.smiles_rf)
    labels = np.zeros(len(smiles), dtype=np.float32)
    skipped = 0
    t1 = time.time()
    for i, s in enumerate(smiles):
        if i % 20000 == 0 and i > 0:
            print(f"    {i}/{len(smiles)}  elapsed={time.time()-t1:.0f}s")
        f = featurize(s)
        if f is None: skipped += 1; continue
        labels[i] = rf.predict_proba(f.reshape(1, -1))[0, 1]
    print(f"  scored {len(labels)-skipped}/{len(smiles)}  skipped={skipped}  "
          f"elapsed={time.time()-t1:.0f}s")
    print(f"  label dist: mean={labels.mean():.3f}  std={labels.std():.3f}")

    # 2) Train latent classifier with noise curriculum
    z_t = torch.from_numpy(z.numpy()).to(args.device)
    y = torch.from_numpy(labels).to(args.device)

    model = LatentViability(latent_dim=z.shape[1]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = len(z_t)
    val_n = max(1000, n // 20)
    perm = torch.randperm(n)
    val_idx = perm[:val_n].to(args.device)
    tr_idx  = perm[val_n:].to(args.device)
    print(f"Training: train={len(tr_idx)} val={len(val_idx)}  "
          f"epochs={args.epochs} bs={args.batch}  n_sigma={args.n_sigma}")

    sigmas = torch.linspace(0, args.sigma_max, args.n_sigma).to(args.device)

    def step(idx, train=True):
        bs = args.batch
        order = idx[torch.randperm(len(idx), device=args.device)] if train else idx
        total = 0.0; steps = 0; correct = 0; n_total = 0
        for i in range(0, len(order), bs):
            ids = order[i:i+bs]
            z_clean = z_t[ids]
            yi = y[ids]
            # sample one sigma per batch (curriculum)
            sigma_idx = torch.randint(0, len(sigmas), (1,), device=args.device).item()
            sigma = sigmas[sigma_idx]
            noise = torch.randn_like(z_clean) * sigma
            z_in = z_clean + noise
            sigma_b = torch.full((len(ids),), sigma.item(), device=args.device)
            logits = model(z_in, sigma_b).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yi)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                p = torch.sigmoid(logits)
                correct += ((p > 0.5) == (yi > 0.5)).sum().item()
            total += loss.item() * len(ids); steps += len(ids); n_total += len(ids)
        return total / max(steps, 1), correct / max(n_total, 1)

    for ep in range(args.epochs):
        model.train()
        tr_loss, tr_acc = step(tr_idx, train=True)
        model.eval()
        with torch.no_grad():
            va_loss, va_acc = step(val_idx, train=False)
        sched.step()
        print(f"  ep {ep+1}/{args.epochs}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.3f}")

    # 3) Save
    torch.save({"state_dict": model.state_dict(),
                 "config": {"latent_dim": z.shape[1], "sigma_max": args.sigma_max}},
               out_dir / "model.pt")
    summary = {"n": int(n), "skipped_featurize": int(skipped),
               "label_mean": float(labels.mean()), "label_std": float(labels.std()),
               "final_tr_acc": float(tr_acc), "final_va_acc": float(va_acc),
               "elapsed_s": time.time() - t0,
               "epochs": args.epochs, "sigma_max": args.sigma_max}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out_dir/'model.pt'}")
    print(f"-> {out_dir/'summary.json'}")
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
