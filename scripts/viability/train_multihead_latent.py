"""Multi-headed latent score model for sampling-time guidance.

Inputs:  experiments/latent_labels_v1.pt  (z_mu + 4 targets + perf)
Output:  experiments/score_model_v1/{model.pt, summary.json}

Architecture:
    z (1024)  ──▶ trunk ──▶ {viab, SA, SC, sens, perf} heads
    sigma         FiLM-conditioning (scale + shift on every trunk layer)

Training:
    - σ ~ U(0, σ_max=2.0) per batch (curriculum)
    - z_t = z_0 + σ * ε,  ε ~ N(0, I)
    - per-head losses normalised against the labels' empirical std
    - perf head masked by cond_valid (Tier-A/B only)
    - early stopping on validation total loss

Sample-time guidance (separate function `guide` for the sampler):
    - autograd through heads w.r.t. z_t
    - per-head gradient norm clamp
    - alpha-anneal: scale_t = scale * alpha_bar_t
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    """Linear + FiLM(sigma_emb) + SiLU + dropout."""
    def __init__(self, dim, sigma_dim, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(sigma_dim, 2*dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, s):
        gamma, beta = self.film(s).chunk(2, dim=-1)
        h = self.norm(self.lin(x))
        h = h * (1 + gamma) + beta
        return x + self.drop(F.silu(h))


class MultiHeadScoreModel(nn.Module):
    def __init__(self, latent_dim=1024, sigma_dim=128, hidden=1024,
                 n_blocks=4, dropout=0.1, n_perf=4):
        super().__init__()
        self.sig_emb = nn.Sequential(
            nn.Linear(1, sigma_dim), nn.SiLU(),
            nn.Linear(sigma_dim, sigma_dim))
        self.input_proj = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList([FiLMBlock(hidden, sigma_dim, dropout)
                                      for _ in range(n_blocks)])
        self.head_viab = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(),
                                        nn.Linear(256, 1))
        self.head_sa   = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(),
                                        nn.Linear(256, 1))
        self.head_sc   = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(),
                                        nn.Linear(256, 1))
        self.head_sens = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(),
                                        nn.Linear(256, 1))
        self.head_perf = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(),
                                        nn.Linear(256, n_perf))

    def forward(self, z, sigma):
        s = self.sig_emb(sigma.view(-1, 1).float())
        h = self.input_proj(z)
        for b in self.blocks:
            h = b(h, s)
        return {
            "viab_logit": self.head_viab(h).squeeze(-1),
            "sa":         self.head_sa(h).squeeze(-1),
            "sc":         self.head_sc(h).squeeze(-1),
            "sens":       self.head_sens(h).squeeze(-1),
            "perf":       self.head_perf(h),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",  required=True)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs",  type=int, default=12)
    ap.add_argument("--batch",   type=int, default=512)
    ap.add_argument("--lr",      type=float, default=2e-4)
    ap.add_argument("--sigma_max", type=float, default=2.0)
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"Device: {args.device}")
    print(f"Loading labels from {args.labels} ...")
    blob = torch.load(args.labels, weights_only=False, map_location="cpu")
    z = blob["z_mu"].float()
    y_viab = blob["y_viab"]
    y_sa   = blob["y_sa"]
    y_sc   = blob["y_sc"]
    y_sens = blob["y_sens"]
    perf   = blob.get("perf_target")
    perf_mask = blob.get("perf_mask")
    print(f"  z={tuple(z.shape)} viab={y_viab.shape} SA={y_sa.shape}")

    if args.max_rows:
        z = z[:args.max_rows]; y_viab = y_viab[:args.max_rows]
        y_sa = y_sa[:args.max_rows]; y_sc = y_sc[:args.max_rows]
        y_sens = y_sens[:args.max_rows]
        if perf is not None: perf = perf[:args.max_rows]
        if perf_mask is not None: perf_mask = perf_mask[:args.max_rows]

    # NaN handling: replace NaNs with 0 and use a label-mask per row per head
    sa_mask = ~torch.isnan(y_sa); y_sa[~sa_mask] = 0.0
    sc_mask = ~torch.isnan(y_sc); y_sc[~sc_mask] = 0.0
    if perf is not None:
        perf_nan_mask = ~torch.isnan(perf); perf[~perf_nan_mask] = 0.0
        if perf_mask is None: perf_mask = perf_nan_mask
        else: perf_mask = perf_mask & perf_nan_mask

    # Normalisation stats for regression heads (computed on the available labels)
    sa_mu, sa_sd   = float(y_sa[sa_mask].mean()), float(y_sa[sa_mask].std() + 1e-6)
    sc_mu, sc_sd   = float(y_sc[sc_mask].mean()), float(y_sc[sc_mask].std() + 1e-6)
    sens_mu, sens_sd = float(y_sens.mean()), float(y_sens.std() + 1e-6)
    perf_mu, perf_sd = None, None
    if perf is not None and isinstance(perf_mask, torch.Tensor) and perf_mask.any():
        # Normalize perf_mask shape: (n,) or (n, n_perf). Broadcast to (n, n_perf).
        if perf_mask.dim() == 1:
            mask2d = perf_mask.unsqueeze(-1).expand_as(perf).float()
        else:
            mask2d = perf_mask.float()
        active_count = mask2d.sum(0).clamp(min=1)        # (n_perf,)
        perf_mu = (perf * mask2d).sum(0) / active_count
        perf_sd = (((perf - perf_mu) ** 2) * mask2d).sum(0) / active_count
        perf_sd = perf_sd.sqrt() + 1e-6
        perf_mask = mask2d.bool()
    print(f"  norm:  SA(mu={sa_mu:.2f} sd={sa_sd:.2f})  SC(mu={sc_mu:.2f} sd={sc_sd:.2f})  "
          f"sens(mu={sens_mu:.3f} sd={sens_sd:.3f})")

    n = z.shape[0]
    perm = torch.randperm(n)
    val_n = max(2000, n // 20)
    val_idx = perm[:val_n]; tr_idx = perm[val_n:]
    print(f"split: train={len(tr_idx)} val={len(val_idx)}")

    z = z.to(args.device)
    y_viab = y_viab.to(args.device)
    y_sa = y_sa.to(args.device); y_sc = y_sc.to(args.device); y_sens = y_sens.to(args.device)
    sa_mask = sa_mask.to(args.device); sc_mask = sc_mask.to(args.device)
    if perf is not None:
        perf = perf.to(args.device)
        if isinstance(perf_mask, torch.Tensor):
            perf_mask = perf_mask.to(args.device)
        if perf_mu is not None:
            perf_mu = perf_mu.to(args.device); perf_sd = perf_sd.to(args.device)

    model = MultiHeadScoreModel(latent_dim=z.shape[1]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    def loss_fn(out, ids, sigma_b):
        # viability: BCE
        l_v = F.binary_cross_entropy_with_logits(out["viab_logit"], y_viab[ids])
        # SA: smooth-L1 on z-scored target
        l_sa = F.smooth_l1_loss(out["sa"], (y_sa[ids] - sa_mu) / sa_sd, reduction="none")
        l_sa = (l_sa * sa_mask[ids].float()).sum() / sa_mask[ids].float().sum().clamp(min=1)
        # SC: similar
        l_sc = F.smooth_l1_loss(out["sc"], (y_sc[ids] - sc_mu) / sc_sd, reduction="none")
        l_sc = (l_sc * sc_mask[ids].float()).sum() / sc_mask[ids].float().sum().clamp(min=1)
        # Sensitivity
        l_sens = F.smooth_l1_loss(out["sens"], (y_sens[ids] - sens_mu) / sens_sd)
        # Perf (masked)
        l_perf = torch.tensor(0.0, device=z.device)
        if perf is not None and perf_mu is not None and isinstance(perf_mask, torch.Tensor):
            target = (perf[ids] - perf_mu) / perf_sd
            l = F.smooth_l1_loss(out["perf"], target, reduction="none")
            mask_b = perf_mask[ids].float()
            if mask_b.dim() == 1: mask_b = mask_b.unsqueeze(-1)
            l_perf = (l * mask_b).sum() / mask_b.sum().clamp(min=1)
        total = l_v + l_sa + l_sc + l_sens + 0.5 * l_perf
        return total, dict(viab=l_v.item(), sa=l_sa.item(), sc=l_sc.item(),
                            sens=l_sens.item(), perf=l_perf.item())

    def step_set(idx, train=True):
        n_ = len(idx); bs = args.batch
        order = idx[torch.randperm(n_)] if train else idx
        agg = {"total": 0.0, "viab": 0.0, "sa": 0.0, "sc": 0.0, "sens": 0.0, "perf": 0.0}
        steps = 0
        for i in range(0, n_, bs):
            ids = order[i:i+bs].to(args.device)
            sigma = (torch.rand(1, device=z.device) * args.sigma_max).item()
            noise = torch.randn(len(ids), z.shape[1], device=z.device) * sigma
            z_in = z[ids] + noise
            sigma_b = torch.full((len(ids),), sigma, device=z.device)
            out = model(z_in, sigma_b)
            total, parts = loss_fn(out, ids, sigma_b)
            if train:
                opt.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            agg["total"] += total.item(); steps += 1
            for k, v in parts.items(): agg[k] += v
        for k in agg: agg[k] /= max(steps, 1)
        return agg

    best_val = float("inf")
    for ep in range(args.epochs):
        model.train(); tr = step_set(tr_idx, train=True)
        model.eval()
        with torch.no_grad():
            va = step_set(val_idx, train=False)
        sched.step()
        marker = ""
        if va["total"] < best_val:
            best_val = va["total"]
            torch.save({"state_dict": model.state_dict(),
                         "config": {"latent_dim": z.shape[1], "sigma_max": args.sigma_max,
                                    "sa_mu": sa_mu, "sa_sd": sa_sd,
                                    "sc_mu": sc_mu, "sc_sd": sc_sd,
                                    "sens_mu": sens_mu, "sens_sd": sens_sd}},
                       out_dir / "model.pt")
            marker = " *"
        print(f"  ep {ep+1}/{args.epochs}  tr_total={tr['total']:.4f}  va_total={va['total']:.4f}{marker}  "
              f"tr_viab={tr['viab']:.4f} va_viab={va['viab']:.4f} | "
              f"va_sa={va['sa']:.4f} va_sc={va['sc']:.4f} va_sens={va['sens']:.4f} va_perf={va['perf']:.4f}")

    summary = {"best_val": best_val, "epochs": args.epochs, "elapsed_s": time.time() - t0,
               "n_train": len(tr_idx), "n_val": len(val_idx)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out_dir/'model.pt'}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
