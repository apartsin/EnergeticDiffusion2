"""Train time-conditional SA + SC surrogates for classifier guidance.

f(z_t, t) → standardised score, where z_t = q_sample(z_mu, t) on a randomly
sampled t per row per epoch. This matches the surrogate's training
distribution to the noisy z the diffusion sampler actually sees, fixing
the noisy-z mismatch that broke the t=0 surrogates.

Output: data/training/guidance/property_heads_t.pt
        contains state_dicts and stats for sa, sc — both time-conditional.
"""
from __future__ import annotations
import sys, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path("E:/Projects/EnergeticDiffusion2")
sys.path.insert(0, str(BASE / "scripts/diffusion"))
from model import NoiseSchedule


class TimeAwareScorePredictor(nn.Module):
    """MLP over (z_t, sinusoidal time embedding) → standardised score."""
    def __init__(self, z_dim: int = 1024, time_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.time_dim = time_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, time_dim))
        self.net = nn.Sequential(
            nn.Linear(z_dim + time_dim, hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1))

    def time_emb(self, t):
        # sinusoidal positional embedding
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000.0) *
                            torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, z_t, t):
        emb = self.t_mlp(self.time_emb(t))
        x = torch.cat([z_t, emb], dim=-1)
        return self.net(x).squeeze(-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    blob = torch.load(BASE / "data/training/diffusion/latents_with_scores.pt",
                       weights_only=False)
    z = blob["z_mu"].float().to(device)
    sa = blob["sa_score"].float().to(device)
    sc = blob["sc_score"].float().to(device)
    N = z.shape[0]
    print(f"N={N:,} z_dim={z.shape[1]}")

    # standardise targets
    sa_mu, sa_sd = float(sa.mean().item()), float(sa.std().item() + 1e-6)
    sc_mu, sc_sd = float(sc.mean().item()), float(sc.std().item() + 1e-6)
    sa_std = (sa - sa_mu) / sa_sd
    sc_std = (sc - sc_mu) / sc_sd

    schedule = NoiseSchedule(T=1000, device=device)
    rng = np.random.default_rng(42)
    val_idx = rng.choice(N, 5000, replace=False)
    val_mask = np.zeros(N, dtype=bool); val_mask[val_idx] = True
    tr_idx = np.where(~val_mask)[0]
    print(f"train={len(tr_idx):,} val={len(val_idx):,}")

    out_dir = BASE / "data/training/guidance"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "sa": {"mu": sa_mu, "sd": sa_sd, "state_dict": None,
                "val_mae": None, "val_r2": None},
        "sc": {"mu": sc_mu, "sd": sc_sd, "state_dict": None,
                "val_mae": None, "val_r2": None},
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schedule_T": 1000,
    }

    bs = 256
    n_steps = 8000

    for label, target_std, mu, sd in [("sa", sa_std, sa_mu, sa_sd),
                                          ("sc", sc_std, sc_mu, sc_sd)]:
        print(f"\n=== training time-aware {label} surrogate ===")
        m = TimeAwareScorePredictor(z_dim=z.shape[1]).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
        t0 = time.time()
        for step in range(n_steps):
            i = torch.from_numpy(rng.choice(tr_idx, bs, replace=False)).long().to(device)
            t = torch.randint(0, schedule.T, (bs,), device=device)
            with torch.no_grad():
                z_t, _ = schedule.q_sample(z[i], t)
            pred = m(z_t, t)
            loss = F.mse_loss(pred, target_std[i])
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 1000 == 0:
                print(f"  step {step:>5d}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")
        # eval on held-out val (sample t per row)
        m.eval()
        with torch.no_grad():
            i = torch.from_numpy(val_idx).long().to(device)
            t = torch.randint(0, schedule.T, (len(val_idx),), device=device)
            z_t, _ = schedule.q_sample(z[i], t)
            pred = m(z_t, t)
            mae = (pred - target_std[i]).abs().mean().item()
            yv = target_std[i].cpu().numpy(); pv = pred.cpu().numpy()
            r = float(np.corrcoef(yv, pv)[0, 1])
            r2 = r*r
        print(f"  val MAE(std)={mae:.3f}  r={r:.3f}  r²={r2:.3f}  total={time.time()-t0:.0f}s")
        bundle[label]["state_dict"] = {k: v.cpu() for k, v in m.state_dict().items()}
        bundle[label]["val_mae"]  = mae
        bundle[label]["val_r2"]   = r2

    out = out_dir / "property_heads_t.pt"
    torch.save(bundle, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    sys.exit(main())
