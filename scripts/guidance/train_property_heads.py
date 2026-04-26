"""Train per-property MLP heads f_p(z) -> property on Tier-A/B latents.

These are used at sample time as classifier guidance: the diffusion sampler
gets a gradient `lambda * grad_z f_p(z_t)` added to the score, pushing z
toward higher predicted property values.

Outputs: data/training/guidance/property_heads.pt
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

BASE = Path("E:/Projects/EnergeticDiffusion2")


class PropHead(nn.Module):
    def __init__(self, d=1024, h=512, time_emb_dim=0):
        super().__init__()
        in_dim = d + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(h, h), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(h, 1))

    def forward(self, z, t_emb=None):
        if t_emb is not None:
            x = torch.cat([z, t_emb], dim=-1)
        else:
            x = z
        return self.net(x).squeeze(-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    blob = torch.load(BASE / "data/training/diffusion/latents_expanded.pt",
                       weights_only=False)
    z_mu = blob["z_mu"].float()
    raw  = blob["values_raw"].float()
    cv   = blob["cond_valid"]
    cw   = blob["cond_weight"]
    prop_names = blob["property_names"]

    out_dir = BASE / "data/training/guidance"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    heads = {}
    metrics = {}
    for j, p in enumerate(prop_names):
        trusted = (cv[:, j] & (cw[:, j] >= 0.99)).numpy()
        idx_all = np.where(trusted)[0]
        if len(idx_all) < 200:
            print(f"  skip {p}: only {len(idx_all)} trusted")
            continue
        rng.shuffle(idx_all)
        n_tr = int(len(idx_all) * 0.8)
        tr_idx = idx_all[:n_tr]; te_idx = idx_all[n_tr:]
        x_tr = z_mu[tr_idx].to(device); x_te = z_mu[te_idx].to(device)
        y_tr = raw[tr_idx, j].to(device); y_te = raw[te_idx, j].to(device)
        mu = y_tr.mean(); sd = y_tr.std() + 1e-6
        y_trn = (y_tr - mu) / sd

        model = PropHead().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        n_steps = 3000 if len(tr_idx) > 5000 else 1500
        bs = 256
        t0 = time.time()
        for s in range(n_steps):
            i = torch.randint(0, len(x_tr), (bs,), device=device)
            loss = ((model(x_tr[i]) - y_trn[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pte = model(x_te) * sd + mu
            mae = (pte - y_te).abs().mean().item()
            yv = y_te.cpu().numpy(); pv = pte.cpu().numpy()
            r = float(np.corrcoef(yv, pv)[0, 1])
        print(f"  {p}: trained on {len(tr_idx):,} rows; r={r:.3f} MAE={mae:.3f} "
              f"({time.time()-t0:.1f}s)")
        heads[p] = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "y_mean":     float(mu),
            "y_std":      float(sd),
        }
        metrics[p] = {"r": r, "mae": mae, "n_train": len(tr_idx),
                      "n_test": len(te_idx)}

    out_path = out_dir / "property_heads.pt"
    torch.save({"heads": heads, "metrics": metrics,
                "property_names": prop_names,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
               out_path)
    print(f"\nSaved {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    sys.exit(main())
