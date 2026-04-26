"""D3: Property predictability from LIMO latents.

Trains a small MLP (1024 -> 512 -> 1) per property on Tier-A/B latents
and reports Pearson r and MAE on a 20% held-out split.

Pass: r >= 0.85 for density / D / P; r >= 0.7 for HOF
Fail: r < 0.5 -> latent space doesn't encode property; need property-aware LIMO
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

BASE = Path("E:/Projects/EnergeticDiffusion2")

class Reg(nn.Module):
    def __init__(self, d=1024, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(h, h), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    blob = torch.load(BASE / "data/training/diffusion/latents_expanded.pt",
                       weights_only=False)
    z = blob["z_mu"].float()
    raw = blob["values_raw"].float()
    cv = blob["cond_valid"]
    cw = blob["cond_weight"]
    rng = np.random.default_rng(42)
    out_md = ["# D3: Property predictability from LIMO latents", "",
              "Small MLP (1024 -> 512 -> 512 -> 1) trained per property on "
              "Tier-A/B latents, 80/20 split.", "",
              "| Property | n_train | n_test | Pearson r | MAE | rel_MAE % | verdict |",
              "|---|---|---|---|---|---|---|"]
    for j, p in enumerate(blob["property_names"]):
        trusted = (cv[:, j] & (cw[:, j] >= 0.99))
        idx = np.where(trusted.numpy())[0]
        if len(idx) < 200:
            print(f"  {p}: only {len(idx)} trusted rows, skip"); continue
        rng.shuffle(idx)
        n_tr = int(len(idx) * 0.8)
        tr_idx = idx[:n_tr]; te_idx = idx[n_tr:]
        x_tr = z[tr_idx].to(device)
        x_te = z[te_idx].to(device)
        y_tr = raw[tr_idx, j].to(device)
        y_te = raw[te_idx, j].to(device)
        # standardise y for stable training
        mu = y_tr.mean(); sd = y_tr.std() + 1e-6
        y_trn = (y_tr - mu) / sd
        y_ten = (y_te - mu) / sd

        model = Reg().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        bs = 256
        n_steps = 800 if len(tr_idx) > 5000 else 400
        t0 = time.time()
        for step in range(n_steps):
            i = torch.randint(0, len(x_tr), (bs,), device=device)
            pred = model(x_tr[i])
            loss = ((pred - y_trn[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        # eval
        model.eval()
        with torch.no_grad():
            pte = model(x_te) * sd + mu
            mae = (pte - y_te).abs().mean().item()
            yv = y_te.cpu().numpy(); pv = pte.cpu().numpy()
            r = float(np.corrcoef(yv, pv)[0, 1])
            rel = 100 * mae / max(abs(yv.mean()), 1e-6)
        ok_thresh = 0.85 if p != "heat_of_formation" else 0.70
        verdict = ("strong" if r >= ok_thresh else
                   "ok" if r >= 0.5 else
                   "weak")
        print(f"  {p:25s}  n_tr={len(tr_idx):,}  n_te={len(te_idx):,}  "
              f"r={r:.3f}  MAE={mae:.3f}  rel={rel:.1f}%  -> {verdict}  "
              f"(in {time.time()-t0:.1f}s)")
        out_md.append(f"| {p} | {len(tr_idx):,} | {len(te_idx):,} | "
                       f"{r:.3f} | {mae:.3f} | {rel:.1f} % | **{verdict}** |")
    out_md += ["", "Verdict thresholds:",
               "- strong = latent encodes property well (r >= 0.85, or 0.7 for HOF)",
               "- ok = usable but limited",
               "- weak = latent doesn't encode property; consider property-aware LIMO retrain"]
    (BASE / "docs/diag_d3.md").write_text("\n".join(out_md), encoding="utf-8")
    print(f"\nSaved {BASE / 'docs/diag_d3.md'}")

if __name__ == "__main__":
    sys.exit(main())
