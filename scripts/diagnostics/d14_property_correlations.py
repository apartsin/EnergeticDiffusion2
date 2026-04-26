"""D14: property correlations on Tier-A/B rows.

Looks for intrinsic anti-correlation that would make joint q90 fundamentally
harder. If high-HOF correlates with low-density (likely — H-rich is high-HOF
but low-ρ), reranking by composite is structurally limited.
"""
import sys
from pathlib import Path
import numpy as np
import torch

BASE = Path("E:/Projects/EnergeticDiffusion2")

def main():
    blob = torch.load(BASE / "data/training/diffusion/latents_expanded.pt",
                       weights_only=False)
    raw = blob["values_raw"].numpy()
    cv  = blob["cond_valid"].numpy().astype(bool)
    cw  = blob["cond_weight"].numpy()
    pn  = blob["property_names"]

    md = ["# D14: property correlations (Tier-A/B only)", ""]
    n = len(pn)
    md.append("## Pearson r between properties")
    md.append("")
    md.append("| | " + " | ".join(pn) + " |")
    md.append("|---|" + "|".join(["---"]*n) + "|")
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            both = (cv[:, i] & cv[:, j] & (cw[:, i] >= 0.99) & (cw[:, j] >= 0.99))
            if both.sum() < 30: R[i, j] = np.nan; continue
            R[i, j] = float(np.corrcoef(raw[both, i], raw[both, j])[0, 1])
    for i, p in enumerate(pn):
        cells = []
        for j in range(n):
            cells.append(f"{R[i, j]:+.2f}" if not np.isnan(R[i, j]) else "-")
        md.append(f"| **{p}** | " + " | ".join(cells) + " |")

    # specifically: HOF vs (density, D, P) high-end relationship
    md.append("")
    md.append("## High-HOF rows: do they have high D / P / density?")
    md.append("")
    md.append("Pick rows in top-10 % of HOF, report mean of other props vs whole-set mean.")
    md.append("")
    md.append("| metric | top-10 % HOF | all Tier-A/B | Δ |")
    md.append("|---|---|---|---|")
    j_h = pn.index("heat_of_formation")
    valid_h = cv[:, j_h] & (cw[:, j_h] >= 0.99)
    h = raw[valid_h, j_h]
    threshold = np.quantile(h, 0.9)
    md.append(f"| HOF cutoff (top-10 %) | {threshold:+.1f} | – | – |")
    high_idx_in_subset = h >= threshold
    full_high = np.where(valid_h)[0][high_idx_in_subset]
    for j, p in enumerate(pn):
        if j == j_h: continue
        # rows in `full_high` AND with this property valid
        ok = cv[full_high, j] & (cw[full_high, j] >= 0.99)
        if ok.sum() < 5:
            md.append(f"| {p} (mean) | n<5 | – | – |"); continue
        v_high = raw[full_high[ok], j].mean()
        all_v = raw[cv[:, j] & (cw[:, j] >= 0.99), j]
        v_all = all_v.mean()
        d = v_high - v_all
        sd = all_v.std()
        md.append(f"| {p} (mean) | {v_high:.3f} | {v_all:.3f} | "
                  f"{d:+.3f} ({d/sd:+.2f} σ) |")

    out = BASE / "docs/diag_d14.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved {out}")

if __name__ == "__main__":
    sys.exit(main())
