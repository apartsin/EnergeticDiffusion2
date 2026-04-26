"""D8: Tier-D label noise.

For each property, find rows where BOTH a Tier-A/B value and a 3DCNN smoke
prediction exist, and quantify how noisy the smoke prediction is as a label.

Threshold: MAE > 50% of property std → Tier-D is poison; consider switching
to Tier-A/B-only conditioning (v4-B).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

BASE = Path("E:/Projects/EnergeticDiffusion2")

def main():
    blob = torch.load(BASE / "data/training/diffusion/latents_expanded.pt",
                       weights_only=False)
    raw = blob["values_raw"].numpy()
    cw  = blob["cond_weight"].numpy()
    cv  = blob["cond_valid"].numpy().astype(bool)
    preds_blob = torch.load(BASE / "data/training/diffusion/preds_3dcnn.pt",
                             weights_only=False)
    preds = preds_blob["predictions"].numpy()
    pred_valid = preds_blob["valid"].numpy().astype(bool)
    prop_names_3dcnn = preds_blob["property_names"]
    prop_names = blob["property_names"]
    # 3DCNN ordering: density, DetoD, DetoP, DetoQ, DetoT, DetoV, HOF_S, BDE
    # Our 4: density, heat_of_formation, det_velocity, det_pressure
    name_map = {
        "density":              "density",
        "heat_of_formation":    "HOF_S",
        "detonation_velocity":  "DetoD",
        "detonation_pressure":  "DetoP",
    }
    print(f"{'property':25s}  {'rows_AB+3DCNN':>13s}  {'std(AB)':>8s}  "
          f"{'MAE_smoke':>10s}  {'rel_MAE%':>9s}  {'verdict':>20s}")
    print("-" * 95)
    rows_md = []
    for j, p in enumerate(prop_names):
        trusted = (cw[:, j] >= 0.99) & cv[:, j]
        # 3DCNN col index
        try:
            j3 = prop_names_3dcnn.index(name_map[p])
        except ValueError:
            print(f"  {p}: not in 3DCNN preds, skip"); continue
        # Tier-D column for this prop has its 3DCNN smoke value substituted into
        # values_raw via expand_conditioning.py; but to compare label-vs-truth
        # we want the smoke prediction at trusted rows ALONE.
        both = trusted & pred_valid
        n = int(both.sum())
        if n < 50:
            print(f"  {p}: only {n} rows with both; skip"); continue
        ab = raw[both, j]
        sm = preds[both, j3]
        # HOF unit fix: smoke gives kcal/mol but I want to make sure
        if p == "heat_of_formation":
            # both values_raw and 3DCNN HOF should already be kcal/mol; verify
            pass
        sd = float(np.std(ab))
        mae = float(np.mean(np.abs(ab - sm)))
        rel = 100 * mae / max(abs(np.mean(ab)), 1e-6)
        rel_to_std = 100 * mae / max(sd, 1e-6)
        verdict = ("OK" if rel_to_std < 30 else
                   "noisy" if rel_to_std < 60 else
                   "poison")
        print(f"{p:25s}  {n:>13,d}  {sd:>8.3f}  {mae:>10.3f}  "
              f"{rel:>8.1f}%  {verdict:>20s}  (MAE/std={rel_to_std:.0f}%)")
        rows_md.append((p, n, sd, mae, rel, rel_to_std, verdict))

    out = BASE / "docs/diag_d8.md"
    md = ["# D8: Tier-D label noise", "",
          "Compares 3DCNN-smoke predictions against Tier-A/B ground truth on"
          " the rows where both exist.", "",
          "| Property | n_rows | std(A/B) | MAE(smoke) | rel_MAE % | MAE/std % | verdict |",
          "|---|---|---|---|---|---|---|"]
    for r in rows_md:
        md.append(f"| {r[0]} | {r[1]:,} | {r[2]:.3f} | {r[3]:.3f} | "
                  f"{r[4]:.1f} % | {r[5]:.0f} % | **{r[6]}** |")
    md += ["",
           "Verdict thresholds (MAE / std):",
           "- < 30 % = OK (smoke labels usable as is)",
           "- 30-60 % = noisy (use, but with weight < 0.7)",
           "- > 60 % = poison (drop or use only as auxiliary)"]
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {out}")

if __name__ == "__main__":
    sys.exit(main())
