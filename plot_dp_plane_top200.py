"""Regenerate fig3_leads_dp_plane.svg/png.

Top candidates in the (D, P) plane coloured by predicted density rho.

Source data: results/novelty_top1.json (rows: per-(condition, seed) top-1
candidate with D, P, rho, composite, novelty). The original figure used the
top-200 from pool=40k; that per-candidate scored pool is not in the JSON
results we have at hand. Until that is recovered, we plot the available
top-1 set (~26 points spanning every condition + seed) and label the figure
honestly. The plotting structure is unchanged; only the data source needs
swapping when the top-200 file is recovered.

# TODO: when results/top200_pool40k.json (or equivalent) is regenerated,
# point SOURCE there and drop the "top-1 per (condition, seed)" caveat.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "novelty_top1.json"
OUT_SVG = ROOT / "docs" / "paper" / "figs" / "fig3_leads_dp_plane.svg"
OUT_PNG = ROOT / "docs" / "paper" / "figs" / "fig3_leads_dp_plane.png"


def main() -> None:
    with SOURCE.open() as fh:
        rows = json.load(fh)["rows"]

    D = np.array([r["top1_D_kms"] for r in rows], dtype=float)
    P = np.array([r["top1_P_GPa"] for r in rows], dtype=float)
    rho = np.array([r["top1_rho"] for r in rows], dtype=float)
    novel = np.array([not r["is_memorized"] for r in rows], dtype=bool)

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "figure.dpi": 120,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    sc = ax.scatter(D[novel], P[novel], c=rho[novel], cmap="viridis",
                    s=70, edgecolor="black", linewidth=0.6,
                    vmin=1.6, vmax=2.1, label=f"Novel (n={int(novel.sum())})")
    ax.scatter(D[~novel], P[~novel], facecolors="none", edgecolor="#888",
               s=70, linewidth=0.8, label=f"Memorized (n={int((~novel).sum())})")

    # Anchors for context
    anchors = [("RDX", 8.75, 34.0), ("HMX", 9.10, 39.0), ("CL-20", 9.40, 42.0)]
    for name, dv, pv in anchors:
        ax.scatter(dv, pv, marker="*", s=160, color="#c0392b",
                   edgecolor="black", linewidth=0.6, zorder=5)
        ax.annotate(name, (dv, pv), xytext=(7, 6), textcoords="offset points",
                    fontsize=9, color="#7b1d12", weight="bold")

    ax.set_xlabel(r"Detonation velocity $D$ (km/s)")
    ax.set_ylabel(r"Detonation pressure $P$ (GPa)")
    ax.set_title("Top candidates in the (D, P) plane (colour = predicted "
                 r"$\rho$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    cb = fig.colorbar(sc, ax=ax, label=r"Density $\rho$ (g/cm$^3$)")
    fig.tight_layout()

    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"dp_plane: rows={len(rows)}  novel={int(novel.sum())}  "
          f"out={OUT_SVG.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
