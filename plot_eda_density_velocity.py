"""Regenerate fig_eda_density_vs_velocity.svg/png.

Density vs detonation-velocity scatter over the labelled corpus, with a few
literature anchors (CL-20, HMX, RDX, TATB, TNT) overlaid as labelled markers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "m6_postprocess_bundle" / "labelled_master.csv"
OUT_SVG = ROOT / "docs" / "paper" / "figs" / "fig_eda_density_vs_velocity.svg"
OUT_PNG = ROOT / "docs" / "paper" / "figs" / "fig_eda_density_vs_velocity.png"

# Literature anchors: (label, density g/cc, D km/s)
ANCHORS = [
    ("TNT",   1.65, 6.95),
    ("RDX",   1.82, 8.75),
    ("HMX",   1.90, 9.10),
    ("CL-20", 2.04, 9.40),
    ("TATB",  1.93, 7.86),
    ("PETN",  1.77, 8.40),
]


def main() -> None:
    df = pd.read_csv(CSV, usecols=["density", "detonation_velocity"])
    sub = df.dropna(subset=["density", "detonation_velocity"]).copy()
    sub = sub[(sub["density"] > 0.5) & (sub["density"] < 3.0)]
    sub = sub[(sub["detonation_velocity"] > 1.0) & (sub["detonation_velocity"] < 12.0)]

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "figure.dpi": 120,
    })

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(
        sub["density"], sub["detonation_velocity"],
        s=4, alpha=0.18, color="#2b6cb0",
        edgecolors="none", rasterized=True, label=f"Labelled corpus (n={len(sub):,})",
    )

    for name, rho, D in ANCHORS:
        ax.scatter(rho, D, s=70, color="#c0392b", edgecolor="black",
                   linewidth=0.8, zorder=5)
        ax.annotate(name, (rho, D), xytext=(6, 4), textcoords="offset points",
                    fontsize=9, color="#7b1d12", weight="bold")

    ax.axhline(9.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(1.85, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(r"Density $\rho$ (g/cm$^3$)")
    ax.set_ylabel(r"Detonation velocity $D$ (km/s)")
    ax.set_title("Labelled corpus: density vs detonation velocity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"density_velocity: rows={len(sub):,}  out={OUT_SVG.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
