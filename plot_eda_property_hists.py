"""Regenerate fig_eda_property_hists.svg/png.

Per-property histograms (density, HOF, detonation velocity, detonation pressure)
over the labelled corpus.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "m6_postprocess_bundle" / "labelled_master.csv"
OUT_SVG = ROOT / "docs" / "paper" / "figs" / "fig_eda_property_hists.svg"
OUT_PNG = ROOT / "docs" / "paper" / "figs" / "fig_eda_property_hists.png"

PROPS = [
    ("density",             r"Density $\rho$ (g/cm$^3$)",      (0.6, 2.6),    "#2b6cb0"),
    ("heat_of_formation",   r"Heat of formation $\Delta H_f$ (kJ/mol)", (-1500, 1500), "#c0392b"),
    ("detonation_velocity", r"Detonation velocity $D$ (km/s)",  (1.0, 11.0),   "#27ae60"),
    ("detonation_pressure", r"Detonation pressure $P$ (GPa)",   (0.0, 50.0),   "#8e44ad"),
]


def main() -> None:
    df = pd.read_csv(CSV, usecols=[p for p, *_ in PROPS])

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "figure.dpi": 120,
    })

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))
    axes = axes.ravel()

    counts = {}
    for ax, (col, label, (lo, hi), color) in zip(axes, PROPS):
        s = df[col].dropna()
        s = s[(s >= lo) & (s <= hi)]
        counts[col] = len(s)
        ax.hist(s, bins=60, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{col}  (n={len(s):,})")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Labelled corpus: per-property distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"property_hists: counts={counts}  out={OUT_SVG.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
