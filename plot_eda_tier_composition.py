"""Regenerate fig_eda_tier_composition.svg/png.

Stacked bar of tier composition (A, B, C, D, missing) per property.
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
OUT_SVG = ROOT / "docs" / "paper" / "figs" / "fig_eda_tier_composition.svg"
OUT_PNG = ROOT / "docs" / "paper" / "figs" / "fig_eda_tier_composition.png"

TIER_COLS = [
    ("density_tier",             "Density"),
    ("heat_of_formation_tier",   "HOF"),
    ("detonation_velocity_tier", "D (km/s)"),
    ("detonation_pressure_tier", "P (GPa)"),
]
TIERS = ["A", "B", "C", "D"]
COLORS = {"A": "#1b7837", "B": "#7fbf7b", "C": "#fdb863", "D": "#e08214",
          "missing": "#cccccc"}


def main() -> None:
    df = pd.read_csv(CSV, usecols=[c for c, _ in TIER_COLS])
    n_total = len(df)

    rows = []
    for col, label in TIER_COLS:
        vc = df[col].fillna("missing").astype(str).value_counts()
        row = {"property": label}
        for t in TIERS + ["missing"]:
            row[t] = int(vc.get(t, 0))
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("property")

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "figure.dpi": 120,
    })

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bottoms = np.zeros(len(tab))
    x = np.arange(len(tab))
    for t in TIERS + ["missing"]:
        vals = tab[t].values
        ax.bar(x, vals, bottom=bottoms, color=COLORS[t], label=t,
               edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(tab.index)
    ax.set_ylabel(f"Number of rows (total {n_total:,})")
    ax.set_title("Label-tier composition by property")
    ax.legend(title="Tier", loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"tier_composition: total={n_total:,}  out={OUT_SVG.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
