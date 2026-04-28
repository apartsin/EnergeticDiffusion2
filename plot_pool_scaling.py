"""Regenerate fig1_pool_scaling.svg/png — pool-size scaling.

# TODO: locate source data.
#
# The paper reports two curves vs pool size in {1k, 2k, 5k, 10k, 20k, 40k}:
#   (i) best composite score over top-1 candidate
#   (ii) number of candidates passing every filter
# (chem + SA/SC + Tanimoto + neutrality)
#
# This raw sweep is not in results/ as a single JSON. m1_summary.json runs
# at fixed pool_per_run=10000 and m1_anneal_clamp_summary.json at
# pool_per_run=40000. The pool-scaling sweep itself is missing.
#
# Required input: a JSON like
#   { "pool":           [1000, 2000, 5000, 10000, 20000, 40000],
#     "n_passing":      [...],
#     "best_composite": [...] }
#
# This script provides the plotting skeleton; replace the loader once the
# pool-scaling sweep file is recovered.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_SVG = ROOT / "docs" / "paper" / "figs" / "fig1_pool_scaling.svg"
OUT_PNG = ROOT / "docs" / "paper" / "figs" / "fig1_pool_scaling.png"

PLACEHOLDER = {
    "pool":           [1000, 2000, 5000, 10000, 20000, 40000],
    "n_passing":      [None] * 6,
    "best_composite": [None] * 6,
}


def load_data() -> dict:
    candidate = ROOT / "results" / "pool_scaling.json"
    if candidate.exists():
        with candidate.open() as fh:
            return json.load(fh)
    return PLACEHOLDER


def main() -> None:
    data = load_data()
    pool = np.array(data["pool"], dtype=float)
    n_pass = data["n_passing"]
    best = data["best_composite"]

    if any(v is None for v in n_pass) or any(v is None for v in best):
        print("pool_scaling: source data missing — wrote skeleton script only, "
              "skipping render.")
        return

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "figure.dpi": 120,
    })

    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax2 = ax1.twinx()

    ax1.plot(pool, n_pass, "o-", color="#2b6cb0", linewidth=2,
             markersize=8, label="Candidates passing all filters")
    ax1.set_xscale("log")
    ax1.set_xlabel("Pool size (samples)")
    ax1.set_ylabel("Candidates passing every filter", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(pool, best, "s--", color="#c0392b", linewidth=2,
             markersize=8, label="Best top-1 composite")
    ax2.set_ylabel("Best top-1 composite score", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    ax1.set_title("Pool size vs filter pass-rate and best top-1 composite")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
               framealpha=0.9)

    fig.tight_layout()
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"pool_scaling: pool={pool.tolist()}  out={OUT_SVG.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
