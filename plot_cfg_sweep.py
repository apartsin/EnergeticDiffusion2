"""Regenerate fig2_cfg_sweep.svg/png — CFG-scale sweep.

# TODO: locate source data.
#
# The paper reports a sweep over classifier-free-guidance scale w in {3,5,7,9,11}
# at pool=8000 each, with v3+v4 reranking, and reports (a) number of candidates
# passing every filter and (b) max composite score per w.
#
# The raw scored output of that sweep (per-w validated counts and top-K
# composites) is not present in results/. m6_post.json contains only the
# main 5-condition sweep at the chosen w=7. m1_summary.json fixes
# cfg_scale=7.0.
#
# Required input: a JSON or CSV like
#   { "w": [3,5,7,9,11],
#     "n_passing_filters": [...],
#     "max_composite": [...],
#     "topN_mean_composite": [...] }
#
# This script provides the plotting skeleton; populate `data` below or
# replace the placeholder loader with the real source path when found.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_SVG = ROOT / "docs" / "paper" / "figs" / "fig2_cfg_sweep.svg"
OUT_PNG = ROOT / "docs" / "paper" / "figs" / "fig2_cfg_sweep.png"

# Placeholder values reflecting the paper's qualitative claims (w=7 sweet spot).
# REPLACE with real data when the CFG-sweep result file is recovered.
PLACEHOLDER = {
    "w":                   [3.0, 5.0, 7.0, 9.0, 11.0],
    "n_passing_filters":   [None, None, None, None, None],
    "max_composite":       [None, None, None, None, None],
    "topN_mean_composite": [None, None, None, None, None],
}


def load_data() -> dict:
    # TODO: point at the real CFG-sweep result file when available.
    candidate = ROOT / "results" / "cfg_sweep.json"
    if candidate.exists():
        with candidate.open() as fh:
            return json.load(fh)
    return PLACEHOLDER


def main() -> None:
    data = load_data()
    w = np.array(data["w"], dtype=float)
    n_pass = data.get("n_passing_filters")
    max_c = data.get("max_composite")
    mean_c = data.get("topN_mean_composite")

    if any(v is None for series in (n_pass, max_c) for v in series):
        print("cfg_sweep: source data missing — wrote skeleton script only, "
              "skipping render.")
        return

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "figure.dpi": 120,
    })

    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax2 = ax1.twinx()

    ax1.plot(w, n_pass, "o-", color="#2b6cb0", linewidth=2,
             markersize=8, label="Candidates passing filters")
    ax1.set_xlabel("Classifier-free guidance scale $w$")
    ax1.set_ylabel("Candidates passing every filter", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.grid(True, alpha=0.3)

    ax2.plot(w, max_c, "s--", color="#c0392b", linewidth=2,
             markersize=8, label="Max composite score")
    if mean_c and not any(v is None for v in mean_c):
        ax2.plot(w, mean_c, "^:", color="#e67e22", linewidth=2,
                 markersize=8, label="Mean top-N composite")
    ax2.set_ylabel("Composite score", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    ax1.axvline(7.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("CFG-scale sweep at pool=8000 (v3+v4 reranking)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center",
               framealpha=0.9)

    fig.tight_layout()
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"cfg_sweep: w={w.tolist()}  out={OUT_SVG.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
