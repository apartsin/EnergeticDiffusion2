"""
Forest plot: guidance ablation effect sizes vs unguided baseline.
Data from Table 5 of the paper (docs/paper/index.html), section 5.5.4.

Table 5. DGLD multi-seed head-to-head at pool=10k per condition across
hazard-axis (Hz-) and SA-axis (SA-) matrices.

Saves:
  docs/paper/figs/fig_forest_ablation.png
  docs/paper/figs/fig_forest_ablation.svg
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Data extracted directly from Table 5 (paper HTML lines 711-724).
# Unguided baseline (Hz-C0 = SA-C0, 6 seeds):
#   top-1 composite  0.451 +/- 0.126
#   top-1 D (km/s)   9.44  +/- 0.07
#   top-1 rho        1.93  +/- 0.01
#   top-1 P (GPa)    39.7  +/- 0.6
#   max-Tani         0.61  +/- 0.10
#
# Guided conditions (3 seeds each, same pool=10k):
#   SA-C1 viab-only:        composite 0.542+/-0.184, D 9.54+/-0.04, rho 1.93+/-0.01, P 39.8+/-0.5, tani 0.63+/-0.06
#   Hz-C1 viab+sens:        composite 0.613+/-0.106, D 9.36+/-0.06, rho 1.93+/-0.01, P 38.7+/-0.4, tani 0.46+/-0.04
#   SA-C2 viab+sens:        composite 0.618+/-0.056, D 9.44+/-0.05, rho 1.94+/-0.01, P 38.5+/-0.4, tani 0.41+/-0.04
#   Hz-C2 viab+sens+hazard: composite 0.485+/-0.152, D 9.39+/-0.04, rho 1.91+/-0.03, P 38.7+/-0.6, tani 0.27+/-0.03
#   Hz-C3 hazard-only:      composite 0.503+/-0.131, D 9.32+/-0.10, rho 1.95+/-0.01, P 38.8+/-0.4, tani 0.44+/-0.05
#   SA-C3 viab+sens+SA:     composite 0.698+/-0.015, D 9.34+/-0.07, rho 1.94+/-0.01, P 38.7+/-0.4, tani 0.49+/-0.06
# ---------------------------------------------------------------------------

# Unguided baseline
BASELINE = {
    "composite": (0.451, 0.126),   # (mean, std)
    "D_kms":     (9.44,  0.07),
    "rho":       (1.93,  0.01),
    "P_GPa":     (39.7,  0.6),
    "tani":      (0.61,  0.10),
}

# Guided conditions (order as we want them on the plot, top-to-bottom)
CONDITIONS = [
    {
        "label": "SA-C1  viab-only",
        "axis":  "SA",
        "composite": (0.542, 0.184),
        "D_kms":     (9.54,  0.04),
        "rho":       (1.93,  0.01),
        "P_GPa":     (39.8,  0.5),
        "tani":      (0.63,  0.06),
    },
    {
        "label": "Hz-C1  viab+sens",
        "axis":  "Hz",
        "composite": (0.613, 0.106),
        "D_kms":     (9.36,  0.06),
        "rho":       (1.93,  0.01),
        "P_GPa":     (38.7,  0.4),
        "tani":      (0.46,  0.04),
    },
    {
        "label": "SA-C2  viab+sens",
        "axis":  "SA",
        "composite": (0.618, 0.056),
        "D_kms":     (9.44,  0.05),
        "rho":       (1.94,  0.01),
        "P_GPa":     (38.5,  0.4),
        "tani":      (0.41,  0.04),
    },
    {
        "label": "Hz-C2  viab+sens+hazard  (prod.)",
        "axis":  "Hz",
        "composite": (0.485, 0.152),
        "D_kms":     (9.39,  0.04),
        "rho":       (1.91,  0.03),
        "P_GPa":     (38.7,  0.6),
        "tani":      (0.27,  0.03),
    },
    {
        "label": "Hz-C3  hazard-only",
        "axis":  "Hz",
        "composite": (0.503, 0.131),
        "D_kms":     (9.32,  0.10),
        "rho":       (1.95,  0.01),
        "P_GPa":     (38.8,  0.4),
        "tani":      (0.44,  0.05),
    },
    {
        "label": "SA-C3  viab+sens+SA",
        "axis":  "SA",
        "composite": (0.698, 0.015),
        "D_kms":     (9.34,  0.07),
        "rho":       (1.94,  0.01),
        "P_GPa":     (38.7,  0.4),
        "tani":      (0.49,  0.06),
    },
]

# Metrics to show as columns in the forest plot.
# For composite: lower is better, so negative delta = improvement.
# For D, rho, P: higher is better.
# For tani: lower is better (more novel), so negative delta = improvement.
METRICS = [
    {
        "key":       "composite",
        "label":     "Top-1 composite\n(delta, lower = better)",
        "baseline_key": "composite",
        "sign":      +1,   # raw difference (we show it signed; negative = better)
        "unit":      "",
        "color_pos": "#d73027",  # red = worse
        "color_neg": "#1a9641",  # green = better
    },
    {
        "key":       "D_kms",
        "label":     "Top-1 D (km/s)\n(delta vs unguided)",
        "baseline_key": "D_kms",
        "sign":      +1,
        "unit":      " km/s",
        "color_pos": "#2166ac",  # blue = better (higher D)
        "color_neg": "#d73027",
    },
    {
        "key":       "tani",
        "label":     "Max-Tanimoto to LM\n(delta, lower = more novel)",
        "baseline_key": "tani",
        "sign":      +1,   # negative delta = lower Tani = more novel = better
        "unit":      "",
        "color_pos": "#d73027",
        "color_neg": "#1a9641",
    },
    {
        "key":       "P_GPa",
        "label":     "Top-1 P (GPa)\n(delta vs unguided)",
        "baseline_key": "P_GPa",
        "sign":      +1,
        "unit":      " GPa",
        "color_pos": "#2166ac",
        "color_neg": "#d73027",
    },
]

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

N_COND = len(CONDITIONS)
N_MET  = len(METRICS)

fig, axes = plt.subplots(
    1, N_MET,
    figsize=(14, 5.5),
    sharey=True,
)
fig.patch.set_facecolor("white")

# Font settings - no LaTeX, serif for readability
LABEL_FONT  = {"fontfamily": "DejaVu Serif", "fontsize": 9}
TITLE_FONT  = {"fontfamily": "DejaVu Serif", "fontsize": 9, "fontweight": "bold"}
TICK_FONT   = {"fontfamily": "DejaVu Serif", "fontsize": 8.5}

y_positions = np.arange(N_COND)

for col_idx, (ax, met) in enumerate(zip(axes, METRICS)):
    ax.set_facecolor("white")
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", zorder=2)

    for row_idx, cond in enumerate(CONDITIONS):
        bval, bstd = BASELINE[met["baseline_key"]]
        cval, cstd = cond[met["key"]]

        delta      = cval - bval
        err_total  = np.sqrt(bstd**2 + cstd**2)  # propagated SE (approximate)

        # Choose color by whether the delta is "good" or "bad"
        # For composite and tani: negative delta = improvement (green)
        # For D and P: positive delta = improvement (blue)
        if met["key"] in ("composite", "tani"):
            bar_color = met["color_neg"] if delta < 0 else met["color_pos"]
        else:
            bar_color = met["color_pos"] if delta > 0 else met["color_neg"]

        y = y_positions[row_idx]
        ax.errorbar(
            delta, y,
            xerr=err_total,
            fmt="D",
            color=bar_color,
            ecolor=bar_color,
            markersize=6,
            capsize=4,
            capthick=1.2,
            linewidth=1.2,
            zorder=3,
        )

        # Value label next to point
        offset = err_total + abs(delta) * 0.05
        ax.text(
            delta + (0.6 if delta >= 0 else -0.6) * err_total + (0.01 if delta >= 0 else -0.01),
            y,
            f"{delta:+.3f}",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=7,
            fontfamily="DejaVu Serif",
            color=bar_color,
            zorder=4,
        )

    ax.set_title(met["label"], **TITLE_FONT, pad=6)
    ax.set_yticks(y_positions)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(col_idx == 0)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)

    if col_idx == 0:
        ax.set_yticklabels(
            [c["label"] for c in CONDITIONS],
            **TICK_FONT,
        )
    else:
        ax.set_yticklabels([])

# Invert so most-novel condition is near top (visual convention for forest plots)
axes[0].invert_yaxis()

# Legend
hz_patch = mpatches.Patch(color="#4393c3", label="Hazard-axis (Hz-) conditions")
sa_patch = mpatches.Patch(color="#f4a582", label="SA-axis (SA-) conditions")
fig.legend(
    handles=[hz_patch, sa_patch],
    loc="lower center",
    ncol=2,
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
    prop={"family": "DejaVu Serif", "size": 8},
)

fig.suptitle(
    "Guidance ablation: effect size vs unguided baseline (Table 5, pool = 10k per condition, 3 seeds)",
    fontfamily="DejaVu Serif",
    fontsize=10,
    fontweight="bold",
    y=1.01,
)

fig.text(
    0.5, -0.05,
    "Unguided baseline (Hz-C0 = SA-C0): composite 0.451 +/- 0.126,  D 9.44 +/- 0.07 km/s,  "
    "max-Tani 0.61 +/- 0.10,  P 39.7 +/- 0.6 GPa",
    ha="center", va="top",
    fontsize=7.5, fontfamily="DejaVu Serif", color="#444444",
)

fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "docs", "paper", "figs"
)
os.makedirs(OUT_DIR, exist_ok=True)

png_path = os.path.join(OUT_DIR, "fig_forest_ablation.png")
svg_path = os.path.join(OUT_DIR, "fig_forest_ablation.svg")

fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved PNG: {png_path}")
print(f"Saved SVG: {svg_path}")
