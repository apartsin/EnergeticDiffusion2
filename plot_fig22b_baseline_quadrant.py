"""Figure 22b: Productive-quadrant snapshot of Table 6a baselines.

Single-panel scatter (novelty, top-1 D) with one inset showing the SELFIES-GA
surrogate -> DFT collapse. Acts as a visual companion to Table 6a so readers
can see at a glance that only DGLD Hz-C2 lives in the productive quadrant
(novel + HMX-class).

Source values: docs/paper/short_paper.html, Table 6a (lines ~588-599).

Output:
  docs/paper/figs/fig22b_baseline_quadrant.png  (220 dpi)
  docs/paper/figs/fig22b_baseline_quadrant.svg
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR     = os.path.join(PROJECT_DIR, "docs", "paper", "figs")

# ---------------------------------------------------------------------------
# Palette (matches plot_fig_quadrant_scatter.py / Figure 23)
# ---------------------------------------------------------------------------
COLOR_GUIDED   = "#d6604d"  # DGLD Hz-C2 (guided)
COLOR_UNGUIDED = "#2166ac"  # DGLD Hz-C0 (unguided)
COLOR_LSTM     = "#a50026"
COLOR_MOLMIM   = "#e08030"
COLOR_GA       = "#7b3294"
COLOR_REINVENT = "#3a8a3a"

# ---------------------------------------------------------------------------
# Method data from Table 6a
# Each entry: (label, x = novelty (1 - max-Tani), y = top-1 D (km/s),
#              memo rate fraction (used for marker area), color, marker)
# ---------------------------------------------------------------------------
METHODS = [
    {
        "label": "SMILES-LSTM",
        "annot": "exact memorisation\n(memo 18.3%; Tani 1.000)",
        "x": 1.0 - 1.000, "y": 9.58,
        "memo": 0.183, "color": COLOR_LSTM, "marker": "X",
        "anchor": (10, -10), "ha": "left", "va": "top",
    },
    {
        "label": "MolMIM 70M",
        "annot": "drug-domain\n(D = 7.70 km/s)",
        "x": 1.0 - 0.625, "y": 7.70,
        "memo": 0.0, "color": COLOR_MOLMIM, "marker": "D",
        "anchor": (10, -4), "ha": "left", "va": "top",
    },
    {
        "label": "REINVENT 4",
        "annot": "N-frac RL\n(D = 9.02 km/s)",
        "x": 1.0 - 0.57, "y": 9.02,
        "memo": 0.0004, "color": COLOR_REINVENT, "marker": "s",
        "anchor": (10, -4), "ha": "left", "va": "top",
    },
    {
        "label": "SELFIES-GA",
        "annot": "surrogate D = 9.74 km/s\n(see inset)",
        "x": 1.0 - 0.35, "y": 9.74,
        "memo": 0.75, "color": COLOR_GA, "marker": "^",
        "anchor": (10, 6), "ha": "left", "va": "bottom",
    },
    {
        "label": "DGLD Hz-C0 unguided",
        "annot": "D = 9.44 km/s\n(memo 0%)",
        "x": 1.0 - 0.61, "y": 9.44,
        "memo": 0.0, "color": COLOR_UNGUIDED, "marker": "o",
        "anchor": (10, 8), "ha": "left", "va": "bottom",
    },
    {
        "label": "DGLD Hz-C2",
        "annot": "novel + HMX-class\nD = 9.39 km/s\n(memo 0%; Tani 0.27)",
        "x": 1.0 - 0.27, "y": 9.39,
        "memo": 0.0, "color": COLOR_GUIDED, "marker": "o",
        "anchor": (-12, -8), "ha": "right", "va": "top",
    },
]

# Marker-area scaling: proportional to (1 - memo). DGLD/MolMIM/REINVENT large,
# SELFIES-GA medium-small, SMILES-LSTM tiny.
SZ_MIN, SZ_MAX = 60, 360
def memo_to_size(memo: float) -> float:
    novelty_factor = max(0.0, 1.0 - memo)
    return SZ_MIN + novelty_factor * (SZ_MAX - SZ_MIN)

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

NOVELTY_FLOOR = 0.45
D_THRESH      = 9.0

# Productive-quadrant tint (novel + HMX-class)
ax.axhspan(D_THRESH, 10.5,
           xmin=(NOVELTY_FLOOR - 0.0) / (1.0 - 0.0),
           xmax=1.0,
           color="#e8f3e8", alpha=0.55, zorder=0)

# Threshold lines
ax.axvline(NOVELTY_FLOOR, color="#444444", linewidth=0.9,
           linestyle="--", zorder=2, alpha=0.75)
ax.axhline(D_THRESH,      color="#444444", linewidth=0.9,
           linestyle="--", zorder=2, alpha=0.75)

# Quadrant labels
ax.text(0.985, 10.40,
        "PRODUCTIVE QUADRANT",
        fontsize=10.0, fontfamily="DejaVu Serif", color="#1a6b1a",
        ha="right", va="top", style="italic", fontweight="bold", zorder=5)
ax.text(0.985, 10.27,
        "novel + HMX-class",
        fontsize=9.0, fontfamily="DejaVu Serif", color="#1a6b1a",
        ha="right", va="top", style="italic", zorder=5)

# Threshold tick labels
ax.text(NOVELTY_FLOOR + 0.005, 6.05, "novelty floor = 0.45",
        fontsize=8.0, fontfamily="DejaVu Serif", color="#555555",
        va="bottom", rotation=90)
ax.text(0.005, D_THRESH + 0.04, "D = 9.0 km/s (HMX-class)",
        fontsize=8.0, fontfamily="DejaVu Serif", color="#555555",
        ha="left", va="bottom")

# ---------------------------------------------------------------------------
# Scatter the methods
# ---------------------------------------------------------------------------
for m in METHODS:
    sz = memo_to_size(m["memo"])
    ax.scatter(m["x"], m["y"], s=sz, c=m["color"], marker=m["marker"],
               edgecolors="white", linewidths=1.0, zorder=6, alpha=0.94)
    label_text = f"{m['label']}\n{m['annot']}"
    ax.annotate(label_text,
                xy=(m["x"], m["y"]),
                xytext=m["anchor"], textcoords="offset points",
                fontsize=8.4, fontfamily="DejaVu Serif",
                color=m["color"], ha=m["ha"], va=m["va"],
                style="italic", fontweight="semibold", zorder=8)

# ---------------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------------
ax.set_xlabel("Novelty (1 - max-Tanimoto to labelled master)",
              fontsize=10.5, fontfamily="DejaVu Serif")
ax.set_ylabel("Top-1 detonation velocity D (km/s)",
              fontsize=10.5, fontfamily="DejaVu Serif")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(6.0, 10.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(color="#eeeeee", linewidth=0.55, zorder=0)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("DejaVu Serif")
    lbl.set_fontsize(9.5)

ax.set_title("Productive-quadrant snapshot: novel + HMX-class top-1 (Table 6a)",
             fontsize=11.5, fontfamily="DejaVu Serif",
             fontweight="bold", pad=12)

# ---------------------------------------------------------------------------
# Inset: SELFIES-GA surrogate -> DFT collapse
# Placed at the upper-left corner of the main panel (high D, low novelty
# region is empty except for the LSTM marker at the floor).
# ---------------------------------------------------------------------------
inset = ax.inset_axes([0.04, 0.55, 0.30, 0.40])
inset.set_facecolor("#fafafa")

GA_X = 1.0 - 0.35  # 0.65 (same as main marker)
GA_D_SURR = 9.74
GA_D_DFT  = 6.28

# HMX-class line in the inset for context
inset.axhline(D_THRESH, color="#888888", linewidth=0.7,
              linestyle=":", alpha=0.7, zorder=2)
inset.text(0.555, D_THRESH + 0.05, "D = 9.0",
           fontsize=7.0, fontfamily="DejaVu Serif", color="#777777",
           ha="left", va="bottom")

# Vertical purple arrow from surrogate to DFT
inset.annotate("",
               xy=(GA_X, GA_D_DFT + 0.18),
               xytext=(GA_X, GA_D_SURR - 0.18),
               arrowprops=dict(arrowstyle="-|>", color=COLOR_GA, lw=2.0,
                               mutation_scale=14, alpha=0.85),
               zorder=4)

# Up-triangle (surrogate) and down-triangle (DFT)
inset.scatter(GA_X, GA_D_SURR, marker="^", s=110, c=COLOR_GA,
              edgecolors="white", linewidths=0.8, zorder=6, alpha=0.95)
inset.scatter(GA_X, GA_D_DFT,  marker="v", s=110, c=COLOR_GA,
              edgecolors="white", linewidths=0.8, zorder=6, alpha=0.95)

inset.annotate("9.74 km/s\n(surrogate)",
               xy=(GA_X, GA_D_SURR), xytext=(6, 0),
               textcoords="offset points",
               fontsize=7.5, fontfamily="DejaVu Serif",
               color=COLOR_GA, ha="left", va="center", style="italic")
inset.annotate("6.28 km/s\n(DFT audit)",
               xy=(GA_X, GA_D_DFT), xytext=(6, 0),
               textcoords="offset points",
               fontsize=7.5, fontfamily="DejaVu Serif",
               color=COLOR_GA, ha="left", va="center", style="italic")

inset.set_xlim(0.55, 0.75)
inset.set_ylim(5.5, 10.0)
inset.set_xticks([0.60, 0.70])
inset.set_yticks([6.0, 7.5, 9.0, 10.0])
for lbl in inset.get_xticklabels() + inset.get_yticklabels():
    lbl.set_fontfamily("DejaVu Serif")
    lbl.set_fontsize(7.5)
inset.set_title("SELFIES-GA: 3.5 km/s surrogate artefact",
                fontsize=8.5, fontfamily="DejaVu Serif",
                fontweight="bold", color=COLOR_GA, pad=4)
inset.set_xlabel("novelty", fontsize=7.5, fontfamily="DejaVu Serif")
inset.set_ylabel("D (km/s)", fontsize=7.5, fontfamily="DejaVu Serif")
inset.tick_params(axis="both", which="major", length=2.5, pad=2)
for sp in inset.spines.values():
    sp.set_color("#999999")
    sp.set_linewidth(0.7)

# ---------------------------------------------------------------------------
# Legend (bottom, horizontal)
# ---------------------------------------------------------------------------
legend_handles = [
    Line2D([0], [0], marker="X", color="w", markerfacecolor=COLOR_LSTM,
           markersize=10, label="SMILES-LSTM"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_MOLMIM,
           markersize=10, label="MolMIM 70M"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_GA,
           markersize=10, label="SELFIES-GA (surrogate)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_REINVENT,
           markersize=10, label="REINVENT 4"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_UNGUIDED,
           markersize=11, label="DGLD Hz-C0 unguided"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_GUIDED,
           markersize=11, label="DGLD Hz-C2 (best novel)"),
]
leg = fig.legend(
    handles=legend_handles,
    loc="lower center", bbox_to_anchor=(0.5, 0.005),
    fontsize=9.0, frameon=True, framealpha=0.96, edgecolor="#cccccc",
    prop={"family": "DejaVu Serif", "size": 9.0},
    handletextpad=0.6, borderpad=0.6, labelspacing=0.4,
    ncol=6, columnspacing=1.4,
)

# Note about marker area
fig.text(0.5, 0.075,
         "Marker area $\\propto$ (1 - memorisation rate); DGLD large, SMILES-LSTM tiny.",
         ha="center", va="bottom",
         fontsize=8.5, fontfamily="DejaVu Serif",
         color="#555555", style="italic")

fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))

os.makedirs(OUT_DIR, exist_ok=True)
png_path = os.path.join(OUT_DIR, "fig22b_baseline_quadrant.png")
svg_path = os.path.join(OUT_DIR, "fig22b_baseline_quadrant.svg")
fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(svg_path,            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {png_path}")
print(f"Saved: {svg_path}")
