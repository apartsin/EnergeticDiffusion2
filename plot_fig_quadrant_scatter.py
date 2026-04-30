"""Productive-quadrant scatter (Figure 23): composite score S vs viability.

Shows the 12 DFT-confirmed DGLD leads against three no-diffusion baselines:
SMILES-LSTM (memorisation), MolMIM 70 M (drug-domain), SELFIES-GA (DFT-audit
collapse), and REINVENT 4 (N-fraction RL). Marker area is proportional to
drop-weight impact sensitivity h50 (cm; larger = safer).

Data sources:
  L1-L5:  composite S and viability from Table 6; h50 from Table D.1c.
  L9-L20: h50 from Table D.1c; composite S and viability not tabulated -
          drawn as hollow markers along the Pareto floor.
  Baselines: composite penalty / top-1 D / novelty from Table 6a.

Output:
  docs/paper/figs/fig_quadrant_scatter.png  (220 dpi)
  docs/paper/figs/fig_quadrant_scatter.svg
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR     = os.path.join(PROJECT_DIR, "docs", "paper", "figs")

# ---------------------------------------------------------------------------
# DGLD lead data
# ---------------------------------------------------------------------------
# L1-L5: real composite score S, real RF viability, real h50 (Table 6 + D.1c).
TOP5 = [
    {"id": "L1", "S": 0.77, "viab": 1.00, "h50": 30.3, "guided": True},
    {"id": "L2", "S": 0.69, "viab": 0.96, "h50": 33.5, "guided": False},
    {"id": "L3", "S": 0.66, "viab": 0.83, "h50": 82.6, "guided": False},
    {"id": "L4", "S": 0.66, "viab": 0.93, "h50": 27.8, "guided": False},
    {"id": "L5", "S": 0.65, "viab": 0.86, "h50": 33.4, "guided": False},
]

# L9-L20: h50 from Table D.1c; S and viab not tabulated. We place them along
# the Pareto-floor band (S between 0.595 and 0.625 jittered) above the
# viability floor (viab=0.83) so the marker cluster does not collapse to a
# single point. Their real ordinate is unknown; the strip is illustrative.
LOWER_LEADS = [
    {"id": "L9",  "h50": 21.9},
    {"id": "L11", "h50": 26.5},
    {"id": "L13", "h50": 31.0},
    {"id": "L16", "h50": 82.6},
    {"id": "L18", "h50": 38.6},
    {"id": "L19", "h50": 45.5},
    {"id": "L20", "h50": 74.1},
]

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 8.4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

S_THRESH    = 0.65
VIAB_THRESH = 0.83

ax.axhline(S_THRESH,    color="#444444", linewidth=0.9, linestyle="--", zorder=2, alpha=0.75)
ax.axvline(VIAB_THRESH, color="#444444", linewidth=0.9, linestyle="--", zorder=2, alpha=0.75)

# Quadrant tints (subtle, helps visual partition)
ax.axhspan(S_THRESH, 0.95, xmin=(VIAB_THRESH - 0.25) / (1.05 - 0.25),
           color="#e8f3e8", alpha=0.45, zorder=0)

# Quadrant labels - placed to fit within plot xlim (right edge = 1.05)
ax.text(0.995, 0.925,
        "PRODUCTIVE QUADRANT",
        fontsize=9.5, fontfamily="DejaVu Serif", color="#1a6b1a",
        ha="right", va="top", style="italic", fontweight="bold", zorder=5)
ax.text(0.995, 0.895,
        "novel + HMX-class",
        fontsize=8.8, fontfamily="DejaVu Serif", color="#1a6b1a",
        ha="right", va="top", style="italic", zorder=5)
ax.text(VIAB_THRESH - 0.008, S_THRESH - 0.012,
        "marginal",
        fontsize=8.5, fontfamily="DejaVu Serif", color="#999999",
        ha="right", va="top", style="italic", zorder=5)

# h50 sizing
H50_MIN, H50_MAX = 20, 90
SZ_MIN,  SZ_MAX  = 60, 360

def h50_size(h50):
    return SZ_MIN + (np.clip(h50, H50_MIN, H50_MAX) - H50_MIN) / (H50_MAX - H50_MIN) * (SZ_MAX - SZ_MIN)

# Palette
COLOR_GUIDED   = "#d6604d"
COLOR_UNGUIDED = "#2166ac"
COLOR_LOWER    = "#5b9bd5"
COLOR_LSTM     = "#a50026"
COLOR_MOLMIM   = "#e08030"
COLOR_GA       = "#7b3294"
COLOR_REINVENT = "#3a8a3a"

# --- L1-L5: filled markers ---
for lead in TOP5:
    color = COLOR_GUIDED if lead["guided"] else COLOR_UNGUIDED
    sz = h50_size(lead["h50"])
    ax.scatter(lead["viab"], lead["S"], s=sz, c=color,
               edgecolors="white", linewidths=1.0, zorder=6, alpha=0.94)
    ax.annotate(lead["id"],
                xy=(lead["viab"], lead["S"]),
                xytext=(9, 4), textcoords="offset points",
                fontsize=10, fontfamily="DejaVu Serif", fontweight="bold",
                color="#111111", zorder=8)

# --- L9-L20: hollow markers spread along the Pareto floor strip ---
rng = np.random.default_rng(seed=42)
n_lower = len(LOWER_LEADS)
xs_floor = np.linspace(VIAB_THRESH + 0.005, 0.985, n_lower)
ys_floor = S_THRESH - 0.045 + rng.uniform(-0.012, 0.012, n_lower)
for (xf, yf, lead) in zip(xs_floor, ys_floor, LOWER_LEADS):
    sz = h50_size(lead["h50"])
    ax.scatter(xf, yf, s=sz, facecolors="none", edgecolors=COLOR_LOWER,
               linewidths=1.4, zorder=4, alpha=0.75, marker="o")
    ax.annotate(lead["id"], xy=(xf, yf),
                xytext=(0, -13), textcoords="offset points",
                fontsize=7.8, fontfamily="DejaVu Serif", color="#345b7a",
                ha="center", va="top", zorder=7)

# Bracket annotation for the L9-L20 strip - placed DIRECTLY BELOW the strip
# at its centre x (~0.91) with a short vertical arrow. This routing avoids
# REINVENT 4 (x=0.78) and SELFIES-GA (x=0.45) entirely.
ax.annotate("L9-L20: h$_{50}$ measured;\nS / viability not tabulated\n(strip placed at Pareto floor)",
            xy=(0.91, S_THRESH - 0.052),
            xytext=(0.91, 0.42),
            fontsize=8.2, fontfamily="DejaVu Serif", color="#345b7a",
            ha="center", va="top",
            style="italic",
            arrowprops=dict(arrowstyle="->", color="#7a9bb5", lw=0.9,
                            shrinkA=2, shrinkB=4),
            zorder=6)

# ---------------------------------------------------------------------------
# Baseline reference markers (different scales for S; placed as annotations)
# ---------------------------------------------------------------------------
# SMILES-LSTM: exact memorisation (Tanimoto 1.0). Placed above L1 with
# annotation going DOWN-LEFT so it doesn't collide with the PRODUCTIVE
# QUADRANT header (y >= 0.89) or with L1 marker at (1.00, 0.77).
ax.scatter(1.005, 0.84, marker="X", s=170, c=COLOR_LSTM,
           edgecolors="white", linewidths=0.9, zorder=5, alpha=0.92)
ax.annotate("SMILES-LSTM\n(18.3% exact memo;\nTanimoto 1.0)",
            xy=(1.005, 0.84), xytext=(-12, -6), textcoords="offset points",
            fontsize=8.2, fontfamily="DejaVu Serif", color=COLOR_LSTM,
            ha="right", va="top", style="italic", zorder=8,
            fontweight="semibold")

# MolMIM 70M: drug-domain pretrain; S not on the same scale, marker placed
# at the low-S, low-viab corner with annotation.
ax.scatter(0.32, 0.20, marker="D", s=120, c=COLOR_MOLMIM,
           edgecolors="white", linewidths=0.9, zorder=5, alpha=0.92)
ax.annotate("MolMIM 70M\n(drug-domain;\nD = 7.70 km/s)",
            xy=(0.32, 0.20), xytext=(12, -2), textcoords="offset points",
            fontsize=8.0, fontfamily="DejaVu Serif", color=COLOR_MOLMIM,
            ha="left", va="top", style="italic", zorder=8,
            fontweight="semibold")

# REINVENT 4: novel high-N heterocycles, peak D=9.02 (top-1 by N-fraction).
# S not on identical scale (N-fraction RL reward); placed in the Tanimoto
# novelty window above the floor.
ax.scatter(0.78, 0.55, marker="s", s=130, c=COLOR_REINVENT,
           edgecolors="white", linewidths=0.9, zorder=5, alpha=0.92)
ax.annotate("REINVENT 4\n(N-frac RL;\ntop-1 D = 9.02 km/s,\nnovel)",
            xy=(0.78, 0.55), xytext=(-12, -3), textcoords="offset points",
            fontsize=8.0, fontfamily="DejaVu Serif", color=COLOR_REINVENT,
            ha="right", va="top", style="italic", zorder=8,
            fontweight="semibold")

# SELFIES-GA: critical comparison - surrogate looks great, DFT collapses.
# Show with a downward arrow from surrogate D = 9.73 km/s position to
# DFT D = 6.28 km/s position (vertical drop annotation).
ax.scatter(0.45, 0.74, marker="^", s=140, c=COLOR_GA,
           edgecolors="white", linewidths=0.9, zorder=5, alpha=0.92)
ax.scatter(0.45, 0.30, marker="v", s=140, c=COLOR_GA,
           edgecolors="white", linewidths=0.9, zorder=5, alpha=0.92)
ax.annotate("",
            xy=(0.45, 0.32), xytext=(0.45, 0.72),
            arrowprops=dict(arrowstyle="-|>", color=COLOR_GA, lw=2.0,
                            mutation_scale=18, alpha=0.85),
            zorder=4)
ax.annotate("SELFIES-GA\nbest novel candidate\nsurrogate: D = 9.73 km/s",
            xy=(0.45, 0.74), xytext=(10, 4), textcoords="offset points",
            fontsize=8.0, fontfamily="DejaVu Serif", color=COLOR_GA,
            ha="left", va="bottom", style="italic", zorder=8,
            fontweight="semibold")
ax.annotate("DFT audit:\nD = 6.28 km/s\n(3.5 km/s artefact)",
            xy=(0.45, 0.30), xytext=(10, -5), textcoords="offset points",
            fontsize=8.0, fontfamily="DejaVu Serif", color=COLOR_GA,
            ha="left", va="top", style="italic", zorder=8,
            fontweight="semibold")

# Axis formatting
ax.set_xlabel("Viability probability (Random-Forest classifier, energetic vs ZINC)",
              fontsize=10.5, fontfamily="DejaVu Serif")
ax.set_ylabel("Composite score S  (higher = better; top-5 threshold = 0.65)",
              fontsize=10.5, fontfamily="DejaVu Serif")
ax.set_xlim(0.25, 1.05)
ax.set_ylim(0.10, 0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(color="#eeeeee", linewidth=0.55, zorder=0)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("DejaVu Serif")
    lbl.set_fontsize(9.5)

# Threshold tick labels
ax.text(0.255, S_THRESH + 0.005, "S = 0.65",
        fontsize=8.0, fontfamily="DejaVu Serif", color="#555555", va="bottom")
ax.text(VIAB_THRESH + 0.003, 0.115, "viab = 0.83",
        fontsize=8.0, fontfamily="DejaVu Serif", color="#555555",
        va="bottom", rotation=90)

# Legend - placed BELOW the x-axis so the full plot area is available for
# data points. Two side-by-side legend boxes: (left) method markers in a
# 4-column grid; (right) h50 size scale in a 3-column row.
legend_methods = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_GUIDED,
           markersize=11, label="L1 - guided C2 (viab+sens+hazard)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_UNGUIDED,
           markersize=11, label="L2-L5 - unguided C0"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
           markeredgecolor=COLOR_LOWER, markersize=11, markeredgewidth=1.5,
           label="L9-L20 (h50 real; S / viab at floor)"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor=COLOR_LSTM,
           markersize=11, label="SMILES-LSTM"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_MOLMIM,
           markersize=11, label="MolMIM 70M"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_REINVENT,
           markersize=11, label="REINVENT 4"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_GA,
           markersize=11,
           label="SELFIES-GA (surrogate $\\to$ DFT collapse)"),
]
leg_methods = fig.legend(
    handles=legend_methods,
    loc="lower left", bbox_to_anchor=(0.06, 0.005),
    fontsize=9.0, frameon=True, framealpha=0.96, edgecolor="#cccccc",
    prop={"family": "DejaVu Serif", "size": 9.0},
    title="Method", title_fontsize=10,
    handletextpad=0.6, borderpad=0.7, labelspacing=0.55,
    ncol=2, columnspacing=1.6,
)
leg_methods.get_title().set_fontfamily("DejaVu Serif")
leg_methods.get_title().set_fontweight("bold")

# Size-scale legend: marker-area encoding for h50 (3-step scale)
def _h50_legend_marker(h50_value):
    # markersize is in points; scatter s=area_pt^2. Convert s -> diameter (pt).
    s_pts2 = h50_size(h50_value)
    diameter_pt = (s_pts2 ** 0.5)
    return Line2D([0], [0], marker="o", color="w",
                  markerfacecolor="#aaaaaa", markeredgecolor="#666666",
                  markersize=diameter_pt, alpha=0.85,
                  label=f"h50 = {h50_value} cm")

legend_size = [
    _h50_legend_marker(25),
    _h50_legend_marker(50),
    _h50_legend_marker(85),
]
fig.legend(
    handles=legend_size,
    loc="lower right", bbox_to_anchor=(0.985, 0.005),
    fontsize=9.0, frameon=True, framealpha=0.96, edgecolor="#cccccc",
    prop={"family": "DejaVu Serif", "size": 9.0},
    title="Marker area = h$_{50}$ (cm); larger = safer",
    title_fontsize=9.5,
    handletextpad=1.0, borderpad=0.7, labelspacing=1.6,
    ncol=3, columnspacing=2.4,
)

# Footer source-note removed; the same information lives in the figcaption
# inside short_paper.html. Removing it eliminates the overlap with the
# bottom-left Method legend and gives the legends the full bottom strip.

# Reserve room at the bottom for the legends
fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))

os.makedirs(OUT_DIR, exist_ok=True)
png_path = os.path.join(OUT_DIR, "fig_quadrant_scatter.png")
svg_path = os.path.join(OUT_DIR, "fig_quadrant_scatter.svg")
fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(svg_path,            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {png_path}")
print(f"Saved: {svg_path}")
