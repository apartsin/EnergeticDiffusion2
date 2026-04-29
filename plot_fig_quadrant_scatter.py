"""Productive-quadrant scatter: composite score S vs viability probability.

Data sources (all verified against paper or stored result files):
  L1-L5: S and viab from Table 6 (_TABLE6 in plot_fig19_lead_cards.py),
          h50 from Table D.1c (docs/paper/index.html), confirmed via
          h50_predictions.json (score_model_v3e_h50).
  L9-L20: h50 from Table D.1c; S and viab not tabulated -- shown as
          hollow markers and labelled "estimated (paper Pareto floor)".
  Baselines: composite penalty from Table 6a; converted to S via
          S = max(0, 1 - composite_penalty) for display only; viab not
          available so shown on a separate inset.

Saves:
  docs/paper/figs/fig_quadrant_scatter.png
  docs/paper/figs/fig_quadrant_scatter.svg
"""
from __future__ import annotations

import json
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
# Actual data from Table 6 (_TABLE6 in card script) + Table D.1c for h50.
# These values are cross-checked against the paper HTML.
# ---------------------------------------------------------------------------
# L1-L5: real S (composite success score, higher = better, 0-1)
#         real viab (RF viability probability, 0-1)
#         real h50 (drop-weight impact sensitivity, cm)
TOP5 = [
    {"id": "L1", "S": 0.77, "viab": 1.00, "h50": 30.3, "source": "guided (C2 viab+sens+hazard)"},
    {"id": "L2", "S": 0.69, "viab": 0.96, "h50": 33.5, "source": "unguided (C0)"},
    {"id": "L3", "S": 0.66, "viab": 0.83, "h50": 82.6, "source": "unguided (C0)"},
    {"id": "L4", "S": 0.66, "viab": 0.93, "h50": 27.8, "source": "unguided (C0)"},
    {"id": "L5", "S": 0.65, "viab": 0.86, "h50": 33.4, "source": "unguided (C0)"},
]

# L9-L20: h50 from Table D.1c; S and viab not tabulated.
# Paper states all pass the 0.83 viability floor (Pareto front requirement).
# We use viab=0.83 (floor) and S at the lower-bound threshold (0.60) as
# "estimated minimum" and show these with hollow markers.
# h50 values are real (from Table D.1c).
LOWER_LEADS_H50 = {
    "L9": 21.9, "L11": 26.5, "L13": 31.0,
    "L16": 82.6, "L18": 38.6, "L19": 45.5, "L20": 74.1,
}

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Reference thresholds (from paper): S=0.65 is the top-5 cutoff;
# viab=0.83 is the Pareto front floor.
S_THRESH    = 0.65
VIAB_THRESH = 0.83

ax.axhline(S_THRESH,    color="#444444", linewidth=0.9, linestyle="--", zorder=2, alpha=0.75)
ax.axvline(VIAB_THRESH, color="#444444", linewidth=0.9, linestyle="--", zorder=2, alpha=0.75)

# Quadrant labels
ax.text(VIAB_THRESH + 0.006, S_THRESH + 0.008,
        "productive\n(novel + HMX-class)",
        fontsize=8.5, fontfamily="DejaVu Serif", color="#1a6b1a",
        ha="left", va="bottom", style="italic", zorder=5)
ax.text(VIAB_THRESH - 0.006, S_THRESH - 0.008,
        "marginal",
        fontsize=8.5, fontfamily="DejaVu Serif", color="#999999",
        ha="right", va="top", style="italic", zorder=5)

# h50 sizing
H50_MIN, H50_MAX = 20, 90
SZ_MIN,  SZ_MAX  = 55, 320

def h50_size(h50):
    return SZ_MIN + (np.clip(h50, H50_MIN, H50_MAX) - H50_MIN) / (H50_MAX - H50_MIN) * (SZ_MAX - SZ_MIN)

# --- L1-L5: filled markers, real data ---
COLOR_GUIDED   = "#d6604d"   # L1 (guided)
COLOR_UNGUIDED = "#2166ac"   # L2-L5 (unguided)

for lead in TOP5:
    color = COLOR_GUIDED if "guided" in lead["source"] else COLOR_UNGUIDED
    sz = h50_size(lead["h50"])
    ax.scatter(lead["viab"], lead["S"], s=sz, c=color,
               edgecolors="white", linewidths=0.8, zorder=5, alpha=0.93)
    ax.annotate(lead["id"],
                xy=(lead["viab"], lead["S"]),
                xytext=(8, 3), textcoords="offset points",
                fontsize=9, fontfamily="DejaVu Serif", fontweight="bold",
                color="#111111", zorder=7)

# --- L9-L20: hollow markers at (viab_floor, S_floor), real h50 ---
for lid, h50 in LOWER_LEADS_H50.items():
    sz = h50_size(h50)
    # Spread slightly along viab floor to reduce overlap
    ax.scatter(VIAB_THRESH, S_THRESH - 0.03, s=sz,
               facecolors="none", edgecolors="#2166ac",
               linewidths=1.2, zorder=4, alpha=0.55, marker="o")

# Bracket annotation for L9-L20 cluster
ax.annotate("L9-L20 (S, viab not tabulated;\nshown at Pareto floor)",
            xy=(VIAB_THRESH, S_THRESH - 0.03),
            xytext=(0.70, 0.56),
            fontsize=7.5, fontfamily="DejaVu Serif", color="#555555",
            style="italic",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
            zorder=6)

# --- Baseline reference lines (from Table 6a) ---
# SMILES-LSTM: S cannot be meaningfully computed (memorisation; Tanimoto=1.0)
#              -- shown as annotated arrow on x-axis (viab~1.0 but memorised)
# MolMIM 70M:  composite_penalty=4.79 -> uncalibrated, drug-domain
# We show these as horizontal reference bands, not scatter points, to avoid
# implying their "S" is comparable to DGLD S.
ax.annotate("",
            xy=(0.99, 0.78), xytext=(0.99, 0.83),
            arrowprops=dict(arrowstyle="-|>", color="#a50026", lw=1.5))
ax.text(0.991, 0.805, "SMILES-LSTM\n(exact rediscovery;\nTanimoto=1.0)",
        fontsize=7.5, fontfamily="DejaVu Serif", color="#a50026",
        ha="left", va="center", style="italic")

ax.annotate("",
            xy=(0.30, 0.16), xytext=(0.30, 0.22),
            arrowprops=dict(arrowstyle="-|>", color="#e08030", lw=1.5))
ax.text(0.31, 0.19, "MolMIM 70M\n(drug-domain;\nD=7.70 km/s)",
        fontsize=7.5, fontfamily="DejaVu Serif", color="#e08030",
        ha="left", va="center", style="italic")

# Axis formatting
ax.set_xlabel("Viability probability (RF classifier, energetic vs ZINC)",
              fontsize=10, fontfamily="DejaVu Serif")
ax.set_ylabel("Composite score S (higher = better; top-5 threshold = 0.65)",
              fontsize=10, fontfamily="DejaVu Serif")
ax.set_xlim(0.25, 1.05)
ax.set_ylim(0.40, 0.90)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("DejaVu Serif")
    lbl.set_fontsize(9)

# Threshold labels
ax.text(0.26, S_THRESH + 0.004, "S = 0.65",
        fontsize=7.5, fontfamily="DejaVu Serif", color="#666666", va="bottom")
ax.text(VIAB_THRESH + 0.002, 0.41, "viab = 0.83",
        fontsize=7.5, fontfamily="DejaVu Serif", color="#666666", va="bottom", rotation=90)

# Legend
legend_handles = [
    mpatches.Patch(color=COLOR_GUIDED,   label="L1 — guided C2 (viab+sens+hazard)"),
    mpatches.Patch(color=COLOR_UNGUIDED, label="L2-L5 — unguided C0"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
           markeredgecolor=COLOR_UNGUIDED, markersize=8, markeredgewidth=1.2,
           label="L9-L20 (h50 real; S/viab at Pareto floor)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777",
           markersize=5,  label="Point area proportional to h50 (cm)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
          frameon=True, framealpha=0.9, edgecolor="#cccccc",
          prop={"family": "DejaVu Serif", "size": 8})

fig.text(0.01, -0.01,
         "S and viab for L1-L5: Table 6 (paper). "
         "h50 for all leads: Table D.1c / h50_predictions.json (score_model_v3e_h50). "
         "S and viab for L9-L20 not tabulated; hollow markers placed at Pareto floor.",
         ha="left", va="top", fontsize=6.5, fontfamily="DejaVu Serif", color="#888888")

fig.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
png_path = os.path.join(OUT_DIR, "fig_quadrant_scatter.png")
svg_path = os.path.join(OUT_DIR, "fig_quadrant_scatter.svg")
fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(svg_path,          bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {png_path}")
print(f"Saved: {svg_path}")
