"""High-quality novelty-vs-multi-property scatter (Fig 7c, 1x3 panel).

For each (method, condition, seed) top-1 candidate, plot novelty (1 - max-Tani
to labelled-master) on a shared x-axis vs three property axes:
  - panel A: predicted detonation velocity D (km/s)
  - panel B: predicted density rho (g/cm^3)
  - panel C: predicted detonation pressure P (GPa)

Each property axis carries the §5.13 caveat that values are 3D-CNN-surrogate
predictions; ground-truth-grade DFT-anchored values for the chem-pass leads
are reported separately in §5.13.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path("docs/paper/figs"); OUT.mkdir(exist_ok=True, parents=True)
data = json.load(open("results/novelty_top1.json"))["rows"]

DGLD_CONDS = {"C0_unguided", "C1_viab", "C1_viab_sens", "C2_viab_sens",
              "C2_viab_sens_hazard", "C3_hazard_only", "C3_viab_sens_sa"}
LSTM_CONDS = {"smiles_lstm_samples"}
MOLMIM_CONDS = {"molmim_samples"}

# Hardcoded baseline points (Table 6a / short_paper.html ~line 605).
# REINVENT 4 top-1 (Table 6a; seed-42 aminotetrazine top-1 by composite):
#   D=9.02 km/s, rho=1.853 g/cm^3, P=34.5 GPa, max-Tanimoto=0.57 -> novelty=0.43.
# SELFIES-GA 40k best-novel from baseline_bundle/results/selfies_ga_competitor_dft.json:
#   D_surrogate=9.737 km/s (=9.74), rho_surrogate=1.994, max-Tanimoto=0.35 -> novelty=0.65;
#   D_DFT=6.28 km/s collapse per Section 5.13 audit.
REINVENT_POINT = {"novelty": 0.43, "D": 9.02, "rho": 1.8527, "P": 34.5151}
SELFIES_GA = {
    "novelty": 0.65,
    "D_surrogate": 9.74,
    "D_DFT": 6.28,
    "rho_surrogate": 1.994,
    # P not anchored against DFT for the novel candidate; omit from rho/P panels.
}
REINVENT_COLOR = "#3a8a3a"
SELFIES_COLOR = "#7b3294"

# Memorisation rates from Table 6a; marker-area encoding: s = base * (1 - memo).
# DGLD = 0.0 (no LM memorisation), MolMIM ~0 (n.d.), REINVENT 4 = 0.001,
# SMILES-LSTM = 0.183, SELFIES-GA = 0.74.
MEMO = {
    "DGLD": 0.0,
    "LSTM": 0.183,
    "MOLMIM": 0.0,
    "REINVENT": 0.001,
    "SELFIES_GA": 0.74,
}

def memo_scale(method: str, base: float) -> float:
    """Marker area scaled by (1 - memorisation rate)."""
    return base * (1.0 - MEMO[method])

# Literature anchors with experimental D, rho, P
ANCHORS = [
    # name,  novelty, D km/s, rho g/cm3, P GPa
    ("RDX",  0.0, 8.75, 1.82, 34.0),
    ("HMX",  0.0, 9.10, 1.91, 39.0),
    ("CL-20",0.0, 9.40, 2.04, 42.5),
    ("TATB", 0.0, 7.86, 1.93, 31.5),
    ("PETN", 0.0, 8.50, 1.77, 33.0),
]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=150, sharex=True,
                         gridspec_kw={"width_ratios": [2, 1, 1]})
axA, axB, axC = axes

PROPS = [
    (axA, "top1_D_kms", "D (km/s)\n[3D-CNN surrogate]", (5.8, 10.0), "D",   3),  # ANCHORS index for D; lowered floor for SELFIES-GA D_DFT=6.28
    (axB, "top1_rho",   r"$\rho$ (g/cm$^3$)" + "\n[3D-CNN surrogate]", (1.55, 2.10), "rho", 4),
    (axC, "top1_P_GPa", "P (GPa)\n[3D-CNN surrogate]", (22.0, 45.0), "P",   5),
]
ANCHOR_IDX = {"D": 2, "rho": 3, "P": 4}

def plot_panel(ax, ykey, ylabel, ylim, prop, _anchor_idx):
    # DGLD (memo = 0; full size)
    for r in data:
        if r["condition"] in DGLD_CONDS and r["novelty"] is not None and r.get(ykey):
            ax.scatter(r["novelty"], r[ykey], s=memo_scale("DGLD", 85), marker="o",
                       facecolor="#2a73c8", edgecolor="black", linewidths=0.6,
                       alpha=0.85, zorder=3)
    # LSTM (memo = 0.183; marker shrunk to ~82% of base)
    for r in data:
        if r["condition"] in LSTM_CONDS and r["novelty"] is not None and r.get(ykey):
            ax.scatter(r["novelty"], r[ykey], s=memo_scale("LSTM", 160), marker="X",
                       facecolor="#d6473a", edgecolor="black", linewidths=0.9,
                       alpha=0.95, zorder=4)
    # MolMIM (memo n.d. ~ 0; full size)
    for r in data:
        if r["condition"] in MOLMIM_CONDS and r["novelty"] is not None and r.get(ykey):
            ax.scatter(r["novelty"], r[ykey], s=memo_scale("MOLMIM", 120), marker="D",
                       facecolor="#e8a830", edgecolor="black", linewidths=0.7,
                       alpha=0.9, zorder=3)
    # REINVENT 4 top-1 (single point on every panel; memo = 0.001 -> full size)
    rx = REINVENT_POINT["novelty"]
    ry = {"D": REINVENT_POINT["D"], "rho": REINVENT_POINT["rho"],
          "P": REINVENT_POINT["P"]}[prop]
    ax.scatter(rx, ry, s=memo_scale("REINVENT", 130), marker="s",
               facecolor=REINVENT_COLOR, edgecolor="black", linewidths=0.7,
               alpha=0.95, zorder=4)

    # SELFIES-GA: D panel shows surrogate->DFT collapse arrow with two triangles;
    # rho panel shows the novel candidate at rho_surrogate; P panel skipped.
    sx = SELFIES_GA["novelty"]
    # SELFIES-GA memo = 0.74; markers shrunk to 26% of base.
    if prop == "D":
        # Main panel A: keep the two markers (surrogate / DFT) so the encoding is
        # consistent across panels; the dedicated collapse inset (added below)
        # carries the explanatory annotation.
        y_top = SELFIES_GA["D_surrogate"]
        y_bot = SELFIES_GA["D_DFT"]
        ax.scatter(sx, y_top, s=memo_scale("SELFIES_GA", 130), marker="^",
                   facecolor=SELFIES_COLOR, edgecolor="black", linewidths=0.7,
                   alpha=0.95, zorder=4)
        ax.scatter(sx, y_bot, s=memo_scale("SELFIES_GA", 130), marker="v",
                   facecolor=SELFIES_COLOR, edgecolor="black", linewidths=0.7,
                   alpha=0.95, zorder=4)
    elif prop == "rho":
        ax.scatter(sx, SELFIES_GA["rho_surrogate"],
                   s=memo_scale("SELFIES_GA", 130), marker="^",
                   facecolor=SELFIES_COLOR, edgecolor="black", linewidths=0.7,
                   alpha=0.95, zorder=4)
    # P panel: SELFIES-GA top-1 was a corpus rediscovery; novel candidate
    # P_surrogate not anchored, so skip per spec.

    # Anchors
    for name, nov, D, rho, P in ANCHORS:
        y = {"D": D, "rho": rho, "P": P}[prop]
        ax.scatter(nov, y, s=110, marker="^",
                   facecolor="#7f7f7f", edgecolor="black", linewidths=0.6,
                   alpha=0.85, zorder=2)
        ax.annotate(name, xy=(nov, y), xytext=(nov - 0.04, y + (ylim[1]-ylim[0])*0.012),
                    fontsize=7.5, ha="right", color="#444")

    ax.axvline(x=0.45, color="grey", linestyle="--", linewidth=0.7, alpha=0.6, zorder=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(*ylim)
    ax.set_xlabel("novelty = 1 \u2212 max Tanimoto to labelled-master", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)


for ax, ykey, ylabel, ylim, prop, idx in PROPS:
    plot_panel(ax, ykey, ylabel, ylim, prop, idx)

# SELFIES-GA collapse inset on panel A (lower-left). Two markers connected by
# an arrow at novelty = 0.65: triangle-up at surrogate D = 9.74, triangle-down
# at DFT-anchored D = 6.28; the 3.5 km/s gap is the headline collapse.
ins = axA.inset_axes([0.04, 0.04, 0.28, 0.40])
sx = SELFIES_GA["novelty"]
y_top = SELFIES_GA["D_surrogate"]
y_bot = SELFIES_GA["D_DFT"]
ins.scatter([sx], [y_top], s=80, marker="^",
            facecolor=SELFIES_COLOR, edgecolor="black", linewidths=0.6, zorder=3)
ins.scatter([sx], [y_bot], s=80, marker="v",
            facecolor=SELFIES_COLOR, edgecolor="black", linewidths=0.6, zorder=3)
ins.annotate("", xy=(sx, y_bot + 0.15), xytext=(sx, y_top - 0.15),
             arrowprops=dict(arrowstyle="->", color=SELFIES_COLOR,
                             lw=1.4, shrinkA=0, shrinkB=0),
             zorder=4)
ins.text(sx + 0.04, (y_top + y_bot) / 2.0,
         "9.74 → 6.28\n(3.5 km/s artefact)",
         fontsize=6.5, color=SELFIES_COLOR, ha="left", va="center")
ins.set_xlim(sx - 0.10, sx + 0.40)
ins.set_ylim(y_bot - 0.6, y_top + 0.6)
ins.set_title("SELFIES-GA: surrogate vs DFT",
              fontsize=7, color=SELFIES_COLOR, pad=2)
ins.tick_params(axis="both", which="both", labelsize=6, length=2)
ins.set_xticks([sx]); ins.set_xticklabels(["0.65"])
ins.set_yticks([y_bot, y_top]); ins.set_yticklabels(["6.28", "9.74"])
ins.grid(True, linestyle=":", alpha=0.35)
for sp in ins.spines.values():
    sp.set_edgecolor(SELFIES_COLOR); sp.set_linewidth(0.6)

# Panel-level highlights for the productive quadrant on each axis
THRESHOLDS = {0: 9.0, 1: 1.85, 2: 35.0}  # D, rho, P "HMX-class" thresholds
for i, (ax, _, _, ylim, prop, _) in enumerate(PROPS):
    ax.axhspan(THRESHOLDS[i], ylim[1], xmin=0.5, xmax=1.0,
               facecolor="#a8d88c", alpha=0.16, zorder=0)
    ax.text(0.97, ylim[1] - (ylim[1]-ylim[0])*0.04,
            "novel + HMX-class",
            fontsize=8, color="#356b1f", ha="right", va="top",
            fontweight="bold", zorder=2)

# Panel labels
for i, (ax, _, _, _, prop, _) in enumerate(PROPS):
    label = ["A. Detonation velocity", "B. Density", "C. Detonation pressure"][i]
    ax.set_title(label, fontsize=10.5, fontweight="bold", loc="left")

# Shared legend at top
handles = [
    mpatches.Patch(facecolor="#2a73c8", edgecolor="black",
                   label="DGLD top-1 (7 conditions \u00d7 3 seeds)"),
    mpatches.Patch(facecolor="#d6473a", edgecolor="black",
                   label="SMILES-LSTM top-1 (1 seed; Tanimoto = 1.000 = exact LM match)"),
    mpatches.Patch(facecolor="#e8a830", edgecolor="black",
                   label="MolMIM 70M top-1 (drug-domain pretrain)"),
    mpatches.Patch(facecolor=REINVENT_COLOR, edgecolor="black",
                   label="REINVENT 4 top-1 (square; N-fraction proxy reward)"),
    mpatches.Patch(facecolor=SELFIES_COLOR, edgecolor="black",
                   label="SELFIES-GA 40k novel (\u25b2 surrogate, \u25bc DFT; D collapse)"),
    mpatches.Patch(facecolor="#7f7f7f", edgecolor="black",
                   label="literature anchors (RDX/HMX/CL-20/TATB/PETN, experimental)"),
    mpatches.Patch(facecolor="none", edgecolor="none",
                   label="marker area \u221d 1 \u2212 memorisation rate (Table 6a)"),
]
fig.legend(handles=handles, loc="upper center", ncol=4, framealpha=0.95,
           fontsize=8.5, bbox_to_anchor=(0.5, 1.04))
fig.suptitle("Top-1 candidates of each method on three target properties:\n"
             "DGLD occupies the productive quadrant (novel + HMX-class) on every axis",
             fontsize=11, fontweight="bold", y=1.10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT / "novelty_vs_D.svg", format="svg", bbox_inches="tight")
plt.savefig(OUT / "novelty_vs_D.png", format="png", dpi=200, bbox_inches="tight")
print(f"[plot] -> {OUT / 'novelty_vs_D.svg'}")
print(f"[plot] -> {OUT / 'novelty_vs_D.png'}")
print("[plot] === DONE ===")
