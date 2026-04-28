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

# Literature anchors with experimental D, rho, P
ANCHORS = [
    # name,  novelty, D km/s, rho g/cm3, P GPa
    ("RDX",  0.0, 8.75, 1.82, 34.0),
    ("HMX",  0.0, 9.10, 1.91, 39.0),
    ("CL-20",0.0, 9.40, 2.04, 42.5),
    ("TATB", 0.0, 7.86, 1.93, 31.5),
    ("PETN", 0.0, 8.50, 1.77, 33.0),
]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=150, sharex=True)
axA, axB, axC = axes

PROPS = [
    (axA, "top1_D_kms", "D (km/s)\n[3D-CNN surrogate]", (7.0, 10.0), "D",   3),  # ANCHORS index for D
    (axB, "top1_rho",   r"$\rho$ (g/cm$^3$)" + "\n[3D-CNN surrogate]", (1.55, 2.10), "rho", 4),
    (axC, "top1_P_GPa", "P (GPa)\n[3D-CNN surrogate]", (22.0, 45.0), "P",   5),
]
ANCHOR_IDX = {"D": 2, "rho": 3, "P": 4}

def plot_panel(ax, ykey, ylabel, ylim, prop, _anchor_idx):
    # DGLD
    for r in data:
        if r["condition"] in DGLD_CONDS and r["novelty"] is not None and r.get(ykey):
            ax.scatter(r["novelty"], r[ykey], s=85, marker="o",
                       facecolor="#2a73c8", edgecolor="black", linewidths=0.6,
                       alpha=0.85, zorder=3)
    # LSTM
    for r in data:
        if r["condition"] in LSTM_CONDS and r["novelty"] is not None and r.get(ykey):
            ax.scatter(r["novelty"], r[ykey], s=160, marker="X",
                       facecolor="#d6473a", edgecolor="black", linewidths=0.9,
                       alpha=0.95, zorder=4)
    # MolMIM
    for r in data:
        if r["condition"] in MOLMIM_CONDS and r["novelty"] is not None and r.get(ykey):
            ax.scatter(r["novelty"], r[ykey], s=120, marker="D",
                       facecolor="#e8a830", edgecolor="black", linewidths=0.7,
                       alpha=0.9, zorder=3)
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
    mpatches.Patch(facecolor="#7f7f7f", edgecolor="black",
                   label="literature anchors (RDX/HMX/CL-20/TATB/PETN, experimental)"),
]
fig.legend(handles=handles, loc="upper center", ncol=4, framealpha=0.95,
           fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Top-1 candidates of each method on three target properties:\n"
             "DGLD occupies the productive quadrant (novel + HMX-class) on every axis",
             fontsize=11, fontweight="bold", y=1.10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT / "novelty_vs_D.svg", format="svg", bbox_inches="tight")
plt.savefig(OUT / "novelty_vs_D.png", format="png", dpi=200, bbox_inches="tight")
print(f"[plot] -> {OUT / 'novelty_vs_D.svg'}")
print(f"[plot] -> {OUT / 'novelty_vs_D.png'}")
print("[plot] === DONE ===")
