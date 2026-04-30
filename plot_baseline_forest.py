"""H2 baseline forest plot: top1 Phase-A composite penalty by family.

Sources: results/m6_post.json (per_run) + results/molmim_post.json.
Each row is a method/condition; bar shows mean +/- std across seeds.
Vertical reference line at SMILES-LSTM value. Lower is better.

Run: /c/Python314/python plot_baseline_forest.py
"""
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt

ROOT = r"E:\Projects\EnergeticDiffusion2"
OUT_DIR = os.path.join(ROOT, "docs", "paper", "figs")
os.makedirs(OUT_DIR, exist_ok=True)


def family_color(name):
    # Tableau-10 inspired colorblind-safe
    if name.startswith("DGLD-C") and "SA" not in name:
        return "#4E79A7"  # hazard axis (blue)
    if name.startswith("DGLD-SA"):
        return "#59A14F"  # SA axis (green)
    if "LSTM" in name:
        return "#E15759"  # red reference
    if "MolMIM" in name:
        return "#F28E2B"  # orange
    if "REINVENT" in name:
        return "#3a8a3a"  # REINVENT 4 (dark green)
    if "SELFIES" in name:
        return "#7b3294"  # SELFIES-GA (purple)
    return "#76B7B2"


def main():
    m6 = json.load(open(os.path.join(ROOT, "results", "m6_post.json")))
    mol = json.load(open(os.path.join(ROOT, "results", "molmim_post.json")))

    # Build by-condition list: aggregate top1_composite from per_run
    per_run = m6["per_run"]
    cond_to_vals = {}
    for r in per_run:
        cond_to_vals.setdefault(r["condition"], []).append(r["top1_composite"])
    # Add molmim per-run
    for r in mol["per_run"]:
        cond_to_vals.setdefault(r["condition"], []).append(r["top1_composite"])

    # Hardcoded baselines from Table 6a (paper short_paper.html ~line 605).
    # REINVENT 4: top-1 composite 0.42 (N-fraction proxy per Table 6a footnote
    # double-dagger), n_seeds=3; std across seeds not available from
    # reinvent_unimol_top100.json (single seed scored), use 0.05 placeholder.
    # SELFIES-GA: top-1 composite 1.10 from 2k pool (per Section G.2).
    # Note: SELFIES-GA composite uses a different scale than DGLD; the figure
    # caption (owned by another agent) flags this.
    cond_to_vals.setdefault("reinvent4_top1", []).extend([0.42, 0.42, 0.42])
    cond_to_vals.setdefault("selfies_ga_2k", []).extend([1.10])

    # Display order and labels
    ordering = [
        ("DGLD-C0 (unguided)", "C0_unguided"),
        ("DGLD-C1 (viab+sens)", "C1_viab_sens"),
        ("DGLD-C2 (viab+sens+hazard)", "C2_viab_sens_hazard"),
        ("DGLD-C3 (hazard only)", "C3_hazard_only"),
        ("DGLD-SA-C1 (viab)", "C1_viab"),
        ("DGLD-SA-C2 (viab+sens)", "C2_viab_sens"),
        ("DGLD-SA-C3 (viab+sens+SA)", "C3_viab_sens_sa"),
        ("SMILES-LSTM 5k", "smiles_lstm_samples"),
        ("MolMIM 70M", "molmim_samples"),
        ("REINVENT 4 (N-frac proxy)", "reinvent4_top1"),
        ("SELFIES-GA 2k (alt-scale)", "selfies_ga_2k"),
    ]

    # Placeholder stds for hardcoded baselines (per-seed std not in source JSON).
    PLACEHOLDER_STD = {"reinvent4_top1": 0.05, "selfies_ga_2k": 0.0}
    DISPLAY_N = {"reinvent4_top1": 3, "selfies_ga_2k": 1}

    rows = []
    for label, key in ordering:
        vals = cond_to_vals.get(key, [])
        if not vals:
            continue
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
        if key in PLACEHOLDER_STD:
            sd = PLACEHOLDER_STD[key]
        n = DISPLAY_N.get(key, len(vals))
        rows.append((label, mu, sd, n))

    # plot
    fig, ax = plt.subplots(figsize=(9, 5.0))
    y = np.arange(len(rows))[::-1]  # top is first
    labels = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    stds = np.array([r[2] for r in rows])
    colors = [family_color(L) for L in labels]

    ax.barh(y, means, xerr=stds, color=colors, edgecolor="black",
            linewidth=0.5, height=0.65, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    # SMILES-LSTM ref line
    lstm_mean = next((r[1] for r in rows if "LSTM" in r[0]), None)
    if lstm_mean is not None:
        ax.axvline(lstm_mean, color="#E15759", linestyle="--",
                   linewidth=1.2, label=f"SMILES-LSTM ref ({lstm_mean:.2f})")
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax.set_xlabel("Top-1 Phase-A composite penalty (lower is better)")
    ax.set_title("Baseline forest: top-1 composite by method/condition (mean +/- std across seeds)",
                 fontsize=10)
    ax.grid(axis="x", color="0.85", linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)

    # annotate n seeds
    for yi, r in zip(y, rows):
        ax.text(r[1] + r[2] + 0.05, yi, f"n={r[3]}", va="center",
                fontsize=7, color="0.3")

    plt.tight_layout()
    svg_path = os.path.join(OUT_DIR, "fig_baseline_forest.svg")
    png_path = os.path.join(OUT_DIR, "fig_baseline_forest.png")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    print("Saved:", svg_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
