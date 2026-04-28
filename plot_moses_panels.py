"""H3 MOSES-style small multiples: 6 panels of distribution-learning metrics.

Panels: validity, uniqueness, novelty, internal-diversity, FCD, scaffold-NN-Tanimoto.
Sources: results/fcd_results.json (per_condition) + results/m6_post.json (per_run aggregates).

For metrics not directly stored (validity, uniqueness, novelty, scaffold-NN-Tanimoto),
we approximate from m6_post.json fields where possible:
- internal-diversity: topN_internal_diversity (mean across seeds)
- FCD: from fcd_results.per_condition (lower is better)
- validity/uniqueness/novelty/scaffold-NN: not directly available so we plot
  proxies with explicit annotation, or skip with a note panel.

Run: /c/Python314/python plot_moses_panels.py
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = r"E:\Projects\EnergeticDiffusion2"
OUT_DIR = os.path.join(ROOT, "docs", "paper", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

CONDITION_LABELS = [
    ("smiles_lstm", "SMILES-LSTM"),
    ("C0_unguided", "DGLD-C0"),
    ("C1_viab", "DGLD-C1v"),
    ("C1_viab_sens", "DGLD-C1vs"),
    ("C2_viab_sens", "DGLD-C2vs"),
    ("C2_viab_sens_hazard", "DGLD-C2vsh"),
    ("C3_hazard_only", "DGLD-C3h"),
    ("C3_viab_sens_sa", "DGLD-C3sa"),
]

COLOR_LSTM = "#E15759"
COLOR_DGLD = "#4E79A7"


def color_for(key):
    return COLOR_LSTM if key == "smiles_lstm" else COLOR_DGLD


def main():
    fcd = json.load(open(os.path.join(ROOT, "results", "fcd_results.json")))
    m6 = json.load(open(os.path.join(ROOT, "results", "m6_post.json")))
    pool = json.load(open(os.path.join(ROOT, "results", "pool_metrics.json")))
    per_run = m6["per_run"]

    # internal diversity by condition (from m6_post per-run aggregates)
    intdiv = {}
    for r in per_run:
        c = r["condition"]
        intdiv.setdefault(c, []).append(r["topN_internal_diversity"])
    if "smiles_lstm_samples" in intdiv:
        intdiv["smiles_lstm"] = intdiv.pop("smiles_lstm_samples")

    # Pool-level metrics (real, computed by compute_pool_metrics.py)
    pool_pc = pool["per_condition"]
    fcd_pc = fcd["per_condition"]

    def from_intdiv(k):
        v = intdiv.get(k, [])
        return float(np.mean(v)) if v else None

    def from_pool(k, metric):
        c = pool_pc.get(k)
        return float(c[f"{metric}_mean"]) if c and f"{metric}_mean" in c else None

    panels = []

    # 1. Validity (real, from pool_metrics; fraction RDKit-parseable per pool)
    vals = []
    for k, lbl in CONDITION_LABELS:
        v = from_pool(k, "validity")
        if v is not None:
            vals.append((lbl, v, color_for(k)))
    panels.append(("RDKit validity (per pool)\n(higher is better)", vals, "fraction"))

    # 2. Uniqueness (real, from pool_metrics; distinct canonical / parseable)
    vals = []
    for k, lbl in CONDITION_LABELS:
        v = from_pool(k, "uniqueness")
        if v is not None:
            vals.append((lbl, v, color_for(k)))
    panels.append(("Pool uniqueness (distinct / parseable)\n(higher is better)",
                   vals, "fraction"))

    # 3. Novelty (real, from pool_metrics; fraction with max-Tanimoto < 0.7 to LM)
    vals = []
    for k, lbl in CONDITION_LABELS:
        v = from_pool(k, "novelty")
        if v is not None:
            vals.append((lbl, v, color_for(k)))
    panels.append(("Novelty fraction\n(rows with max-Tani < 0.7 to labelled master)",
                   vals, "fraction"))

    # 4. Internal diversity (m6_post topN_internal_diversity per condition)
    vals = []
    for k, lbl in CONDITION_LABELS:
        v = from_intdiv(k)
        if v is not None:
            vals.append((lbl, v, color_for(k)))
    panels.append(("Internal diversity (top-100, Tanimoto)\n(higher is better)",
                   vals, "fraction"))

    # 5. FCD (real, from fcd_results per_condition; lower is better)
    vals = []
    for k, lbl in CONDITION_LABELS:
        if k in fcd_pc:
            vals.append((lbl, fcd_pc[k]["mean"], color_for(k)))
    panels.append(("FCD vs labelled master\n(lower is better)", vals, "fcd"))

    # 6. Scaffold-NN Tanimoto (real, from pool_metrics; mean of per-row max
    #    Tanimoto on Bemis-Murcko scaffold FPs against the LM scaffold bank)
    vals = []
    for k, lbl in CONDITION_LABELS:
        v = from_pool(k, "scaffold_nn")
        if v is not None:
            vals.append((lbl, v, color_for(k)))
    panels.append(("Scaffold-NN Tanimoto vs labelled master\n(lower = more novel scaffolds)",
                   vals, "fraction"))

    # Render
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    for i, (title, data, kind) in enumerate(panels):
        ax = axes[i // 3, i % 3]
        ax.set_title(title, fontsize=9.5)
        if data is None:
            ax.text(0.5, 0.5, "No data available\n(see caption)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="0.4")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        labs = [d[0] for d in data]
        v = np.array([d[1] for d in data])
        cs = [d[2] for d in data]
        y = np.arange(len(v))[::-1]
        ax.barh(y, v, color=cs, edgecolor="black", linewidth=0.4, height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=8)
        for yi, vv in zip(y, v):
            ax.text(vv, yi, f" {vv:.2f}" if kind != "count" else f" {vv:.0f}",
                    va="center", fontsize=7)
        ax.grid(axis="x", color="0.88", linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)

    fig.suptitle("Distribution-learning small-multiples (DGLD vs SMILES-LSTM)",
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    svg_path = os.path.join(OUT_DIR, "fig_moses_small_multiples.svg")
    png_path = os.path.join(OUT_DIR, "fig_moses_small_multiples.png")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    print("Saved:", svg_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
