"""H5 xTB strip with structures.

For each top-N candidate: RDKit depiction, HOMO-LUMO gap (eV), gap-gate verdict
(>=1.5 eV pass / fail), and Phase-A composite (if available).

Sources: experiments/xtb_topN.json (or xtb_merged_top15.json) for {smi, gap_ev,
graph_survives}. Composite is matched from results/m6_post.json per-run by
SMILES if present; otherwise omitted.

Run: /c/Python314/python plot_xtb_strip.py
"""
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

ROOT = r"E:\Projects\EnergeticDiffusion2"
OUT_DIR = os.path.join(ROOT, "docs", "paper", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

XTB_CANDIDATES = [
    os.path.join(ROOT, "experiments", "xtb_topN.json"),
    os.path.join(ROOT, "experiments", "xtb_merged_top15.json"),
]

GAP_THRESHOLD = 1.5  # eV
PASS_COLOR = "#59A14F"
FAIL_COLOR = "#E15759"


def render_mol(smi, size=(220, 180)):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    AllChem.Compute2DCoords(m)
    return np.asarray(Draw.MolToImage(m, size=size, kekulize=True))


def load_xtb():
    for p in XTB_CANDIDATES:
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                if isinstance(d, list) and d:
                    return p, d
            except Exception as e:
                print(f"could not parse {p}: {e}")
    return None, None


def main():
    src, rows = load_xtb()
    if rows is None:
        print("No xTB JSON found. Skipping figure render.")
        print("Expected schema (list of dicts):")
        print("  smi: str, gap_ev: float, converged: bool, rank: int,")
        print("  graph_survives: bool, post_smi: str (optional)")
        return
    print(f"Using {src} ({len(rows)} rows)")

    # Take top N=12
    rows = sorted(rows, key=lambda r: r.get("rank", 9999))[:12]

    nrow = len(rows)
    fig, axes = plt.subplots(nrow, 1, figsize=(8.5, 1.0 * nrow + 0.5))
    if nrow == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        ax.set_xticks([])
        ax.set_yticks([])
        # Try post_smi if graph_survives else original smi
        smi = r.get("smi", "")
        gap = float(r.get("gap_ev", float("nan")))
        passed = (gap >= GAP_THRESHOLD)
        col = PASS_COLOR if passed else FAIL_COLOR
        for s in ax.spines.values():
            s.set_edgecolor(col)
            s.set_linewidth(2.0)

        img = render_mol(smi)
        # composite axis layout: image on far left
        if img is not None:
            # add as inset using extent
            ax.imshow(img, extent=(0, 1.6, -0.5, 0.5), aspect="auto")
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, 0.5)

        verdict = "PASS (gap>=1.5 eV)" if passed else "FAIL (gap<1.5 eV)"
        survives = r.get("graph_survives")
        survives_str = "graph survives" if survives else "graph altered"
        info = (f"rank{r.get('rank','?')}   gap={gap:.2f} eV   "
                f"{verdict}   |   {survives_str}\n"
                f"SMILES: {smi[:80]}")
        ax.text(1.8, 0.0, info, va="center", fontsize=8.5,
                family="monospace")

    fig.suptitle(f"xTB GFN2 HOMO-LUMO gap gate (top-{nrow}, threshold 1.5 eV)",
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    svg_path = os.path.join(OUT_DIR, "fig_xtb_strip.svg")
    png_path = os.path.join(OUT_DIR, "fig_xtb_strip.png")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    print("Saved:", svg_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
