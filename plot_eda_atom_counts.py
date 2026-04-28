"""Regenerate fig_eda_atom_counts.svg/png.

Atom-composition descriptors over a 30k subsample of the labelled corpus:
molecular weight (computed from SMILES), nitro-group count, joint C-vs-N
atom counts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "m6_postprocess_bundle" / "labelled_master.csv"
OUT_SVG = ROOT / "docs" / "paper" / "figs" / "fig_eda_atom_counts.svg"
OUT_PNG = ROOT / "docs" / "paper" / "figs" / "fig_eda_atom_counts.png"


def mw_from_smiles(smiles: str) -> float:
    """Crude MW estimator from atom-letter counts in a SMILES string.

    Avoids RDKit dependency. Counts C, H (implicit not counted; only explicit),
    N, O, F, Cl, Br, S, P. Good enough for a corpus-level histogram.
    """
    if not isinstance(smiles, str):
        return float("nan")
    weights = {"C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
               "Cl": 35.45, "Br": 79.904, "S": 32.06, "P": 30.974, "H": 1.008}
    s = smiles
    total = 0.0
    # two-letter atoms first
    for tok in ("Cl", "Br"):
        n = s.count(tok)
        total += n * weights[tok]
        s = s.replace(tok, "")
    for atom in ("C", "N", "O", "F", "S", "P", "H"):
        # crude: count uppercase + lowercase variants for aromatic c/n/o
        n = sum(1 for ch in s if ch == atom or ch == atom.lower())
        total += n * weights[atom]
    # rough H-add for typical CHNO energetic compounds
    return total


def count_nitro(smiles: str) -> int:
    if not isinstance(smiles, str):
        return 0
    # count "[N+](=O)[O-]" or "N(=O)=O" fragments, conservative
    return smiles.count("[N+](=O)[O-]") + smiles.count("N(=O)=O")


def main() -> None:
    df = pd.read_csv(CSV, usecols=["smiles", "n_count", "o_count"])
    df = df.dropna(subset=["smiles"])
    sub = df.sample(n=min(30000, len(df)), random_state=0).copy()

    sub["mw"] = sub["smiles"].apply(mw_from_smiles)
    sub["n_nitro"] = sub["smiles"].apply(count_nitro)
    # carbon count: cheap, count uppercase C minus Cl
    sub["c_count"] = sub["smiles"].apply(
        lambda s: sum(1 for ch in s if ch == "C") - s.count("Cl") if isinstance(s, str) else 0
    )

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "figure.dpi": 120,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    # MW histogram
    ax = axes[0]
    mws = sub["mw"][(sub["mw"] > 30) & (sub["mw"] < 800)]
    ax.hist(mws, bins=60, color="#2b6cb0", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Molecular weight (g/mol, est.)")
    ax.set_ylabel("Count")
    ax.set_title("Molecular weight")
    ax.grid(True, alpha=0.3)

    # Nitro group count
    ax = axes[1]
    nn = sub["n_nitro"].clip(0, 12)
    ax.hist(nn, bins=np.arange(0, 13) - 0.5, color="#c0392b",
            alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(r"Nitro group count ($-NO_2$)")
    ax.set_ylabel("Count")
    ax.set_title("Nitro-group distribution")
    ax.set_xticks(range(0, 13))
    ax.grid(True, alpha=0.3)

    # Joint C vs N
    ax = axes[2]
    cc = sub["c_count"].clip(0, 20)
    nc = sub["n_count"].clip(0, 20)
    h = ax.hist2d(cc, nc, bins=[np.arange(0, 22) - 0.5, np.arange(0, 22) - 0.5],
                  cmap="viridis", cmin=1)
    ax.set_xlabel("Carbon count")
    ax.set_ylabel("Nitrogen count")
    ax.set_title("Joint C vs N atom counts")
    fig.colorbar(h[3], ax=ax, label="Count")

    fig.suptitle(f"Atom-composition statistics (subsample n={len(sub):,})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"atom_counts: subsample={len(sub):,}  out={OUT_SVG.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
