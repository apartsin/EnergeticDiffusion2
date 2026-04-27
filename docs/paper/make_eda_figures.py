"""EDA + dataset figures for the paper. Reads the labelled master + tier
distribution + property histograms + scaffold counts."""
from __future__ import annotations
import csv, sys, json
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("E:/Projects/EnergeticDiffusion2")
OUT  = ROOT / "docs/paper/figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                      "axes.spines.top": False, "axes.spines.right": False,
                      "figure.dpi": 130})


def load_master():
    rows = []
    with open(ROOT / "data/training/master/labeled_master.csv", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            d = {}
            for k in ("density", "heat_of_formation", "detonation_velocity", "detonation_pressure"):
                v = r.get(k); d[k] = float(v) if v else None
            for k in ("density_tier", "heat_of_formation_tier", "detonation_velocity_tier", "detonation_pressure_tier"):
                d[k] = r.get(k) or ""
            d["smiles"] = r.get("smiles", "")
            rows.append(d)
    return rows


def fig_property_hists():
    rows = load_master()
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
    props = [("density", "ρ (g/cm³)", (0.5, 2.5)),
             ("heat_of_formation", "HOF (kJ/mol)", (-1500, 1500)),
             ("detonation_velocity", "D (km/s)", (2, 12)),
             ("detonation_pressure", "P (GPa)", (0, 60))]
    for ax, (k, lbl, (lo, hi)) in zip(axes.flat, props):
        vals = [r[k] for r in rows if r[k] is not None and lo <= r[k] <= hi]
        ax.hist(vals, bins=60, color="#1f77b4", edgecolor="white", alpha=0.85)
        ax.set_xlabel(lbl); ax.set_ylabel("# molecules")
        ax.set_title(f"{lbl} distribution (n={len(vals)})")
        ax.axvline(np.median(vals), color="#d62728", ls="--", lw=1,
                   label=f"median = {np.median(vals):.2f}")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Labelled energetic-materials corpus: per-property distributions", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_eda_property_hists.svg")
    plt.close(fig)
    print(f"  fig_eda_property_hists.svg")


def fig_tier_pie():
    rows = load_master()
    counts = Counter()
    for r in rows:
        for k in ("density_tier", "heat_of_formation_tier", "detonation_velocity_tier", "detonation_pressure_tier"):
            t = r[k]
            if t: counts[(k.replace("_tier",""), t)] += 1
    # 2x2 pie per property
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    props = ["density", "heat_of_formation", "detonation_velocity", "detonation_pressure"]
    cmap = {"A": "#2ca02c", "B": "#1f77b4", "C": "#ff7f0e", "D": "#d62728", "": "#cccccc"}
    for ax, p in zip(axes.flat, props):
        sub = {t: c for (pp, t), c in counts.items() if pp == p}
        labels = sorted(sub.keys())
        sizes = [sub[k] for k in labels]
        colors = [cmap.get(k, "#888") for k in labels]
        ax.pie(sizes, labels=[f"{l}: {s}" for l, s in zip(labels, sizes)],
                colors=colors, startangle=90, autopct="%1.0f%%",
                textprops={"fontsize": 8})
        ax.set_title(p.replace("_", " "))
    fig.suptitle("Label-tier composition by property (A: experimental; B: DFT; C: K-J; D: 3D-CNN smoke)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_eda_tier_composition.svg")
    plt.close(fig)
    print(f"  fig_eda_tier_composition.svg")


def fig_scaffold_atom_counts():
    rows = load_master()
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mws = []; nC = []; nN = []; nO = []; nNO2 = []
    nitro_p = Chem.MolFromSmarts("[N+](=O)[O-]")
    for r in rows[:30000]:  # cap for speed
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None: continue
        mws.append(Descriptors.MolWt(m))
        nC.append(sum(1 for a in m.GetAtoms() if a.GetSymbol()=="C"))
        nN.append(sum(1 for a in m.GetAtoms() if a.GetSymbol()=="N"))
        nO.append(sum(1 for a in m.GetAtoms() if a.GetSymbol()=="O"))
        nNO2.append(len(m.GetSubstructMatches(nitro_p)))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    axes[0].hist(mws, bins=50, color="#1f77b4", edgecolor="white")
    axes[0].set_xlabel("MW"); axes[0].set_title(f"MW (n={len(mws)})"); axes[0].set_ylabel("# molecules")
    axes[1].hist(nNO2, bins=range(0, 12), color="#ff7f0e", edgecolor="white", align="left")
    axes[1].set_xlabel("nitro count"); axes[1].set_title(f"# of [N+](=O)[O-] groups")
    axes[1].set_xticks(range(0, 12))
    axes[2].hist2d(nC, nN, bins=[range(0, 25), range(0, 18)], cmap="viridis")
    axes[2].set_xlabel("# C"); axes[2].set_ylabel("# N")
    axes[2].set_title("C vs N atom counts")
    fig.suptitle("Atom-composition descriptors over the labelled corpus")
    fig.tight_layout()
    fig.savefig(OUT / "fig_eda_atom_counts.svg")
    plt.close(fig)
    print(f"  fig_eda_atom_counts.svg")


def fig_property_pairplot():
    rows = load_master()
    import numpy as np
    # density vs D
    pts = [(r["density"], r["detonation_velocity"]) for r in rows
           if r["density"] is not None and r["detonation_velocity"] is not None]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if pts:
        d, v = zip(*pts)
        ax.scatter(d, v, s=4, alpha=0.4, color="#1f77b4")
        ax.set_xlim(0.5, 2.5); ax.set_ylim(2, 12)
        ax.set_xlabel("density (g/cm³)"); ax.set_ylabel("detonation velocity (km/s)")
        ax.set_title(f"Joint density vs detonation-velocity (n={len(pts)})")
        ax.grid(True, alpha=0.3)
        # Plot anchors
        anchors = {"TNT": (1.65, 6.95), "RDX": (1.81, 8.75), "HMX": (1.91, 9.10),
                   "PETN": (1.77, 8.30), "TATB": (1.93, 7.95), "CL-20": (2.04, 9.66)}
        for n, (dd, vv) in anchors.items():
            ax.scatter([dd], [vv], marker="*", s=160, c="white", edgecolor="black", linewidth=1.5, zorder=3)
            ax.annotate(n, (dd, vv), xytext=(7, 4), textcoords="offset points", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_eda_density_vs_velocity.svg")
    plt.close(fig)
    print(f"  fig_eda_density_vs_velocity.svg")


if __name__ == "__main__":
    print("Generating EDA figures...")
    fig_property_hists()
    fig_tier_pie()
    fig_scaffold_atom_counts()
    fig_property_pairplot()
    print("done.")
