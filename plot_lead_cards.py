"""H1 lead-card grid: 12 chem-pass leads from M2 bundle.

Cell shows: RDKit 2D depiction, lead ID + chemotype, formula,
anchor-calibrated rho/D/P, color-coded by HMX-class threshold
(rho>=1.85, D>=9.0, P>=35).

Run: /c/Python314/python plot_lead_cards.py
"""
import json
import glob
import os
import math
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

LEAD_DIR = r"E:\Projects\EnergeticDiffusion2\m2_bundle\results"
OUT_DIR = r"E:\Projects\EnergeticDiffusion2\docs\paper\figs"
os.makedirs(OUT_DIR, exist_ok=True)

# Tableau-10 friendly: green pass, orange partial, red fail
PASS_COLOR = "#59A14F"
PARTIAL_COLOR = "#F28E2B"
FAIL_COLOR = "#E15759"
NEUTRAL = "#4E79A7"

# HMX-class thresholds
RHO_T = 1.85
D_T = 9.0
P_T = 35.0


def status_color(rho, D, P):
    flags = []
    if rho is not None and not math.isnan(rho):
        flags.append(rho >= RHO_T)
    if D is not None and not math.isnan(D):
        flags.append(D >= D_T)
    if P is not None and not math.isnan(P):
        flags.append(P >= P_T)
    if not flags:
        return FAIL_COLOR
    if all(flags):
        return PASS_COLOR
    if any(flags):
        return PARTIAL_COLOR
    return FAIL_COLOR


def render_mol_image(smiles, size=(280, 220)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size, kekulize=True)
    return np.asarray(img)


def main():
    summary = json.load(open(os.path.join(LEAD_DIR, "m2_summary.json")))
    by_id = {row["id"]: row for row in summary}

    leads = []
    for f in sorted(glob.glob(os.path.join(LEAD_DIR, "m2_lead_L*.json"))):
        d = json.load(open(f))
        lid = d["id"]
        s = by_id.get(lid, {})
        # anchor-calibrated values
        rho_cal = s.get("rho_cal")
        kjcal = s.get("kj_dft_cal", {}) or {}
        D_cal = kjcal.get("D_kms")
        P_cal = kjcal.get("P_GPa")
        # Replace NaN with raw kj_dft as fallback
        kjraw = s.get("kj_dft", {}) or {}
        if D_cal is None or (isinstance(D_cal, float) and math.isnan(D_cal)):
            D_cal = kjraw.get("D_kms")
        if P_cal is None or (isinstance(P_cal, float) and math.isnan(P_cal)):
            P_cal = kjraw.get("P_GPa")
        leads.append({
            "id": lid,
            "smiles": d["smiles"],
            "name": d.get("name", ""),
            "formula": d.get("formula", ""),
            "rho": rho_cal,
            "D": D_cal,
            "P": P_cal,
        })
    # sort by lead id numeric
    def key(L):
        try:
            return int(L["id"][1:])
        except Exception:
            return 999
    leads.sort(key=key)
    leads = leads[:12]
    print(f"Rendering {len(leads)} lead cards.")

    nrow, ncol = 3, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.0, nrow * 3.4))
    for i, lead in enumerate(leads):
        ax = axes[i // ncol, i % ncol]
        ax.set_xticks([])
        ax.set_yticks([])
        # color border
        col = status_color(lead["rho"], lead["D"], lead["P"])
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(3.0)

        img = render_mol_image(lead["smiles"])
        if img is not None:
            ax.imshow(img)
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
        # title
        title = f"{lead['id']}: {lead['name'][:32]}"
        ax.set_title(title, fontsize=9, loc="left", pad=4)
        # labels at bottom
        rho = lead["rho"]; D = lead["D"]; P = lead["P"]
        rho_s = f"{rho:.2f}" if isinstance(rho, (int, float)) and not math.isnan(rho) else "n/a"
        D_s = f"{D:.2f}" if isinstance(D, (int, float)) and not math.isnan(D) else "n/a"
        P_s = f"{P:.1f}" if isinstance(P, (int, float)) and not math.isnan(P) else "n/a"
        txt = (f"{lead['formula']}\n"
               f"rho={rho_s} g/cc, D={D_s} km/s, P={P_s} GPa")
        ax.text(0.5, -0.18, txt, transform=ax.transAxes,
                ha="center", va="top", fontsize=8,
                family="monospace")

    fig.suptitle("DGLD lead cards (anchor-calibrated DFT KJ): green=HMX-class pass on rho/D/P",
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=(0, 0.01, 1, 0.97))
    svg_path = os.path.join(OUT_DIR, "fig_lead_cards_grid.svg")
    png_path = os.path.join(OUT_DIR, "fig_lead_cards_grid.png")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    print("Saved:", svg_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
