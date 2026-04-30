"""H4 DFT dumbbell + N-fraction residual scatter.

Left: per-lead dumbbell connecting CNN-predicted D vs anchor-calibrated DFT KJ D.
Right: scatter of N-fraction (N atoms / total atoms) vs (D_KJ_DFT_cal - D_CNN)
       residual, with linear fit and Pearson r.

Sources: m2_summary.json + m2_lead_*.json (for predicted.D and formula).
We also annotate aggregate Pearson r from results/d9_kj_nfrac_table.json
in the title for context (a different binning).

Run: /c/Python314/python plot_dft_dumbbell.py
"""
import json
import os
import glob
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scistats

ROOT = r"E:\Projects\EnergeticDiffusion2"
LEAD_DIR = os.path.join(ROOT, "m2_bundle", "results")
OUT_DIR = os.path.join(ROOT, "docs", "paper", "figs")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_formula(s):
    if not s:
        return {}
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", s):
        if not el:
            continue
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def n_fraction(formula):
    f = parse_formula(formula)
    tot = sum(f.values())
    if tot == 0:
        return None
    return f.get("N", 0) / tot


def main():
    summary = json.load(open(os.path.join(LEAD_DIR, "m2_summary.json")))
    leads_by_id = {}
    for f in glob.glob(os.path.join(LEAD_DIR, "m2_lead_*.json")):
        d = json.load(open(f))
        leads_by_id[d["id"]] = d

    rows = []
    for s in summary:
        lid = s["id"]
        lead = leads_by_id.get(lid, {})
        pred = (lead.get("predicted") or {})
        D_cnn = pred.get("D")
        kjcal = s.get("kj_dft_cal", {}) or {}
        D_dft_cal = kjcal.get("D_kms")
        if D_dft_cal is None or (isinstance(D_dft_cal, float) and math.isnan(D_dft_cal)):
            continue
        if D_cnn is None:
            continue
        formula = lead.get("formula", "")
        nfrac = n_fraction(formula)
        rows.append({
            "id": lid,
            "D_cnn": float(D_cnn),
            "D_dft_cal": float(D_dft_cal),
            "nfrac": nfrac,
            "formula": formula,
        })

    if not rows:
        print("No usable rows; aborting.")
        return

    rows.sort(key=lambda r: r["D_cnn"])
    print(f"Plotting {len(rows)} leads.")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6))
    # ---- Left: dumbbell ----
    y = np.arange(len(rows))
    for yi, r in zip(y, rows):
        axL.plot([r["D_cnn"], r["D_dft_cal"]], [yi, yi],
                 "-", color="0.6", linewidth=1.2, zorder=1)
    axL.scatter([r["D_cnn"] for r in rows], y, color="#4E79A7",
                label="3D-CNN predicted D", zorder=2, s=42)
    axL.scatter([r["D_dft_cal"] for r in rows], y, color="#F28E2B",
                label="DFT KJ (anchor-cal) D", zorder=3, s=42)
    axL.set_yticks(y)
    axL.set_yticklabels([r["id"] for r in rows], fontsize=9)
    axL.set_xlabel("Detonation velocity D (km/s)")
    axL.axvline(9.0, color="#59A14F", linestyle=":", linewidth=1.0,
                label="HMX-class threshold (9.0 km/s)")
    # L1 K-J calibrated D reference line
    axL.axvline(8.25, color="0.45", linestyle="--", linewidth=1.0,
                label=r"L1 $D_{K\!-\!J,\mathrm{cal}}$ = 8.25 km/s")
    # Inline label at top of L1 reference line for visual clarity
    ymax = len(rows) - 0.5
    axL.text(8.25, ymax, r" L1 $D_{K\!-\!J,\mathrm{cal}}$=8.25",
             fontsize=7.5, color="0.35", ha="left", va="top",
             rotation=90)
    axL.legend(loc="lower right", fontsize=8, frameon=False)
    axL.set_title("Per-lead dumbbell: CNN-predicted vs DFT-KJ (calibrated) D",
                  fontsize=10)
    axL.grid(axis="x", color="0.9", linestyle=":")
    axL.set_axisbelow(True)

    # ---- Right: residual vs N-frac ----
    rows2 = [r for r in rows if r["nfrac"] is not None]
    nf = np.array([r["nfrac"] for r in rows2])
    resid = np.array([r["D_dft_cal"] - r["D_cnn"] for r in rows2])
    axR.scatter(nf, resid, color="#E15759", s=55, edgecolor="black",
                linewidth=0.5)
    for r in rows2:
        axR.annotate(r["id"], (r["nfrac"], r["D_dft_cal"] - r["D_cnn"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
    if len(nf) >= 3:
        slope, intercept, rval, pval, _ = scistats.linregress(nf, resid)
        xx = np.linspace(nf.min(), nf.max(), 50)
        axR.plot(xx, slope * xx + intercept, "-", color="0.3",
                 linewidth=1.2,
                 label=f"linear fit, Pearson r={rval:.2f}, p={pval:.2g}")
        axR.legend(loc="best", fontsize=8, frameon=False)
    axR.axhline(0, color="0.5", linestyle="--", linewidth=0.8)

    # Reference anchors (literature K-J calibrated values for canonical
    # energetics). f_N = 0.378 for all three since they share C/H/N/O
    # stoichiometry of (3,6,6,6); residual = D_KJ_cal - D_CNN.
    anchors = [
        ("RDX",   0.378, -1.32),
        ("HMX",   0.378, -1.58),
        ("FOX-7", 0.378, -1.17),
    ]
    ax_offsets = {
        "RDX":   (8, 4),
        "HMX":   (8, -10),
        "FOX-7": (-46, -4),
    }
    for nm, fN, res in anchors:
        axR.scatter([fN], [res], marker="*", s=160,
                    facecolor="#FFD24A", edgecolor="black",
                    linewidth=0.8, zorder=5)
        dx, dy = ax_offsets.get(nm, (8, 4))
        axR.annotate(nm, (fN, res),
                     fontsize=8, fontweight="bold",
                     xytext=(dx, dy), textcoords="offset points",
                     color="#1f2c3a", zorder=6)

    axR.set_xlabel("N fraction (N atoms / total atoms)")
    axR.set_ylabel("D residual: DFT-KJ-cal minus CNN (km/s)")
    axR.set_title("Residual vs N fraction (per lead)", fontsize=10)
    axR.grid(color="0.9", linestyle=":")
    axR.set_axisbelow(True)

    # Pull global r from d9 table for caption text
    try:
        d9 = json.load(open(os.path.join(ROOT, "results", "d9_kj_nfrac_table.json")))
        gr = d9.get("pearson_resid_vs_Nfrac", {}).get("r")
        if gr is not None:
            fig.suptitle(
                f"DFT validation: 3D-CNN vs anchor-calibrated KJ "
                f"(global N-frac residual Pearson r={gr:.2f} on n={d9.get('n_total','?')})",
                fontsize=11, y=0.995)
    except Exception:
        pass

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    svg_path = os.path.join(OUT_DIR, "fig_dft_dumbbell.svg")
    png_path = os.path.join(OUT_DIR, "fig_dft_dumbbell.png")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    print("Saved:", svg_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
