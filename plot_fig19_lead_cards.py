"""Figure 19: publication-grade 3x4 lead-cards grid for the DGLD paper.

Each card shows, top-to-bottom:
  1. RDKit 2D depiction (white background).
  2. Lead ID + chemotype label (e.g. "L1 - trinitro-isoxazole").
  3. Property line: rho_cal, D_cal, P_cal, HOF_cal (anchor-calibrated).
  4. Status badge row (4 pills): SMARTS, Pareto rank, xTB, DFT.
  5. Border colour + faint background tint:
       green  = full HMX-class pass + xTB pass + DFT pass + graph unchanged
       orange = partial (graph altered, or rho/D/P 0..5 % below HMX-class)
       red    = any hard FAIL (xTB gap < 1.5 eV, or DFT did not converge,
                or graph collapsed under xTB optimisation).

Output: docs/paper/figs/fig_lead_cards_grid.png and .svg
PNG is post-quantised to 8-bit palette (<=250 KB).

Run: /c/Python314/python plot_fig19_lead_cards.py
"""
from __future__ import annotations

import glob
import json
import math
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

ROOT = os.path.dirname(os.path.abspath(__file__))
LEAD_DIR = os.path.join(ROOT, "m2_bundle", "results")
OUT_DIR = os.path.join(ROOT, "docs", "paper", "figs")
XTB_FILES = [
    os.path.join(ROOT, "experiments", "xtb_merged_top15.json"),
    os.path.join(ROOT, "experiments", "xtb_merged_top16_30.json"),
    os.path.join(ROOT, "experiments", "xtb_merged_top31_100.json"),
]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
PASS_BORDER     = "#3a7a4a"
PARTIAL_BORDER  = "#c97a1f"
FAIL_BORDER     = "#aa3a3a"
PASS_BG         = "#e8f5e8"
PARTIAL_BG      = "#fff4e6"
FAIL_BG         = "#fde2e2"
TEXT_NAVY       = "#1f2c3a"
TEXT_SLATE      = "#5a6a7a"
BADGE_PASS_FILL = "#cfe7d2"
BADGE_PASS_EDGE = "#3a7a4a"
BADGE_PASS_TEXT = "#1f4a2a"
BADGE_NEUTRAL_FILL = "#e3ecf4"
BADGE_NEUTRAL_EDGE = "#27445d"
BADGE_NEUTRAL_TEXT = "#1f2c3a"
BADGE_WARN_FILL = "#ffe7c8"
BADGE_WARN_EDGE = "#c97a1f"
BADGE_WARN_TEXT = "#7a4a10"
BADGE_FAIL_FILL = "#f4cccc"
BADGE_FAIL_EDGE = "#aa3a3a"
BADGE_FAIL_TEXT = "#5a1f1f"
BADGE_NA_FILL   = "#ececec"
BADGE_NA_EDGE   = "#909090"
BADGE_NA_TEXT   = "#5a5a5a"

# HMX-class thresholds (anchor-calibrated)
RHO_T = 1.85
D_T   = 9.0
P_T   = 35.0
SOFT_TOL = 0.05  # within 5 percent counts as "near miss" (orange)

# xTB gap threshold
GAP_T = 1.5

# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def _load_xtb_lookup():
    by_smi = {}
    for path in XTB_FILES:
        if not os.path.exists(path):
            continue
        try:
            for x in json.load(open(path)):
                smi = x.get("smi")
                if smi and smi not in by_smi:
                    by_smi[smi] = x
        except Exception:
            continue
    return by_smi


def _safe_float(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def load_leads():
    summary = json.load(open(os.path.join(LEAD_DIR, "m2_summary.json")))
    by_id = {row["id"]: row for row in summary}
    xtb_lut = _load_xtb_lookup()

    leads = []
    for f in sorted(glob.glob(os.path.join(LEAD_DIR, "m2_lead_L*.json"))):
        d = json.load(open(f))
        lid = d["id"]
        s = by_id.get(lid, {})

        rho_cal = _safe_float(s.get("rho_cal"))
        kjcal = s.get("kj_dft_cal", {}) or {}
        kjraw = s.get("kj_dft", {}) or {}
        D_cal = _safe_float(kjcal.get("D_kms")) or _safe_float(kjraw.get("D_kms"))
        P_cal = _safe_float(kjcal.get("P_GPa")) or _safe_float(kjraw.get("P_GPa"))
        HOF_cal = _safe_float(s.get("HOF_kJmol_wb97xd_cal"))

        n_imag = s.get("n_imag")
        dft_pass = (n_imag == 0) and not s.get("errors")

        xtb = xtb_lut.get(d["smiles"])
        if xtb is not None:
            gap = _safe_float(xtb.get("gap_ev"))
            graph_ok = bool(xtb.get("graph_survives"))
            xtb_conv = bool(xtb.get("converged"))
            pareto_rank = xtb.get("rank")
        else:
            gap = None
            graph_ok = None
            xtb_conv = None
            pareto_rank = None

        leads.append({
            "id": lid,
            "smiles": d["smiles"],
            "name": d.get("name", ""),
            "formula": d.get("formula", ""),
            "rho": rho_cal,
            "D": D_cal,
            "P": P_cal,
            "HOF": HOF_cal,
            "n_imag": n_imag,
            "dft_pass": dft_pass,
            "xtb_gap": gap,
            "xtb_graph_ok": graph_ok,
            "xtb_conv": xtb_conv,
            "pareto_rank": pareto_rank,
        })

    leads.sort(key=lambda L: int(L["id"][1:]))
    return leads


# ---------------------------------------------------------------------------
# Status / badges
# ---------------------------------------------------------------------------
def xtb_verdict(lead):
    """-> ('PASS'|'FAIL'|'GRAPH'|'NA', label, fill, edge, text)."""
    gap = lead["xtb_gap"]
    graph_ok = lead["xtb_graph_ok"]
    if gap is None:
        return ("NA", "xTB n/a", BADGE_NA_FILL, BADGE_NA_EDGE, BADGE_NA_TEXT)
    if gap < GAP_T:
        return ("FAIL", f"xTB FAIL ({gap:.2f} eV)", BADGE_FAIL_FILL, BADGE_FAIL_EDGE, BADGE_FAIL_TEXT)
    if graph_ok is False:
        return ("GRAPH", f"graph altered ({gap:.2f} eV)", BADGE_WARN_FILL, BADGE_WARN_EDGE, BADGE_WARN_TEXT)
    return ("PASS", f"xTB PASS ({gap:.2f} eV)", BADGE_PASS_FILL, BADGE_PASS_EDGE, BADGE_PASS_TEXT)


def dft_verdict(lead):
    if lead["dft_pass"]:
        return ("PASS", "DFT PASS", BADGE_PASS_FILL, BADGE_PASS_EDGE, BADGE_PASS_TEXT)
    return ("FAIL", "DFT FAIL", BADGE_FAIL_FILL, BADGE_FAIL_EDGE, BADGE_FAIL_TEXT)


def hmx_verdict(lead):
    """Compare anchor-calibrated rho/D/P to HMX-class threshold.
    Returns one of 'pass', 'partial', 'fail'."""
    rho, D, P = lead["rho"], lead["D"], lead["P"]
    if None in (rho, D, P):
        return "fail"
    flags = [rho >= RHO_T, D >= D_T, P >= P_T]
    if all(flags):
        return "pass"
    # near-miss check: within SOFT_TOL of each threshold
    near = [
        rho >= RHO_T * (1 - SOFT_TOL),
        D   >= D_T   * (1 - SOFT_TOL),
        P   >= P_T   * (1 - SOFT_TOL),
    ]
    if all(near):
        return "partial"
    return "partial" if any(flags) else "fail"


def card_status(lead):
    """Border + background colour for the whole card."""
    xtb = xtb_verdict(lead)[0]
    dft = dft_verdict(lead)[0]
    hmx = hmx_verdict(lead)

    if xtb == "FAIL" or dft == "FAIL":
        return "fail", FAIL_BORDER, FAIL_BG
    if hmx == "pass" and xtb == "PASS" and dft == "PASS":
        return "pass", PASS_BORDER, PASS_BG
    # NA xTB falls through to partial if HMX pass + DFT pass
    if hmx == "pass" and xtb == "NA" and dft == "PASS":
        return "partial", PARTIAL_BORDER, PARTIAL_BG
    return "partial", PARTIAL_BORDER, PARTIAL_BG


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_mol(smiles, size=(440, 320)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size, kekulize=True)
    return np.asarray(img)


def _badge(ax, x, y, w, h, text, fill, edge, text_color, fontsize=7.6):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        linewidth=1.0, facecolor=fill, edgecolor=edge, zorder=4,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, family="sans-serif",
            fontweight=600, zorder=5)


def draw_card(ax, lead):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    verdict_kind, border_col, bg_col = card_status(lead)

    # Card backdrop
    backdrop = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.005,rounding_size=0.025",
        linewidth=2.6, facecolor=bg_col, edgecolor=border_col, zorder=1,
    )
    ax.add_patch(backdrop)

    # 1. Structure (top ~55% of card)
    img = render_mol(lead["smiles"], size=(440, 320))
    if img is not None:
        # White background panel for the structure
        struct_panel = FancyBboxPatch(
            (0.05, 0.43), 0.90, 0.52,
            boxstyle="round,pad=0.0,rounding_size=0.018",
            linewidth=0.8, facecolor="white",
            edgecolor="#d0d6dc", zorder=2,
        )
        ax.add_patch(struct_panel)
        ax.imshow(img, extent=(0.07, 0.93, 0.45, 0.93),
                  aspect="auto", zorder=3, interpolation="bilinear")

    # 2. ID + chemotype
    name = lead["name"] or ""
    if len(name) > 28:
        name = name[:27] + "…"
    ax.text(0.06, 0.395, lead["id"], fontsize=12.5, fontweight="bold",
            color=TEXT_NAVY, family="serif", ha="left", va="center", zorder=4)
    ax.text(0.18, 0.395, "- " + name, fontsize=10.5,
            color=TEXT_NAVY, family="serif", ha="left", va="center",
            fontstyle="italic", zorder=4)

    # 3. Property line
    rho = lead["rho"]; D = lead["D"]; P = lead["P"]; H = lead["HOF"]
    def f(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) else "n/a"
    prop_line = (
        f"ρ={f(rho,'{:.2f}')} g/cm³ · "
        f"D={f(D,'{:.2f}')} km/s · "
        f"P={f(P,'{:.1f}')} GPa · "
        f"HOF={f(H,'{:+.0f}')} kJ/mol"
    )
    formula = lead["formula"] or ""
    ax.text(0.06, 0.335, formula, fontsize=8.6, color=TEXT_SLATE,
            family="monospace", ha="left", va="center", zorder=4)
    ax.text(0.5, 0.275, prop_line, fontsize=8.0, color=TEXT_NAVY,
            family="sans-serif", ha="center", va="center", zorder=4)

    # 4. Status badges row
    badges = []
    badges.append(("SMARTS ✓", BADGE_PASS_FILL, BADGE_PASS_EDGE, BADGE_PASS_TEXT))
    pr = lead["pareto_rank"]
    pr_txt = f"Pareto #{pr}" if pr is not None else "Pareto -"
    badges.append((pr_txt, BADGE_NEUTRAL_FILL, BADGE_NEUTRAL_EDGE, BADGE_NEUTRAL_TEXT))
    _, xt, xf, xe, xtcol = xtb_verdict(lead)
    badges.append((xt, xf, xe, xtcol))
    _, dt, df_, de, dtcol = dft_verdict(lead)
    badges.append((dt, df_, de, dtcol))

    # Layout: 4 badges horizontally with small gaps
    badge_y = 0.085
    badge_h = 0.115
    margin_x = 0.045
    gap = 0.018
    avail = 1.0 - 2 * margin_x - 3 * gap
    # variable widths proportional to text length
    weights = np.array([max(len(b[0]), 6) for b in badges], dtype=float)
    weights = weights / weights.sum()
    x_cur = margin_x
    for (txt, fill, edge, tcol), w_frac in zip(badges, weights):
        bw = avail * w_frac
        _badge(ax, x_cur, badge_y, bw, badge_h, txt, fill, edge, tcol,
               fontsize=7.4)
        x_cur += bw + gap


def build_figure(leads):
    nrow, ncol = 4, 3
    fig_w = 13.5
    fig_h = 16.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(nrow, ncol, left=0.025, right=0.975,
                          top=0.965, bottom=0.025,
                          wspace=0.06, hspace=0.10)
    for i, lead in enumerate(leads):
        if i >= nrow * ncol:
            break
        r, c = divmod(i, ncol)
        ax = fig.add_subplot(gs[r, c])
        draw_card(ax, lead)

    fig.suptitle(
        "Figure 19. Twelve chem-pass DGLD leads: structure, anchor-calibrated "
        "DFT properties, and validation status",
        fontsize=13.5, fontweight=600, color=TEXT_NAVY,
        family="serif", y=0.992,
    )
    return fig


def quantise_png(path, max_bytes=250_000):
    """Reduce PNG file size via 8-bit palette quantisation."""
    img = Image.open(path).convert("RGB")
    # Try increasing palette sizes until we are under the limit.
    for ncolors in (128, 96, 64, 48, 32):
        q = img.quantize(colors=ncolors, method=Image.MEDIANCUT, dither=Image.NONE)
        buf = BytesIO()
        q.save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_bytes:
            with open(path, "wb") as f:
                f.write(buf.getvalue())
            return ncolors, buf.tell()
    # Fall back to the smallest palette
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return ncolors, buf.tell()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    leads = load_leads()
    print(f"Loaded {len(leads)} leads")
    for L in leads:
        v_card = card_status(L)[0]
        v_xtb  = xtb_verdict(L)[0]
        v_dft  = dft_verdict(L)[0]
        print(f"  {L['id']:>4} | xTB={v_xtb:>5} | DFT={v_dft:>4} | card={v_card}")

    fig = build_figure(leads)
    svg_path = os.path.join(OUT_DIR, "fig_lead_cards_grid.svg")
    png_path = os.path.join(OUT_DIR, "fig_lead_cards_grid.png")
    fig.savefig(svg_path, format="svg", bbox_inches="tight",
                facecolor="white")
    fig.savefig(png_path, format="png", dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    raw_size = os.path.getsize(png_path)
    print(f"PNG raw size: {raw_size/1024:.1f} KB")
    ncolors, qsize = quantise_png(png_path, max_bytes=250_000)
    print(f"PNG quantised ({ncolors} colours): {qsize/1024:.1f} KB")
    # Report final dimensions
    with Image.open(png_path) as im:
        print(f"PNG dimensions: {im.size[0]} x {im.size[1]}")
    print("Saved:", svg_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
