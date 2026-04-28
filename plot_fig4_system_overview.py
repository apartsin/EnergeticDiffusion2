"""Figure 4: DGLD system overview diagram.

Redraw of the inline SVG using matplotlib for editorial-quality output
(PNG at 300 dpi). Two-row layout:
  Row 1 (CONDITIONING):           Cond. mask m  ->  Conditional DDPM
  Row 2 (ENCODE / DECODE):        SMILES -> LIMO encoder -> Latent z -> LIMO decoder -> SMILES'
Conditioning arrow (gold dashed) feeds the DDPM; DDPM arrow (navy)
feeds the latent z block. All arrows axis-parallel.

Outputs: docs/paper/figs/fig4_system_overview.{png,svg}
"""
from __future__ import annotations
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as patheffects

OUT_DIR = os.path.join("docs", "paper", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# Palette (consistent with prior SVG)
NAVY        = "#27445d"
GOLD        = "#b88a25"
PALE_GOLD   = "#fdf3d8"
PALE_GREY   = "#dde6e9"
CREAM       = "#f6f3ed"
TEXT_NAVY   = "#1f2c3a"
TEXT_SLATE  = "#5a6a7a"
TEXT_LIGHT  = "#8a9aaa"

# Layout (figure-coord units; 0..16 wide, 0..8 tall)
FIG_W, FIG_H = 16.0, 6.4

# Two row centres (y midpoints in axis units). Box heights matched across rows.
ROW_TOP_Y = 4.7
ROW_BOT_Y = 1.5
BOX_H_TOP = 1.4
BOX_H_BOT = 1.4

# Layout uses a 5-column architecture with uniform horizontal gaps so arrows
# have visible length. Column centres: 1.6, 4.6, 8.0, 11.4, 14.4. The top-row
# Cond mask sits over column 2 (LIMO encoder) and the Conditional DDPM sits
# over column 3 (Latent z), so the gating arrow drops vertically into the
# state node. All boxes are 2.6 fig units wide except the latent-z state
# (3.2) and the DDPM that feeds it (3.2), which are emphasised by width.

BOX_SPECS_TOP = [
    (4.6,  2.6, PALE_GOLD,  GOLD, "Cond. mask m",       "Tier-A/B → 1, else 0"),
    (8.0,  3.2, PALE_GREY,  NAVY, "Conditional DDPM",   "FiLM ResNet, latent diffusion"),
]

BOX_SPECS_BOT = [
    (1.6,  2.6, CREAM,      NAVY, "SMILES",             "SELFIES tokens"),
    (4.6,  2.6, PALE_GREY,  NAVY, "LIMO encoder",       "frozen, fine-tuned"),
    (8.0,  3.2, PALE_GOLD,  GOLD, "Latent z (1024-d)",  "N(μ, σ²)"),
    (11.4, 2.6, PALE_GREY,  NAVY, "LIMO decoder",       "non-autoregressive"),
    (14.4, 2.6, CREAM,      NAVY, "SMILES′",            "candidate"),
]


def add_box(ax, x_c, w, h, y_c, fill, edge, title, subtitle):
    """Add a rounded box with title and subtitle, plus a soft drop shadow."""
    x = x_c - w / 2
    y = y_c - h / 2
    # Drop shadow: a slightly-offset, slightly-blurred grey rectangle behind the real one
    shadow = FancyBboxPatch(
        (x + 0.04, y - 0.07), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=0,
        facecolor="#0a1620",
        alpha=0.13,
        zorder=1,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.6,
        facecolor=fill,
        edgecolor=edge,
        zorder=2,
    )
    ax.add_patch(box)
    # Vertical placement: title +0.22 from box centre, subtitle -0.28; with
    # box_h=1.4 this leaves >=0.40 of box-edge clearance at both top and bottom.
    ax.text(
        x_c, y_c + 0.22, title,
        ha="center", va="center",
        fontsize=12.5, fontweight=600, color=TEXT_NAVY,
        family="serif", zorder=3,
    )
    ax.text(
        x_c, y_c - 0.28, subtitle,
        ha="center", va="center",
        fontsize=9.5, fontstyle="italic", color=TEXT_SLATE,
        family="serif", zorder=3,
    )


def add_arrow(ax, x0, y0, x1, y1, color=NAVY, dashed=False, lw=2.0):
    """Add an axis-parallel arrow. Use ax.annotate so head size is in points
    (independent of axis units) and we don't get the FancyArrowPatch +
    mutation_scale interaction that exploded the head in cycle 3."""
    if x0 == x1:  # vertical
        dy = 0.05 if y1 > y0 else -0.05
        start = (x0, y0 + dy); end = (x1, y1 - dy)
    else:  # horizontal
        dx = 0.05 if x1 > x0 else -0.05
        start = (x0 + dx, y0); end = (x1 - dx, y1)
    ax.annotate(
        "", xy=end, xytext=start,
        xycoords="data", textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=(0, (5, 3)) if dashed else "-",
            shrinkA=0, shrinkB=0,
            mutation_scale=12,
        ),
        zorder=4,
    )


def build_figure():
    fig, ax = plt.subplots(figsize=(FIG_W * 0.65, FIG_H * 0.65), dpi=300)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Row banners (placed in vertical gaps ABOVE each row, NOT overlapping boxes).
    # The banners use small all-caps letter-spaced text for an editorial feel.
    banner_y_top = ROW_TOP_Y + BOX_H_TOP / 2 + 0.45
    banner_y_bot = ROW_BOT_Y + BOX_H_BOT / 2 + 0.45
    ax.text(
        0.4, banner_y_top, "CONDITIONING",
        ha="left", va="center",
        fontsize=9, color=TEXT_LIGHT, family="serif",
        fontweight="normal", zorder=3,
    )
    ax.text(
        0.4, banner_y_bot, "ENCODE  /  DECODE  PIPELINE",
        ha="left", va="center",
        fontsize=9, color=TEXT_LIGHT, family="serif",
        fontweight="normal", zorder=3,
    )

    # Top row
    for (x_c, w, fill, edge, title, sub) in BOX_SPECS_TOP:
        add_box(ax, x_c, w, BOX_H_TOP, ROW_TOP_Y, fill, edge, title, sub)

    # Bottom row
    for (x_c, w, fill, edge, title, sub) in BOX_SPECS_BOT:
        add_box(ax, x_c, w, BOX_H_BOT, ROW_BOT_Y, fill, edge, title, sub)

    # Pipeline arrows (left -> right) in the bottom row at y = ROW_BOT_Y
    for i in range(len(BOX_SPECS_BOT) - 1):
        x0_c, w0, *_ = BOX_SPECS_BOT[i]
        x1_c, w1, *_ = BOX_SPECS_BOT[i + 1]
        x0 = x0_c + w0 / 2
        x1 = x1_c - w1 / 2
        add_arrow(ax, x0, ROW_BOT_Y, x1, ROW_BOT_Y, color=NAVY, lw=1.8)

    # Conditioning arrow (top row, gold dashed): mask -> DDPM
    x0_c, w0, *_ = BOX_SPECS_TOP[0]
    x1_c, w1, *_ = BOX_SPECS_TOP[1]
    x0 = x0_c + w0 / 2
    x1 = x1_c - w1 / 2
    add_arrow(ax, x0, ROW_TOP_Y, x1, ROW_TOP_Y, color=GOLD, dashed=True, lw=1.8)

    # DDPM -> Latent z (vertical down). DDPM x = 8.0 (top), latent z x = 8.0 (bottom).
    add_arrow(ax, 8.0, ROW_TOP_Y - BOX_H_TOP / 2, 8.0, ROW_BOT_Y + BOX_H_BOT / 2,
              color=NAVY, lw=1.8)

    plt.tight_layout(pad=0.4)
    base = os.path.join(OUT_DIR, "fig4_system_overview")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".svg", bbox_inches="tight", facecolor="white")
    print("Saved:", base + ".png")
    print("Saved:", base + ".svg")
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
