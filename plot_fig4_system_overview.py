"""Figure 4: DGLD system overview.

Two-panel figure:
  4a TRAINING  : SMILES -> LIMO encoder -> z; (target props p, trust mask m) feed
                 the Conditional DDPM noise-predictor; loss = MSE(eps - eps_theta).
  4b SAMPLING  : z_T ~ N(0,I); per DDIM step, eps_theta is corrected by a
                 multi-head classifier-guidance gradient (viab, sens, perf, hazard);
                 z_0 decoded via LIMO; chemistry filter + Phase-A reranker pick top-K.

Outputs:
  docs/paper/figs/fig4a_training.{png,svg}
  docs/paper/figs/fig4b_sampling.{png,svg}
"""
from __future__ import annotations
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join("docs", "paper", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# Palette
NAVY        = "#27445d"
GOLD        = "#b88a25"
RED         = "#aa3a3a"
GREEN       = "#3a7a4a"
PALE_GOLD   = "#fdf3d8"
PALE_GREY   = "#dde6e9"
PALE_RED    = "#f7e6e6"
PALE_GREEN  = "#e3f1e3"
CREAM       = "#f6f3ed"
TEXT_NAVY   = "#1f2c3a"
TEXT_SLATE  = "#5a6a7a"
TEXT_LIGHT  = "#8a9aaa"


def add_box(ax, x_c, w, h, y_c, fill, edge, title, subtitle, title_size=12.5):
    x = x_c - w / 2
    y = y_c - h / 2
    # drop shadow
    sh = FancyBboxPatch(
        (x + 0.04, y - 0.07), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=0, facecolor="#0a1620", alpha=0.13, zorder=1,
    )
    ax.add_patch(sh)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.6, facecolor=fill, edgecolor=edge, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x_c, y_c + 0.22, title, ha="center", va="center",
            fontsize=title_size, fontweight=600, color=TEXT_NAVY,
            family="serif", zorder=3)
    if subtitle:
        ax.text(x_c, y_c - 0.28, subtitle, ha="center", va="center",
                fontsize=9.5, fontstyle="italic", color=TEXT_SLATE,
                family="serif", zorder=3)


def add_arrow(ax, x0, y0, x1, y1, color=NAVY, dashed=False, lw=2.0):
    if x0 == x1:
        dy = 0.05 if y1 > y0 else -0.05
        start, end = (x0, y0 + dy), (x1, y1 - dy)
    else:
        dx = 0.05 if x1 > x0 else -0.05
        start, end = (x0 + dx, y0), (x1 - dx, y1)
    ax.annotate("", xy=end, xytext=start,
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=(0, (5, 3)) if dashed else "-",
                                shrinkA=0, shrinkB=0, mutation_scale=12),
                zorder=4)


def add_label(ax, x, y, text, color=TEXT_LIGHT, size=9, bold=False, italic=False):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=size, color=color, family="serif",
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal", zorder=3)


def setup_axes(ax, xmax=16.0, ymax=6.4):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────
def fig4a_training():
    """Training-time data flow: data → encoder → z; (p, m) → DDPM."""
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=300)
    setup_axes(ax, xmax=16.0, ymax=6.4)

    # Banners
    add_label(ax, 0.4, 6.2, "TRAINING",  color=TEXT_LIGHT, size=10, bold=True)
    add_label(ax, 0.4, 5.6, "(noise prediction in latent space; conditioning gated by trust mask)",
              color=TEXT_LIGHT, size=8.5, italic=True)

    # Data inputs row (top)
    add_box(ax, 1.7,  2.6, 1.2, 4.7, CREAM,     NAVY, "SMILES",            "SELFIES tokens")
    add_box(ax, 4.7,  2.6, 1.2, 4.7, PALE_GREY, NAVY, "LIMO encoder",      "frozen, fine-tuned")
    add_box(ax, 8.0,  3.0, 1.2, 4.7, PALE_GOLD, GOLD, "Latent z (1024-d)", "cached, deterministic μ")

    # Conditioning inputs (right side, into DDPM)
    add_box(ax, 12.0, 2.4, 1.0, 5.6, PALE_GOLD, GOLD, "Target props p",    "ρ, HOF, D, P (z-scored)")
    add_box(ax, 14.6, 2.4, 1.0, 5.6, PALE_GOLD, GOLD, "Trust mask m",      "Tier-A/B → 1, else 0")

    # Conditional DDPM (centre, lower row)
    add_box(ax, 8.0, 3.0, 1.4, 1.7, PALE_GREY, NAVY,
            r"Conditional DDPM $\varepsilon_\theta$",
            r"FiLM ResNet, $\varepsilon_\theta(z_t, t, p, m)$")

    # Loss block (bottom)
    add_box(ax, 12.5, 4.0, 1.0, 1.7, PALE_RED, RED, "Training loss",
            r"$\mathbb{E}\,\Vert\varepsilon - \varepsilon_\theta(z_t, t, p, m)\Vert^2$",
            title_size=11.5)

    # Pipeline arrows (top row)
    add_arrow(ax, 2.6, 4.7, 4.1, 4.7)               # SMILES → encoder
    add_arrow(ax, 5.3, 4.7, 7.4, 4.7)               # encoder → z

    # z (top) → DDPM (bottom): vertical arrow from latent z at (8.0, 4.7-1.5=3.2) to DDPM at (8.0, 1.7+0.85=2.55)
    add_arrow(ax, 8.0, 4.0, 8.0, 2.55)              # z → DDPM
    # p, m → DDPM: arrows from right boxes (top row) into DDPM
    add_arrow(ax, 11.5, 5.6, 8.7, 2.0,  color=GOLD, dashed=True)   # props → DDPM
    add_arrow(ax, 14.1, 5.6, 8.7, 1.7,  color=GOLD, dashed=True)   # mask → DDPM
    # DDPM → loss
    add_arrow(ax, 8.7, 1.7, 12.0, 1.7,  color=RED, lw=1.8)

    # Mini-caption inside DDPM region
    add_label(ax, 7.0, 0.55, "Tier-C/D rows: m = 0 (loss enters unconditional branch via cfg-dropout)",
              color=TEXT_LIGHT, size=8.5, italic=True)

    plt.tight_layout(pad=0.5)
    base = os.path.join(OUT_DIR, "fig4a_training")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", base + ".{png,svg}")


# ─────────────────────────────────────────────────────────────────────────
def fig4b_sampling():
    """Sampling: 3-row layout. Row 1 (top): 4 classifier-guidance heads.
    Row 2 (middle, main pipeline): z_T -> DDIM -> z_0 -> LIMO decoder.
    Row 3 (bottom): Pool of SMILES -> chem filter -> reranker -> top-K leads.
    Heads inject gradients down into DDIM; main pipeline flows right; filter
    chain flows right; LIMO decoder drops down into the pool."""
    fig, ax = plt.subplots(figsize=(14, 6.0), dpi=300)
    setup_axes(ax, xmax=20.0, ymax=8.0)

    # Banners
    add_label(ax, 0.4, 7.7, "SAMPLING / INFERENCE", color=TEXT_LIGHT, size=10, bold=True)
    add_label(ax, 0.4, 7.2,
              "(DDIM denoise with multi-head classifier guidance; chemistry-credibility filter)",
              color=TEXT_LIGHT, size=8.5, italic=True)

    # ── Row 1 (top): 4 classifier-guidance heads ───────────────────
    add_label(ax, 0.4, 6.5, "MULTI-HEAD CLASSIFIER GUIDANCE",
              color=TEXT_LIGHT, size=8.5, bold=True)
    head_y = 5.6
    head_specs = [
        (4.5,  "Viability",   "RandomForest on z"),
        (8.0,  "Sensitivity", r"h$_{50}$ head"),
        (11.5, "Performance", r"$\rho$, D, P"),
        (15.0, "Hazard",      "SMARTS-aware head"),
    ]
    for x_c, t, sub in head_specs:
        add_box(ax, x_c, 2.5, 1.0, head_y, PALE_GREEN, GREEN, t, sub, title_size=11.0)

    # ── Row 2 (middle): main pipeline z_T -> DDIM -> z_0 -> LIMO decoder ──
    row2_y = 3.6
    add_box(ax, 1.6,  1.6, 1.2, row2_y, CREAM,     NAVY, r"$z_T$",          r"$\mathcal{N}(0, I)$")
    add_box(ax, 6.5,  3.6, 1.2, row2_y, PALE_GREY, NAVY, "DDIM denoise",
            r"$\hat\varepsilon = \varepsilon_\theta + \sum_h s_h \nabla\log p_h(c\mid z)$",
            title_size=11.0)
    add_box(ax, 11.5, 1.6, 1.2, row2_y, PALE_GOLD, GOLD, r"$z_0$",          "denoised latent")
    add_box(ax, 15.5, 2.4, 1.2, row2_y, PALE_GREY, NAVY, "LIMO decoder",    "non-autoregressive")
    add_box(ax, 18.6, 1.6, 1.2, row2_y, CREAM,     NAVY, "Pool",            r"$\sim$10$^4$ SMILES")

    # Row-2 arrows (left to right)
    add_arrow(ax, 2.2,  row2_y, 4.7, row2_y)        # z_T → DDIM
    add_arrow(ax, 8.3,  row2_y, 10.9, row2_y)       # DDIM → z_0
    add_arrow(ax, 12.1, row2_y, 14.3, row2_y)       # z_0 → decoder
    add_arrow(ax, 16.7, row2_y, 18.0, row2_y)       # decoder → pool

    # Heads → DDIM: 4 gold dashed arrows from row-1 boxes pointing into the
    # DDIM box top edge (single sink). Anchor x at 6.5 (DDIM centre) y top of DDIM.
    ddim_top = row2_y + 1.6 / 2  # = 4.4
    for x_c, *_ in head_specs:
        add_arrow(ax, x_c, head_y - 0.5, 6.5, ddim_top + 0.05,
                  color=GOLD, dashed=True, lw=1.4)

    # ── Row 3 (bottom): pool → filter → reranker → top-K ──
    row3_y = 1.2
    add_box(ax, 4.5,  2.6, 1.2, row3_y, PALE_RED,   RED,   "Chem-rules filter", "energetic-SMARTS")
    add_box(ax, 9.0,  2.6, 1.2, row3_y, PALE_GREEN, GREEN, "Phase-A reranker",  "scaffold composite")
    add_box(ax, 13.5, 2.6, 1.2, row3_y, PALE_GOLD,  GOLD,  "Top-K leads",       r"$K \approx 100$")

    # Pool → filter (long arrow back-left across the page)
    add_arrow(ax, 18.0, row2_y - 0.8, 5.8, row3_y, color=NAVY, lw=1.5)
    # filter → reranker → top-K
    add_arrow(ax, 5.8,  row3_y, 7.7,  row3_y, color=NAVY)
    add_arrow(ax, 10.3, row3_y, 12.2, row3_y, color=NAVY)

    # Mini-caption (kept as text below the bottom row)
    add_label(ax, 0.4, 0.3,
              r"At each DDIM step $t$: $\hat\varepsilon = \varepsilon_\theta(z_t, t, c) + \sum_h s_h \nabla_{z_t} \log p_h(c\mid z_t)$",
              color=TEXT_LIGHT, size=8.5, italic=True)

    plt.tight_layout(pad=0.4)
    base = os.path.join(OUT_DIR, "fig4b_sampling")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", base + ".{png,svg}")


if __name__ == "__main__":
    fig4a_training()
    fig4b_sampling()
