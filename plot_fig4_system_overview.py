"""Figure 4: DGLD system overview.

Two-panel figure:
  4a TRAINING  : SMILES -> LIMO encoder -> cached (z_0, p, m); per-step loop
                 samples t, eps; forward-diffuses z_0 -> z_t; cfg-dropout on (p,m);
                 eps_theta predicts noise; masked MSE loss; AdamW+EMA backprop.
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
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = os.path.join("docs", "paper", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# Palette
NAVY        = "#27445d"
GOLD        = "#b88a25"
RED         = "#aa3a3a"
GREEN       = "#3a7a4a"
PURPLE      = "#6a4a8a"
PALE_GOLD   = "#fdf3d8"
PALE_GREY   = "#dde6e9"
PALE_RED    = "#f7e6e6"
PALE_GREEN  = "#e3f1e3"
PALE_PURPLE = "#ece4f3"
CREAM       = "#f6f3ed"
TEXT_NAVY   = "#1f2c3a"
TEXT_SLATE  = "#5a6a7a"
TEXT_LIGHT  = "#8a9aaa"


def add_box(ax, x_c, w, h, y_c, fill, edge, title, subtitle, title_size=12.0,
            sub_size=9.0):
    x = x_c - w / 2
    y = y_c - h / 2
    # drop shadow
    sh = FancyBboxPatch(
        (x + 0.04, y - 0.07), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=0, facecolor="#0a1620", alpha=0.13, zorder=1,
    )
    ax.add_patch(sh)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.5, facecolor=fill, edgecolor=edge, zorder=2,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x_c, y_c + h * 0.18, title, ha="center", va="center",
                fontsize=title_size, fontweight=600, color=TEXT_NAVY,
                family="serif", zorder=3)
        ax.text(x_c, y_c - h * 0.22, subtitle, ha="center", va="center",
                fontsize=sub_size, fontstyle="italic", color=TEXT_SLATE,
                family="serif", zorder=3)
    else:
        ax.text(x_c, y_c, title, ha="center", va="center",
                fontsize=title_size, fontweight=600, color=TEXT_NAVY,
                family="serif", zorder=3)


def add_arrow(ax, x0, y0, x1, y1, color=NAVY, dashed=False, lw=2.0,
              shrink=0.06):
    if x0 == x1:
        dy = shrink if y1 > y0 else -shrink
        start, end = (x0, y0 + dy), (x1, y1 - dy)
    elif y0 == y1:
        dx = shrink if x1 > x0 else -shrink
        start, end = (x0 + dx, y0), (x1 - dx, y1)
    else:
        # diagonal: shrink along both axes
        dx = shrink if x1 > x0 else -shrink
        dy = shrink if y1 > y0 else -shrink
        start, end = (x0 + dx, y0 + dy), (x1 - dx, y1 - dy)
    ax.annotate("", xy=end, xytext=start,
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=(0, (5, 3)) if dashed else "-",
                                shrinkA=0, shrinkB=0, mutation_scale=12),
                zorder=4)


def add_label(ax, x, y, text, color=TEXT_LIGHT, size=9, bold=False, italic=False,
              ha="left"):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=size, color=color, family="serif",
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal", zorder=3)


def add_phase_band(ax, x0, x1, y0, y1, label, color, alpha=0.45,
                   label_x=None):
    """Translucent banded background for a phase region."""
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                     facecolor=color, edgecolor="none", alpha=alpha, zorder=0)
    ax.add_patch(rect)
    if label_x is None:
        label_x = x0 + 0.15
    ax.text(label_x, y1 - 0.22, label, ha="left", va="top",
            fontsize=8.5, fontweight="bold", color=TEXT_SLATE,
            family="serif", zorder=1)


def setup_axes(ax, xmax=16.0, ymax=6.4):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────
def fig4a_training():
    """Training: Phase 1 data prep (cached) + Phase 2 per-step loop.

    Layout (two horizontal stripes):
      Phase 1 (top): SMILES -> LIMO encoder -> (z_0, p, m) cache.
      Phase 2 (bottom): minibatch from cache; sample t, eps; forward-diffuse z_t;
        cfg-dropout on (p,m); eps_theta predicts noise; masked MSE; AdamW+EMA.
    """
    return _fig4a_training_v3()


def _fig4a_training_v3():
    fig, ax = plt.subplots(figsize=(15, 8.0), dpi=300)
    setup_axes(ax, xmax=22.0, ymax=11.0)

    # Title banner
    add_label(ax, 0.3, 10.7, "TRAINING",  color=TEXT_NAVY, size=11, bold=True)
    add_label(ax, 0.3, 10.3,
              "Phase 1: precompute cached triples;  Phase 2: per-step noise-prediction loop",
              color=TEXT_SLATE, size=9, italic=True)

    # Phase bands — use distinctly different tints for clear visual separation.
    # Phase 1 is a cool grey-blue (data prep, run once); Phase 2 is a warm cream
    # (per-step loop) so the eye reads them as different stages immediately.
    add_phase_band(ax, 0.1, 21.9, 8.05, 10.05,
                   "PHASE 1: data preparation (run once)",
                   "#dbe5ec", alpha=0.75)
    add_phase_band(ax, 0.1, 21.9, 0.30, 7.95,
                   "PHASE 2: per-step training loop",
                   "#fcefcc", alpha=0.55, label_x=3.4)

    # ─── Phase 1: top stripe ───────────────────────────────────────
    p1_y = 9.0
    add_box(ax, 1.7,  2.2, 1.1, p1_y, CREAM,     NAVY, "SMILES",       "SELFIES tokens")
    add_box(ax, 5.0,  2.4, 1.1, p1_y, PALE_GREY, NAVY, "LIMO encoder", "frozen (fine-tuned)")
    add_box(ax, 8.6,  2.4, 1.1, p1_y, PALE_GOLD, GOLD, r"$z_0$ (1024-d)",
            r"deterministic $\mu$")
    add_box(ax, 12.4, 2.6, 1.1, p1_y, PALE_GOLD, GOLD, r"props $p$",
            r"$\rho,\ \mathrm{HOF},\ D,\ P$  z-scored")
    add_box(ax, 15.8, 2.6, 1.1, p1_y, PALE_GOLD, GOLD, r"trust mask $m$",
            "Tier-A/B = 1, else 0")
    add_box(ax, 19.4, 2.8, 1.1, p1_y, CREAM,     NAVY, "Cached triples",
            r"$\{(z_0, p, m)\}$")

    add_arrow(ax, 2.8,  p1_y, 3.8,  p1_y)
    add_arrow(ax, 6.2,  p1_y, 7.4,  p1_y)
    add_arrow(ax, 9.8,  p1_y, 11.1, p1_y)
    add_arrow(ax, 13.7, p1_y, 14.5, p1_y)
    add_arrow(ax, 17.1, p1_y, 18.0, p1_y)

    # ─── Phase 2: per-step training loop ────────────────────────────
    # 6 columns: minibatch | samplers | forward-diffuse | eps_theta | loss | adamw
    col_x  = [1.7, 5.0, 8.6, 12.4, 16.0, 19.5]
    mid_y  = 4.4   # main data-flow row
    top_y  = 6.6   # samplers row
    bot_y  = 2.4   # cfg-dropout row

    # Minibatch (col 1)
    add_box(ax, col_x[0], 2.4, 1.4, mid_y, PALE_GOLD, GOLD,
            "Mini-batch", r"$(z_0, p, m) \sim \mathcal{D}_\mathrm{cache}$",
            title_size=11.5, sub_size=9.5)

    # Samplers (col 2, top row)
    add_box(ax, col_x[1] - 0.85, 1.5, 0.95, top_y, PALE_PURPLE, PURPLE,
            r"$t$", r"$\mathcal{U}\{1{:}T\}$",
            title_size=11.0, sub_size=9.0)
    add_box(ax, col_x[1] + 0.85, 1.5, 0.95, top_y, PALE_PURPLE, PURPLE,
            r"$\varepsilon$", r"$\mathcal{N}(0, I)$",
            title_size=11.0, sub_size=9.0)

    # Forward diffusion (col 3, mid row)
    add_box(ax, col_x[2], 2.8, 1.6, mid_y, PALE_GREY, NAVY,
            "Forward diffusion",
            r"$z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$",
            title_size=11.5, sub_size=9.5)

    # cfg-dropout (col 2, bottom row)
    add_box(ax, col_x[1], 2.6, 1.4, bot_y, PALE_RED, RED,
            "cfg-dropout",
            r"$(p,m)\to\varnothing$ w.p. 0.10",
            title_size=11.0, sub_size=9.5)

    # Noise predictor (col 4, mid row)
    add_box(ax, col_x[3], 3.0, 1.6, mid_y, PALE_GREY, NAVY,
            r"Noise predictor  $\varepsilon_\theta$",
            r"FiLM ResNet, 44.6 M params",
            title_size=11.5, sub_size=9.5)

    # Loss (col 5, mid row)
    add_box(ax, col_x[4], 2.8, 1.6, mid_y, PALE_RED, RED,
            "Masked MSE loss",
            r"$\mathcal{L} = \Vert m\odot(\varepsilon - \hat\varepsilon)\Vert^2$",
            title_size=11.5, sub_size=10.0)

    # Optimizer (col 6, mid row)
    add_box(ax, col_x[5], 2.4, 1.6, mid_y, PALE_PURPLE, PURPLE,
            "AdamW + EMA",
            r"$\theta \leftarrow \theta - \eta\nabla_\theta\mathcal{L}$",
            title_size=11.5, sub_size=9.5)

    # Phase 1 -> Phase 2 link: a compact stacked-cache icon directly ABOVE the
    # mini-batch column (col_x[0]=1.7). Short down-arrow from cache icon to the
    # mini-batch box.  The Phase-1 "Cached triples" box reaches the cache via a
    # short, contained L-route in the gutter strip between the two phase bands
    # (band boundary at y=8.0). The connector hugs the gutter so it does not
    # span the figure.
    cache_x = col_x[0]            # 1.7 — directly above mini-batch
    cache_y_bot = 7.05            # nestled in the upper Phase-2 strip
    gutter_y = 7.85               # routed through the inter-phase gutter
    # Stacked-disc icon
    for dy in [0.00, 0.18, 0.36]:
        rect = FancyBboxPatch(
            (cache_x - 0.55, cache_y_bot + dy), 1.10, 0.16,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=1.0, facecolor=PALE_GOLD, edgecolor=GOLD, zorder=3,
        )
        ax.add_patch(rect)
    ax.text(cache_x + 1.35, cache_y_bot + 0.30,
            r"cache $\mathcal{D}_{\mathrm{cache}}$",
            ha="left", va="center", fontsize=9, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=4)
    # Phase-1 "Cached triples" -> gutter -> cache icon (compact L)
    ax.plot([19.4, 19.4], [p1_y - 0.6, gutter_y], color=NAVY, lw=1.2, zorder=4)
    ax.plot([19.4, cache_x], [gutter_y, gutter_y], color=NAVY, lw=1.2, zorder=4)
    add_arrow(ax, cache_x, gutter_y, cache_x, cache_y_bot + 0.55,
              color=NAVY, lw=1.2)
    add_label(ax, 10.5, gutter_y + 0.18,
              "written once  ->  read each step",
              color=TEXT_LIGHT, size=8.2, italic=True, ha="center")
    # Cache icon -> mini-batch (short)
    add_arrow(ax, cache_x, cache_y_bot - 0.05, cache_x, mid_y + 0.85,
              color=NAVY, lw=1.4)

    # ── Arrows inside Phase 2 ───────────────────────────────
    # minibatch -> forward diffusion (z_0)
    add_arrow(ax, col_x[0] + 0.75, mid_y, col_x[2] - 1.45, mid_y, lw=1.6)
    add_label(ax, col_x[0] + 1.0, mid_y + 0.30, r"$z_0$",
              color=TEXT_NAVY, size=10.5, italic=True)
    # t sampler -> forward diffusion (drop in)
    add_arrow(ax, col_x[1] - 0.85, top_y - 0.78, col_x[2] - 0.6, mid_y + 0.85,
              color=PURPLE, lw=1.2)
    # eps sampler -> forward diffusion
    add_arrow(ax, col_x[1] + 0.85, top_y - 0.78, col_x[2] + 0.0, mid_y + 0.85,
              color=PURPLE, lw=1.2)

    # eps sampler -> loss (true epsilon).  Route as L: up from sampler top to
    # y=top_y+0.45, across to above the loss box, then drop in from above.
    eps_x = col_x[1] + 0.85
    skip_y = top_y + 0.85
    loss_top = mid_y + 0.8
    ax.plot([eps_x, eps_x], [top_y + 0.78, skip_y],
            color=PURPLE, lw=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax.plot([eps_x, col_x[4]], [skip_y, skip_y],
            color=PURPLE, lw=1.2, linestyle=(0, (4, 3)), zorder=4)
    add_arrow(ax, col_x[4], skip_y, col_x[4], loss_top + 0.05,
              color=PURPLE, dashed=True, lw=1.2)
    add_label(ax, (eps_x + col_x[4]) / 2, skip_y - 0.3,
              r"true $\varepsilon$ (skip to loss)",
              color=PURPLE, size=9.5, italic=True, ha="center")

    # forward diffusion -> noise predictor (z_t)
    add_arrow(ax, col_x[2] + 1.45, mid_y, col_x[3] - 1.55, mid_y, lw=1.6)
    add_label(ax, (col_x[2] + col_x[3]) / 2 - 0.1, mid_y + 0.30, r"$z_t$",
              color=TEXT_NAVY, size=10.5, italic=True)

    # minibatch -> cfg-dropout (p, m down)
    add_arrow(ax, col_x[0] + 0.75, mid_y - 0.85, col_x[1] - 1.35, bot_y,
              color=GOLD, dashed=True, lw=1.4)
    add_label(ax, col_x[0] + 0.95, mid_y - 1.25, r"$p, m$",
              color=GOLD, size=10, italic=True)
    # cfg-dropout -> noise predictor (right then up)
    ax.plot([col_x[1] + 1.35, col_x[3]], [bot_y, bot_y],
            color=GOLD, lw=1.4, linestyle=(0, (5, 3)), zorder=4)
    add_arrow(ax, col_x[3], bot_y, col_x[3], mid_y - 0.85,
              color=GOLD, dashed=True, lw=1.4)
    add_label(ax, (col_x[1] + col_x[3]) / 2, bot_y + 0.30, r"$p', m'$",
              color=GOLD, size=10, italic=True, ha="center")

    # noise predictor -> loss
    add_arrow(ax, col_x[3] + 1.55, mid_y, col_x[4] - 1.45, mid_y, lw=1.6)
    add_label(ax, (col_x[3] + col_x[4]) / 2 - 0.1, mid_y + 0.30,
              r"$\hat\varepsilon$", color=TEXT_NAVY, size=10.5, italic=True)

    # loss -> optimizer (gradient)
    add_arrow(ax, col_x[4] + 1.45, mid_y, col_x[5] - 1.25, mid_y,
              color=RED, lw=1.6)
    add_label(ax, (col_x[4] + col_x[5]) / 2 - 0.1, mid_y + 0.30,
              r"$\nabla_\theta\mathcal{L}$", color=RED, size=10, italic=True)

    # optimizer -> noise predictor (param update, L-route below mid row)
    upd_y = 1.05
    ax.plot([col_x[5], col_x[5]], [mid_y - 0.85, upd_y],
            color=PURPLE, lw=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax.plot([col_x[5], col_x[3]], [upd_y, upd_y],
            color=PURPLE, lw=1.2, linestyle=(0, (4, 3)), zorder=4)
    add_arrow(ax, col_x[3], upd_y, col_x[3], mid_y - 0.85,
              color=PURPLE, dashed=True, lw=1.2)
    add_label(ax, (col_x[3] + col_x[5]) / 2, upd_y - 0.30,
              r"update $\theta$  (EMA decay 0.999)",
              color=PURPLE, size=9, italic=True, ha="center")

    # Footer caption
    add_label(ax, 0.3, 0.05,
              r"AdamW, peak LR $10^{-4}$, cosine decay, batch 256, EMA decay 0.999.  "
              r"Tier-C/D rows have $m=0$ on missing properties; cfg-dropout drops $(p,m)$ entirely 10% of steps.",
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
    Row 2 (middle, main pipeline): z_T -> DDIM -> z_0 -> LIMO decoder -> Pool.
    Row 3 (bottom): chem filter -> reranker -> top-K leads.
    Heads inject gradients down into DDIM (gold dashed); pool drops down into
    the filter chain via a clean L-shaped connector that does not cross other boxes.
    """
    fig, ax = plt.subplots(figsize=(14, 6.4), dpi=300)
    setup_axes(ax, xmax=20.0, ymax=8.4)

    # Banner
    add_label(ax, 0.3, 8.15, "SAMPLING / INFERENCE", color=TEXT_NAVY, size=11, bold=True)
    add_label(ax, 0.3, 7.75,
              "DDIM denoise with multi-head classifier guidance, then chemistry-credibility filter",
              color=TEXT_SLATE, size=9, italic=True)

    # ── Row 1 (top): 4 classifier-guidance heads ───────────────────
    add_label(ax, 0.3, 6.95, "MULTI-HEAD CLASSIFIER GUIDANCE",
              color=TEXT_LIGHT, size=8.5, bold=True)
    head_y = 6.1
    head_specs = [
        (4.5,  "Viability",   "RandomForest on z"),
        (8.0,  "Sensitivity", r"$h_{50}$ on Huang-Massa"),
        (11.5, "Performance", r"$\rho$, $D$, $P$ heads"),
        (15.0, "Hazard",      "SMARTS-aware learned"),
    ]
    for x_c, t, sub in head_specs:
        add_box(ax, x_c, 2.6, 1.05, head_y, PALE_GREEN, GREEN, t, sub,
                title_size=11.0, sub_size=9.0)

    # row2_y declared early so the bundled bus drop can reference it.
    row2_y = 4.0
    # Bundled-gradient bus: a horizontal gold line collects all four head
    # outputs into a single stream that drops into the DDIM box top via one
    # arrow.  Bus sits between the heads (head_y=6.1) and the DDIM top
    # (row2_y + 0.75 = 4.75).
    bus_y = 5.40
    bus_x_left  = 4.5
    bus_x_right = 15.0
    # vertical drops from each head bottom down to the bus
    for x_c, *_ in head_specs:
        ax.plot([x_c, x_c], [head_y - 0.55, bus_y + 0.02],
                color=GOLD, lw=1.3, linestyle=(0, (4, 3)), zorder=4)
        # join-dot where each head meets the bus
        ax.plot([x_c], [bus_y], marker="o", markersize=3.5,
                markerfacecolor=GOLD, markeredgecolor=GOLD, zorder=5)
    # Horizontal bus line joining all four drops
    ax.plot([bus_x_left, bus_x_right], [bus_y, bus_y],
            color=GOLD, lw=2.0, zorder=4)
    # Single bundled drop from bus into DDIM top — clearly visible gap
    ddim_top_y = row2_y + 1.5 / 2  # = 4.75
    bundle_x = 6.5  # DDIM x_c
    add_arrow(ax, bundle_x, bus_y, bundle_x, ddim_top_y + 0.05,
              color=GOLD, lw=2.2)
    # Equation label tucked to the right of the bundled drop
    add_label(ax, bundle_x + 0.30, (bus_y + ddim_top_y) / 2 + 0.05,
              r"$\sum_h s_h\,\nabla_{z_t}\log p_h(c\!\mid\!z_t)$",
              color=GOLD, size=9.5, italic=True, bold=True)

    # ── Row 2 (middle): main pipeline ──
    add_box(ax, 1.6,  1.6, 1.2, row2_y, CREAM,     NAVY, r"$z_T$",
            r"$\mathcal{N}(0, I_{1024})$")
    add_box(ax, 6.5,  3.6, 1.5, row2_y, PALE_GREY, NAVY, "DDIM denoise",
            r"$\hat\varepsilon = \varepsilon_\theta(z_t,t,c)+\sum_h s_h\,\nabla_{z_t}\log p_h(c\mid z_t)$",
            title_size=11.0, sub_size=9.0)
    add_box(ax, 11.5, 1.6, 1.2, row2_y, PALE_GOLD, GOLD, r"$z_0$", "denoised latent")
    add_box(ax, 14.7, 2.4, 1.2, row2_y, PALE_GREY, NAVY, "LIMO decoder",
            "non-autoregressive")
    add_box(ax, 18.3, 2.0, 1.2, row2_y, CREAM,     NAVY, "Pool",
            r"$\sim 10^4$ SMILES")

    add_arrow(ax, 2.4,  row2_y, 4.7, row2_y)
    add_arrow(ax, 8.3,  row2_y, 10.9, row2_y)
    add_arrow(ax, 12.1, row2_y, 13.5, row2_y)
    add_arrow(ax, 15.9, row2_y, 17.3, row2_y)
    add_label(ax, 3.3, row2_y + 0.25, r"$t = T \to 1$", color=TEXT_SLATE,
              size=9, italic=True)

    # (Heads -> DDIM gradients now go via the bundled gold bus drawn above.)
    add_label(ax, 19.7, 6.95,
              "head outputs share one\nguidance bus into DDIM",
              color=TEXT_LIGHT, size=8.2, italic=True, ha="right")

    # ── Row 3 (bottom): pool → filter → reranker → top-K ──
    row3_y = 1.4
    add_box(ax, 13.5, 2.6, 1.2, row3_y, PALE_RED,   RED,
            "Chem-rules filter", "energetic-SMARTS, hard reject")
    add_box(ax, 9.0,  2.6, 1.2, row3_y, PALE_GREEN, GREEN,
            "Phase-A reranker", "scaffold-aware composite")
    add_box(ax, 4.5,  2.6, 1.2, row3_y, PALE_GOLD,  GOLD,
            "Top-K leads", r"$K \approx 100$")

    # Pool -> chem-rules: L-shaped connector (down then left only over its own column)
    # Pool is at (18.3, row2_y). Filter is at (13.5, row3_y).
    # First go down from pool to y=2.6 staying at x=18.3, then left to chem-filter at y=row3_y.
    # Use plot lines + a final arrow.
    ax.plot([18.3, 18.3], [row2_y - 0.8, 2.6], color=NAVY, lw=1.6, zorder=4)
    ax.plot([18.3, 14.8], [2.6, 2.6],          color=NAVY, lw=1.6, zorder=4)
    add_arrow(ax, 14.8, 2.6, 14.8, row3_y + 0.55, color=NAVY, lw=1.6)
    # filter -> reranker -> top-K (right-to-left)
    add_arrow(ax, 12.2, row3_y, 10.3, row3_y, color=NAVY, lw=1.8)
    add_arrow(ax, 7.7,  row3_y, 5.8,  row3_y, color=NAVY, lw=1.8)

    # Mini-caption
    add_label(ax, 0.3, 0.25,
              r"DDIM with $n=40$ steps; conditioning $c=(p_\mathrm{target},\,m=\mathbf{1})$.  "
              r"Per-head scales $s_h$ tabulated in Appendix C.",
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
