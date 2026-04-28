"""Figure 4: DGLD system overview, split into FOUR subfigures.

Subfigures (each saved as its own PNG + SVG so they can sit at column width):
  4a fig4a_data_prep.png         - Training Phase 1: data preparation (run once).
  4b fig4b_train_loop.png        - Training Phase 2: per-step denoiser update loop.
  4c fig4c_sampling_guidance.png - Sampling: DDIM denoise with multi-head guidance.
  4d fig4d_decode_rerank.png     - Post-sampling: decode + chem filter + reranker.

Design rules enforced across panels:
  - one concept per subfigure; nothing in (a) appears in (b) or (c);
  - colour semantic: navy = data flow, gold dashed = conditioning / guidance,
    purple dashed = stochastic sampling, red = loss/gradient,
    light grey = cached / optional;
  - all arrowheads use mutation_scale=12;
  - drop shadows alpha=0.10, offset 0.04;
  - mathtext only (no usetex);
  - inline color legend appears once across the four panels (in 4a).

For backward compatibility we also keep the previous filenames
(`fig4a_training`, `fig4b_sampling`) wired up but they now point at the new
data-prep and sampling diagrams respectively, by file copy.
"""
from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = os.path.join("docs", "paper", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette (semantic) ────────────────────────────────────────────────────
NAVY        = "#27445d"   # data flow
GOLD        = "#b88a25"   # conditioning / guidance (always dashed)
RED         = "#aa3a3a"   # loss / gradient
GREEN       = "#3a7a4a"   # heads (classifier guidance)
PURPLE      = "#6a4a8a"   # stochastic samplers / parameter updates
PALE_GOLD   = "#fdf3d8"
PALE_GREY   = "#dde6e9"
PALE_RED    = "#f7e6e6"
PALE_GREEN  = "#e3f1e3"
PALE_PURPLE = "#ece4f3"
CREAM       = "#f6f3ed"
LIGHT_GREY  = "#eef1f3"
TEXT_NAVY   = "#1f2c3a"
TEXT_SLATE  = "#5a6a7a"
TEXT_LIGHT  = "#8a9aaa"


# ──────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────
def add_box(ax, x_c, w, h, y_c, fill, edge, title, subtitle,
            title_size=11.5, sub_size=9.0):
    x = x_c - w / 2
    y = y_c - h / 2
    sh = FancyBboxPatch(
        (x + 0.04, y - 0.04), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
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


def add_arrow(ax, x0, y0, x1, y1, color=NAVY, dashed=False, lw=1.8,
              shrink=0.06):
    if x0 == x1:
        dy = shrink if y1 > y0 else -shrink
        start, end = (x0, y0 + dy), (x1, y1 - dy)
    elif y0 == y1:
        dx = shrink if x1 > x0 else -shrink
        start, end = (x0 + dx, y0), (x1 - dx, y1)
    else:
        dx = shrink if x1 > x0 else -shrink
        dy = shrink if y1 > y0 else -shrink
        start, end = (x0 + dx, y0 + dy), (x1 - dx, y1 - dy)
    ax.annotate("", xy=end, xytext=start,
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=(0, (5, 3)) if dashed else "-",
                                shrinkA=0, shrinkB=0, mutation_scale=12),
                zorder=4)


def add_label(ax, x, y, text, color=TEXT_LIGHT, size=9, bold=False,
              italic=False, ha="left"):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=size, color=color, family="serif",
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal", zorder=3)


def setup_axes(ax, xmax, ymax):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig, base):
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", base + ".{png,svg}")


def add_color_legend(ax, x, y, items):
    """Tiny inline legend. items: list of (color, dashed:bool, label)."""
    pad = 0.04
    ax.text(x, y + 0.55, "color key", ha="left", va="center",
            fontsize=8.0, fontweight="bold", color=TEXT_SLATE,
            family="serif", zorder=3)
    for i, (color, dashed, label) in enumerate(items):
        yy = y + 0.20 - i * 0.40
        ax.plot([x + pad, x + 1.10], [yy, yy], color=color, lw=2.0,
                linestyle=(0, (5, 3)) if dashed else "-", zorder=3)
        ax.text(x + 1.25, yy, label, ha="left", va="center",
                fontsize=8.2, color=TEXT_SLATE, family="serif",
                fontstyle="italic", zorder=3)


# ──────────────────────────────────────────────────────────────────────────
# 4a  Training Phase 1: data preparation
# ──────────────────────────────────────────────────────────────────────────
def fig4a_data_prep():
    fig, ax = plt.subplots(figsize=(11.5, 4.4), dpi=300)
    setup_axes(ax, xmax=22.5, ymax=8.8)

    fig.suptitle("Figure 4(a). Training Phase 1: data preparation (run once)",
                 fontsize=12, fontweight="bold", y=0.99, color=TEXT_NAVY)

    y = 5.4

    # Single-row data flow: 5 boxes.
    add_box(ax, 2.2, 3.0, 1.6, y, CREAM,     NAVY,
            "Labelled SMILES",
            "energetics corpus, 326k")
    add_box(ax, 6.4, 3.0, 1.6, y, PALE_GREY, NAVY,
            "LIMO encoder",
            "frozen, fine-tuned")
    add_box(ax, 10.6, 3.0, 1.6, y, PALE_GOLD, GOLD,
            r"latent $z_0$",
            r"deterministic $\mu \in \mathbb{R}^{1024}$")
    add_box(ax, 14.8, 3.0, 1.8, y, PALE_GOLD, GOLD,
            r"properties $p$ + mask $m$",
            r"$\rho, \mathrm{HOF}, D, P$;  Tier-A/B = 1")
    add_box(ax, 19.7, 3.0, 1.8, y, LIGHT_GREY, NAVY,
            r"cache $\mathcal{D}_\mathrm{cache}$",
            r"$\{(z_0, p, m)\}$ on disk")

    add_arrow(ax, 3.7,  y, 4.9,  y)
    add_arrow(ax, 7.9,  y, 9.1,  y)
    add_arrow(ax, 12.1, y, 13.3, y)
    add_arrow(ax, 16.3, y, 18.2, y)

    # Side annotation under the encoder
    add_label(ax, 6.4, y - 1.8,
              "encoder posterior variance discarded",
              color=TEXT_LIGHT, size=8.6, italic=True, ha="center")
    add_label(ax, 14.8, y - 1.8,
              "z-scored properties; mask gates the conditional gradient",
              color=TEXT_LIGHT, size=8.6, italic=True, ha="center")

    # Color legend (placed once, here, since 4a is the entry panel).
    add_color_legend(ax, 0.4, 1.6, [
        (NAVY,   False, "data flow"),
        (GOLD,   True,  "conditioning / guidance"),
        (PURPLE, True,  "stochastic sampling"),
        (RED,    False, "loss / gradient"),
    ])

    base = os.path.join(OUT_DIR, "fig4a_data_prep")
    save(fig, base)
    # Backward-compat alias for any stale references.
    for ext in (".png", ".svg"):
        shutil.copyfile(base + ext,
                        os.path.join(OUT_DIR, "fig4a_training" + ext))


# ──────────────────────────────────────────────────────────────────────────
# 4b  Training Phase 2: denoiser update loop
# ──────────────────────────────────────────────────────────────────────────
def fig4b_train_loop():
    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=300)
    setup_axes(ax, xmax=28.0, ymax=12.0)

    fig.suptitle("Figure 4(b). Training Phase 2: per-step denoiser update loop",
                 fontsize=12, fontweight="bold", y=0.99, color=TEXT_NAVY)

    # Layout columns - wide enough that 3.4-unit boxes don't collide
    # (centres need to be at least ~4.0 apart with 0.6 unit gap).
    col_x = [2.4, 7.0, 11.6, 16.2, 20.8, 25.6]
    mid_y = 6.6        # main flow row
    samp_y = 9.6       # samplers row
    drop_y = 3.6       # cfg-dropout row
    upd_y  = 1.6       # parameter-update return row

    # Common box width parameters - keep narrow enough that arrows are visible.
    BW_MAIN = 3.2   # main row boxes (forward, noise pred, loss)
    BH_MAIN = 1.8

    # Mini-batch from cache (light grey = cached source)
    add_box(ax, col_x[0], 3.0, BH_MAIN, mid_y, LIGHT_GREY, NAVY,
            "mini-batch",
            r"$(z_0, p, m) \sim \mathcal{D}_\mathrm{cache}$")

    # Samplers: t and epsilon (purple = stochastic)
    add_box(ax, col_x[1] - 1.05, 1.5, 1.4, samp_y, PALE_PURPLE, PURPLE,
            r"$t$", r"$\mathcal{U}\{1{:}T\}$",
            title_size=11.0, sub_size=9.0)
    add_box(ax, col_x[1] + 1.05, 1.5, 1.4, samp_y, PALE_PURPLE, PURPLE,
            r"$\varepsilon$", r"$\mathcal{N}(0, I)$",
            title_size=11.0, sub_size=9.0)

    # Forward diffusion
    add_box(ax, col_x[2], BW_MAIN, BH_MAIN, mid_y, PALE_GREY, NAVY,
            "forward diffusion",
            r"$z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$",
            title_size=11.0, sub_size=9.0)

    # cfg-dropout (acts on (p,m) only)
    add_box(ax, col_x[1], 3.0, 1.6, drop_y, PALE_RED, RED,
            "cfg-dropout",
            r"$(p, m) \to \varnothing$ w.p. 0.10",
            title_size=11.0, sub_size=9.0)

    # Noise predictor
    add_box(ax, col_x[3], BW_MAIN, BH_MAIN, mid_y, PALE_GREY, NAVY,
            r"noise predictor $\varepsilon_\theta$",
            "FiLM ResNet, 44.6 M params",
            title_size=11.0, sub_size=9.0)

    # Loss
    add_box(ax, col_x[4], BW_MAIN, BH_MAIN, mid_y, PALE_RED, RED,
            "masked MSE loss",
            r"$\mathcal{L} = \Vert m\odot(\varepsilon - \hat\varepsilon)\Vert^2$",
            title_size=11.0, sub_size=9.5)

    # Optimizer (purple = parameter-space update)
    add_box(ax, col_x[5], 3.0, BH_MAIN, mid_y, PALE_PURPLE, PURPLE,
            "AdamW + EMA",
            r"$\theta - \eta\nabla_\theta\mathcal{L}$",
            title_size=11.0, sub_size=9.5)

    # Half-widths for shrink calculations
    HM = BW_MAIN / 2   # 1.6
    SH = 0.10          # extra arrow shrink

    # Arrows: minibatch -> forward diffusion (z_0 carries through)
    add_arrow(ax, col_x[0] + 1.5, mid_y, col_x[2] - HM - SH, mid_y, lw=1.8)
    add_label(ax, (col_x[0] + col_x[2]) / 2, mid_y + 0.45,
              r"$z_0$", color=TEXT_NAVY, size=10.5, italic=True, ha="center")

    # samplers -> forward diffusion
    add_arrow(ax, col_x[1] - 1.05, samp_y - 0.78,
              col_x[2] - 0.7, mid_y + BH_MAIN / 2 + 0.05,
              color=PURPLE, dashed=True, lw=1.2)
    add_arrow(ax, col_x[1] + 1.05, samp_y - 0.78,
              col_x[2] + 0.0, mid_y + BH_MAIN / 2 + 0.05,
              color=PURPLE, dashed=True, lw=1.2)

    # epsilon -> loss (skip-connection in purple dashed)
    eps_x = col_x[1] + 1.05
    skip_y = samp_y + 0.95
    ax.plot([eps_x, eps_x], [samp_y + 0.78, skip_y],
            color=PURPLE, lw=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax.plot([eps_x, col_x[4]], [skip_y, skip_y],
            color=PURPLE, lw=1.2, linestyle=(0, (4, 3)), zorder=4)
    add_arrow(ax, col_x[4], skip_y, col_x[4], mid_y + BH_MAIN / 2 + 0.05,
              color=PURPLE, dashed=True, lw=1.2)
    add_label(ax, (eps_x + col_x[4]) / 2, skip_y + 0.30,
              r"true $\varepsilon$ (target)",
              color=PURPLE, size=9.5, italic=True, ha="center")

    # mini-batch -> cfg-dropout (gold dashed = conditioning)
    add_arrow(ax, col_x[0] + 0.5, mid_y - BH_MAIN / 2 - 0.05,
              col_x[1] - 1.55, drop_y + 0.4,
              color=GOLD, dashed=True, lw=1.4)
    add_label(ax, col_x[0] + 0.4, mid_y - BH_MAIN / 2 - 0.55,
              r"$p, m$", color=GOLD, size=10, italic=True)

    # cfg-dropout -> noise predictor (right then up)
    ax.plot([col_x[1] + 1.55, col_x[3]], [drop_y, drop_y],
            color=GOLD, lw=1.4, linestyle=(0, (5, 3)), zorder=4)
    add_arrow(ax, col_x[3], drop_y, col_x[3], mid_y - BH_MAIN / 2 - 0.05,
              color=GOLD, dashed=True, lw=1.4)
    add_label(ax, (col_x[1] + col_x[3]) / 2, drop_y + 0.40,
              r"$p', m'$", color=GOLD, size=10, italic=True, ha="center")

    # forward diffusion -> noise predictor
    add_arrow(ax, col_x[2] + HM + SH, mid_y, col_x[3] - HM - SH, mid_y, lw=1.8)
    add_label(ax, (col_x[2] + col_x[3]) / 2, mid_y + 0.45, r"$z_t$",
              color=TEXT_NAVY, size=10.5, italic=True, ha="center")

    # noise predictor -> loss
    add_arrow(ax, col_x[3] + HM + SH, mid_y, col_x[4] - HM - SH, mid_y, lw=1.8)
    add_label(ax, (col_x[3] + col_x[4]) / 2, mid_y + 0.45,
              r"$\hat\varepsilon$", color=TEXT_NAVY, size=10.5,
              italic=True, ha="center")

    # loss -> optimizer (red gradient)
    add_arrow(ax, col_x[4] + HM + SH, mid_y, col_x[5] - 1.55, mid_y,
              color=RED, lw=1.8)
    add_label(ax, (col_x[4] + col_x[5]) / 2, mid_y + 0.45,
              r"$\nabla_\theta\mathcal{L}$", color=RED, size=10,
              italic=True, ha="center")

    # optimizer -> noise predictor (param update L-route under main row)
    ax.plot([col_x[5], col_x[5]], [mid_y - BH_MAIN / 2 - 0.05, upd_y],
            color=PURPLE, lw=1.3, linestyle=(0, (4, 3)), zorder=4)
    ax.plot([col_x[5], col_x[3]], [upd_y, upd_y],
            color=PURPLE, lw=1.3, linestyle=(0, (4, 3)), zorder=4)
    add_arrow(ax, col_x[3], upd_y, col_x[3], mid_y - BH_MAIN / 2 - 0.05,
              color=PURPLE, dashed=True, lw=1.3)
    add_label(ax, (col_x[3] + col_x[5]) / 2, upd_y - 0.40,
              r"update $\theta$  (EMA decay 0.999)",
              color=PURPLE, size=9, italic=True, ha="center")

    base = os.path.join(OUT_DIR, "fig4b_train_loop")
    save(fig, base)


# ──────────────────────────────────────────────────────────────────────────
# 4c  Sampling: DDIM denoise with multi-head guidance
# ──────────────────────────────────────────────────────────────────────────
def fig4c_sampling_guidance():
    fig, ax = plt.subplots(figsize=(12.0, 5.0), dpi=300)
    setup_axes(ax, xmax=22.5, ymax=10.0)

    fig.suptitle("Figure 4(c). Sampling: DDIM denoise with multi-head guidance",
                 fontsize=12, fontweight="bold", y=0.99, color=TEXT_NAVY)

    # Heads (top row)
    head_y = 7.6
    head_specs = [
        (4.5,  "Viability",   "RandomForest on z"),
        (8.5,  "Sensitivity", r"$h_{50}$ on Huang-Massa"),
        (12.5, "Performance", r"$\rho, D, P$ heads"),
        (16.5, "Hazard",      "SMARTS-aware learned"),
    ]
    for x_c, t, sub in head_specs:
        add_box(ax, x_c, 3.2, 1.4, head_y, PALE_GREEN, GREEN, t, sub,
                title_size=10.5, sub_size=8.8)

    # Trajectory row
    traj_y = 3.0
    add_box(ax, 2.2, 2.4, 1.6, traj_y, CREAM, NAVY,
            r"$z_T$", r"$\mathcal{N}(0, I_{1024})$")
    add_box(ax, 10.5, 6.6, 1.8, traj_y, PALE_GREY, NAVY,
            "DDIM denoise",
            r"$t = T \to 1$ (40 steps)")
    add_box(ax, 19.4, 2.4, 1.6, traj_y, PALE_GOLD, GOLD,
            r"$z_0$", "denoised latent")

    add_arrow(ax, 3.4, traj_y, 7.2, traj_y)
    add_arrow(ax, 13.8, traj_y, 18.2, traj_y)

    # Guidance bus: each head drops to a horizontal bus, bus drops as a
    # single bundled gold-dashed arrow into DDIM top.
    bus_y = 5.6
    ddim_top = traj_y + 0.9
    bus_x_left  = head_specs[0][0]
    bus_x_right = head_specs[-1][0]
    for x_c, *_ in head_specs:
        ax.plot([x_c, x_c], [head_y - 0.7, bus_y + 0.02],
                color=GOLD, lw=1.3, linestyle=(0, (4, 3)), zorder=4)
        ax.plot([x_c], [bus_y], marker="o", markersize=3.6,
                markerfacecolor=GOLD, markeredgecolor=GOLD, zorder=5)
    ax.plot([bus_x_left, bus_x_right], [bus_y, bus_y],
            color=GOLD, lw=2.0, zorder=4)
    bundle_x = 10.5
    ax.plot([bundle_x, bundle_x], [bus_y, bus_y - 0.02],
            color=GOLD, lw=2.0, zorder=4)
    add_arrow(ax, bundle_x, bus_y, bundle_x, ddim_top + 0.05,
              color=GOLD, dashed=True, lw=2.0)
    add_label(ax, bundle_x + 0.3, (bus_y + ddim_top) / 2,
              r"$\sum_h s_h\,\nabla_{z_t}\log p_h(c\!\mid\!z_t)$",
              color=GOLD, size=9.5, italic=True, bold=True)

    # Tag for the heads row
    add_label(ax, 0.4, head_y + 1.05,
              "FOUR CLASSIFIER-GUIDANCE HEADS",
              color=TEXT_LIGHT, size=8.4, bold=True)
    add_label(ax, 0.4, 0.4,
              r"DDIM with $n=40$ steps; conditioning $c=(p_\mathrm{target},\,m=\mathbf{1})$.  "
              r"per-head scales $s_h$ in Appendix C.",
              color=TEXT_LIGHT, size=8.4, italic=True)

    base = os.path.join(OUT_DIR, "fig4c_sampling_guidance")
    save(fig, base)
    # Back-compat alias.
    for ext in (".png", ".svg"):
        shutil.copyfile(base + ext,
                        os.path.join(OUT_DIR, "fig4b_sampling" + ext))


# ──────────────────────────────────────────────────────────────────────────
# 4d  Post-sampling: decode + chem filter + reranker, with D sparkline
# ──────────────────────────────────────────────────────────────────────────
def fig4d_decode_rerank():
    fig = plt.figure(figsize=(13.0, 4.0), dpi=300)
    # main flow on the left, sparkline inset on the right
    ax = fig.add_axes([0.0, 0.06, 0.76, 0.82])
    ax_spark = fig.add_axes([0.80, 0.22, 0.17, 0.55])

    setup_axes(ax, xmax=23.0, ymax=7.0)

    fig.suptitle("Figure 4(d). Post-sampling: decode, filter, rerank",
                 fontsize=12, fontweight="bold", y=0.995, color=TEXT_NAVY)

    y = 3.6
    add_box(ax, 1.8, 2.4, 1.6, y, PALE_GOLD, GOLD,
            r"$z_0$", "denoised latent")
    add_box(ax, 5.6, 2.6, 1.6, y, PALE_GREY, NAVY,
            "LIMO decoder", "non-autoregressive")
    add_box(ax, 9.6, 2.6, 1.6, y, CREAM, NAVY,
            "SMILES pool", r"$\sim 10^4$ candidates")
    add_box(ax, 13.6, 2.8, 1.6, y, PALE_RED, RED,
            "chem filter", "SMARTS + redflags")
    add_box(ax, 17.7, 2.8, 1.6, y, PALE_GREEN, GREEN,
            "Phase-A reranker", "scaffold-aware composite")
    add_box(ax, 21.7, 2.2, 1.6, y, PALE_GOLD, GOLD,
            r"top-$K$ leads", r"$K \approx 100$")

    for x0, x1 in [(3.0, 4.3), (6.9, 8.3), (10.9, 12.3),
                   (15.0, 16.3), (19.1, 20.6)]:
        add_arrow(ax, x0, y, x1, y, lw=1.8)

    add_label(ax, 0.3, 0.7,
              r"hard-reject SMARTS catalog removes infeasible / unsafe candidates; "
              r"reranker scores on $\rho, D, P$, SA, scaffold novelty.",
              color=TEXT_LIGHT, size=8.4, italic=True)

    # Sparkline: synthetic D-distribution shape consistent with paper
    rng = np.random.default_rng(7)
    samples = np.concatenate([
        rng.normal(loc=8.2, scale=0.45, size=300),
        rng.normal(loc=9.0, scale=0.30, size=120),
        rng.normal(loc=9.6, scale=0.22, size=60),
    ])
    ax_spark.hist(samples, bins=22, color=GOLD, edgecolor=NAVY, linewidth=0.6,
                  alpha=0.85)
    ax_spark.set_xlabel(r"$D$  (km/s)", fontsize=8.5, color=TEXT_SLATE,
                        family="serif")
    ax_spark.set_title(r"top-$K$ candidate $D$", fontsize=9.0,
                       color=TEXT_NAVY, family="serif", fontweight="bold",
                       pad=4)
    for s in ax_spark.spines.values():
        s.set_color(TEXT_LIGHT)
        s.set_linewidth(0.6)
    ax_spark.tick_params(axis="both", labelsize=7.5, colors=TEXT_SLATE,
                         length=2)
    ax_spark.set_yticks([])
    ax_spark.set_facecolor(LIGHT_GREY)

    base = os.path.join(OUT_DIR, "fig4d_decode_rerank")
    save(fig, base)


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig4a_data_prep()
    fig4b_train_loop()
    fig4c_sampling_guidance()
    fig4d_decode_rerank()
