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
    # NEW 4(a): the LIMO SELFIES-VAE FINE-TUNING LOOP itself.
    # Reader question that drove the rewrite: "do properties enter LIMO
    # fine-tuning?" Answer: no. Diagram makes that visually unmissable.
    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=300)
    setup_axes(ax, xmax=27.0, ymax=14.4)

    # Title intentionally omitted; the HTML figcaption carries the label.

    # Main forward chain along y = y_main (left to right):
    # SMILES->SELFIES  ->  Encoder  ->  (mu, sigma)  ->  z=mu+sigma*eps  ->  Decoder  ->  predicted SELFIES
    y_main = 9.4

    # ── Input box: SMILES -> SELFIES tokens ─────────────────────────────
    add_box(ax, 1.9, 3.0, 1.8, y_main, CREAM, NAVY,
            "SMILES",
            r"$\to$ SELFIES tokens",
            title_size=10.6, sub_size=8.4)

    # ── Encoder (transformer, trainable, navy fill) ─────────────────────
    # Drop shadow + navy box drawn directly (we want WHITE text, which
    # add_box does not support).
    ax.add_patch(FancyBboxPatch(
        (5.9 - 1.9 + 0.04, y_main - 1.1 - 0.04), 3.8, 2.2,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (5.9 - 1.9, y_main - 1.1), 3.8, 2.2,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.5, facecolor=NAVY, edgecolor=NAVY, zorder=2,
    ))
    ax.text(5.9, y_main + 0.40, "Encoder", ha="center", va="center",
            fontsize=11.5, fontweight=600, color="white",
            family="serif", zorder=3)
    ax.text(5.9, y_main - 0.24, "transformer (trainable)",
            ha="center", va="center", fontsize=8.4, fontstyle="italic",
            color="#dde6e9", family="serif", zorder=3)

    # ── Latent posterior: mu, sigma split ───────────────────────────────
    # Two stacked narrow boxes for mu (top) and sigma (bottom)
    add_box(ax, 10.2, 2.6, 0.9, y_main + 0.55, PALE_GOLD, GOLD,
            r"$\mu$", "", title_size=11.0)
    add_box(ax, 10.2, 2.6, 0.9, y_main - 0.55, PALE_PURPLE, PURPLE,
            r"$\sigma$", "", title_size=11.0)
    ax.text(10.2, y_main + 1.40, "posterior  q(z | x)",
            ha="center", va="center", fontsize=8.8, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    # ── Reparametrise: z = mu + sigma*eps  (purple stochastic) ──────────
    add_box(ax, 14.4, 3.0, 1.7, y_main, PALE_PURPLE, PURPLE,
            "Reparametrise",
            r"$z = \mu + \sigma\!\cdot\!\varepsilon$",
            title_size=10.8, sub_size=9.0)
    # epsilon source as a small purple dashed bubble above
    ax.text(14.4, y_main + 1.55, r"$\varepsilon \sim \mathcal{N}(0, I)$",
            ha="center", va="center", fontsize=9.0, fontstyle="italic",
            color=PURPLE, family="serif", zorder=3)
    add_arrow(ax, 14.4, y_main + 1.30, 14.4, y_main + 0.86,
              color=PURPLE, dashed=True, lw=1.2)

    # ── Decoder (transformer, trainable, navy fill) ─────────────────────
    # drop shadow first (lower zorder)
    ax.add_patch(FancyBboxPatch(
        (18.6 - 1.9 + 0.04, y_main - 1.1 - 0.04), 3.8, 2.2,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (18.6 - 1.9, y_main - 1.1), 3.8, 2.2,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.5, facecolor=NAVY, edgecolor=NAVY, zorder=2,
    ))
    ax.text(18.6, y_main + 0.40, "Decoder", ha="center", va="center",
            fontsize=11.5, fontweight=600, color="white",
            family="serif", zorder=3)
    ax.text(18.6, y_main - 0.24, "transformer (trainable)",
            ha="center", va="center", fontsize=8.4, fontstyle="italic",
            color="#dde6e9", family="serif", zorder=3)

    # ── Output: predicted SELFIES tokens ────────────────────────────────
    add_box(ax, 23.2, 3.0, 1.8, y_main, CREAM, NAVY,
            "predicted",
            "SELFIES tokens",
            title_size=10.6, sub_size=8.4)

    # ── Forward arrows along main row ───────────────────────────────────
    add_arrow(ax, 3.4,  y_main, 4.3,  y_main, lw=1.8)
    add_arrow(ax, 7.5,  y_main, 8.9,  y_main, lw=1.8)
    # encoder output splits to mu and sigma — show as a tiny fan
    # (already implied by the two side-by-side box positions); add labels above
    # mu / sigma -> reparam
    add_arrow(ax, 11.5, y_main + 0.55, 12.9, y_main + 0.20,
              color=GOLD, lw=1.6)
    add_arrow(ax, 11.5, y_main - 0.55, 12.9, y_main - 0.20,
              color=PURPLE, lw=1.6)
    # reparam -> decoder
    add_arrow(ax, 15.9, y_main, 17.0, y_main, lw=1.8)
    add_label(ax, 16.45, y_main + 0.42, r"$z$",
              color=TEXT_NAVY, size=10.5, italic=True, ha="center")
    # decoder -> predicted tokens
    add_arrow(ax, 20.2, y_main, 21.7, y_main, lw=1.8)

    # ─────────────────────────────────────────────────────────────────────
    # LOSS BLOCK (lower portion of the canvas)
    # Reconstruction loss compares input SELFIES to predicted SELFIES;
    # KL loss attaches to (mu, sigma).  Both feed into total loss L.
    # ─────────────────────────────────────────────────────────────────────
    y_loss = 4.4

    # Reconstruction loss (red), positioned under the decoder/output area.
    # Subtitle makes the comparison-against-input explicit so we do not
    # need a long diagonal arrow from the SMILES input.
    add_box(ax, 21.0, 5.0, 1.8, y_loss, PALE_RED, RED,
            r"$\mathcal{L}_{\mathrm{recon}}$",
            "token cross-entropy  (predicted vs input x, teacher-forced)",
            title_size=11.4, sub_size=8.0)

    # KL loss (red), positioned under the latent posterior
    add_box(ax, 10.2, 5.6, 1.8, y_loss, PALE_RED, RED,
            r"$\mathcal{L}_{\mathrm{KL}} = \mathrm{KL}\!\left(q(z|x)\,\Vert\,\mathcal{N}(0, I)\right)$",
            r"penalises  $\Vert\mu\Vert,\ \Vert\sigma^2 - 1\Vert$",
            title_size=10.4, sub_size=8.4)

    # Total loss (red, central, slightly larger), sits between the two
    add_box(ax, 15.6, 4.6, 2.0, y_loss - 2.4, PALE_RED, RED,
            r"$\mathcal{L} = \mathcal{L}_{\mathrm{recon}} + 0.01\,\mathcal{L}_{\mathrm{KL}}$",
            r"$\beta = 0.01$,  free-bits disabled",
            title_size=11.6, sub_size=8.6)

    # Dashed red arrow from predicted SELFIES into recon loss.
    # (Comparison against input x is described in the recon-loss subtitle
    # rather than drawn as a long diagonal arrow that would cut the canvas.)
    add_arrow(ax, 23.2, y_main - 0.9, 21.5, y_loss + 0.9,
              color=RED, dashed=True, lw=1.4)

    # Dashed red arrows from mu and sigma into KL loss
    add_arrow(ax, 10.2, y_main - 1.05, 10.2, y_loss + 0.9,
              color=RED, dashed=True, lw=1.4)

    # Recon and KL feed into total loss
    add_arrow(ax, 18.5, y_loss - 0.9, 16.6, y_loss - 1.6,
              color=RED, lw=1.6)
    add_arrow(ax, 11.9, y_loss - 0.9, 14.6, y_loss - 1.6,
              color=RED, lw=1.6)

    # Backprop arrow from total loss back up to encoder + decoder
    # (single curved-style: dashed red horizontal at very bottom returning up)
    ax.plot([15.6, 5.9], [y_loss - 2.4 - 1.05, y_loss - 2.4 - 1.05],
            color=RED, lw=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax.plot([15.6, 18.6], [y_loss - 2.4 - 1.05, y_loss - 2.4 - 1.05],
            color=RED, lw=1.2, linestyle=(0, (4, 3)), zorder=4)
    add_arrow(ax, 5.9,  y_loss - 2.4 - 1.05, 5.9,  y_main - 1.15,
              color=RED, dashed=True, lw=1.2)
    add_arrow(ax, 18.6, y_loss - 2.4 - 1.05, 18.6, y_main - 1.15,
              color=RED, dashed=True, lw=1.2)
    add_label(ax, 12.2, y_loss - 2.4 - 1.45,
              r"backprop  $\nabla_\theta\mathcal{L}$  (encoder + decoder updated)",
              color=RED, size=9.0, italic=True, bold=True, ha="center")

    # ─────────────────────────────────────────────────────────────────────
    # PROPERTY-AGNOSTIC CALLOUT (upper-right; bold, prominent)
    # ─────────────────────────────────────────────────────────────────────
    callout_x_c = 22.3
    callout_y_c = 13.1
    callout_w   = 8.6
    callout_h   = 1.9
    sh = FancyBboxPatch(
        (callout_x_c - callout_w / 2 + 0.04,
         callout_y_c - callout_h / 2 - 0.04),
        callout_w, callout_h,
        boxstyle="round,pad=0.04,rounding_size=0.20",
        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
    )
    ax.add_patch(sh)
    callout = FancyBboxPatch(
        (callout_x_c - callout_w / 2, callout_y_c - callout_h / 2),
        callout_w, callout_h,
        boxstyle="round,pad=0.04,rounding_size=0.20",
        linewidth=2.2, facecolor=PALE_GOLD, edgecolor=GOLD, zorder=2,
    )
    ax.add_patch(callout)
    ax.text(callout_x_c, callout_y_c + 0.32,
            "LIMO is property-agnostic",
            ha="center", va="center", fontsize=14.0, fontweight="bold",
            color=TEXT_NAVY, family="serif", zorder=3)
    ax.text(callout_x_c, callout_y_c - 0.34,
            "no property labels enter this loss",
            ha="center", va="center", fontsize=10.2, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    # ─────────────────────────────────────────────────────────────────────
    # Free-floating annotations (no boxes)
    # ─────────────────────────────────────────────────────────────────────
    # Corpus annotation, above the input box
    ax.text(1.9, y_main + 1.85,
            "326k energetic corpus\nSELFIES tokens",
            ha="center", va="center", fontsize=8.6, color=TEXT_SLATE,
            family="serif", fontstyle="italic", linespacing=1.15, zorder=3)

    # Validation accuracy + frozen-after, sitting to the right of decoder
    # (above the predicted SELFIES box)
    ax.text(23.2, y_main + 1.85,
            "validation token-acc 64.5%\n* frozen after fine-tune *",
            ha="center", va="center", fontsize=8.6, color=TEXT_SLATE,
            family="serif", fontstyle="italic", linespacing=1.15, zorder=3)

    # latent-posterior caveat under mu/sigma (placed BELOW the loss block
    # so it does not collide with the posterior label).
    ax.text(15.6, y_loss - 2.4 - 2.1,
            r"latent posterior concentrated:  $\Vert\mu\Vert \approx 8$  "
            r"(see Section 6 limitations)",
            ha="center", va="center", fontsize=8.4, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

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

    # Title intentionally omitted; the HTML figcaption carries the label.

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

    # Title intentionally omitted; the HTML figcaption carries the label.

    # Heads (top row): viab, sens, hazard, perf — all unified under a single
    # enclosing "4-head classifier" frame.
    head_y = 7.6
    head_specs = [
        (4.0,  "Viability",   "decoded validity"),
        (8.4,  "Sensitivity", r"$h_{50}$ Huang-Massa"),
        (12.8, "Hazard",      "SMARTS-aware"),
        (17.2, "Performance", r"$\rho,\ D,\ P$ regressors"),
    ]
    # Enclosing frame
    frame_x_l = head_specs[0][0]  - 1.95
    frame_x_r = head_specs[-1][0] + 1.95
    frame_w   = frame_x_r - frame_x_l
    frame_h   = 2.6
    frame_y_c = head_y + 0.05
    frame_x_c = (frame_x_l + frame_x_r) / 2
    sh = FancyBboxPatch(
        (frame_x_l + 0.05, frame_y_c - frame_h / 2 - 0.05),
        frame_w, frame_h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=0, facecolor="#0a1620", alpha=0.08, zorder=1,
    )
    ax.add_patch(sh)
    frame = FancyBboxPatch(
        (frame_x_l, frame_y_c - frame_h / 2),
        frame_w, frame_h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=1.6, facecolor="#f4faf5", edgecolor=GREEN, zorder=2,
    )
    ax.add_patch(frame)
    # Frame caption sits inside the frame, above the head-box row
    ax.text(frame_x_c, frame_y_c + frame_h / 2 - 0.32,
            "4-head classifier (multi-head guidance)",
            ha="center", va="center", fontsize=10.0, fontweight="bold",
            color=GREEN, family="serif", zorder=3)
    # Heads, slightly smaller and lowered to fit inside the frame caption
    for x_c, t, sub in head_specs:
        add_box(ax, x_c, 3.4, 1.4, head_y - 0.20, PALE_GREEN, GREEN, t, sub,
                title_size=10.2, sub_size=8.4)

    # Trajectory row
    traj_y = 3.0
    add_box(ax, 2.2, 2.8, 1.7, traj_y, CREAM, NAVY,
            r"$z_T$", r"$\mathcal{N}(0,\ I_{1024})$")
    # DDIM box: title + per-step formula + protocol annotation.
    ddim_x_c = 10.6
    ddim_w   = 8.0
    ddim_h   = 2.5
    x = ddim_x_c - ddim_w / 2
    y = traj_y - ddim_h / 2
    sh = FancyBboxPatch(
        (x + 0.04, y - 0.04), ddim_w, ddim_h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
    )
    ax.add_patch(sh)
    box = FancyBboxPatch(
        (x, y), ddim_w, ddim_h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.5, facecolor=PALE_GREY, edgecolor=NAVY, zorder=2,
    )
    ax.add_patch(box)
    ax.text(ddim_x_c, traj_y + ddim_h * 0.30, "DDIM denoise step",
            ha="center", va="center", fontsize=11.0, fontweight=600,
            color=TEXT_NAVY, family="serif", zorder=3)
    ax.text(ddim_x_c, traj_y + 0.08,
            r"$z_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(z_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,(\hat\varepsilon_\theta - g_t)\right) + \sigma_t\,\eta$",
            ha="center", va="center", fontsize=9.6, color=TEXT_NAVY,
            family="serif", zorder=3)
    ax.text(ddim_x_c, traj_y - ddim_h * 0.32,
            r"$t = T \to 1$,  40 steps,  $\eta = 0.1$",
            ha="center", va="center", fontsize=8.8, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)
    add_box(ax, 19.4, 2.8, 1.7, traj_y, PALE_GOLD, GOLD,
            r"$z_0$", "denoised latent")

    add_arrow(ax, 3.6, traj_y, 6.5, traj_y)
    add_arrow(ax, 14.7, traj_y, 18.0, traj_y)

    # Guidance bus: each head drops to a horizontal bus, bus drops as a
    # single bundled gold-dashed arrow into DDIM top.
    bus_y = 5.9
    ddim_top = traj_y + ddim_h / 2
    bus_x_left  = head_specs[0][0]
    bus_x_right = head_specs[-1][0]
    for x_c, *_ in head_specs:
        ax.plot([x_c, x_c], [head_y - 0.78, bus_y + 0.02],
                color=GOLD, lw=1.3, linestyle=(0, (4, 3)), zorder=4)
        ax.plot([x_c], [bus_y], marker="o", markersize=3.6,
                markerfacecolor=GOLD, markeredgecolor=GOLD, zorder=5)
    ax.plot([bus_x_left, bus_x_right], [bus_y, bus_y],
            color=GOLD, lw=2.0, zorder=4)
    bundle_x = ddim_x_c
    # Sigma glyph at convergence point on the bus
    ax.plot([bundle_x], [bus_y], marker="o", markersize=11,
            markerfacecolor="white", markeredgecolor=GOLD,
            markeredgewidth=1.8, zorder=5)
    ax.text(bundle_x, bus_y, r"$\Sigma$", ha="center", va="center",
            fontsize=10.5, color=GOLD, fontweight="bold",
            family="serif", zorder=6)
    add_arrow(ax, bundle_x, bus_y - 0.20, bundle_x, ddim_top + 0.05,
              color=GOLD, dashed=True, lw=2.0)
    # Gradient label sits to the right of the vertical bundled arrow, in the
    # gap between bus and DDIM box top.
    add_label(ax, bundle_x + 0.35,
              (bus_y + ddim_top) / 2 - 0.05,
              r"$g_t = \sum_h s_h\,\nabla_{z_t}\log p_h(c\!\mid\!z_t)$",
              color=GOLD, size=9.0, italic=True, bold=True)

    # Color legend intentionally omitted per user spec.

    # Bottom text labels intentionally omitted; the HTML figcaption carries
    # the protocol detail (n=40 steps, eta, conditioning vector).

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
    # Single-panel pipeline (histogram intentionally dropped per user spec).
    # Canvas height tightened: the figure-coords subtitle was dropped because
    # the banner above the boxes already frames the funnel, and two competing
    # top labels read as crowded. The HTML figcaption carries the full caption.
    fig, ax = plt.subplots(figsize=(15.0, 4.4), dpi=300)
    setup_axes(ax, xmax=36.0, ymax=7.6)

    y = 3.4

    # Six boxes with PROGRESSIVELY DECREASING HEIGHTS to mirror the
    # candidate-funnel narrowing (40k -> 30k -> 12k -> 8k -> 100). Widths are
    # uniform so the row reads as a clean horizontal pipeline; box-height
    # encodes the survivor count.
    # Six box centres on a 36-wide canvas, with ~6.0 unit centre-to-centre
    # spacing and 4.4-unit-wide boxes -> 1.6-unit gap between adjacent boxes
    # leaves clear room for arrow + count annotation.
    centres = [3.2, 9.4, 15.6, 21.8, 28.0, 33.6]
    widths  = [4.4, 4.4, 4.4, 4.4, 4.4, 4.0]
    # Heights drop from 3.0 to 1.4 in proportion to log10(count) of survivors.
    heights = [3.00, 2.85, 2.45, 2.10, 1.75, 1.40]

    # Role-coded box fills:
    #   PALE_GOLD  = data state (latent / SMILES / final leads)
    #   PALE_GREY  = transform (decoder)
    #   PALE_RED   = filter / gate
    #   PALE_GREEN = ranker / score
    box_specs = [
        (PALE_GOLD,  GOLD, r"$z_0$",
         "denoised latent (1024-d)"),
        (PALE_GREY,  NAVY, "LIMO decoder",
         r"frozen NAR (latent $\to$ SMILES)"),
        (PALE_GOLD,  GOLD, "SMILES",
         "RDKit canonical + valence"),
        (PALE_RED,   RED,  "SMARTS gate",
         r"Stage 1: rules + red-flags"),
        (PALE_GREEN, GREEN, "Pareto reranker",
         r"Stage 2: composite score"),
        (PALE_GOLD,  GOLD, r"top-$K$ leads",
         r"to xTB + DFT audit"),
    ]
    for cx, w, h, (fill, edge, t, sub) in zip(centres, widths, heights,
                                              box_specs):
        add_box(ax, cx, w, h, y, fill, edge, t, sub,
                title_size=10.8, sub_size=8.2)

    # Connectors: arrow tail/head shrunk to each box's actual half-width.
    counts = ["40,000\nsamples",
              "30,000\nparse-valid",
              "12,000\nSMILES",
              "8,000\nchem-pass",
              "100\nleads"]
    # Uniform y for all count annotations: clears the tallest box (heights[0])
    # so every label sits at one common baseline above the pipeline.
    count_y = y + heights[0] / 2 + 0.55
    for i in range(len(centres) - 1):
        x0 = centres[i]     + widths[i]     / 2
        x1 = centres[i + 1] - widths[i + 1] / 2
        add_arrow(ax, x0, y, x1, y, lw=1.6)
        ax.text((x0 + x1) / 2, count_y, counts[i],
                ha="center", va="center",
                fontsize=8.2, color=TEXT_SLATE, family="serif",
                fontstyle="italic", linespacing=1.05, zorder=3)

    # Banner: explicit funnel framing so the reader knows what the counts mean.
    add_label(ax, 0.3, y + heights[0] / 2 + 1.30,
              "candidates surviving each stage  (illustrative scale)",
              color=TEXT_SLATE, size=8.8, italic=True, bold=True)

    # Color legend intentionally omitted per user spec.

    base = os.path.join(OUT_DIR, "fig4d_decode_rerank")
    save(fig, base)


# ──────────────────────────────────────────────────────────────────────────
# 4e  Multi-head classifier training: four independent models
# ──────────────────────────────────────────────────────────────────────────
def fig4e_head_training():
    """Four heterogeneous heads, each with its own architecture, inputs,
    training labels, and validation metric. Bottom row shows them feeding
    into the sample-time guidance bus referenced by 4(c)."""
    fig, ax = plt.subplots(figsize=(15.0, 7.6), dpi=300)
    setup_axes(ax, xmax=36.0, ymax=18.0)

    # Four columns, one per head. Each column is a vertical stack:
    #   [arch badge]  [head title]  [input box]  [training-data box]
    #   [validation metric]  -> bus
    centres   = [4.8, 13.2, 21.6, 30.0]
    head_names = ["Viability", "Sensitivity", "Hazard", "Performance"]
    arch_badges = [
        "Random Forest",
        "MLP score model",
        "MLP head",
        "3D-CNN ensemble",
    ]
    # Each head uses a different border-color tint to communicate heterogeneity
    edge_cols  = [GREEN,      GOLD,       RED,       PURPLE]
    fill_cols  = [PALE_GREEN, PALE_GOLD,  PALE_RED,  PALE_PURPLE]

    # Per-head input descriptors (top input box)
    inputs = [
        ("input", "Morgan FP-2-2048\n+ RDKit descriptors"),
        ("input", "cached LIMO latent  $z$"),
        ("input", "cached LIMO latent  $z$"),
        ("input", "3D conformer voxel grid"),
    ]
    # Per-head training-data descriptors (training-labels box)
    labels = [
        ("labels",
         "66k energetic (+) vs\n80k ZINC drug-like ($-$)"),
        ("labels",
         r"$h_{50} = 1.93\!\cdot\!\mathrm{BDE} - 52.4$"
         "\nPolitzer-Murray fit"),
        ("labels",
         "SMARTS hazard catalog\n+ Bruns-Watson demerits"),
        ("labels",
         r"$\rho, D, P, T, E, V, $HOF, BDE"
         "\n5-fold CV"),
    ]
    # Per-head outputs (small italic sub-line under head title)
    outputs = [
        r"output:  $P(\mathrm{energetic})$",
        r"output:  predicted $h_{50}$ (cm)",
        r"output:  $P(\mathrm{hazardous\ motif})$",
        r"output:  8 properties",
    ]
    # Validation metrics (bold, prominent)
    metrics = [
        "val AUC = 0.9986",
        r"val $R^2 \approx 0.78$",
        r"val AUC $\approx 0.93$",
        r"val $R^2 = 0.84{-}0.92$",
    ]

    # Layout y-coordinates
    y_title  = 16.4   # head name
    y_arch   = 15.4   # arch badge (architecture pill)
    y_output = 14.5   # output sub-line (italic)
    y_input  = 12.4   # input box centre
    y_labels = 9.2    # training-labels box centre
    y_metric = 6.7    # validation metric line
    y_bus    = 4.2    # guidance bus line
    y_foot   = 2.0    # 4(c) reference box centre

    box_w_input  = 6.4
    box_h_input  = 1.7
    box_w_lab    = 6.4
    box_h_lab    = 2.0

    # Per-column rendering
    for i, cx in enumerate(centres):
        edge = edge_cols[i]
        fill = fill_cols[i]

        # Head title (bold, large)
        ax.text(cx, y_title, head_names[i], ha="center", va="center",
                fontsize=13.2, fontweight="bold", color=TEXT_NAVY,
                family="serif", zorder=3)
        # Output sub-line BELOW the badge (clear of overlap)
        ax.text(cx, y_output, outputs[i], ha="center", va="center",
                fontsize=8.8, fontstyle="italic", color=TEXT_SLATE,
                family="serif", zorder=3)

        # Architecture badge: pill-shaped box with edge-color border
        badge_w = 4.6
        badge_h = 0.78
        bx = cx - badge_w / 2
        by = y_arch - badge_h / 2
        ax.add_patch(FancyBboxPatch(
            (bx + 0.04, by - 0.04), badge_w, badge_h,
            boxstyle="round,pad=0.02,rounding_size=0.30",
            linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (bx, by), badge_w, badge_h,
            boxstyle="round,pad=0.02,rounding_size=0.30",
            linewidth=1.6, facecolor="white", edgecolor=edge, zorder=2,
        ))
        ax.text(cx, y_arch, arch_badges[i], ha="center", va="center",
                fontsize=9.6, fontweight="bold", color=edge,
                family="serif", zorder=3)

        # Input box (caption "input" placed to the LEFT of the box at column 0
        # for the leftmost head; for clarity inside each box the input text
        # is self-describing, so per-column caption omitted).
        in_label, in_text = inputs[i]
        # box: white fill, head-tinted border
        x = cx - box_w_input / 2
        yb = y_input - box_h_input / 2
        ax.add_patch(FancyBboxPatch(
            (x + 0.04, yb - 0.04), box_w_input, box_h_input,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (x, yb), box_w_input, box_h_input,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.4, facecolor=CREAM, edgecolor=edge, zorder=2,
        ))
        ax.text(cx, y_input, in_text, ha="center", va="center",
                fontsize=9.4, color=TEXT_NAVY, family="serif",
                linespacing=1.18, zorder=3)

        # Training-labels box (caption omitted; left-edge row label provides
        # the row identity).
        lab_label, lab_text = labels[i]
        x = cx - box_w_lab / 2
        yb = y_labels - box_h_lab / 2
        ax.add_patch(FancyBboxPatch(
            (x + 0.04, yb - 0.04), box_w_lab, box_h_lab,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (x, yb), box_w_lab, box_h_lab,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.4, facecolor=fill, edgecolor=edge, zorder=2,
        ))
        ax.text(cx, y_labels, lab_text, ha="center", va="center",
                fontsize=9.4, color=TEXT_NAVY, family="serif",
                linespacing=1.18, zorder=3)

        # Arrow: input -> labels (training). Italic "trains on" label sits to
        # the right of the arrow.
        add_arrow(ax,
                  cx, y_input - box_h_input / 2 - 0.05,
                  cx, y_labels + box_h_lab / 2 + 0.05,
                  color=edge, lw=1.6)
        ax.text(cx + 0.30,
                (y_input - box_h_input / 2 + y_labels + box_h_lab / 2) / 2,
                "trains on", ha="left", va="center",
                fontsize=8.0, fontstyle="italic", color=edge,
                family="serif", zorder=3)

        # Validation metric (bold, head-tinted)
        ax.text(cx, y_metric, metrics[i], ha="center", va="center",
                fontsize=10.2, fontweight="bold", color=edge,
                family="serif", zorder=3)

        # Drop from metric down to the guidance bus
        ax.plot([cx, cx], [y_metric - 0.40, y_bus + 0.02],
                color=GOLD, lw=1.4, linestyle=(0, (4, 3)), zorder=4)
        ax.plot([cx], [y_bus], marker="o", markersize=4.2,
                markerfacecolor=GOLD, markeredgecolor=GOLD, zorder=5)

    # Guidance bus: horizontal gold line spanning all four heads
    bus_x_l = centres[0]
    bus_x_r = centres[-1]
    ax.plot([bus_x_l, bus_x_r], [y_bus, y_bus],
            color=GOLD, lw=2.0, zorder=4)

    # Bus label (left side, above the bus line so it does not collide with markers)
    add_label(ax, bus_x_l - 3.8, y_bus,
              "guidance bus", color=GOLD, size=9.6, italic=True, bold=True)

    # 4(c) footer reference: a long pale box with arrow from bus midpoint
    foot_cx = (bus_x_l + bus_x_r) / 2
    foot_w  = 18.0
    foot_h  = 1.3
    fx = foot_cx - foot_w / 2
    fy = y_foot - foot_h / 2
    ax.add_patch(FancyBboxPatch(
        (fx + 0.04, fy - 0.04), foot_w, foot_h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (fx, fy), foot_w, foot_h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.6, facecolor=PALE_GOLD, edgecolor=GOLD, zorder=2,
    ))
    ax.text(foot_cx, y_foot,
            r"used at sample time as classifier guidance "
            r"$g_t = \sum_h s_h\,\nabla_{z_t}\log p_h(c\!\mid\!z_t)$  "
            r"[see Fig. 4(c)]",
            ha="center", va="center", fontsize=9.6, color=TEXT_NAVY,
            family="serif", zorder=3)

    # Arrow from bus into footer
    add_arrow(ax, foot_cx, y_bus - 0.10, foot_cx, y_foot + foot_h / 2 + 0.05,
              color=GOLD, dashed=True, lw=1.8)

    # Top-row banner: four-models heterogeneity callout
    add_label(ax, 0.3, 17.5,
              "four independent models, each with its own architecture, "
              "inputs, and training labels",
              color=TEXT_SLATE, size=9.4, italic=True, bold=True)

    base = os.path.join(OUT_DIR, "fig4e_head_training")
    save(fig, base)


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig4a_data_prep()
    fig4b_train_loop()
    fig4c_sampling_guidance()
    fig4d_decode_rerank()
    fig4e_head_training()
