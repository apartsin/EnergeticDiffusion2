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
    """Per-step denoiser update for one cached row.

    Story: the trust mask m, derived from the four-tier label hierarchy,
    gates the conditional gradient. Tier-A/B rows feed real (p, m=1) into
    epsilon_theta and contribute to the masked MSE; Tier-C/D rows have
    m=0 on the missing properties so the conditional gradient is zero.
    """
    fig, ax = plt.subplots(figsize=(14.0, 7.4), dpi=300)
    setup_axes(ax, xmax=28.0, ymax=14.0)

    # ── Layout anchors ────────────────────────────────────────────────
    mid_y = 8.0    # main flow row (forward diff, eps_theta, loss)
    samp_y = 11.5  # t / eps samplers row
    upd_y  = 3.4   # theta-update return row

    BW_MAIN = 3.6
    BH_MAIN = 1.8

    # Column centres
    cx_cache = 3.2      # cached row + tier panel
    cx_fwd   = 10.0     # forward diffusion
    cx_eps   = 15.2     # noise predictor eps_theta
    cx_loss  = 20.4     # masked MSE
    cx_ema   = 25.4     # EMA

    HM = BW_MAIN / 2
    SH = 0.10

    # ── LEFT: tier hierarchy panel (the source of m) ──────────────────
    # Outer container framing the tier story.
    panel_x = cx_cache
    panel_w = 6.0
    panel_y = 7.6
    panel_h = 9.0
    px = panel_x - panel_w / 2
    py = panel_y - panel_h / 2
    container = FancyBboxPatch(
        (px, py), panel_w, panel_h,
        boxstyle="round,pad=0.04,rounding_size=0.20",
        linewidth=1.0, facecolor=CREAM, edgecolor=TEXT_LIGHT,
        zorder=1,
    )
    ax.add_patch(container)
    ax.text(panel_x, py + panel_h - 0.45,
            "four-tier label hierarchy",
            ha="center", va="center", fontsize=10.5, fontweight=600,
            color=TEXT_NAVY, family="serif", zorder=3)
    ax.text(panel_x, py + panel_h - 0.95,
            r"trust mask $m \in \{0,1\}^4$ per row",
            ha="center", va="center", fontsize=9.0, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    # FOUR ILLUSTRATIVE PER-PROPERTY MASK STATES.
    # The mask is set per-property, not per-row: a row's m-vector reflects
    # the tier of EACH property independently. We show the spectrum:
    # full-trust (all-1s); typical "exp rho only" partial (m=(1,0,0,0));
    # DFT-trust on the energetic-pair (1,1,0,0); fully-untrusted (all-0s).
    tier_rows = [
        ("All A/B",   "exp/DFT all 4",         PALE_GOLD, GOLD,       "m = (1, 1, 1, 1)", True),
        ("rho only",  r"exp $\rho$ + K-J D,P", PALE_GOLD, GOLD,       "m = (1, 0, 0, 0)", True),
        ("rho/HOF",   r"DFT $\rho$/HOF + K-J", PALE_GOLD, GOLD,       "m = (1, 1, 0, 0)", True),
        ("All C/D",   "K-J / 3D-CNN only",     PALE_GREY, TEXT_LIGHT, "m = (0, 0, 0, 0)", False),
    ]
    tier_top = py + panel_h - 1.60
    tier_h = 1.20
    tier_gap = 0.14
    tier_w = 5.2
    for i, (name, src, fill, edge, mtxt, trusted) in enumerate(tier_rows):
        ty = tier_top - i * (tier_h + tier_gap) - tier_h / 2
        tx = panel_x - tier_w / 2
        sh = FancyBboxPatch(
            (tx + 0.04, ty - tier_h / 2 - 0.04), tier_w, tier_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0, facecolor="#0a1620", alpha=0.08, zorder=1,
        )
        ax.add_patch(sh)
        box = FancyBboxPatch(
            (tx, ty - tier_h / 2), tier_w, tier_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.2, facecolor=fill, edgecolor=edge, zorder=2,
        )
        ax.add_patch(box)
        # Top line: tier name (left) | trust-mask vector (right).
        ax.text(tx + 0.30, ty + 0.28, name,
                ha="left", va="center", fontsize=10.2, fontweight=600,
                color=TEXT_NAVY, family="serif", zorder=3)
        ax.text(tx + tier_w - 0.25, ty + 0.28, mtxt,
                ha="right", va="center", fontsize=9.8,
                color=(GOLD if trusted else TEXT_SLATE),
                family="serif", zorder=3,
                fontweight=("bold" if trusted else "normal"))
        # Bottom line: source label, full-width italic.
        ax.text(tx + 0.30, ty - 0.30, src,
                ha="left", va="center", fontsize=8.6, fontstyle="italic",
                color=TEXT_SLATE, family="serif", zorder=3)

    # Caption under tier rows: arrow from the tier panel feeds m into
    # the cached-row tuple, then into the main flow.
    cache_label_y = py + 0.55
    ax.text(panel_x, cache_label_y + 0.30,
            r"cached row: $(z_0,\; p,\; m)$",
            ha="center", va="center", fontsize=10.5, fontweight=600,
            color=TEXT_NAVY, family="serif", zorder=3)
    ax.text(panel_x, cache_label_y - 0.20,
            r"$p = (\rho,\; \mathrm{HOF},\; D,\; P)$",
            ha="center", va="center", fontsize=9.0, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    # ── MIDDLE: samplers, forward diffusion, noise predictor ──────────
    # Stochastic samplers (purple)
    add_box(ax, cx_fwd - 1.20, 1.6, 1.5, samp_y, PALE_PURPLE, PURPLE,
            r"$t$", r"$\mathcal{U}\{1{:}T\}$",
            title_size=11.0, sub_size=9.0)
    add_box(ax, cx_fwd + 1.20, 1.6, 1.5, samp_y, PALE_PURPLE, PURPLE,
            r"$\varepsilon$", r"$\mathcal{N}(0, I)$",
            title_size=11.0, sub_size=9.0)

    # Forward diffusion
    add_box(ax, cx_fwd, BW_MAIN, BH_MAIN, mid_y, PALE_GREY, NAVY,
            "forward diffusion",
            r"$z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$",
            title_size=11.0, sub_size=9.0)

    # Noise predictor (NAVY, taller to make it the visual centrepiece)
    EPS_W = 3.8
    EPS_H = 2.2
    add_box(ax, cx_eps, EPS_W, EPS_H, mid_y, PALE_GREY, NAVY,
            r"noise predictor  $\varepsilon_\theta$",
            r"inputs: $(z_t,\; t,\; p,\; m)$",
            title_size=11.5, sub_size=9.5)

    # Loss (RED) — plain full-latent-dim MSE; m gates FiLM, not the loss.
    # Optional row-weight omega from tier weights w (see Fig 4(g)).
    add_box(ax, cx_loss, BW_MAIN, BH_MAIN, mid_y, PALE_RED, RED,
            "MSE loss",
            r"$\mathcal{L} = \frac{1}{1024}\,\Vert\varepsilon - \hat\varepsilon\Vert^2$",
            title_size=11.0, sub_size=10.0)

    # EMA (purple)
    add_box(ax, cx_ema, 2.8, BH_MAIN, mid_y, PALE_PURPLE, PURPLE,
            "EMA",
            r"$\bar\theta \leftarrow 0.999\,\bar\theta + 0.001\,\theta$",
            title_size=11.0, sub_size=9.0)

    # ── Arrows: tier panel -> forward diffusion (z_0) ─────────────────
    # z_0 from the cached-row caption into forward diffusion
    add_arrow(ax,
              panel_x + panel_w / 2, cache_label_y,
              cx_fwd - HM - SH, mid_y - 0.40,
              lw=1.8)
    add_label(ax,
              (panel_x + panel_w / 2 + cx_fwd) / 2 + 0.2,
              (cache_label_y + mid_y) / 2 + 0.55,
              r"$z_0$", color=TEXT_NAVY, size=10.5,
              italic=True, ha="center")

    # samplers -> forward diffusion (purple dashed)
    add_arrow(ax, cx_fwd - 1.20, samp_y - 0.78,
              cx_fwd - 0.6, mid_y + BH_MAIN / 2 + 0.05,
              color=PURPLE, dashed=True, lw=1.2)
    add_arrow(ax, cx_fwd + 1.20, samp_y - 0.78,
              cx_fwd + 0.0, mid_y + BH_MAIN / 2 + 0.05,
              color=PURPLE, dashed=True, lw=1.2)

    # forward diffusion -> noise predictor (z_t)
    add_arrow(ax, cx_fwd + HM + SH, mid_y, cx_eps - EPS_W / 2 - SH, mid_y,
              lw=1.8)
    add_label(ax, (cx_fwd + cx_eps) / 2, mid_y + 0.45,
              r"$z_t$", color=TEXT_NAVY, size=10.5,
              italic=True, ha="center")

    # ── KEY VISUAL: m gates the (p) input into eps_theta ──────────────
    # Conditioning route: (p, m) leave the cached-row caption, run along
    # the bottom, then up into eps_theta. Gold dashed = conditioning.
    cond_y = mid_y - 3.2  # below main row
    pm_start_x = panel_x + panel_w / 2
    pm_start_y = cache_label_y - 0.55
    # Down from cached caption
    ax.plot([pm_start_x, pm_start_x], [pm_start_y, cond_y],
            color=GOLD, lw=1.6, linestyle=(0, (5, 3)), zorder=4)
    # Across to under eps_theta
    ax.plot([pm_start_x, cx_eps], [cond_y, cond_y],
            color=GOLD, lw=1.6, linestyle=(0, (5, 3)), zorder=4)
    # Up into eps_theta
    add_arrow(ax, cx_eps, cond_y, cx_eps, mid_y - EPS_H / 2 - 0.05,
              color=GOLD, dashed=True, lw=1.6)
    # Label on the horizontal: "p gated by m"
    add_label(ax, (pm_start_x + cx_eps) / 2, cond_y + 0.40,
              r"$p$  gated by trust mask  $m$",
              color=GOLD, size=10.0, italic=True, ha="center")
    # Small inline reminder: m dashed-gold tag near eps_theta
    add_label(ax, cx_eps + 0.05, cond_y + 0.40,
              "", color=GOLD)

    # eps_theta -> loss (hat eps)
    add_arrow(ax, cx_eps + EPS_W / 2 + SH, mid_y,
              cx_loss - HM - SH, mid_y, lw=1.8)
    add_label(ax, (cx_eps + cx_loss) / 2, mid_y + 0.45,
              r"$\hat\varepsilon$", color=TEXT_NAVY, size=10.5,
              italic=True, ha="center")

    # ── true epsilon skip from sampler to loss (purple dashed) ────────
    eps_x = cx_fwd + 1.20
    skip_y = samp_y + 1.10
    ax.plot([eps_x, eps_x], [samp_y + 0.78, skip_y],
            color=PURPLE, lw=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax.plot([eps_x, cx_loss], [skip_y, skip_y],
            color=PURPLE, lw=1.2, linestyle=(0, (4, 3)), zorder=4)
    add_arrow(ax, cx_loss, skip_y, cx_loss, mid_y + BH_MAIN / 2 + 0.05,
              color=PURPLE, dashed=True, lw=1.2)
    add_label(ax, (eps_x + cx_loss) / 2, skip_y + 0.32,
              r"true $\varepsilon$ (target)",
              color=PURPLE, size=9.5, italic=True, ha="center")

    # NOTE: m does NOT enter the loss; it only gates the property vector p
    # at the FiLM input of eps_theta (gold dashed route above). The loss is
    # plain full-latent-dim MSE between epsilon and hat-epsilon. Optional
    # row-weight omega = alpha + (1-alpha)*mean(w*m) is described in caption.

    # ── Loss -> EMA (red gradient) ────────────────────────────────────
    add_arrow(ax, cx_loss + HM + SH, mid_y, cx_ema - 1.40 - SH, mid_y,
              color=RED, lw=1.8)
    add_label(ax, (cx_loss + cx_ema) / 2, mid_y + 0.45,
              r"$\nabla_\theta\mathcal{L}$", color=RED, size=10,
              italic=True, ha="center")

    # ── theta-update feedback loop: EMA -> eps_theta ──────────────────
    ax.plot([cx_ema, cx_ema], [mid_y - BH_MAIN / 2 - 0.05, upd_y],
            color=PURPLE, lw=1.4, linestyle=(0, (4, 3)), zorder=4)
    ax.plot([cx_ema, cx_eps], [upd_y, upd_y],
            color=PURPLE, lw=1.4, linestyle=(0, (4, 3)), zorder=4)
    add_arrow(ax, cx_eps, upd_y, cx_eps, mid_y - EPS_H / 2 - 0.05,
              color=PURPLE, dashed=True, lw=1.4)
    add_label(ax, (cx_eps + cx_ema) / 2, upd_y - 0.40,
              r"update $\theta$",
              color=PURPLE, size=9.5, italic=True, ha="center")

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
    # Full inference pipeline: decode + ALL FOUR STAGES of the credibility funnel.
    # Stage 1 SMARTS gate (rules), Stage 2 Pareto reranker, Stage 3 semi-empirical
    # triage (xTB), Stage 4 first-principles audit (DFT).
    fig, ax = plt.subplots(figsize=(18.5, 4.8), dpi=300)
    setup_axes(ax, xmax=46.0, ymax=8.2)

    y = 3.6

    # Eight boxes spanning the canvas, with PROGRESSIVELY DECREASING HEIGHTS
    # to mirror the candidate-funnel narrowing (40k -> ... -> ~12 leads).
    # Box-height encodes survivor count on a log10 scale.
    centres = [2.8, 8.4, 14.0, 19.6, 25.2, 30.8, 36.4, 42.6]
    widths  = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.8]
    heights = [3.20, 3.05, 2.70, 2.40, 2.05, 1.75, 1.45, 1.20]

    # Role-coded box fills:
    #   PALE_GOLD  = data state (latent / SMILES / final leads)
    #   PALE_GREY  = transform (decoder)
    #   PALE_RED   = filter / gate              (Stage 1 SMARTS gate)
    #   PALE_GREEN = ranker / score             (Stage 2 Pareto reranker)
    #   PALE_PURPLE = semi-empirical physics    (Stage 3 xTB)
    #   PALE_PURPLE deeper edge = ab-initio     (Stage 4 DFT)
    box_specs = [
        (PALE_GOLD,   GOLD,   r"$z_0$",
         "denoised latent (1024-d)"),
        (PALE_GREY,   NAVY,   "LIMO decoder",
         r"frozen NAR (latent $\to$ SMILES)"),
        (PALE_GOLD,   GOLD,   "SMILES",
         "RDKit canonical + valence"),
        (PALE_RED,    RED,    "SMARTS gate",
         r"Stage 1: rules + red-flags"),
        (PALE_GREEN,  GREEN,  "Pareto reranker",
         r"Stage 2: composite score"),
        (PALE_PURPLE, PURPLE, "xTB triage",
         r"Stage 3: GFN2 + gap $\geq$ 1.5 eV"),
        (PALE_PURPLE, NAVY,   "DFT audit",
         r"Stage 4: B3LYP + $\omega$B97X-D"),
        (PALE_GOLD,   GOLD,   "validated leads",
         r"6-anchor cal. (Tab D.2)"),
    ]
    for cx, w, h, (fill, edge, t, sub) in zip(centres, widths, heights,
                                              box_specs):
        add_box(ax, cx, w, h, y, fill, edge, t, sub,
                title_size=10.4, sub_size=7.9)

    # Connectors: arrow tail/head shrunk to each box's actual half-width.
    counts = ["40,000\nsamples",
              "30,000\nparse-valid",
              "12,000\nSMILES",
              "8,000\nchem-pass",
              "100\nleads",
              "85\nxTB-pass",
              "12\nDFT-validated"]
    # Uniform y for count labels above the tallest box.
    count_y = y + heights[0] / 2 + 0.55
    for i in range(len(centres) - 1):
        x0 = centres[i]     + widths[i]     / 2
        x1 = centres[i + 1] - widths[i + 1] / 2
        add_arrow(ax, x0, y, x1, y, lw=1.6)
        ax.text((x0 + x1) / 2, count_y, counts[i],
                ha="center", va="center",
                fontsize=7.8, color=TEXT_SLATE, family="serif",
                fontstyle="italic", linespacing=1.05, zorder=3)

    # Banner: explicit funnel framing.
    add_label(ax, 0.3, y + heights[0] / 2 + 1.30,
              "candidates surviving each stage of the four-stage pipeline  (illustrative scale)",
              color=TEXT_SLATE, size=8.8, italic=True, bold=True)

    # Color legend intentionally omitted per user spec.

    base = os.path.join(OUT_DIR, "fig4d_decode_rerank")
    save(fig, base)


# ──────────────────────────────────────────────────────────────────────────
# 4e1  Data labeling: how per-row training labels are produced
# ──────────────────────────────────────────────────────────────────────────
def fig4e1_data_labeling():
    """Three-stage diagram: shared input column (SMILES + LIMO encoder + cached
    latent z) feeds four label-source models, which produce four per-row label
    vectors consumed by the score-model trainer in 4(e2)."""
    fig, ax = plt.subplots(figsize=(20.0, 9.6), dpi=300)
    setup_axes(ax, xmax=52.0, ymax=24.0)

    # ── Stage A (left): shared input column ──────────────────────────────
    col_x    = 4.2
    col_w    = 6.4
    smiles_y = 19.6
    enc_y    = 16.4
    latent_y = 13.2
    bypass_y = 10.6

    add_box(ax, col_x, col_w, 1.8, smiles_y, CREAM, NAVY,
            "326k energetic SMILES",
            "labelled corpus", title_size=11.0, sub_size=8.6)

    add_box(ax, col_x, col_w, 1.8, enc_y, PALE_GREY, NAVY,
            "LIMO encoder",
            "frozen, fine-tuned", title_size=11.0, sub_size=8.6)
    ax.text(col_x + col_w / 2 - 0.55, enc_y + 0.32, "*",
            ha="center", va="center", fontsize=16, color=NAVY,
            family="serif", zorder=4)

    add_box(ax, col_x, col_w, 1.8, latent_y, PALE_GOLD, GOLD,
            r"$z = \mu \in \mathbb{R}^{1024}$",
            "(cached)", title_size=11.0, sub_size=8.6)

    add_arrow(ax, col_x, smiles_y - 0.95, col_x, enc_y + 0.95, lw=1.8)
    add_arrow(ax, col_x, enc_y - 0.95,    col_x, latent_y + 0.95, lw=1.8)

    bypass_x = col_x + col_w / 2 + 0.6
    ax.plot([col_x + col_w / 2, bypass_x], [smiles_y, smiles_y],
            color=GOLD, lw=1.4, linestyle=(0, (5, 3)), zorder=4)
    ax.plot([bypass_x, bypass_x], [smiles_y, bypass_y],
            color=GOLD, lw=1.4, linestyle=(0, (5, 3)), zorder=4)
    add_label(ax, bypass_x + 0.25, (smiles_y + bypass_y) / 2,
              "SMILES (bypasses z)",
              color=GOLD, size=8.6, italic=True, bold=True)

    ax.text(col_x, latent_y - 1.55,
            r"$z$ is NOT consumed by label sources",
            ha="center", va="center", fontsize=8.8, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    # ── Stage B (middle): four label-source models ───────────────────────
    src_y_c   = 5.8
    src_w     = 6.4
    src_h     = 7.6
    src_centres = [14.0, 21.4, 28.8, 36.2]
    edge_cols = [GREEN, GOLD, RED, PURPLE]
    fill_cols = [PALE_GREEN, PALE_GOLD, PALE_RED, PALE_PURPLE]

    src_titles = [
        "Random Forest",
        r"Politzer-Murray BDE fit",
        "SMARTS + Bruns-Watson",
        "3D-CNN smoke ensemble",
    ]
    src_arch = [
        "arch: Random Forest",
        "arch: linear fit",
        "arch: SMARTS rules",
        "arch: Uni-Mol v1 ensemble",
    ]
    src_inputs = [
        "input: Morgan FP\n+ RDKit descriptors\nof SMILES",
        "input: chemotype class\nof weakest bond",
        "input: SMILES patterns",
        "input: 3D conformer\nvoxel grid",
    ]
    src_subset = [
        "training subset:\n66k energetic vs 80k ZINC",
        "training subset: 306 pairs\n(Huang & Massa)",
        "no training subset;\nrules-based",
        "training subset:\n9k Tier-A/B rows",
    ]
    src_eq = [
        r"$\to P(\mathrm{viable})$",
        r"$h_{50} = 1.93\!\cdot\!\mathrm{BDE} - 52.4$",
        r"pattern match $\to$ hazard",
        r"8 outputs ($\rho, D, P, T,$ ...)",
    ]
    src_metrics = [
        "AUC = 0.9986",
        r"Pearson $r = +0.71$",
        "(rules; no metric)",
        r"$R^2 = 0.84{-}0.92$",
    ]

    for i, cx in enumerate(src_centres):
        edge = edge_cols[i]
        fill = fill_cols[i]
        x  = cx - src_w / 2
        yb = src_y_c - src_h / 2
        ax.add_patch(FancyBboxPatch(
            (x + 0.04, yb - 0.04), src_w, src_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (x, yb), src_w, src_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.8, facecolor=fill, edgecolor=edge, zorder=2,
        ))
        ax.text(cx, src_y_c + src_h / 2 - 0.55, src_titles[i],
                ha="center", va="center", fontsize=10.0,
                fontweight="bold", color=edge, family="serif", zorder=3)
        ax.text(cx, src_y_c + src_h / 2 - 1.30, src_arch[i],
                ha="center", va="center", fontsize=8.2,
                fontstyle="italic", color=TEXT_NAVY, family="serif", zorder=3)
        ax.text(cx, src_y_c + 1.05, src_inputs[i],
                ha="center", va="center", fontsize=8.0,
                color=TEXT_NAVY, family="serif",
                linespacing=1.20, zorder=3)
        ax.text(cx, src_y_c - 0.55, src_eq[i],
                ha="center", va="center", fontsize=8.4,
                color=edge, family="serif", zorder=3)
        ax.text(cx, src_y_c - 1.85, src_subset[i],
                ha="center", va="center", fontsize=7.8,
                color=TEXT_SLATE, family="serif",
                linespacing=1.20, zorder=3)
        ax.text(cx, src_y_c - src_h / 2 + 0.45, src_metrics[i],
                ha="center", va="center", fontsize=8.4,
                fontstyle="italic", color=TEXT_SLATE, family="serif",
                zorder=3)

        ax.plot([bypass_x, cx], [bypass_y, bypass_y],
                color=GOLD, lw=1.2, linestyle=(0, (5, 3)), zorder=3)
        add_arrow(ax, cx, bypass_y, cx, src_y_c + src_h / 2 + 0.05,
                  color=GOLD, dashed=True, lw=1.4)

    # ── Stage C (right): per-row label vectors ───────────────────────────
    label_x  = 47.4
    label_w  = 5.6
    label_h  = 1.6
    label_specs = [
        (GREEN,  PALE_GREEN,  r"$y_{\mathrm{viab}} \in [0, 1]$"),
        (GOLD,   PALE_GOLD,   r"$y_{\mathrm{sens}} \in \mathbb{R}$"),
        (RED,    PALE_RED,    r"$y_{\mathrm{haz}} \in \{0, 1\}$"),
        (PURPLE, PALE_PURPLE, r"$y_{\mathrm{perf}} \in \mathbb{R}^4$"),
    ]
    label_y_centres = np.linspace(src_y_c + 2.7, src_y_c - 2.7, 4)

    for i, (edge, fill, txt) in enumerate(label_specs):
        ly = label_y_centres[i]
        x = label_x - label_w / 2
        yb = ly - label_h / 2
        ax.add_patch(FancyBboxPatch(
            (x + 0.04, yb - 0.04), label_w, label_h,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (x, yb), label_w, label_h,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=1.6, facecolor=fill, edgecolor=edge, zorder=2,
        ))
        ax.text(label_x, ly, txt, ha="center", va="center",
                fontsize=10.4, fontweight=600, color=TEXT_NAVY,
                family="serif", zorder=3)

        cx = src_centres[i]
        add_arrow(ax, cx + src_w / 2 + 0.05, src_y_c,
                  label_x - label_w / 2 - 0.05, ly,
                  color=edge_cols[i], lw=1.8)

    ax.text(label_x, label_y_centres[-1] - 1.7,
            r"$\to$ used in 4(e2) for",
            ha="center", va="center", fontsize=9.4, fontweight="bold",
            color=TEXT_NAVY, family="serif", zorder=3)
    ax.text(label_x, label_y_centres[-1] - 2.3,
            r"score-model head training",
            ha="center", va="center", fontsize=9.4, fontweight="bold",
            color=TEXT_NAVY, family="serif", zorder=3)

    add_label(ax, col_x, 23.0, "Stage A: shared input",
              color=TEXT_NAVY, size=11.0, italic=True, bold=True, ha="center")
    add_label(ax, (src_centres[0] + src_centres[-1]) / 2, 23.0,
              "Stage B: four label-source models",
              color=TEXT_NAVY, size=11.0, italic=True, bold=True, ha="center")
    add_label(ax, label_x, 23.0, "Stage C: per-row labels",
              color=TEXT_NAVY, size=11.0, italic=True, bold=True, ha="center")

    base = os.path.join(OUT_DIR, "fig4e1_data_labeling")
    save(fig, base)


# ──────────────────────────────────────────────────────────────────────────
# 4e2  Score-model head training
# ──────────────────────────────────────────────────────────────────────────
def fig4e2_score_training():
    """Cached (z, sigma_t) -> trunk -> 4 heads -> 4 losses -> sum loss; with a
    hard-negative self-distillation feedback loop on the right margin and a
    gold-dashed handoff to the guidance bus / 4(c)."""
    fig, ax = plt.subplots(figsize=(16.0, 10.4), dpi=300)
    setup_axes(ax, xmax=40.0, ymax=26.0)

    y_in = 23.0
    add_box(ax, 4.0, 4.0, 1.8, y_in, CREAM, NAVY,
            r"cached $z$",
            "LIMO latent (1024-d)", title_size=10.8, sub_size=8.4)
    add_box(ax, 12.0, 6.0, 2.2, y_in, PALE_PURPLE, PURPLE,
            "Forward diffusion",
            r"$z_t = \sqrt{\bar\alpha_t}\,\mu + \sqrt{1-\bar\alpha_t}\,\varepsilon$",
            title_size=10.6, sub_size=8.8)
    add_box(ax, 20.6, 6.4, 2.2, y_in, PALE_PURPLE, PURPLE,
            r"$\sigma_t = \sqrt{1 - \bar\alpha_t}$",
            "128-d sinusoidal embed", title_size=10.6, sub_size=8.6)

    add_arrow(ax, 4.0 + 2.0, y_in, 12.0 - 3.0, y_in, lw=1.8)
    add_arrow(ax, 12.0 + 3.0, y_in, 20.6 - 3.2, y_in,
              color=PURPLE, dashed=True, lw=1.6)

    trunk_x_c = 14.4
    trunk_y_c = 18.4
    trunk_w   = 16.4
    trunk_h   = 2.8
    tx = trunk_x_c - trunk_w / 2
    ty = trunk_y_c - trunk_h / 2
    ax.add_patch(FancyBboxPatch(
        (tx + 0.04, ty - 0.04), trunk_w, trunk_h,
        boxstyle="round,pad=0.02,rounding_size=0.20",
        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (tx, ty), trunk_w, trunk_h,
        boxstyle="round,pad=0.02,rounding_size=0.20",
        linewidth=1.8, facecolor=NAVY, edgecolor=NAVY, zorder=2,
    ))
    ax.text(trunk_x_c, trunk_y_c + 0.55,
            "4-block FiLM-MLP trunk", ha="center", va="center",
            fontsize=12.4, fontweight="bold", color="white",
            family="serif", zorder=3)
    ax.text(trunk_x_c, trunk_y_c - 0.18,
            r"1024-d hidden;  inputs $(z_t, \sigma_t)$",
            ha="center", va="center", fontsize=10.0, fontstyle="italic",
            color="#dde6e9", family="serif", zorder=3)

    add_arrow(ax, 12.0, y_in - 1.1,
              trunk_x_c - 3.0, trunk_y_c + trunk_h / 2 + 0.05,
              color=NAVY, lw=1.8)
    add_arrow(ax, 20.6, y_in - 1.1,
              trunk_x_c + 3.0, trunk_y_c + trunk_h / 2 + 0.05,
              color=PURPLE, lw=1.8)

    bus_y = trunk_y_c - trunk_h / 2 - 1.0
    head_centres = [5.4, 12.6, 19.8, 27.0]
    ax.plot([trunk_x_c, trunk_x_c],
            [trunk_y_c - trunk_h / 2 - 0.05, bus_y],
            color=NAVY, lw=2.4, zorder=4)
    ax.plot([head_centres[0], head_centres[-1]], [bus_y, bus_y],
            color=NAVY, lw=2.4, zorder=4)
    add_label(ax, trunk_x_c, bus_y + 0.40,
              "1024-d feature bus", color=TEXT_SLATE, size=9.0,
              italic=True, ha="center")

    head_y_c = 12.4
    head_w   = 5.2
    head_h   = 2.4
    head_names = ["Viability", "Sensitivity", "Hazard", "Performance"]
    head_arch  = [
        r"Linear(1024 $\to$ 1) + sigmoid",
        r"Linear(1024 $\to$ 1)",
        r"Linear(1024 $\to$ 1) + sigmoid",
        r"Linear(1024 $\to$ 4)",
    ]
    head_outputs = [
        r"$\hat y_{\mathrm{viab}}$",
        r"$\hat y_{\mathrm{sens}}$  ($h_{50}$)",
        r"$\hat y_{\mathrm{haz}}$",
        r"$\hat y_{\mathrm{perf}} = (\rho, D, P, \mathrm{HOF})$",
    ]
    edge_cols2 = [GREEN, GOLD, RED, PURPLE]
    fill_cols2 = [PALE_GREEN, PALE_GOLD, PALE_RED, PALE_PURPLE]

    for i, cx in enumerate(head_centres):
        edge = edge_cols2[i]
        fill = fill_cols2[i]
        add_arrow(ax, cx, bus_y, cx, head_y_c + head_h / 2 + 0.05,
                  color=NAVY, lw=1.8)
        x = cx - head_w / 2
        yb = head_y_c - head_h / 2
        ax.add_patch(FancyBboxPatch(
            (x + 0.04, yb - 0.04), head_w, head_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (x, yb), head_w, head_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.8, facecolor=fill, edgecolor=edge, zorder=2,
        ))
        ax.text(cx, head_y_c + 0.65, head_names[i] + " head",
                ha="center", va="center", fontsize=11.2,
                fontweight="bold", color=TEXT_NAVY, family="serif", zorder=3)
        ax.text(cx, head_y_c + 0.05, head_arch[i],
                ha="center", va="center", fontsize=8.4,
                color=TEXT_SLATE, family="serif", zorder=3)
        ax.text(cx, head_y_c - 0.65, head_outputs[i],
                ha="center", va="center", fontsize=8.8,
                fontstyle="italic", color=edge, family="serif", zorder=3)

    loss_y_c = 7.6
    loss_w   = 5.2
    loss_h   = 1.6
    loss_specs = [
        r"$\mathcal{L}_{\mathrm{viab}} = \mathrm{BCE}$",
        r"$\mathcal{L}_{\mathrm{sens}} = \mathrm{SmoothL1}$",
        r"$\mathcal{L}_{\mathrm{haz}} = \mathrm{BCE}$",
        r"$\mathcal{L}_{\mathrm{perf}} = \mathrm{SmoothL1}$",
    ]
    for i, cx in enumerate(head_centres):
        add_arrow(ax, cx, head_y_c - head_h / 2 - 0.05,
                  cx, loss_y_c + loss_h / 2 + 0.05,
                  color=RED, dashed=True, lw=1.5)
        x = cx - loss_w / 2
        yb = loss_y_c - loss_h / 2
        ax.add_patch(FancyBboxPatch(
            (x + 0.04, yb - 0.04), loss_w, loss_h,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (x, yb), loss_w, loss_h,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=1.5, facecolor=PALE_RED, edgecolor=RED, zorder=2,
        ))
        ax.text(cx, loss_y_c, loss_specs[i],
                ha="center", va="center", fontsize=9.4,
                color=TEXT_NAVY, family="serif", zorder=3)

    sum_x_c = trunk_x_c
    sum_y_c = 3.6
    sum_w   = 14.0
    sum_h   = 1.8
    x = sum_x_c - sum_w / 2
    yb = sum_y_c - sum_h / 2
    ax.add_patch(FancyBboxPatch(
        (x + 0.04, yb - 0.04), sum_w, sum_h,
        boxstyle="round,pad=0.04,rounding_size=0.20",
        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (x, yb), sum_w, sum_h,
        boxstyle="round,pad=0.04,rounding_size=0.20",
        linewidth=2.0, facecolor=PALE_RED, edgecolor=RED, zorder=2,
    ))
    ax.text(sum_x_c, sum_y_c + 0.20,
            r"$\mathcal{L}_{\mathrm{score}} = \sum_k a_k \, w_k \, \mathcal{L}_k$",
            ha="center", va="center", fontsize=12.0, fontweight="bold",
            color=TEXT_NAVY, family="serif", zorder=3)
    ax.text(sum_x_c, sum_y_c - 0.40,
            "AdamW, LR 1e-4 cosine, batch 1024, ~40k steps, EMA 0.999",
            ha="center", va="center", fontsize=8.6, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    for cx in head_centres:
        add_arrow(ax, cx, loss_y_c - loss_h / 2 - 0.05,
                  sum_x_c, sum_y_c + sum_h / 2 + 0.05,
                  color=RED, lw=1.5)

    hn_x = 35.6
    ax.plot([trunk_x_c + trunk_w / 2 + 0.05, hn_x],
            [trunk_y_c, trunk_y_c],
            color=GOLD, lw=1.4, linestyle=(0, (2, 3)), zorder=4)
    ax.plot([hn_x, hn_x], [trunk_y_c, y_in + 1.4],
            color=GOLD, lw=1.4, linestyle=(0, (2, 3)), zorder=4)
    ax.plot([hn_x, 4.0 + 2.1], [y_in + 1.4, y_in + 1.4],
            color=GOLD, lw=1.4, linestyle=(0, (2, 3)), zorder=4)
    add_arrow(ax, 4.0 + 2.1, y_in + 1.4, 4.0 + 2.1, y_in + 0.95,
              color=GOLD, dashed=True, lw=1.4)
    ax.text(hn_x + 0.4, (trunk_y_c + y_in) / 2 + 0.4,
            "self-distillation:",
            ha="left", va="center", fontsize=8.6, fontweight="bold",
            color=GOLD, family="serif", zorder=3)
    ax.text(hn_x + 0.4, (trunk_y_c + y_in) / 2 - 0.10,
            r"137 $\to$ 918 hard negatives",
            ha="left", va="center", fontsize=8.4,
            color=GOLD, family="serif", zorder=3)
    ax.text(hn_x + 0.4, (trunk_y_c + y_in) / 2 - 0.55,
            "encoded as viab=0",
            ha="left", va="center", fontsize=8.4, fontstyle="italic",
            color=GOLD, family="serif", zorder=3)
    ax.text(hn_x + 0.4, (trunk_y_c + y_in) / 2 - 1.10,
            "3 rounds; anchor/cheat probe stops",
            ha="left", va="center", fontsize=8.0, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    handoff_y = head_y_c
    add_arrow(ax, head_centres[-1] + head_w / 2 + 0.05, handoff_y,
              hn_x - 0.4, handoff_y, color=GOLD, dashed=True, lw=1.8)
    ax.text(hn_x + 0.4, handoff_y + 0.55,
            r"$\to$ guidance bus",
            ha="left", va="center", fontsize=10.0, fontweight="bold",
            color=GOLD, family="serif", zorder=3)
    ax.text(hn_x + 0.4, handoff_y - 0.05,
            r"$\to$ 4(c)",
            ha="left", va="center", fontsize=10.0, fontweight="bold",
            color=GOLD, family="serif", zorder=3)

    add_label(ax, 0.6, loss_y_c + 1.05,
              r"per-row labels  $y_k$  from 4(e1)",
              color=RED, size=9.0, italic=True, bold=True)
    add_arrow(ax, 2.4, loss_y_c, head_centres[0] - loss_w / 2 - 0.05,
              loss_y_c, color=RED, dashed=True, lw=1.4)

    base = os.path.join(OUT_DIR, "fig4e2_score_training")
    save(fig, base)


# ──────────────────────────────────────────────────────────────────────────
# (legacy) 4e  Multi-head classifier training: kept for reference only.
# ──────────────────────────────────────────────────────────────────────────
def fig4e_head_training():
    """ONE shared 4-block FiLM-MLP trunk takes (z, sigma); four heads branch
    off and produce viability / sensitivity / hazard / performance outputs.

    The four EXTERNAL label sources (RF, Politzer-Murray, SMARTS+Bruns-Watson,
    3D-CNN ensemble) live in a separate lower zone and supply training labels
    only; they are NOT consumed at sample time. At sample time, every head
    takes (z, sigma) through the shared trunk; gradients combine on the
    guidance bus consumed by 4(c)."""
    fig, ax = plt.subplots(figsize=(16.0, 9.6), dpi=300)
    setup_axes(ax, xmax=40.0, ymax=24.0)

    # ── Zone divider (subtle horizontal rule) ──────────────────────────────
    ax.plot([0.6, 39.4], [9.8, 9.8], color=TEXT_LIGHT, lw=0.8,
            linestyle=(0, (2, 3)), zorder=1)

    # Zone banners
    add_label(ax, 0.6, 23.0,
              "Zone 1  -  Score model (sample time + training)",
              color=TEXT_NAVY, size=10.6, italic=True, bold=True)
    add_label(ax, 0.6, 9.3,
              "Zone 2  -  Label sources (offline; training only)",
              color=TEXT_SLATE, size=10.2, italic=True, bold=True)

    # ──────────────────────────────────────────────────────────────────────
    # ZONE 1: SHARED TRUNK + 4 HEADS
    # ──────────────────────────────────────────────────────────────────────
    # Inputs (z, sigma) feeding into the trunk
    y_inputs = 20.6
    add_box(ax, 3.4, 2.4, 1.3, y_inputs, CREAM, NAVY,
            r"$z$", "cached LIMO latent",
            title_size=12.0, sub_size=8.4)
    add_box(ax, 6.8, 2.4, 1.3, y_inputs, PALE_PURPLE, PURPLE,
            r"$\sigma$", "noise level",
            title_size=12.0, sub_size=8.4)

    # Shared trunk: ONE big rectangle. Navy fill, white text.
    trunk_x_c = 14.4
    trunk_y_c = 17.8
    trunk_w   = 11.2
    trunk_h   = 2.8
    tx = trunk_x_c - trunk_w / 2
    ty = trunk_y_c - trunk_h / 2
    ax.add_patch(FancyBboxPatch(
        (tx + 0.04, ty - 0.04), trunk_w, trunk_h,
        boxstyle="round,pad=0.02,rounding_size=0.20",
        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (tx, ty), trunk_w, trunk_h,
        boxstyle="round,pad=0.02,rounding_size=0.20",
        linewidth=1.8, facecolor=NAVY, edgecolor=NAVY, zorder=2,
    ))
    ax.text(trunk_x_c, trunk_y_c + 0.55,
            "Shared trunk", ha="center", va="center",
            fontsize=13.2, fontweight="bold", color="white",
            family="serif", zorder=3)
    ax.text(trunk_x_c, trunk_y_c - 0.10,
            "4-block FiLM-MLP, 1024-d hidden",
            ha="center", va="center", fontsize=10.0, fontstyle="italic",
            color="#dde6e9", family="serif", zorder=3)
    ax.text(trunk_x_c, trunk_y_c - 0.80,
            r"input  $(z, \sigma)$  shared by ALL heads at sample time",
            ha="center", va="center", fontsize=8.8,
            color="#cdd6da", family="serif", zorder=3)

    # Arrows: (z, sigma) into trunk
    add_arrow(ax, 3.4, y_inputs - 0.70,
              trunk_x_c - trunk_w / 2 + 1.0, trunk_y_c + trunk_h / 2 + 0.05,
              color=NAVY, lw=1.8)
    add_arrow(ax, 6.8, y_inputs - 0.70,
              trunk_x_c - trunk_w / 2 + 2.6, trunk_y_c + trunk_h / 2 + 0.05,
              color=PURPLE, lw=1.8)

    # Trunk feature label between trunk and heads
    feat_y = trunk_y_c - trunk_h / 2 - 0.55
    add_label(ax, trunk_x_c, feat_y, "1024-d trunk features",
              color=TEXT_SLATE, size=9.0, italic=True, ha="center")

    # Four head boxes branching off the trunk
    head_y_c = 13.0
    head_w   = 5.2
    head_h   = 2.2
    head_centres = [5.4, 12.6, 19.8, 27.0]
    head_names = ["Viability", "Sensitivity", "Hazard", "Performance"]
    head_arches = [
        r"Linear(1024 $\to$ 1) + sigmoid",
        r"Linear(1024 $\to$ 1)",
        r"Linear(1024 $\to$ 1) + sigmoid",
        r"Linear(1024 $\to$ 4)",
    ]
    head_outputs = [
        r"$P(\mathrm{energetic})$",
        r"predicted $h_{50}$",
        r"$P(\mathrm{hazardous})$",
        r"$\rho,\ D,\ P,\ \mathrm{HOF}$",
    ]
    edge_cols = [GREEN,      GOLD,       RED,       PURPLE]
    fill_cols = [PALE_GREEN, PALE_GOLD,  PALE_RED,  PALE_PURPLE]

    # Single big arrow trunk -> heads (fanout): a horizontal feature bus + drops
    fan_y = trunk_y_c - trunk_h / 2 - 1.05
    fan_x_l = head_centres[0]
    fan_x_r = head_centres[-1]
    # Vertical stem from trunk centre
    ax.plot([trunk_x_c, trunk_x_c],
            [trunk_y_c - trunk_h / 2 - 0.05, fan_y],
            color=NAVY, lw=2.4, zorder=4)
    # Horizontal bus
    ax.plot([fan_x_l, fan_x_r], [fan_y, fan_y],
            color=NAVY, lw=2.4, zorder=4)

    for i, cx in enumerate(head_centres):
        edge = edge_cols[i]
        fill = fill_cols[i]
        # Branch arrow down into head
        add_arrow(ax, cx, fan_y, cx, head_y_c + head_h / 2 + 0.05,
                  color=NAVY, lw=1.8)
        # Head box
        x = cx - head_w / 2
        yb = head_y_c - head_h / 2
        ax.add_patch(FancyBboxPatch(
            (x + 0.04, yb - 0.04), head_w, head_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (x, yb), head_w, head_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.8, facecolor=fill, edgecolor=edge, zorder=2,
        ))
        ax.text(cx, head_y_c + 0.55, head_names[i] + " head",
                ha="center", va="center", fontsize=11.4,
                fontweight="bold", color=TEXT_NAVY, family="serif", zorder=3)
        ax.text(cx, head_y_c - 0.05, head_arches[i],
                ha="center", va="center", fontsize=8.6,
                color=TEXT_SLATE, family="serif", zorder=3)
        ax.text(cx, head_y_c - 0.60, head_outputs[i],
                ha="center", va="center", fontsize=9.2,
                fontstyle="italic", color=edge, family="serif", zorder=3)

    # ── Sample-time guidance bus annotation (right of zone 1) ──────────────
    # Each head emits a gold-dashed gradient down to a small bus that arrows
    # toward Fig 4(c).
    sg_bus_y = 11.0
    for cx in head_centres:
        ax.plot([cx, cx], [head_y_c - head_h / 2 - 0.05, sg_bus_y + 0.02],
                color=GOLD, lw=1.3, linestyle=(0, (4, 3)), zorder=4)
        ax.plot([cx], [sg_bus_y], marker="o", markersize=3.6,
                markerfacecolor=GOLD, markeredgecolor=GOLD, zorder=5)
    ax.plot([head_centres[0], head_centres[-1]], [sg_bus_y, sg_bus_y],
            color=GOLD, lw=2.0, zorder=4)
    # Bus -> Fig 4(c) arrow (rightward)
    add_arrow(ax, head_centres[-1] + 0.05, sg_bus_y,
              35.6, sg_bus_y, color=GOLD, dashed=True, lw=1.8)
    ax.text(37.6, sg_bus_y + 0.55,
            "guidance bus", ha="center", va="center",
            fontsize=9.6, fontweight="bold", color=GOLD,
            family="serif", zorder=3)
    ax.text(37.6, sg_bus_y - 0.55,
            r"$\to$ Fig. 4(c)", ha="center", va="center",
            fontsize=9.8, fontweight="bold", color=GOLD,
            family="serif", zorder=3)
    # Sample-time annotation, centred between Viability and Sensitivity drops
    # at a y just above the bus so it does not clash with the Zone 2 banner.
    sg_label_x = (head_centres[0] + head_centres[1]) / 2
    ax.text(sg_label_x, sg_bus_y + 0.50,
            r"sample time:  $\nabla_z h_k(z, \sigma)$ from each head",
            ha="center", va="center", fontsize=9.0,
            fontstyle="italic", fontweight="bold",
            color=GOLD, family="serif", zorder=5,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.85))

    # ──────────────────────────────────────────────────────────────────────
    # ZONE 2: EXTERNAL LABEL SOURCES
    # ──────────────────────────────────────────────────────────────────────
    src_y_c = 5.0
    src_w   = 6.8
    src_h   = 3.6
    src_centres = head_centres  # vertical alignment with heads above

    src_titles = [
        "Random Forest",
        "Politzer-Murray BDE fit",
        "SMARTS + Bruns-Watson",
        "3D-CNN smoke ensemble",
    ]
    src_inputs = [
        "input: Morgan FP\n+ RDKit descriptors of SMILES",
        "input: chemotype class\n(Ar-NO$_2$, R$_2$N-NO$_2$, ...)",
        "input: SMILES pattern\nmatches",
        "input: 3D conformer\nvoxel grid (Uni-Mol v1)",
    ]
    src_metrics = [
        "AUC = 0.9986",
        r"$h_{50} = 1.93\!\cdot\!\mathrm{BDE} - 52.4$",
        "demerit catalog",
        r"5-fold CV $R^2 = 0.84{-}0.92$",
    ]

    for i, cx in enumerate(src_centres):
        edge = edge_cols[i]
        fill = fill_cols[i]
        x  = cx - src_w / 2
        yb = src_y_c - src_h / 2
        ax.add_patch(FancyBboxPatch(
            (x + 0.04, yb - 0.04), src_w, src_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (x, yb), src_w, src_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.6, facecolor=fill, edgecolor=edge, zorder=2,
        ))
        ax.text(cx, src_y_c + 1.15, src_titles[i],
                ha="center", va="center", fontsize=10.0,
                fontweight="bold", color=edge, family="serif", zorder=3)
        ax.text(cx, src_y_c + 0.10, src_inputs[i],
                ha="center", va="center", fontsize=8.6,
                color=TEXT_NAVY, family="serif",
                linespacing=1.20, zorder=3)
        ax.text(cx, src_y_c - 1.20, src_metrics[i],
                ha="center", va="center", fontsize=8.8,
                fontstyle="italic", color=TEXT_SLATE, family="serif",
                zorder=3)

        # Red dashed "training labels" arrow up into the corresponding head
        add_arrow(ax, cx, src_y_c + src_h / 2 + 0.05,
                  cx, head_y_c - head_h / 2 - 0.05,
                  color=RED, dashed=True, lw=1.6)
        # "training labels" italic tag, placed just above the source-box top,
        # within Zone 2 so it does not collide with the sample-time bus.
        ax.text(cx + 0.65, src_y_c + src_h / 2 + 0.45,
                "training labels",
                ha="left", va="center", fontsize=8.0,
                fontstyle="italic", fontweight="bold",
                color=RED, family="serif", zorder=3)

    # ── "offline only" callout (lower-right, prominent gold-bordered note) ─
    note_x_c = 33.2
    note_y_c = 1.6
    note_w   = 11.6
    note_h   = 1.8
    nx = note_x_c - note_w / 2
    ny = note_y_c - note_h / 2
    ax.add_patch(FancyBboxPatch(
        (nx + 0.04, ny - 0.04), note_w, note_h,
        boxstyle="round,pad=0.04,rounding_size=0.20",
        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (nx, ny), note_w, note_h,
        boxstyle="round,pad=0.04,rounding_size=0.20",
        linewidth=2.2, facecolor=PALE_RED, edgecolor=RED, zorder=2,
    ))
    ax.text(note_x_c, note_y_c + 0.32,
            "label sources run ONCE offline",
            ha="center", va="center", fontsize=11.2,
            fontweight="bold", color=RED, family="serif", zorder=3)
    ax.text(note_x_c, note_y_c - 0.34,
            "not used at sample time",
            ha="center", va="center", fontsize=9.6,
            fontstyle="italic", color=TEXT_NAVY, family="serif", zorder=3)

    base = os.path.join(OUT_DIR, "fig4e_head_training")
    save(fig, base)


# ──────────────────────────────────────────────────────────────────────────
# 4f  Mask construction: per-property inputs (value, availability, tier)
#                       -> binary eligibility mask + tier weight
# ──────────────────────────────────────────────────────────────────────────
def fig4g_mask_sampling():
    """Per-step training mask construction (Fig 4g).

    Five-stage left-to-right pipeline showing how the per-step
    conditioning mask m is sampled fresh each gradient step from the
    fixed per-row eligibility c and tier weight w:

      Stage 1: subset_size k ~ Categorical(subset_size_probs)
      Stage 2: chosen ~ multinomial(w[eligible], k)  [or uniform]
      Stage 3: tentative mask m_hat = onehot(chosen)
      Stage 4: property dropout (per entry, p=0.30)
      Stage 5: cfg dropout (whole row to 0, p=0.10)
      Output : m + row weight omega = alpha + (1-alpha) mean(w*m)
    """
    fig, ax = plt.subplots(figsize=(18.0, 7.5), dpi=300)
    setup_axes(ax, xmax=42.0, ymax=14.0)

    # Stage anchor x-centres (gap of ~5.6, box ~4.4 -> arrow gap ~1.2)
    in_x = 3.4
    s1_x = 9.4
    s2_x = 15.0
    s3_x = 20.6
    s4_x = 26.4
    s5_x = 32.0
    out_x = 38.4

    # Common box dims
    box_w = 4.4
    box_h = 3.6
    bar_y = 9.0   # main row of stage boxes (centre y)

    def stage_box(x, fill, edge, title, y=bar_y, w=box_w, h=box_h):
        sh = FancyBboxPatch((x - w / 2 + 0.04, y - h / 2 - 0.04), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.18",
                            linewidth=0, facecolor="#0a1620", alpha=0.10,
                            zorder=1)
        ax.add_patch(sh)
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.18",
                           linewidth=1.5, facecolor=fill, edgecolor=edge,
                           zorder=2)
        ax.add_patch(b)
        ax.text(x, y + h / 2 - 0.30, title, ha="center", va="center",
                fontsize=10.0, fontweight=700, color=TEXT_NAVY,
                family="serif", zorder=3)

    def vec_cells(xc, y, vals, fill_fn, edge_fn, color_fn, label=None,
                  cell_w=0.55, cell_h=0.55, gap=0.04, size=8.6):
        n = len(vals)
        total_w = n * cell_w + (n - 1) * gap
        x0 = xc - total_w / 2 + cell_w / 2
        for i, v in enumerate(vals):
            xi = x0 + i * (cell_w + gap)
            f = fill_fn(v, i)
            e = edge_fn(v, i)
            c = color_fn(v, i)
            b = FancyBboxPatch((xi - cell_w / 2, y - cell_h / 2),
                               cell_w, cell_h,
                               boxstyle="round,pad=0.01,rounding_size=0.06",
                               linewidth=0.9, facecolor=f, edgecolor=e,
                               zorder=3)
            ax.add_patch(b)
            ax.text(xi, y, str(v), ha="center", va="center",
                    fontsize=size, color=c, family="serif", zorder=4)
        if label is not None:
            ax.text(xc - total_w / 2 - 0.18, y, label, ha="right",
                    va="center", fontsize=9.0, color=TEXT_NAVY,
                    family="serif", zorder=3, fontstyle="italic")

    # ── INPUTS column (c, w) ──
    in_w, in_h = 3.4, 4.6
    sh = FancyBboxPatch((in_x - in_w / 2 + 0.04, bar_y - in_h / 2 - 0.04),
                        in_w, in_h,
                        boxstyle="round,pad=0.02,rounding_size=0.18",
                        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1)
    ax.add_patch(sh)
    b = FancyBboxPatch((in_x - in_w / 2, bar_y - in_h / 2), in_w, in_h,
                       boxstyle="round,pad=0.02,rounding_size=0.18",
                       linewidth=1.5, facecolor=CREAM, edgecolor=NAVY,
                       zorder=2)
    ax.add_patch(b)
    ax.text(in_x, bar_y + in_h / 2 - 0.32, "fixed per-row inputs",
            ha="center", va="center", fontsize=10.6, fontweight=700,
            color=TEXT_NAVY, family="serif", zorder=3)
    ax.text(in_x, bar_y + in_h / 2 - 0.78,
            "(cached at data prep)", ha="center", va="center",
            fontsize=8.6, fontstyle="italic", color=TEXT_SLATE,
            family="serif", zorder=3)

    # property-name header above the input vectors (aligned with cell centres)
    ax.text(in_x + 0.50, bar_y + 0.95, r"$\rho$    HOF    $D$    $P$",
            ha="center", va="center", fontsize=9.5, fontweight=700,
            color=TEXT_NAVY, family="serif", zorder=3)

    # c vector
    c_vals = [1, 1, 0, 0]
    vec_cells(in_x + 0.50, bar_y + 0.40, c_vals,
              fill_fn=lambda v, i: PALE_GREEN if v == 1 else PALE_GREY,
              edge_fn=lambda v, i: GREEN if v == 1 else TEXT_LIGHT,
              color_fn=lambda v, i: GREEN if v == 1 else TEXT_SLATE,
              label=r"$c$")
    # w vector
    w_vals = ["1.0", "1.0", "0.3", "0.1"]
    vec_cells(in_x + 0.50, bar_y - 0.30, w_vals,
              fill_fn=lambda v, i: PALE_GOLD if float(v) >= 0.5 else PALE_GREY,
              edge_fn=lambda v, i: GOLD if float(v) >= 0.5 else TEXT_LIGHT,
              color_fn=lambda v, i: GOLD if float(v) >= 0.5 else TEXT_SLATE,
              label=r"$w$")
    # legends below (kept inside the input box footprint)
    ax.text(in_x + 0.50, bar_y - 1.20,
            r"$c$: Tier-A/B $\wedge$ present",
            ha="center", va="center", fontsize=8.0,
            fontstyle="italic", color=GREEN, family="serif", zorder=3)
    ax.text(in_x + 0.50, bar_y - 1.55,
            r"$w$: A 1.0 / B 0.7 / C 0.3 / D 0.1",
            ha="center", va="center", fontsize=8.0,
            fontstyle="italic", color=GOLD, family="serif", zorder=3)

    # ── STAGE 1: subset-size sampler (purple stochastic) ──
    stage_box(s1_x, PALE_PURPLE, PURPLE,
              "Stage 1: subset size")
    # Categorical bars (k = 0..4)
    cat_x0 = s1_x - 1.40
    cat_y0 = bar_y - 0.95
    cat_dx = 0.62
    cat_probs = [0.10, 0.25, 0.30, 0.25, 0.10]
    cat_max_h = 1.40
    for i, p in enumerate(cat_probs):
        bx = cat_x0 + i * cat_dx
        h = cat_max_h * (p / 0.30)
        rect = Rectangle((bx - 0.20, cat_y0), 0.40, h,
                         facecolor=PURPLE, edgecolor=PURPLE,
                         linewidth=0.5, alpha=0.85, zorder=3)
        ax.add_patch(rect)
        ax.text(bx, cat_y0 - 0.22, f"{i}", ha="center", va="center",
                fontsize=8.4, color=TEXT_SLATE, family="serif", zorder=3)
        ax.text(bx, cat_y0 + h + 0.18, f"{p:.2f}", ha="center", va="center",
                fontsize=7.4, color=TEXT_SLATE, family="serif", zorder=3)
    ax.text(s1_x, cat_y0 - 0.62,
            r"$k \sim \mathrm{Cat}(0.10, 0.25, 0.30, 0.25, 0.10)$",
            ha="center", va="center", fontsize=8.6, color=PURPLE,
            family="serif", fontstyle="italic", zorder=3)
    # sampled k callout (placed below title, above bars)
    ax.text(s1_x, bar_y + 0.95, r"sampled  $k = 2$",
            ha="center", va="center", fontsize=9.6, fontweight=700,
            color=PURPLE, family="serif", zorder=3)

    # ── STAGE 2: weighted pick (gold conditioning) ──
    stage_box(s2_x, PALE_GOLD, GOLD,
              "Stage 2: weighted pick")
    # NOTE: shortened titles below keep within box width.
    # Show formula and resulting indices
    ax.text(s2_x, bar_y + 0.65,
            r"chosen $\sim$ Multinom",
            ha="center", va="center", fontsize=9.2, color=TEXT_NAVY,
            family="serif", zorder=3)
    ax.text(s2_x, bar_y + 0.25,
            r"$(w[c{=}1],\, k)$",
            ha="center", va="center", fontsize=9.2, color=TEXT_NAVY,
            family="serif", zorder=3)
    ax.text(s2_x, bar_y - 0.10,
            r"(uniform if unweighted)",
            ha="center", va="center", fontsize=8.0, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)
    # eligible set graphic: positions where c=1 -> {0, 1}
    ax.text(s2_x, bar_y - 0.55,
            r"eligible $= \{0, 1\}$",
            ha="center", va="center", fontsize=8.6, color=GREEN,
            family="serif", zorder=3)
    ax.text(s2_x, bar_y - 0.95,
            r"chosen $= \{0, 1\}$",
            ha="center", va="center", fontsize=9.4, fontweight=700,
            color=GOLD, family="serif", zorder=3)
    ax.text(s2_x, bar_y - 1.40,
            r"($w$ favours Tier-A)",
            ha="center", va="center", fontsize=8.0, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    # ── STAGE 3: tentative mask m_hat ──
    stage_box(s3_x, PALE_GREY, NAVY,
              r"Stage 3: tentative $\hat m$")
    mhat_vals = [1, 1, 0, 0]
    vec_cells(s3_x, bar_y + 0.10, mhat_vals,
              fill_fn=lambda v, i: PALE_GREEN if v == 1 else "#ffffff",
              edge_fn=lambda v, i: NAVY if v == 1 else TEXT_LIGHT,
              color_fn=lambda v, i: NAVY if v == 1 else TEXT_LIGHT,
              cell_w=0.62, cell_h=0.62, size=9.6)
    ax.text(s3_x, bar_y - 0.55,
            r"$\hat m_i = \mathbb{1}[i \in \mathrm{chosen}]$",
            ha="center", va="center", fontsize=9.0, color=TEXT_NAVY,
            family="serif", zorder=3)
    ax.text(s3_x, bar_y - 1.05,
            "one-hot over chosen", ha="center", va="center",
            fontsize=8.2, fontstyle="italic", color=TEXT_SLATE,
            family="serif", zorder=3)

    # ── STAGE 4: property dropout (purple stochastic) ──
    stage_box(s4_x, PALE_PURPLE, PURPLE,
              "Stage 4: prop dropout")
    # show coin flips per entry, then result
    coin_vals = ["1", "0", "1", "1"]   # rand >= 0.30 ? (per-entry retain)
    ax.text(s4_x, bar_y + 0.95, r"per entry: $u_i \sim U[0,1)$",
            ha="center", va="center", fontsize=8.6, color=PURPLE,
            family="serif", fontstyle="italic", zorder=3)
    vec_cells(s4_x, bar_y + 0.40,
              [r"$u\!\geq\!.3$", r"$u\!<\!.3$", r"$u\!\geq\!.3$", r"$u\!\geq\!.3$"],
              fill_fn=lambda v, i: PALE_PURPLE,
              edge_fn=lambda v, i: PURPLE,
              color_fn=lambda v, i: PURPLE,
              cell_w=0.78, cell_h=0.45, size=7.0)
    # outgoing intermediate mask after dropout: 1*1, 1*0, 0*1, 0*1 -> [1,0,0,0]
    after_vals = [1, 0, 0, 0]
    vec_cells(s4_x, bar_y - 0.30, after_vals,
              fill_fn=lambda v, i: PALE_GREEN if v == 1 else "#ffffff",
              edge_fn=lambda v, i: NAVY if v == 1 else TEXT_LIGHT,
              color_fn=lambda v, i: NAVY if v == 1 else TEXT_LIGHT,
              cell_w=0.62, cell_h=0.55, size=9.0)
    ax.text(s4_x, bar_y - 0.95,
            r"$\hat m_i \;\leftarrow\; \hat m_i \cdot \mathbb{1}[u_i \geq 0.30]$",
            ha="center", va="center", fontsize=8.6, color=TEXT_NAVY,
            family="serif", zorder=3)
    ax.text(s4_x, bar_y - 1.35,
            "(zero each entry w.p. 0.30)",
            ha="center", va="center", fontsize=8.0, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    # ── STAGE 5: cfg-dropout (purple stochastic) ──
    stage_box(s5_x, PALE_PURPLE, PURPLE,
              "Stage 5: CFG dropout")
    # short title fits 4.4-wide box at fontsize 10
    ax.text(s5_x, bar_y + 0.95,
            r"$v \sim U[0,1)$",
            ha="center", va="center", fontsize=9.0, color=PURPLE,
            family="serif", fontstyle="italic", zorder=3)
    ax.text(s5_x, bar_y + 0.40,
            r"if $v < 0.10$:  $m \leftarrow 0$",
            ha="center", va="center", fontsize=9.0, color=TEXT_NAVY,
            family="serif", zorder=3)
    ax.text(s5_x, bar_y - 0.10,
            r"else:  $m \leftarrow \hat m$",
            ha="center", va="center", fontsize=9.0, color=TEXT_NAVY,
            family="serif", zorder=3)
    ax.text(s5_x, bar_y - 0.70,
            "(whole row unconditional)",
            ha="center", va="center", fontsize=8.0, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)
    ax.text(s5_x, bar_y - 1.20,
            "trains the unconditional branch",
            ha="center", va="center", fontsize=8.0, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    # ── OUTPUT: final m + omega callout (red/loss colour for grad weight) ──
    out_w, out_h = 3.4, 4.6
    sh = FancyBboxPatch((out_x - out_w / 2 + 0.04, bar_y - out_h / 2 - 0.04),
                        out_w, out_h,
                        boxstyle="round,pad=0.02,rounding_size=0.18",
                        linewidth=0, facecolor="#0a1620", alpha=0.10, zorder=1)
    ax.add_patch(sh)
    b = FancyBboxPatch((out_x - out_w / 2, bar_y - out_h / 2), out_w, out_h,
                       boxstyle="round,pad=0.02,rounding_size=0.18",
                       linewidth=1.5, facecolor=PALE_RED, edgecolor=RED,
                       zorder=2)
    ax.add_patch(b)
    ax.text(out_x, bar_y + out_h / 2 - 0.32, "output (this step)",
            ha="center", va="center", fontsize=10.6, fontweight=700,
            color=TEXT_NAVY, family="serif", zorder=3)
    # final m (assume CFG kept it: v >= 0.1)
    ax.text(out_x, bar_y + 0.95, r"final mask  $m$",
            ha="center", va="center", fontsize=9.6, fontweight=700,
            color=NAVY, family="serif", zorder=3)
    final_m = [1, 0, 0, 0]
    vec_cells(out_x, bar_y + 0.40, final_m,
              fill_fn=lambda v, i: PALE_GREEN if v == 1 else "#ffffff",
              edge_fn=lambda v, i: NAVY if v == 1 else TEXT_LIGHT,
              color_fn=lambda v, i: NAVY if v == 1 else TEXT_LIGHT,
              cell_w=0.62, cell_h=0.62, size=10.0)
    ax.text(out_x, bar_y - 0.20, r"$m \in \{0,1\}^4$  to FiLM",
            ha="center", va="center", fontsize=8.6, fontstyle="italic",
            color=NAVY, family="serif", zorder=3)
    # omega
    ax.text(out_x, bar_y - 0.80,
            r"row weight  $\omega$",
            ha="center", va="center", fontsize=9.6, fontweight=700,
            color=RED, family="serif", zorder=3)
    ax.text(out_x, bar_y - 1.30,
            r"$\omega = \alpha + (1{-}\alpha)\,\overline{w \odot m}$",
            ha="center", va="center", fontsize=8.8, color=TEXT_NAVY,
            family="serif", zorder=3)
    ax.text(out_x, bar_y - 1.75,
            r"$\alpha{=}1$: $\omega{=}1$ (plain MSE)",
            ha="center", va="center", fontsize=8.0, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    # ── connecting arrows between stages ──
    arrow_pairs = [
        (in_x + in_w / 2,  s1_x - box_w / 2,  NAVY),
        (s1_x + box_w / 2, s2_x - box_w / 2,  PURPLE),
        (s2_x + box_w / 2, s3_x - box_w / 2,  GOLD),
        (s3_x + box_w / 2, s4_x - box_w / 2,  NAVY),
        (s4_x + box_w / 2, s5_x - box_w / 2,  PURPLE),
        (s5_x + box_w / 2, out_x - out_w / 2, PURPLE),
    ]
    for (x0, x1, col) in arrow_pairs:
        add_arrow(ax, x0, bar_y, x1, bar_y, color=col,
                  dashed=(col != NAVY), lw=1.6, shrink=0.10)

    # arrow labels (data passing along)
    ax.text((in_x + in_w / 2 + s1_x - box_w / 2) / 2, bar_y + 0.30,
            r"$c, w$", ha="center", va="center", fontsize=8.6,
            color=TEXT_SLATE, family="serif", fontstyle="italic", zorder=3)
    ax.text((s1_x + box_w / 2 + s2_x - box_w / 2) / 2, bar_y + 0.30,
            r"$k$", ha="center", va="center", fontsize=8.6,
            color=PURPLE, family="serif", fontstyle="italic", zorder=3)
    ax.text((s2_x + box_w / 2 + s3_x - box_w / 2) / 2, bar_y + 0.30,
            "chosen", ha="center", va="center", fontsize=8.4,
            color=GOLD, family="serif", fontstyle="italic", zorder=3)
    ax.text((s3_x + box_w / 2 + s4_x - box_w / 2) / 2, bar_y + 0.30,
            r"$\hat m$", ha="center", va="center", fontsize=8.6,
            color=NAVY, family="serif", fontstyle="italic", zorder=3)
    ax.text((s4_x + box_w / 2 + s5_x - box_w / 2) / 2, bar_y + 0.30,
            r"$\hat m'$", ha="center", va="center", fontsize=8.6,
            color=PURPLE, family="serif", fontstyle="italic", zorder=3)
    ax.text((s5_x + box_w / 2 + out_x - out_w / 2) / 2, bar_y + 0.30,
            r"$m$", ha="center", va="center", fontsize=8.6,
            color=PURPLE, family="serif", fontstyle="italic", zorder=3)

    # ── Title and legend strip ──
    ax.text(21.0, 13.3,
            "Per-step training mask construction (Fig 4g)",
            ha="center", va="center", fontsize=14.0, fontweight=700,
            color=TEXT_NAVY, family="serif", zorder=5)
    ax.text(21.0, 12.55,
            r"per gradient step: sample fresh $m \in \{0,1\}^4$ "
            r"from fixed $(c, w)$; $m$ feeds FiLM, $\omega$ scales the row's MSE",
            ha="center", va="center", fontsize=10.2, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=5)

    # color key strip near bottom-left
    add_color_legend(ax, x=1.0, y=2.4,
                     items=[
                         (NAVY,   False, "data flow"),
                         (GOLD,   True,  "conditioning (weighted pick)"),
                         (PURPLE, True,  "stochastic samplers"),
                         (RED,    False, "loss / row weight"),
                     ])

    base = os.path.join(OUT_DIR, "fig4g_mask_sampling")
    save(fig, base)


def fig4f_mask_construction_DEPRECATED():
    """DEPRECATED: replaced by fig4g_mask_sampling. Kept for archival.

    Reads top-to-bottom for ONE example row across the 4 properties:
    value -> avail -> trust -> eligibility c_k, then a parallel
    tier-weight lookup w_k. Then per-step mask m_k is sampled from the
    eligibility column, and the row weight omega is shown at the bottom.
    """
    fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=300)
    setup_axes(ax, xmax=22.5, ymax=11.0)

    # 4 property columns at x = 6, 10, 14, 18; row-label column at x = 2.6
    prop_x = [6.0, 10.0, 14.0, 18.0]
    prop_names = [r"$\rho$", "HOF", r"$D$", r"$P$"]
    rowlabel_x = 2.6

    # Concrete example row: experimental rho + HOF (Tier-A), missing D, K-J P
    example_value = ["1.82",  "+70",  "(NaN)", "34"]
    example_avail = ["yes",   "yes",  "no",    "yes"]
    example_tier  = ["A",     "A",    "—",     "C"]

    # Row layout (top -> bottom)
    rows = [
        # (label, values, fill, edge, text_color, italic_label?)
        ("value (raw)",      example_value, CREAM,      NAVY,      TEXT_NAVY, False),
        ("availability $a$", example_avail, PALE_GREY,  NAVY,      TEXT_NAVY, False),
        ("tier",             example_tier,  PALE_GREY,  NAVY,      TEXT_NAVY, False),
    ]
    row_y = [9.6, 8.4, 7.2]
    cell_w = 1.7
    cell_h = 0.9

    def cell(x, y, w, h, fill, edge, text, color=TEXT_NAVY, bold=False,
             italic=False, size=10.0):
        sh = FancyBboxPatch((x - w / 2 + 0.03, y - h / 2 - 0.03), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            linewidth=0, facecolor="#0a1620", alpha=0.08,
                            zorder=1)
        ax.add_patch(sh)
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.10",
                           linewidth=1.0, facecolor=fill, edgecolor=edge,
                           zorder=2)
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center", fontsize=size,
                color=color, family="serif",
                fontweight="bold" if bold else "normal",
                fontstyle="italic" if italic else "normal", zorder=3)

    # Top section: input rows (value, avail, tier)
    for (lbl, vals, fill, edge, _, _), ry in zip(rows, row_y):
        ax.text(rowlabel_x, ry, lbl, ha="right", va="center", fontsize=10.5,
                fontweight=600, color=TEXT_NAVY, family="serif")
        for x, v in zip(prop_x, vals):
            cell(x, ry, cell_w, cell_h, fill, edge, v)

    # Property column headers above
    for x, name in zip(prop_x, prop_names):
        ax.text(x, 10.6, name, ha="center", va="center", fontsize=12,
                fontweight="bold", color=TEXT_NAVY, family="serif")

    # Divider line
    ax.plot([rowlabel_x - 0.5, prop_x[-1] + cell_w / 2 + 0.3],
            [6.55, 6.55], color=TEXT_LIGHT, lw=0.8, linestyle=":", zorder=2)

    # Mid section: derivation rules (with formulas) and outputs
    # trust_k = (tier_k in {A,B})
    ry_trust = 5.8
    ax.text(rowlabel_x, ry_trust, "trust  $t = (\\,\\mathrm{tier} \\in \\{A,B\\}\\,)$",
            ha="right", va="center", fontsize=10.5, fontweight=600,
            color=TEXT_NAVY, family="serif")
    trust_vals = ["1", "1", "—", "0"]
    for x, v, t in zip(prop_x, trust_vals, example_tier):
        c_color = GOLD if v == "1" else (TEXT_LIGHT if v == "—" else TEXT_SLATE)
        cell(x, ry_trust, cell_w, cell_h, "#ffffff",
             c_color, v, color=c_color, bold=(v == "1"))

    # eligibility c = trust AND avail
    ry_elig = 4.6
    ax.text(rowlabel_x, ry_elig,
            r"eligibility  $c = t \wedge a$",
            ha="right", va="center", fontsize=10.5, fontweight=600,
            color=GREEN, family="serif")
    elig_vals = ["1", "1", "0", "0"]
    for x, v in zip(prop_x, elig_vals):
        is_one = (v == "1")
        cell(x, ry_elig, cell_w, cell_h, PALE_GREEN if is_one else PALE_GREY,
             GREEN if is_one else TEXT_LIGHT, v,
             color=GREEN if is_one else TEXT_SLATE, bold=is_one, size=11)

    # tier weight w = lookup(tier)
    ry_w = 3.4
    ax.text(rowlabel_x, ry_w,
            "tier weight  $w$",
            ha="right", va="center", fontsize=10.5, fontweight=600,
            color=GOLD, family="serif")
    ax.text(rowlabel_x, ry_w - 0.45,
            r"(A:1.0  B:0.7  C:0.3  D:0.1)",
            ha="right", va="center", fontsize=8.4, fontstyle="italic",
            color=TEXT_SLATE, family="serif")
    weight_vals = ["1.0", "1.0", "0.0", "0.3"]
    for x, v in zip(prop_x, weight_vals):
        is_strong = float(v) >= 0.5
        cell(x, ry_w, cell_w, cell_h, PALE_GOLD if is_strong else PALE_GREY,
             GOLD if is_strong else TEXT_LIGHT, v,
             color=GOLD if is_strong else TEXT_SLATE, bold=is_strong, size=11)

    # per-step sampled mask m
    ry_m = 2.0
    ax.text(rowlabel_x, ry_m,
            r"sampled mask  $m$  (per step)",
            ha="right", va="center", fontsize=10.5, fontweight=600,
            color=NAVY, family="serif")
    ax.text(rowlabel_x, ry_m - 0.45,
            r"(random subset of eligible)",
            ha="right", va="center", fontsize=8.4, fontstyle="italic",
            color=TEXT_SLATE, family="serif")
    mask_vals = ["1", "1", "0", "0"]
    for x, v in zip(prop_x, mask_vals):
        is_one = (v == "1")
        cell(x, ry_m, cell_w, cell_h, PALE_GREY if not is_one else "#dde6e9",
             NAVY if is_one else TEXT_LIGHT, v,
             color=NAVY if is_one else TEXT_SLATE, bold=is_one, size=11)

    # Per-row sample weight, formula at the bottom
    ax.add_patch(FancyBboxPatch(
        (rowlabel_x - 1.0, 0.4), prop_x[-1] + cell_w / 2 + 0.3 - (rowlabel_x - 1.0), 0.95,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, facecolor=PALE_GOLD, edgecolor=GOLD, zorder=2,
    ))
    ax.text((rowlabel_x - 1.0 + prop_x[-1] + cell_w / 2 + 0.3) / 2, 0.85,
            r"row weight  $\omega = \alpha + (1 - \alpha)\,\mathrm{mean}(w \odot m)$"
            r" $\;\;=\;\; \alpha + (1-\alpha)\,\mathrm{mean}(1.0,\,1.0,\,0,\,0) "
            r"= \alpha + 0.5\,(1-\alpha)$  for this row",
            ha="center", va="center", fontsize=10.6, color=TEXT_NAVY,
            family="serif", zorder=3)

    # Right margin: outputs note
    ax.text(prop_x[-1] + cell_w / 2 + 0.5, ry_elig,
            r"$\to$ feeds FiLM",
            ha="left", va="center", fontsize=9.0, fontstyle="italic",
            color=GREEN, family="serif")
    ax.text(prop_x[-1] + cell_w / 2 + 0.5, ry_w,
            r"$\to$ row weight $\omega$",
            ha="left", va="center", fontsize=9.0, fontstyle="italic",
            color=GOLD, family="serif")
    ax.text(prop_x[-1] + cell_w / 2 + 0.5, ry_m,
            r"$\to$ FiLM gate",
            ha="left", va="center", fontsize=9.0, fontstyle="italic",
            color=NAVY, family="serif")

    base = os.path.join(OUT_DIR, "fig_mask_construction_aside")
    save(fig, base)


# ──────────────────────────────────────────────────────────────────────────
# 4f  Self-distillation: 3 rounds of (mine-with-current-model, retrain)
#                       to grow the viability head's hard-negative set.
# ──────────────────────────────────────────────────────────────────────────
def fig4f_self_distillation():
    """Three sequential rounds of viability-head self-distillation.
    Each row = one round. Round 0 trains on corpus only, mines 137 cheats.
    Round 1 trains on corpus + 137, mines additional cheats up to 918 total.
    Round 2 trains on corpus + 918 and is the final score model.
    """
    fig, ax = plt.subplots(figsize=(15.5, 7.5), dpi=300)
    setup_axes(ax, xmax=29.0, ymax=11.5)

    # Three rows, top-to-bottom.
    rows_y = [9.4, 6.4, 3.4]
    round_labels = ["Round 0", "Round 1", "Round 2 (final)"]

    # x-positions for the 5 boxes per row:
    # train_data  ->  train  ->  v_n  ->  sample  ->  mine -> hard-negative count
    train_x = 3.0
    fit_x   = 8.0
    model_x = 12.5
    samp_x  = 17.0
    gate_x  = 21.0
    out_x   = 25.5

    bw_data = 4.4
    bw_step = 2.4
    bw_model = 2.6
    bh = 1.55

    train_data_specs = [
        ("corpus only",          "66k labelled rows"),
        ("corpus + 137",         "66k + 137 cheats"),
        ("corpus + 918",         "66k + 918 cheats"),
    ]
    out_specs = [
        ("137 cheats",  PALE_RED,  RED,    True),
        ("918 cheats",  PALE_RED,  RED,    True),
        ("(final v2)",  PALE_GOLD, GOLD,   False),  # Round 2 doesn't mine more
    ]

    for i, (y, lbl, td, out) in enumerate(zip(rows_y, round_labels,
                                              train_data_specs, out_specs)):
        # Row label on the left
        ax.text(0.6, y, lbl, ha="left", va="center", fontsize=11.5,
                fontweight="bold", color=TEXT_NAVY, family="serif")
        if i == 2:
            ax.text(0.6, y - 0.55,
                    "stops; passes anchor/cheat probe",
                    ha="left", va="center", fontsize=8.4, fontstyle="italic",
                    color=GREEN, family="serif")

        # Train-data box
        add_box(ax, train_x, bw_data, bh, y, CREAM, NAVY, td[0], td[1],
                title_size=10.2, sub_size=8.6)
        # train arrow + box
        add_arrow(ax, train_x + bw_data / 2, y, fit_x - bw_step / 2, y, lw=1.6)
        add_box(ax, fit_x, bw_step, bh, y, PALE_GREY, NAVY,
                "train", r"$\sim$40k steps",
                title_size=10.2, sub_size=8.4)
        # model arrow + box
        add_arrow(ax, fit_x + bw_step / 2, y, model_x - bw_model / 2, y, lw=1.6)
        # Score model box (navy fill, white text)
        sh = FancyBboxPatch((model_x - bw_model / 2 + 0.04,
                             y - bh / 2 - 0.04), bw_model, bh,
                            boxstyle="round,pad=0.02,rounding_size=0.14",
                            linewidth=0, facecolor="#0a1620", alpha=0.10,
                            zorder=1)
        ax.add_patch(sh)
        b = FancyBboxPatch((model_x - bw_model / 2, y - bh / 2),
                           bw_model, bh,
                           boxstyle="round,pad=0.02,rounding_size=0.14",
                           linewidth=1.4, facecolor=NAVY, edgecolor=NAVY,
                           zorder=2)
        ax.add_patch(b)
        ax.text(model_x, y + 0.16, f"score model v{i}",
                ha="center", va="center", fontsize=10.4, fontweight="bold",
                color="white", family="serif", zorder=3)
        ax.text(model_x, y - 0.30, ("FINAL" if i == 2 else "intermediate"),
                ha="center", va="center", fontsize=8.4, fontstyle="italic",
                color=("#ffd54a" if i == 2 else "#dde6e9"),
                family="serif", zorder=3)

        if i < 2:
            # sample step (purple, stochastic)
            add_arrow(ax, model_x + bw_model / 2, y, samp_x - bw_step / 2, y,
                      color=PURPLE, dashed=True, lw=1.6)
            add_box(ax, samp_x, bw_step, bh, y, PALE_PURPLE, PURPLE,
                    "sample", r"diffusion + v$_{}$".format(i),
                    title_size=10.2, sub_size=8.4)
            # SMARTS gate
            add_arrow(ax, samp_x + bw_step / 2, y, gate_x - bw_step / 2, y,
                      lw=1.6)
            add_box(ax, gate_x, bw_step, bh, y, PALE_RED, RED,
                    "SMARTS gate",
                    "false-positive\nfilter",
                    title_size=10.2, sub_size=8.0)
            # output (mined hard negatives)
            add_arrow(ax, gate_x + bw_step / 2, y, out_x - bw_step / 2, y,
                      color=RED, dashed=True, lw=1.6)
            add_box(ax, out_x, bw_step, bh, y, *out_specs[i][1:3],
                    f"mine\n{out_specs[i][0]}",
                    "encode via LIMO",
                    title_size=10.2, sub_size=8.0)
            # vertical feedback to next round's train data
            ax.annotate("", xy=(train_x, rows_y[i + 1] + bh / 2 + 0.05),
                        xytext=(out_x - bw_step / 2, y - bh / 2 - 0.05),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                                        linestyle=(0, (4, 3)),
                                        shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3,rad=0.18",
                                        mutation_scale=12),
                        zorder=4)
            # label on feedback arrow
            ax.text((out_x + train_x) / 2 + 1.5,
                    (y + rows_y[i + 1]) / 2 + 0.5,
                    f"feed back as viab=0",
                    ha="center", va="center", fontsize=8.6,
                    fontstyle="italic", color=RED,
                    family="serif", zorder=5)

    # Right margin: anchor/cheat probe note
    ax.add_patch(FancyBboxPatch(
        (24.3, 0.4), 4.4, 1.4,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.2, facecolor=PALE_GREEN, edgecolor=GREEN, zorder=2,
    ))
    ax.text(26.5, 1.4, "stop criterion",
            ha="center", va="center", fontsize=9.6, fontweight="bold",
            color=GREEN, family="serif", zorder=3)
    ax.text(26.5, 0.85,
            r"7 anchors $\geq 0.86$;  5 cheats $\leq 0.84$",
            ha="center", va="center", fontsize=8.6, fontstyle="italic",
            color=TEXT_SLATE, family="serif", zorder=3)

    base = os.path.join(OUT_DIR, "fig4f_self_distillation")
    save(fig, base)


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # NOTE: fig4a_data_prep(), fig4c_sampling_guidance(), and
    # fig4d_decode_rerank() are intentionally NOT called here. Their PNGs are
    # sourced from hand-authored assets:
    #   assets/4a_LIMO_TRAIN.png        -> figs/fig4a_data_prep.png
    #   assets/4C_classifier_guide.png  -> figs/fig4c_sampling_guidance.png
    #   assets/_4d sampling.png         -> figs/fig4d_decode_rerank.png
    # Re-running the matplotlib renderer would clobber those assets. The
    # functions are preserved as documentation of alternative layouts.
    # fig4e1_data_labeling() is also asset-sourced now:
    #   assets/4e1_score_labels.png -> figs/fig4e1_data_labeling.png
    fig4b_train_loop()
    fig4e2_score_training()
    fig4g_mask_sampling()
    fig4f_self_distillation()
