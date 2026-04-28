"""Generate figures for the paper. Runs against the project's actual data."""
from __future__ import annotations
import json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("E:/Projects/EnergeticDiffusion2")
OUT  = ROOT / "docs/paper/figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})


def parse_rerank_md(path):
    """Extract (rank, composite, rho, hof, d, p, sa, sc, maxtan, source, smiles)."""
    if not Path(path).exists():
        return []
    md = Path(path).read_text(encoding="utf-8")
    rows = []
    for line in md.split("\n"):
        if not line.startswith("| "):
            continue
        if "rank" in line or "---" in line:
            continue
        cells = [c.strip(" `") for c in line.split("|")]
        if len(cells) < 12:
            continue
        try:
            rows.append({
                "rank":   int(cells[1]),
                "score":  float(cells[2]),
                "rho":    float(cells[3]),
                "hof":    float(cells[4].replace("+", "")),
                "d":      float(cells[5]),
                "p":      float(cells[6]),
                "sa":     float(cells[7]),
                "sc":     float(cells[8]),
                "maxtan": float(cells[9]),
                "src":    cells[10],
                "smi":    cells[11],
            })
        except (ValueError, IndexError):
            continue
    return rows


# ── Figure 1: Pool-size scaling ─────────────────────────────────────────
def fig_pool_scaling():
    runs = [
        ("Initial (pool=1.5k)", 1500, 4.11, 80, "registry initial"),
        ("pool=8k",   8000,  0.79, 196, "this work"),
        ("pool=20k", 20000, 0.70, 983, "this work"),
        ("pool=40k", 40000, 0.92, 1667, "this work"),
    ]
    pools  = [r[1] for r in runs]
    scores = [r[2] for r in runs]
    finals = [r[3] for r in runs]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    l1 = ax1.plot(pools, scores, "o-", color="#1f77b4", label="best composite score (lower = better)")
    l2 = ax2.plot(pools, finals, "s--", color="#d62728", label="candidates kept after all filters")
    ax1.set_xscale("log")
    ax1.set_xlabel("pool size (samples per denoiser)")
    ax1.set_ylabel("best composite score", color="#1f77b4")
    ax2.set_ylabel("# candidates after chem + SA/SC + Tanimoto", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title("Pool size scaling (joint v3 + v4-B, cfg=7)")
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_pool_scaling.svg")
    plt.close(fig)
    print(f"  fig1: {OUT / 'fig1_pool_scaling.svg'}")


# ── Figure 2: CFG sweep ─────────────────────────────────────────────────
def fig_cfg_sweep():
    runs = [(5, 802, 758, 528, 0.92), (7, 1533, 1429, 983, 0.70), (9, 673, 650, 427, 0.79)]
    cfgs = [r[0] for r in runs]
    final = [r[3] for r in runs]
    score = [r[4] for r in runs]
    fig, ax1 = plt.subplots(figsize=(6, 3.6))
    ax2 = ax1.twinx()
    l1 = ax1.bar([c - 0.18 for c in cfgs], final, width=0.36,
                 color="#2ca02c", alpha=0.7, label="# final candidates")
    l2 = ax2.plot(cfgs, score, "o-", color="#d62728", label="best composite", markersize=8)
    ax1.set_xticks(cfgs)
    ax1.set_xlabel("CFG scale")
    ax1.set_ylabel("# final candidates", color="#2ca02c")
    ax2.set_ylabel("best composite score (lower = better)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Classifier-free guidance scale sweep (pool=8k each)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cfg_sweep.svg")
    plt.close(fig)
    print(f"  fig2: {OUT / 'fig2_cfg_sweep.svg'}")


# ── Figure 3: Top leads in (D, P) plane vs known anchors (2-panel) ────────
def fig_top_leads_plane():
    leads = parse_rerank_md(
        ROOT / "experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_pool40k.md"
    )[:200]
    # Known anchors (literature values)
    anchors = {
        "TNT":   {"rho": 1.65, "d": 6.95, "p": 21.0},
        "RDX":   {"rho": 1.81, "d": 8.75, "p": 34.9},
        "HMX":   {"rho": 1.91, "d": 9.10, "p": 39.0},
        "PETN":  {"rho": 1.77, "d": 8.30, "p": 31.5},
        "CL-20": {"rho": 2.04, "d": 9.66, "p": 46.0},
        "TATB":  {"rho": 1.93, "d": 7.95, "p": 31.5},
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    ds  = [l["d"] for l in leads]
    ps  = [l["p"] for l in leads]
    rhos = [l["rho"] for l in leads]
    novelties = [1.0 - l["maxtan"] for l in leads]

    # Panel A: colour = predicted density rho
    axA = axes[0]
    scA = axA.scatter(ds, ps, c=rhos, cmap="viridis", s=22, alpha=0.75,
                       edgecolor="black", linewidths=0.3)
    cbA = fig.colorbar(scA, ax=axA, label=r"predicted density $\rho$ (g/cm$^3$)")
    axA.set_title("A. Coloured by predicted density")

    # Panel B: colour = novelty (1 - max Tanimoto to labelled-master)
    axB = axes[1]
    scB = axB.scatter(ds, ps, c=novelties, cmap="plasma", vmin=0.0, vmax=1.0,
                       s=22, alpha=0.85, edgecolor="black", linewidths=0.3)
    cbB = fig.colorbar(scB, ax=axB, label="novelty (1 - max Tanimoto to labelled master)")
    axB.set_title("B. Coloured by novelty")

    # Anchors + targets in both panels
    for ax in (axA, axB):
        for name, a in anchors.items():
            ax.scatter(a["d"], a["p"], marker="*", s=200, edgecolor="black",
                       facecolor="white", linewidth=1.4, zorder=4)
            ax.annotate(name, (a["d"], a["p"]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=9, fontweight="bold")
        ax.axvline(9.5, color="grey", linestyle=":", alpha=0.5)
        ax.axhline(40,  color="grey", linestyle=":", alpha=0.5)
        ax.set_xlabel("detonation velocity D (km/s)")
        ax.set_ylabel("detonation pressure P (GPa)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Generated leads vs. known energetic anchors (pool=40k, top-200)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_leads_dp_plane.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig3_leads_dp_plane.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3: {OUT / 'fig3_leads_dp_plane.svg'} (+ .png)")


# ── Figure 4: Self-consistency comparison v1 vs v3.1 ─────────────────────
def fig_self_consistency():
    seeds = ["TNT", "PETN", "TATB", "RDX", "FOX-7", "HMX", "CL-20"]
    v1   = [0.13, 0.46, 0.13, 0.50, 0.04, 0.30, 0.55]
    v3_1 = [1.00, 1.00, 1.00, 0.57, 0.17, 0.30, 0.50]
    x = np.arange(len(seeds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x - w/2, v1,   w, label="LIMO v1 (production)", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, v3_1, w, label="LIMO v3.1 (AR + skip)", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(seeds, rotation=0)
    ax.set_ylabel("encode→decode self-consistency (Tanimoto)")
    ax.set_title("Per-seed reconstruction fidelity: v1 vs v3.1")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="k", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_self_consistency.svg")
    plt.close(fig)
    print(f"  fig4: {OUT / 'fig4_self_consistency.svg'}")


# ── Figure 5: Composite score distribution (pool=40k top 200) ────────────
def fig_score_distribution():
    leads = parse_rerank_md(
        ROOT / "experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_pool40k.md"
    )
    if not leads: return
    scores = [l["score"] for l in leads]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.hist(scores, bins=40, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(scores), color="#d62728", linestyle="--",
               label=f"median = {np.median(scores):.2f}")
    ax.set_xlabel("composite score (lower = better, hits all 4 targets)")
    ax.set_ylabel("# candidates")
    ax.set_title(f"Distribution of top-{len(leads)} composite scores (pool=40k)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_score_dist.svg")
    plt.close(fig)
    print(f"  fig5: {OUT / 'fig5_score_dist.svg'}")


def fig_pareto_v2():
    """Pareto-front view: viability vs performance, color = sensitivity.
    Uses pool=40k_v3b (post-Phase-A) for more candidates and clearer Pareto front."""
    md = ROOT / "experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_guided_pool40k_v3b.md"
    if not md.exists():
        md = ROOT / "experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_pool40k_v2.md"
    if not md.exists(): return
    rows = []
    text = md.read_text(encoding="utf-8")
    for line in text.split("\n"):
        if not line.startswith("| ") or "rank" in line or "---" in line:
            continue
        cells = [c.strip(" `") for c in line.split("|")]
        if len(cells) < 17: continue
        try:
            rows.append({
                "comp":  float(cells[2]),
                "perf":  float(cells[3]),
                "viab":  float(cells[4]),
                "sens":  float(cells[5]),
                "rho":   float(cells[7]),
                "d":     float(cells[9]),
                "p":     float(cells[10]),
                "pareto": "*" in cells[15] or "★" in cells[15],
                "smiles": cells[16] if len(cells) > 16 else "",
            })
        except Exception:
            continue
    if not rows: return
    pareto_pts = sorted([r for r in rows if r["pareto"]], key=lambda r: -r["comp"])
    other_pts  = [r for r in rows if not r["pareto"]]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    # background: filtered (non-Pareto) candidates as light grey dots
    if other_pts:
        ax.scatter([r["perf"] for r in other_pts], [r["viab"] for r in other_pts],
                    color="#bbbbbb", s=22, alpha=0.55, edgecolor="none",
                    label=f"non-Pareto ({len(other_pts)})", zorder=1)
    # Pareto front: stars colored by sensitivity (low=blue=safer, high=red=risky)
    sc = ax.scatter([r["perf"] for r in pareto_pts], [r["viab"] for r in pareto_pts],
                    c=[r["sens"] for r in pareto_pts], cmap="RdYlBu_r", vmin=0, vmax=1,
                    s=110, alpha=0.95, edgecolor="black", linewidth=0.8,
                    marker="*", label=f"Pareto front ({len(pareto_pts)})", zorder=3)
    cb = fig.colorbar(sc, label="sensitivity proxy   (low = safer)")
    # Annotate top-3 Pareto by composite
    top3 = pareto_pts[:3]
    for i, r in enumerate(top3, 1):
        ax.annotate(f"#{i}", (r["perf"], r["viab"]),
                    xytext=(8, -2), textcoords="offset points",
                    fontsize=10, fontweight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.85))
    ax.set_xlabel("saturating performance score (banded ramp)")
    ax.set_ylabel("viability classifier  $P(\\mathrm{EM\\text{-}like})$")
    # Useful ranges given the post-filter density
    pf_min = min((r["perf"] for r in rows), default=0)
    vb_min = min((r["viab"] for r in rows), default=0.5)
    ax.set_xlim(max(0, pf_min - 0.05), 1.03)
    ax.set_ylim(max(0.5, vb_min - 0.02), 1.03)
    ax.set_title("Pareto front of post-Phase-A candidates (pool=40k)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_pareto_v2.svg")
    plt.close(fig)
    print(f"  fig6: {OUT / 'fig6_pareto_v2.svg'}")


def fig_guided_vs_unguided():
    """Figure 7: composite-score distribution, guided vs unguided pool=10k."""
    def parse(path, n_skip_header=4):
        if not Path(path).exists(): return []
        scores = []
        for line in Path(path).read_text(encoding="utf-8").split("\n"):
            if not line.startswith("| ") or "rank" in line or "---" in line:
                continue
            cells = [c.strip(" `") for c in line.split("|")]
            if len(cells) < 12: continue
            try: scores.append(float(cells[2]))
            except Exception: pass
        return scores
    g = parse(ROOT / "experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_guided_pool10k_v2.md")
    u = parse(ROOT / "experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_unguided_pool10k_v2.md")
    if not g or not u: return
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(0, max(max(g), max(u)) * 1.05, 35)
    ax.hist(u, bins=bins, alpha=0.55, label=f"unguided (n={len(u)}, max={max(u):.2f})",
             color="#1f77b4", edgecolor="white")
    ax.hist(g, bins=bins, alpha=0.7, label=f"guided (n={len(g)}, max={max(g):.2f})",
             color="#d62728", edgecolor="white")
    ax.axvline(max(u), color="#1f77b4", linestyle=":", linewidth=1)
    ax.axvline(max(g), color="#d62728", linestyle="--", linewidth=1)
    ax.set_xlabel("composite v2 score (higher = better)")
    ax.set_ylabel("# candidates")
    ax.set_title("Multi-head classifier guidance vs unguided baseline (pool=10k matched compute)")
    # Place legend top-left so it doesn't obscure the right tail of the
    # distribution (which is where the headline guided-shift evidence lives).
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_guided_vs_unguided.svg")
    fig.savefig(OUT / "fig7_guided_vs_unguided.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig7: {OUT / 'fig7_guided_vs_unguided.svg'} (+ .png)")


def fig_head_sweep():
    """Figure 8: per-condition top composite + chemistry-class label."""
    base = ROOT / "experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z"
    conds = [
        ("ungUNG",  "ung", 0, 0,   0,   "#888"),
        ("default", "def", 1.0, 0.3, 0,  "#1f77b4"),
        ("low",     "low", 0.5, 0.2, 0,  "#9467bd"),
        ("viabhi",  "v↑",  1.5, 0.3, 0,  "#d62728"),
        ("senshi",  "s↑",  1.0, 0.6, 0,  "#ff7f0e"),
        ("withSA",  "+SA", 1.0, 0.3, 0.3,"#2ca02c"),
    ]
    points = []
    for name, lbl, v, s, a, color in conds:
        f = base / f"sweep_head_{name}_v2.md"
        if not f.exists(): continue
        for line in f.read_text(encoding="utf-8").split("\n"):
            if not line.startswith("| 1 "): continue
            cells = [c.strip(" `") for c in line.split("|")]
            try:
                comp = float(cells[2]); perf = float(cells[3])
                d = float(cells[9]); p = float(cells[10])
                points.append((lbl, comp, perf, d, p, color))
            except Exception: pass
            break
    if not points: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [p[0] for p in points]
    comps  = [p[1] for p in points]
    perfs  = [p[2] for p in points]
    ds     = [p[3] for p in points]
    cols   = [p[5] for p in points]
    axes[0].bar(labels, comps, color=cols, edgecolor="black", linewidth=0.4)
    axes[0].set_ylabel("top-1 composite v2 (post-strict)")
    axes[0].set_title("Top-1 composite by guidance condition")
    axes[0].set_ylim(0, max(comps)*1.15)
    for i, c in enumerate(comps):
        axes[0].text(i, c + 0.01, f"{c:.2f}", ha="center", fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].scatter(perfs, ds, c=cols, s=160, edgecolor="black", linewidth=0.5)
    for lbl, p, d in zip(labels, perfs, ds):
        axes[1].annotate(lbl, (p, d), xytext=(7, -3), textcoords="offset points", fontsize=10)
    axes[1].set_xlabel("top-1 perf score (saturating-band)")
    axes[1].set_ylabel("top-1 detonation velocity D (km/s)")
    axes[1].set_title("Top-1 perf-vs-D by guidance condition")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_head_sweep.svg")
    plt.close(fig)
    print(f"  fig8: {OUT / 'fig8_head_sweep.svg'}")


def fig_merged_provenance():
    """Bar of how each source pool contributed to the merged top-100."""
    md = ROOT / "experiments/final_merged_topN.md"
    if not md.exists(): return
    src_count = {}
    for line in md.read_text(encoding="utf-8").split("\n")[5:]:
        if not line.startswith("| "): continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 14:
            src = cells[13]
            for s in src.split(","):
                s = s.strip()
                if s: src_count[s] = src_count.get(s, 0) + 1
    items = sorted(src_count.items(), key=lambda x: -x[1])
    if not items: return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names, counts = zip(*items)
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"][:len(names)]
    ax.bar(names, counts, color=colors, edgecolor="black", linewidth=0.4)
    for i, c in enumerate(counts):
        ax.text(i, c + 1, str(c), ha="center", fontsize=10)
    ax.set_ylabel("# of merged top-100 from this source")
    ax.set_title("Merged top-100 paper-ready set by source pool")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_merged_provenance.svg")
    plt.close(fig)
    print(f"  fig9: {OUT / 'fig9_merged_provenance.svg'}")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_pool_scaling()
    fig_cfg_sweep()
    fig_top_leads_plane()
    fig_self_consistency()
    fig_score_distribution()
    fig_pareto_v2()
    fig_guided_vs_unguided()
    fig_head_sweep()
    fig_merged_provenance()
    print("done.")
