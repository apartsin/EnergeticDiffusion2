"""
Audit of 3DCNN-generated property predictions.

Inputs:
  - data/training/diffusion/preds_3dcnn.pt    (3DCNN predictions for 235k molecules)
  - data/training/diffusion/latents.pt        (SMILES + Tier A+B values)
  - data/training/master/labeled_master.csv   (per-SMILES source_dataset for grouping)

Produces eda_3dcnn_audit.html with:
  1. Coverage by source_dataset
  2. Property distributions (per-source overlays)
  3. Outliers per property
  4. Cross-property correlations (3DCNN-internal)
  5. K-J self-consistency (3DCNN's DetoD vs K-J from its own ρ + HOF)
  6. Validation against Tier A experimental ground truth
"""
from __future__ import annotations
import argparse
import html
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


PROPS_3DCNN = ["density", "DetoD", "DetoP", "DetoQ", "DetoT", "DetoV", "HOF_S", "BDE"]
PROP_TO_TIER_NAME = {
    "density": "density",
    "HOF_S":   "heat_of_formation",
    "DetoD":   "detonation_velocity",
    "DetoP":   "detonation_pressure",
}
PROP_UNITS = {
    "density": "g/cm³", "DetoD": "km/s", "DetoP": "GPa", "DetoQ": "kJ/g",
    "DetoT": "kK", "DetoV": "L/kg", "HOF_S": "kJ/mol", "BDE": "kcal/mol",
}
PROP_VALID_RANGE = {
    "density":  (0.5, 3.5),
    "DetoD":    (1.0, 12.0),
    "DetoP":    (0.0, 80.0),
    "DetoQ":    (0.0, 12.0),
    "DetoT":    (0.0, 8.0),
    "DetoV":    (200.0, 1500.0),
    "HOF_S":    (-3000.0, 3000.0),
    "BDE":      (0.0, 500.0),
}


# ── K-J formula (CHNO only) ─────────────────────────────────────────────────
HOF_H2O, HOF_CO2, HOF_CO = -57798.0, -94051.0, -26416.0   # cal/mol
KJ_TO_CAL = 239.006

def kj_dp(smi: str, density: float, hof_kj_mol: float):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mol_h = Chem.AddHs(mol)
    c = {"C":0, "H":0, "N":0, "O":0, "OTHER":0}
    for a in mol_h.GetAtoms():
        s = a.GetSymbol()
        c[s if s in c else "OTHER"] += 1
    if c["OTHER"] > 0: return None
    a, b, g, d = c["C"], c["H"], c["N"], c["O"]
    MW = 12.011*a + 1.008*b + 14.007*g + 15.999*d
    if MW <= 0: return None
    n2 = g / 2.0
    h_rem, o_rem, c_rem = b, d, a
    h2o = min(h_rem/2, o_rem); o_rem -= h2o; h_rem -= 2*h2o
    co2 = min(c_rem, o_rem/2); o_rem -= 2*co2; c_rem -= co2
    co  = min(c_rem, o_rem); o_rem -= co; c_rem -= co
    o2  = max(o_rem/2, 0.0)
    n_gas = n2 + h2o + co2 + co + o2
    if n_gas <= 0: return None
    N = n_gas / MW
    M = (n2*28 + h2o*18 + co2*44 + co*28 + o2*32) / n_gas
    Q = (hof_kj_mol*KJ_TO_CAL - (h2o*HOF_H2O + co2*HOF_CO2 + co*HOF_CO)) / MW
    if Q <= 0: return None
    phi = N * np.sqrt(M * Q)
    return float(1.01 * np.sqrt(phi) * (1 + 1.3*density)), float(1.558 * density**2 * phi)


def fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _esc(s): return html.escape(str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--out",  default="eda_3dcnn_audit.html")
    args = ap.parse_args()
    base = Path(args.base)

    print("Loading data …")
    latents_blob = torch.load(base / "data/training/diffusion/latents.pt",
                               weights_only=False, mmap=True)
    preds_blob   = torch.load(base / "data/training/diffusion/preds_3dcnn.pt",
                               weights_only=False)
    smiles = latents_blob["smiles"]
    valid  = preds_blob["valid"].numpy()
    preds  = preds_blob["predictions"].numpy()
    N      = preds_blob["n"]
    n_valid = int(valid.sum())
    print(f"  N={N:,}  valid 3DCNN preds = {n_valid:,} ({100*n_valid/N:.1f}%)")

    # Tier A+B values from latents (4 props in our standard order)
    tier_props = ["density", "heat_of_formation", "detonation_velocity", "detonation_pressure"]
    tier_vals = latents_blob["values_raw"].numpy()    # (N, 4)
    tier_valid = latents_blob["cond_valid"].numpy()   # (N, 4)
    tier_codes = latents_blob["tiers"].numpy()        # (N, 4)
    stats_orig = latents_blob["stats"]

    # source_dataset per SMILES — load from labeled_master + flag unlabeled rows
    print("  loading source_dataset per SMILES …")
    lm = pd.read_csv(base / "data/training/master/labeled_master.csv",
                     low_memory=False, usecols=["smiles", "source_dataset"])
    lm = lm.drop_duplicates("smiles")
    smi_to_src = dict(zip(lm["smiles"], lm["source_dataset"]))
    sources = np.array([smi_to_src.get(s, "(unlabeled)") for s in smiles])
    print(f"  unique source_dataset values: {len(set(sources))}")

    # Slice to valid 3DCNN predictions
    idx = np.where(valid)[0]
    src_v = sources[idx]
    smi_v = [smiles[i] for i in idx]
    p_v = preds[idx]                         # (n_valid, 8)

    # ── Section 1: coverage by source ────────────────────────────────────────
    print("\n[1/6] Coverage by source …")
    src_counts_total = pd.Series(sources).value_counts()
    src_counts_valid = pd.Series(src_v).value_counts()
    cov = pd.DataFrame({"total": src_counts_total, "valid_3dcnn": src_counts_valid})
    cov["valid_pct"] = (100 * cov["valid_3dcnn"] / cov["total"]).round(1)
    cov = cov.fillna(0).astype({"total": int, "valid_3dcnn": int})
    cov = cov.sort_values("total", ascending=False).head(20)
    cov_html = cov.to_html(classes="kv", border=0)

    # ── Section 2: property distributions overlaid by top sources ───────────
    print("[2/6] Distributions per source …")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    top_sources = src_counts_valid.head(8).index.tolist()
    # always include any energetic-tagged ones
    for s in ("3DCNN", "compendium_energetics_zenodo_3727033", "rnnmgm_ds9", "Dm",
              "det_dataset_08-02-2022", "MDGNN", "(unlabeled)"):
        if s in src_counts_valid.index and s not in top_sources:
            top_sources.append(s)

    fig_dist = make_subplots(rows=2, cols=4,
        subplot_titles=[f"{p} ({PROP_UNITS[p]})" for p in PROPS_3DCNN])
    palette = ["#4f8ef7", "#4caf50", "#ff9800", "#9c27b0", "#ef5350",
               "#00bcd4", "#ffc107", "#8bc34a", "#607d8b", "#e91e63"]
    for j, p in enumerate(PROPS_3DCNN):
        r = j // 4 + 1; c = j % 4 + 1
        rng = PROP_VALID_RANGE[p]
        for k, src in enumerate(top_sources[:8]):
            mask = (src_v == src)
            vals = p_v[mask, j]
            vals = vals[(vals > rng[0]) & (vals < rng[1])]
            if len(vals) < 5:
                continue
            fig_dist.add_trace(go.Histogram(
                x=vals, name=src if j == 0 else None,
                showlegend=(j == 0), histnorm="probability density",
                marker_color=palette[k % len(palette)],
                opacity=0.55, nbinsx=40,
                legendgroup=src), row=r, col=c)
    fig_dist.update_layout(
        barmode="overlay", height=620,
        title_text="3DCNN predictions distribution by source (overlaid, normalized)",
        margin=dict(t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10)))

    # ── Section 3: outliers ─────────────────────────────────────────────────
    print("[3/6] Outliers …")
    outliers_rows = []
    for j, p in enumerate(PROPS_3DCNN):
        rng = PROP_VALID_RANGE[p]
        mask_oor = (p_v[:, j] < rng[0]) | (p_v[:, j] > rng[1])
        n_oor = int(mask_oor.sum())
        # Plus extreme tail (within range, but >3 std from median)
        in_range = p_v[:, j][~mask_oor]
        med = float(np.median(in_range))
        sd = float(np.std(in_range))
        mask_3sd = np.abs(p_v[:, j] - med) > 3 * sd
        n_3sd = int((mask_3sd & ~mask_oor).sum())
        outliers_rows.append({
            "Property":         p,
            "Range expected":   f"[{rng[0]}, {rng[1]}]",
            "Out of range":     n_oor,
            ">3σ from median (within range)": n_3sd,
            "Median":           f"{med:.3f}",
            "Std":              f"{sd:.3f}",
        })
    outliers_df = pd.DataFrame(outliers_rows)
    outliers_html = outliers_df.to_html(classes="kv", border=0, index=False)

    # ── Section 4: cross-property correlations ──────────────────────────────
    print("[4/6] Cross-property correlations …")
    # Take only in-range values
    in_range_mask = np.ones(len(p_v), dtype=bool)
    for j, p in enumerate(PROPS_3DCNN):
        rng = PROP_VALID_RANGE[p]
        in_range_mask &= (p_v[:, j] > rng[0]) & (p_v[:, j] < rng[1])
    p_clean = p_v[in_range_mask]
    print(f"  in-range subset: {len(p_clean):,}")
    corr = pd.DataFrame(p_clean, columns=PROPS_3DCNN).corr().round(3)
    import plotly.express as px
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                          color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          title="3DCNN-predicted property correlations (n={:,})".format(len(p_clean)))
    fig_corr.update_layout(height=480, margin=dict(t=70, b=40))

    # ── Section 5: K-J self-consistency ─────────────────────────────────────
    print("[5/6] K-J self-consistency check …")
    # Compute K-J(ρ_3DCNN, HOF_3DCNN) and compare to DetoD/DetoP
    ji_d = PROPS_3DCNN.index("DetoD")
    ji_p = PROPS_3DCNN.index("DetoP")
    ji_rho = PROPS_3DCNN.index("density")
    ji_hof = PROPS_3DCNN.index("HOF_S")
    sample_size = min(8000, len(p_v))
    rng_idx = np.random.default_rng(42).choice(len(p_v), sample_size, replace=False)

    kj_d_arr, kj_p_arr, cnn_d_arr, cnn_p_arr = [], [], [], []
    for k in rng_idx:
        smi = smi_v[k]
        rho = float(p_v[k, ji_rho])
        hof = float(p_v[k, ji_hof])
        if not (PROP_VALID_RANGE["density"][0] < rho < PROP_VALID_RANGE["density"][1]):
            continue
        kj = kj_dp(smi, rho, hof)
        if kj is None: continue
        kj_d_arr.append(kj[0])
        kj_p_arr.append(kj[1])
        cnn_d_arr.append(float(p_v[k, ji_d]))
        cnn_p_arr.append(float(p_v[k, ji_p]))
    kj_d_arr = np.array(kj_d_arr); kj_p_arr = np.array(kj_p_arr)
    cnn_d_arr = np.array(cnn_d_arr); cnn_p_arr = np.array(cnn_p_arr)
    n_kj = len(kj_d_arr)
    print(f"  K-J vs 3DCNN comparison on n={n_kj:,}")

    if n_kj > 10:
        d_mae = float(np.mean(np.abs(cnn_d_arr - kj_d_arr)))
        p_mae = float(np.mean(np.abs(cnn_p_arr - kj_p_arr)))
        d_corr = float(np.corrcoef(cnn_d_arr, kj_d_arr)[0, 1])
        p_corr = float(np.corrcoef(cnn_p_arr, kj_p_arr)[0, 1])
        d_rel = float(np.mean(np.abs(cnn_d_arr - kj_d_arr) / (cnn_d_arr + 1e-6)))
        p_rel = float(np.mean(np.abs(cnn_p_arr - kj_p_arr) / (cnn_p_arr + 1e-6)))
    else:
        d_mae = p_mae = d_corr = p_corr = d_rel = p_rel = float("nan")

    fig_kj = make_subplots(rows=1, cols=2,
        subplot_titles=[f"DetoD: 3DCNN vs K-J (r={d_corr:.3f}, MAE={d_mae:.2f} km/s)",
                        f"DetoP: 3DCNN vs K-J (r={p_corr:.3f}, MAE={p_mae:.2f} GPa)"])
    fig_kj.add_trace(go.Scatter(x=kj_d_arr, y=cnn_d_arr, mode="markers",
                                  marker=dict(color="#4f8ef7", size=4, opacity=0.4),
                                  showlegend=False, name="D"), row=1, col=1)
    fig_kj.add_trace(go.Scatter(x=[2, 12], y=[2, 12], mode="lines",
                                  line=dict(color="white", dash="dash"),
                                  showlegend=False), row=1, col=1)
    fig_kj.add_trace(go.Scatter(x=kj_p_arr, y=cnn_p_arr, mode="markers",
                                  marker=dict(color="#ff9800", size=4, opacity=0.4),
                                  showlegend=False, name="P"), row=1, col=2)
    fig_kj.add_trace(go.Scatter(x=[0, 50], y=[0, 50], mode="lines",
                                  line=dict(color="white", dash="dash"),
                                  showlegend=False), row=1, col=2)
    fig_kj.update_xaxes(title="K-J prediction (km/s)", row=1, col=1)
    fig_kj.update_yaxes(title="3DCNN prediction (km/s)", row=1, col=1)
    fig_kj.update_xaxes(title="K-J prediction (GPa)", row=1, col=2)
    fig_kj.update_yaxes(title="3DCNN prediction (GPa)", row=1, col=2)
    fig_kj.update_layout(height=460, margin=dict(t=80, b=40))

    # ── Section 6: validation vs Tier A experimental ────────────────────────
    print("[6/6] Validation vs Tier A experimental …")
    val_rows = []
    val_figs = []
    for cnn_name, our_name in PROP_TO_TIER_NAME.items():
        cnn_idx = PROPS_3DCNN.index(cnn_name)
        our_idx = tier_props.index(our_name)
        # rows: have Tier A AND have valid 3DCNN
        is_a = (tier_codes[:, our_idx] == 0)   # tier code 0 = A
        is_v = valid
        m = is_a & is_v
        n_pairs = int(m.sum())
        if n_pairs < 5:
            val_rows.append({"Property": cnn_name, "n": n_pairs, "Tier A MAE": "—", "RMSE": "—", "r": "—", "bias": "—"})
            continue
        x = preds[m, cnn_idx]
        y = tier_vals[m, our_idx]
        diffs = x - y
        mae = float(np.mean(np.abs(diffs)))
        rmse = float(np.sqrt(np.mean(diffs**2)))
        r = float(np.corrcoef(x, y)[0, 1]) if n_pairs > 2 else float("nan")
        bias = float(np.mean(diffs))
        val_rows.append({
            "Property": cnn_name,
            "n": n_pairs,
            "Tier A MAE": f"{mae:.3f}",
            "RMSE": f"{rmse:.3f}",
            "r": f"{r:.3f}",
            "Bias (3DCNN−exp)": f"{bias:+.3f}",
        })
        # Make scatter
        unit = PROP_UNITS[cnn_name]
        f = go.Figure()
        f.add_trace(go.Scatter(x=y, y=x, mode="markers",
                                marker=dict(color="#4f8ef7", size=5, opacity=0.5),
                                name="3DCNN vs experimental"))
        lo = float(min(x.min(), y.min())) * 0.95
        hi = float(max(x.max(), y.max())) * 1.05
        f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="white", dash="dash"),
                                 showlegend=False, name="y=x"))
        f.update_layout(
            title=f"{cnn_name} — 3DCNN vs Tier A experimental (n={n_pairs:,}, "
                  f"MAE={mae:.3f} {unit}, r={r:.3f})",
            xaxis_title=f"Tier A experimental ({unit})",
            yaxis_title=f"3DCNN prediction ({unit})",
            height=380, margin=dict(t=60, b=40))
        val_figs.append(f)
    val_df = pd.DataFrame(val_rows)
    val_html = val_df.to_html(classes="kv", border=0, index=False)

    # ── HTML assembly ───────────────────────────────────────────────────────
    print("\nBuilding HTML …")
    out_html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>3DCNN prediction audit</title>
<style>
 :root {{ --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3e;
          --text: #e0e4f0; --muted: #8892b0; --accent: #4f8ef7;
          --green: #4caf50; --orange: #ff9800; --red: #ef5350; }}
 * {{ box-sizing: border-box; margin: 0; padding: 0; }}
 body {{ background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', system-ui, sans-serif;
         font-size: 14px; line-height: 1.6; }}
 header {{ background: linear-gradient(135deg, #1a1d27, #0d1b3e);
           padding: 24px 36px; border-bottom: 1px solid var(--border); }}
 header h1 {{ font-size: 1.7rem; color: #fff; }}
 header p  {{ color: var(--muted); margin-top: 4px; }}
 main {{ max-width: 1300px; margin: 0 auto; padding: 32px 36px 80px; }}
 section {{ margin-bottom: 48px; }}
 section h2 {{ font-size: 1.2rem; color: #fff;
                border-left: 3px solid var(--accent); padding-left: 10px;
                margin-bottom: 14px; }}
 table.kv {{ width: 100%; border-collapse: collapse; background: var(--surface);
              border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
              font-size: 0.86rem; }}
 table.kv th, table.kv td {{ padding: 6px 14px;
              border-bottom: 1px solid var(--border); }}
 table.kv th {{ background: #1e2235; color: var(--muted);
                 text-align: left; font-size: 0.78rem; }}
 .audit-box {{ border-radius: 8px; padding: 14px 20px; margin: 12px 0 18px;
                border-left: 4px solid; line-height: 1.7; }}
 .audit-box.info    {{ background: #0d1929; border-color: #64b5f6; }}
 .audit-box.warning {{ background: #1a1500; border-color: var(--orange); }}
 .audit-box.error   {{ background: #1f0f0f; border-color: var(--red); }}
</style></head><body>
<header>
  <h1>3DCNN prediction audit</h1>
  <p>EMDP smoke_model (Uni-Mol v1, 84M params, 8-target multi-task regressor) —
     {n_valid:,} of {N:,} molecules predicted ({100*n_valid/N:.1f}% coverage).</p>
</header>
<main>

<section>
  <h2>1. Coverage by source_dataset</h2>
  <p>Top 20 sources by row count, with 3DCNN prediction validity rate:</p>
  {cov_html}
</section>

<section>
  <h2>2. Distribution of 3DCNN predictions by source</h2>
  <div class="audit-box info">
    Histograms (probability density) overlaid for the top 8 sources by row count.
    Use to spot source-specific systematic biases — e.g. if 3DCNN-source rows
    cluster differently from de-novo-generated rows.
  </div>
  {fig_html(fig_dist)}
</section>

<section>
  <h2>3. Outliers</h2>
  <div class="audit-box warning">
    <strong>Out of range</strong> = predictions outside physically-plausible bounds
    (these flag model failures or non-CHNO molecules);
    <strong>&gt;3σ from median</strong> = within range but extreme tail.
  </div>
  {outliers_html}
</section>

<section>
  <h2>4. Cross-property correlations</h2>
  <p>Pearson correlations among 3DCNN's 8 outputs (in-range subset).
  Strong positive ρ-D-P expected from physics; weak or wrong-signed correlation
  indicates model deficiency.</p>
  {fig_html(fig_corr)}
</section>

<section>
  <h2>5. K-J self-consistency: 3DCNN's D, P vs K-J(its own ρ, HOF)</h2>
  <div class="audit-box info">
    For each molecule, compute K-J(ρ_3DCNN, HOF_3DCNN, formula) and compare to
    3DCNN's direct DetoD / DetoP outputs. <strong>Low MAE / high correlation</strong>
    indicates 3DCNN's D/P are physically consistent with its own ρ/HOF —
    a self-consistency check, not an absolute-quality metric.
    <br><br>
    Sample size: {n_kj:,} molecules. D MAE = <strong>{d_mae:.2f} km/s</strong>
    (mean rel. err {100*d_rel:.1f}%, r = {d_corr:.3f});
    P MAE = <strong>{p_mae:.2f} GPa</strong>
    (mean rel. err {100*p_rel:.1f}%, r = {p_corr:.3f}).
  </div>
  {fig_html(fig_kj)}
</section>

<section>
  <h2>6. Validation vs Tier A experimental ground truth</h2>
  <div class="audit-box info">
    Per property: 3DCNN prediction compared to actual experimental Tier A value
    (the same molecule appears in both labeled_master and the 3DCNN dataset).
    This is the <strong>real</strong> accuracy of 3DCNN on our energetic distribution.
  </div>
  {val_html}
  {''.join(fig_html(f) for f in val_figs)}
</section>

</main></body></html>
"""
    out_path = base / args.out
    out_path.write_text(out_html, encoding="utf-8")
    print(f"\nWrote → {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
