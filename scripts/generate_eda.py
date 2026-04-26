"""
Exploratory Data Analysis generator for EnergeticDiffusion2.
Produces a self-contained HTML report with Plotly figures and styled tables.
"""
import json
import os
import textwrap
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

BASE = "E:/Projects/EnergeticDiffusion2"
OUT  = f"{BASE}/eda_report.html"

# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv(path, **kw):
    full = os.path.join(BASE, path)
    print(f"  loading {path} …")
    return pd.read_csv(full, low_memory=False, **kw)

def fig_to_html(fig, div_id=""):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       div_id=div_id or None, config={"responsive": True})

def table_html(df, title="", max_rows=None, classes=""):
    if max_rows:
        df = df.head(max_rows)
    th = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, r in df.iterrows():
        rows += "<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
    label = f"<p class='tbl-title'>{title}</p>" if title else ""
    return f"{label}<div class='tbl-wrap {classes}'><table><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>"

def section(title, content, sec_id=""):
    sid = f' id="{sec_id}"' if sec_id else ""
    return f"""
<section{sid}>
  <h2>{title}</h2>
  {content}
</section>"""

def card(label, value, sub=""):
    return f"""<div class="card"><div class="card-val">{value}</div>
<div class="card-label">{label}</div>
{'<div class="card-sub">'+sub+'</div>' if sub else ''}</div>"""

# ── load data ─────────────────────────────────────────────────────────────────

print("Loading datasets …")

lm = load_csv("data/training/master/labeled_master.csv")
# unlabeled: only read the source_dataset column to save memory
um_src = load_csv("data/training/master/unlabeled_master.csv",
                  usecols=["source_dataset"])
um = load_csv("data/training/master/unlabeled_master.csv",
              usecols=["smiles", "source_dataset"])

emdb = load_csv("data/raw/energetic_external/EMDB_public/emdb_v21_molecules_pubchem.csv")
emdp_tr = load_csv("data/raw/energetic_external/EMDP/Data/train_set.csv")
emdp_te = load_csv("data/raw/energetic_external/EMDP/Data/test_set.csv")

manifest = load_csv("data/training/metadata/data_split_manifest.csv")

with open(f"{BASE}/data/training/metadata/property_normalization.json") as f:
    norm_json = json.load(f)

print("All data loaded.\n")

# ── derived variables ─────────────────────────────────────────────────────────

PROPS = ["density", "heat_of_formation", "detonation_velocity",
         "detonation_pressure"]
# explosion_heat dropped from EDA (redundant — derivable from HOF + formula
# via Kamlet-Jacobs product balance). Master CSV still retains the column.

PROP_LABELS = {
    "density":              "Density (g/cm³)",
    "heat_of_formation":    "Heat of Formation (kJ/mol)",
    "detonation_velocity":  "Detonation Velocity (km/s)",
    "detonation_pressure":  "Detonation Pressure (GPa)",
    "explosion_heat":       "Explosion Heat (MJ/kg)",
}

COLORS = px.colors.qualitative.Plotly

# label_source_type is now clean in the CSV — use it directly
lm["label_source_type_refined"] = lm["label_source_type"]

SOURCE_TYPES   = ["compiled_observed", "model_predicted", "kj_calculated", "qsar_predicted"]
SOURCE_COLORS  = {"compiled_observed": "#4CAF50",
                  "model_predicted":   "#2196F3",
                  "kj_calculated":     "#FF9800",
                  "qsar_predicted":    "#9C27B0"}
SOURCE_LABELS  = {"compiled_observed": "Experimental (observed)",
                  "model_predicted":   "ML model predicted",
                  "kj_calculated":     "Kamlet-Jacobs calculated",
                  "qsar_predicted":    "QSAR predicted"}

# ── coverage ──────────────────────────────────────────────────────────────────

cov = {p: lm[p].notna().sum() for p in PROPS}
cov_pct = {p: round(100 * v / len(lm), 1) for p, v in cov.items()}

# ── 1. EXECUTIVE SUMMARY ──────────────────────────────────────────────────────

total_raw_bytes   = 5_147_814_512
total_train_bytes = 1_752_705_899

summary_cards = "".join([
    card("Unlabeled molecules", f"{len(um_src)+len(lm):,}", "diffusion pretraining"),
    card("Labeled molecules",   f"{len(lm):,}",  "with ≥1 property"),
    card("Target properties",   "5",             "density · HOF · det-vel · det-P · expl-heat"),
    card("Raw data sources",    "6",             "PubChem · ChEMBL · QM9 · GuacaMol · EMDB · EMDP"),
    card("Raw data size",       "4.79 GiB",      f"{total_raw_bytes:,} bytes"),
    card("Training data size",  "1.63 GiB",      f"{total_train_bytes:,} bytes"),
])
sec1 = f"<div class='cards'>{summary_cards}</div>"

# ── 2. RAW DATA INVENTORY ─────────────────────────────────────────────────────

raw_inv = pd.DataFrame([
    ("benchmarks",         1,   61_841_218,
     "~1.27M",  "None (SMILES only)",
     "Generic (drug-like)", "–",
     "GuacaMol v1 benchmark set; diverse drug-like molecules for generative pretraining"),
    ("energetic",        125,  119_517_377,
     "~132K",   "None (code/descriptors)",
     "Energetic (code + descriptors)", "–",
     "DeepEMs research repo: t-SNE descriptor embeddings, MD simulation scripts, NNP model code"),
    ("energetic_external", 976, 523_718_535,
     "~16K+", "density · HOF · det-vel · det-P · det-heat · BDE · sensitivity",
     "Energetic", "Mixed: experimental (EMDB) + GNN/3DCNN model predictions (EMDP/MDGNN)",
     "EMDB (~656 exp.), EMDP train/test (~2K exp.), 3DCNN/MDGNN predictions (~20K ea.), "
     "ML impact sensitivity notebooks, NIST-CAMEO compatibility, de novo RL/TL sampling"),
    ("general",            4, 4_397_909_915,
     "~700K used", "None (structure only)",
     "Generic (diverse)", "–",
     "PubChem CID-SMILES (~500K cap), ChEMBL 36 SQLite + SDF (~200K cap), PubChem identifiers"),
    ("quantum",            1,   44_827_467,
     "~132K",  "QM9 quantum properties (not energetic targets)",
     "Small organic", "QM/DFT (B3LYP/6-31G*)",
     "QM9 dataset: 134K small CHNO molecules with 12 quantum mechanical properties"),
], columns=["Subdirectory", "Files", "Size (bytes)",
            "~Compounds", "Properties available",
            "Domain", "Label type", "Notes"])

raw_inv["Size (MiB)"] = (raw_inv["Size (bytes)"] / 1024**2).round(1)

fig_raw = px.bar(raw_inv[raw_inv["Size (bytes)"] > 0],
                 x="Subdirectory", y="Size (MiB)",
                 color="Subdirectory",
                 color_discrete_sequence=COLORS,
                 title="Raw Data Size by Subdirectory",
                 text_auto=".1f")
fig_raw.update_layout(showlegend=False, height=380, margin=dict(t=50,b=40))
fig_raw.update_traces(textposition="outside")

sec2 = (table_html(
    raw_inv[["Subdirectory","~Compounds","Properties available","Domain","Label type","Size (MiB)","Notes"]],
    "Raw data subdirectories — compound counts, available properties, label origin") +
    fig_to_html(fig_raw, "fig_raw"))

# ── 3. MASTER DATASET OVERVIEW ────────────────────────────────────────────────

master_overview = pd.DataFrame([
    ("Unlabeled master",  "data/training/master/unlabeled_master.csv",
     f"{len(um_src)+len(lm):,}",
     "molecule_id · smiles · selfies · source_dataset · source_path + 6 more",
     "Diffusion pretraining — all generic + energetic molecules without numeric property labels",
     "699,969"),
    ("Labeled master",   "data/training/master/labeled_master.csv",
     f"{len(lm):,}",
     "molecule_id · smiles · selfies · 5 numeric targets · structural features · provenance (40 cols)",
     "Property prediction and guided diffusion — all molecules with ≥1 numeric property label",
     "65,960"),
], columns=["File", "Path", "Rows", "Key columns", "Role", "Confirmed rows"])

# Labeled master label coverage bar
fam_rows = pd.DataFrame([
    ("Unlabeled (pretraining)", len(um_src) + len(lm) - len(lm), "#607D8B"),
    ("Labeled (≥1 property)",   len(lm),                         "#2196F3"),
    ("4-target complete",        33_348,                          "#4CAF50"),
    ("5-target complete",        20_409,                          "#FF9800"),
])
fam_rows.columns = ["Dataset", "Molecules", "Color"]

fig_master = go.Figure(go.Bar(
    x=fam_rows["Molecules"], y=fam_rows["Dataset"],
    orientation="h",
    marker_color=fam_rows["Color"],
    text=fam_rows["Molecules"].map("{:,}".format),
    textposition="outside",
))
fig_master.update_layout(
    height=320, margin=dict(t=60, l=200, b=40, r=80),
    title="Master dataset molecule counts (before any train/test split)",
    xaxis_title="Molecules",
)

# Target statistics by the rigorous per-property TIER (not row-level label):
#   A = experimental / literature
#   B = quantum or physics simulation (DFT, EXPLO5)
#   C = Kamlet-Jacobs empirical formula and group-contribution
#   D = ML / QSAR / data-driven regression (3DCNN, MDGNN, q-RASPR, denovo)
prop_src_rows = []
for p in PROPS:
    tc = f"{p}_tier"
    nA = int(((lm[tc] == "A") & lm[p].notna()).sum())
    nB = int(((lm[tc] == "B") & lm[p].notna()).sum())
    nC = int(((lm[tc] == "C") & lm[p].notna()).sum())
    nD = int(((lm[tc] == "D") & lm[p].notna()).sum())
    prop_src_rows.append({
        "Property":           PROP_LABELS[p],
        "A (experimental)":   f"{nA:,}",
        "B (simulation)":     f"{nB:,}",
        "C (empirical formula)": f"{nC:,}",
        "D (ML / QSAR)":      f"{nD:,}",
        "Total labeled":      f"{nA+nB+nC+nD:,}",
    })
prop_src_df = pd.DataFrame(prop_src_rows)

# Stacked-within-tier, property-faceted bar:
#   one subplot per property (4 subplots, horizontally)
#   x-axis of each subplot = Tier A / B / C / D
#   each tier bar is STACKED by contributing source_dataset
# This shows, per property, which methods contribute to each tier.
TIER_BASE_COLOR = {"A": "#4CAF50", "B": "#2196F3", "C": "#FF9800", "D": "#9C27B0"}
TIER_PALETTE = {
    "A": ["#1B5E20","#2E7D32","#388E3C","#43A047","#4CAF50","#66BB6A",
          "#81C784","#A5D6A7","#C8E6C9","#689F38","#7CB342","#9CCC65"],
    "B": ["#0D47A1","#1565C0","#1976D2","#1E88E5","#2196F3","#42A5F5","#64B5F6"],
    "C": ["#E65100","#EF6C00","#F57C00","#FB8C00","#FF9800","#FFA726","#FFB74D"],
    "D": ["#4A148C","#6A1B9A","#7B1FA2","#8E24AA","#9C27B0","#AB47BC",
          "#BA68C8","#CE93D8","#7E57C2","#5E35B1","#673AB7"],
}

# Map source_dataset → human-readable method name.
# (source_dataset, property) -> method label (some sources produce multiple methods per property)
def _method_label(source_dataset, property_name):
    s = source_dataset
    if s == "3DCNN":
        return "3DCNN (DFT-surrogate ML)"
    if s == "cm4c01978_si_001":
        if property_name == "density":
            return "XRD crystallography (CSD)"
        if property_name == "heat_of_formation":
            return "Benson-style group contribution"
        return "Kamlet-Jacobs formula"
    if s == "kj_from_explo5_hof" or s == "kj_row_impute":
        return "Kamlet-Jacobs formula (row-imputed)"
    if s in ("train_set", "test_set"):
        if property_name == "density":
            return "XRD crystallography (EMDP)"
        return "EXPLO5 thermochemistry code"
    if s == "q-RASPR":
        return "q-RASPR QSAR"
    if s == "MDGNN":
        return "MDGNN (generative + predictor)"
    if s == "generation":
        return "Generative model + predictor"
    if s and s.startswith("denovo_sampling_rl"):
        return "De-novo RL generator + predictor"
    if s and s.startswith("denovo_sampling_tl"):
        return "De-novo TL generator + predictor"
    # All experimental literature compilations → single unified segment
    LIT_SOURCES = {
        "kroonblawd_2025_jctc", "5039", "det_dataset_08-02-2022", "Dm",
        "combined_data", "Huang_Massa_data_with_all_SMILES",
        "emdb_v21_molecules_pubchem", "nist_cameo_enrichment", "CHNOClF_dataset",
    }
    if s in LIT_SOURCES:
        return "Experimental literature compilations"
    return s if isinstance(s, str) else "unknown"

fig_prop_src = make_subplots(rows=1, cols=len(PROPS),
                              subplot_titles=[PROP_LABELS[p] for p in PROPS],
                              shared_yaxes=False, horizontal_spacing=0.07)
legend_added = set()
for col_idx, p in enumerate(PROPS, start=1):
    tc, sc = f"{p}_tier", f"{p}_source_dataset"
    for tier_code in ("A", "B", "C", "D"):
        sub = lm[(lm[tc] == tier_code) & lm[p].notna()].copy()
        if sub.empty:
            continue
        sub["_method"] = sub[sc].apply(lambda s: _method_label(s, p))
        counts = sub["_method"].value_counts()
        palette = TIER_PALETTE[tier_code]
        for i, (method, n) in enumerate(counts.items()):
            key = f"{tier_code}::{method}"
            show = key not in legend_added
            legend_added.add(key)
            fig_prop_src.add_trace(go.Bar(
                name=method,
                x=[f"Tier {tier_code}"],
                y=[int(n)],
                marker_color=palette[i % len(palette)],
                marker_line=dict(color=TIER_BASE_COLOR[tier_code], width=0.8),
                legendgroup=f"tier_{tier_code}",
                legendgrouptitle_text=f"Tier {tier_code}",
                showlegend=show,
                text=[f"{int(n):,}"] if n >= 300 else [""],
                textposition="inside",
                insidetextfont=dict(size=9, color="white"),
                hovertemplate=f"<b>Tier {tier_code} — {method}</b><br>"
                              f"{PROP_LABELS[p]}: %{{y:,}}<extra></extra>",
            ), row=1, col=col_idx)
fig_prop_src.update_layout(
    barmode="stack",
    height=520, margin=dict(t=90, b=60, r=260),
    title=dict(text="Property labels by reliability tier — stacked by contributing source",
               y=0.98, yanchor="top"),
    legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01,
                font=dict(size=9), groupclick="toggleitem", traceorder="grouped"),
)
for col_idx in range(1, len(PROPS) + 1):
    fig_prop_src.update_yaxes(title_text="Molecules" if col_idx == 1 else "",
                              row=1, col=col_idx)

sec3 = (table_html(master_overview, "Master files — all data before any train/test split") +
        fig_to_html(fig_master, "fig_master") +
        "<h3>Property Labels by Source Type</h3>" +
        fig_to_html(fig_prop_src, "fig_prop_src") +
        table_html(prop_src_df, "Labels per property: experimental vs predicted/calculated"))

# ── 4. LABEL COVERAGE ────────────────────────────────────────────────────────

cov_df = pd.DataFrame({
    "Property":  [PROP_LABELS[p] for p in PROPS],
    "Labeled":   [cov[p] for p in PROPS],
    "Missing":   [len(lm) - cov[p] for p in PROPS],
    "Coverage %":[f"{cov_pct[p]}%" for p in PROPS],
})

fig_cov = go.Figure()
fig_cov.add_bar(name="Labeled",
                x=cov_df["Property"], y=cov_df["Labeled"],
                marker_color="#2196F3")
fig_cov.add_bar(name="Missing",
                x=cov_df["Property"], y=cov_df["Missing"],
                marker_color="#EF9A9A")
fig_cov.update_layout(barmode="stack", height=400,
                      title="Label Coverage per Property (65,960 molecules)",
                      yaxis_title="Molecules", margin=dict(t=50,b=50))

# per-molecule coverage count — side-by-side by source type
lm["n_labels"] = lm[PROPS].notna().sum(axis=1)

fig_nlabels = go.Figure()
all_counts = lm["n_labels"].value_counts().sort_index()
fig_nlabels.add_bar(
    x=all_counts.index, y=all_counts.values,
    name="All sources", marker_color="#607D8B", opacity=0.5,
    showlegend=True)
for src in SOURCE_TYPES:
    sub = lm[lm["label_source_type_refined"] == src]["n_labels"].value_counts().sort_index()
    if sub.empty:
        continue
    fig_nlabels.add_bar(
        x=sub.index, y=sub.values,
        name=SOURCE_LABELS[src],
        marker_color=SOURCE_COLORS[src],
        opacity=0.85,
        showlegend=True)

# Experimental + K-J imputed: for each compiled_observed molecule,
# fill missing properties from a kj_calculated row with the same SMILES
_exp_sub = lm[lm["label_source_type_refined"] == "compiled_observed"].copy()
_kj_sub  = lm[lm["label_source_type_refined"] == "kj_calculated"][["smiles"] + PROPS].copy()
_kj_sub  = _kj_sub.rename(columns={p: f"_kj_{p}" for p in PROPS})
_kj_sub  = _kj_sub.drop_duplicates("smiles")
_exp_kj  = _exp_sub.merge(_kj_sub, on="smiles", how="left")
for _p in PROPS:
    _exp_kj[_p] = _exp_kj[_p].fillna(_exp_kj[f"_kj_{_p}"])
_imputed_counts = _exp_kj[PROPS].notna().sum(axis=1).value_counts().sort_index()
fig_nlabels.add_bar(
    x=_imputed_counts.index, y=_imputed_counts.values,
    name="Experimental + K-J imputed",
    marker_color="#00BCD4",
    opacity=0.85,
    showlegend=True)

# All K-J imputed: group by SMILES across ALL sources (union of labels),
# then apply K-J to fill missing D from (ρ,P) and missing P from (ρ,D).
_KJ_K1_e, _KJ_K2_e = 1.01, 1.558
_uni = lm.groupby("smiles")[PROPS].agg(
    lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
_rho_u = _uni["density"]
_D_u   = _uni["detonation_velocity"]
_P_u   = _uni["detonation_pressure"]
_fill_P = _P_u.isna() & _D_u.notna() & _rho_u.notna() & (_rho_u > 0.3)
_uni.loc[_fill_P, "detonation_pressure"] = (
    _KJ_K2_e * _rho_u[_fill_P]**2 *
    (_D_u[_fill_P] / (_KJ_K1_e * (1 + 1.3 * _rho_u[_fill_P])))**2)
_fill_D = _D_u.isna() & _P_u.notna() & _rho_u.notna() & (_rho_u > 0.3)
_uni.loc[_fill_D, "detonation_velocity"] = (
    _KJ_K1_e * (1 + 1.3 * _rho_u[_fill_D]) *
    np.sqrt(np.maximum(_P_u[_fill_D] / (_KJ_K2_e * _rho_u[_fill_D]**2), 0)))
_uni_counts = _uni[PROPS].notna().sum(axis=1).value_counts().sort_index()
fig_nlabels.add_bar(
    x=_uni_counts.index, y=_uni_counts.values,
    name="All K-J imputed (union by SMILES)",
    marker_color="#FFD54F",
    opacity=0.90,
    showlegend=True)

# Annotation: highlight 4 and 5 property counts after full imputation
_full5 = int(_uni_counts.get(5, 0))
_full4 = int(_uni_counts.get(4, 0))
_note_label = (f"After full K-J imputation: {_full5:,} molecules have all 5 properties, "
               f"{_full4:,} have 4 of 5")

fig_nlabels.update_layout(
    barmode="group",
    title="Distribution of Label Completeness per Molecule — by source type",
    xaxis_title="Number of properties labeled (out of 5)",
    yaxis_title="Molecules",
    height=420, margin=dict(t=100, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    xaxis=dict(tickmode="linear", dtick=1),
)

# ── 3-tier reliability summary ────────────────────────────────────────────────
# Per-property tiers (rigorous): each <prop>_tier column is populated by
# assign_per_property_tiers.py — respects that train_set density is XRD
# (Tier A) while its D/P/Q come from EXPLO5 (Tier B), etc.
tier_rows = []
for p in PROPS:
    tc = f"{p}_tier"
    if tc not in lm.columns:
        # fallback: row-level tier
        tier_rows.append({
            "Property": PROP_LABELS[p],
            "Tier A only":       f"{lm[(lm['tier']=='A') & lm[p].notna()].shape[0]:,}",
            "A + B (cumulative)":f"{lm[lm['tier'].isin(['A','B']) & lm[p].notna()].shape[0]:,}",
            "A + B + C":         f"{lm[lm['tier'].isin(['A','B','C']) & lm[p].notna()].shape[0]:,}",
            "Excluded (ML/QSAR)":f"{lm[lm['tier'].isna() & lm[p].notna()].shape[0]:,}",
        })
    else:
        a  = (lm[tc] == "A").sum()
        b  = (lm[tc] == "B").sum()
        c  = (lm[tc] == "C").sum()
        d  = (lm[tc] == "D").sum()
        tier_rows.append({
            "Property":                 PROP_LABELS[p],
            "Tier A (experimental)":    f"{a:,}",
            "Tier B (simulation)":      f"{b:,}",
            "Tier C (K-J / group contribution)": f"{c:,}",
            "Tier D (ML / QSAR / 3DCNN)":        f"{d:,}",
            "A+B+C+D total":            f"{a+b+c+d:,}",
        })
tier_prop_df = pd.DataFrame(tier_rows)

# Per-property bar chart (stacked): A / B / C / D per property
fig_tier = go.Figure()
if all(f"{p}_tier" in lm.columns for p in PROPS):
    tier_layers = [("Tier A (experimental)",     "#4CAF50", "A"),
                   ("Tier B (quantum/EXPLO5)",   "#2196F3", "B"),
                   ("Tier C (K-J / group contribution)", "#FF9800", "C"),
                   ("Tier D (ML / QSAR / 3DCNN)",        "#9E9E9E", "D")]
    for name, color, code in tier_layers:
        ys = []
        for p in PROPS:
            tc = f"{p}_tier"
            y = int((lm[tc] == code).sum())
            ys.append(y)
        fig_tier.add_bar(
            x=[PROP_LABELS[p] for p in PROPS], y=ys,
            name=name, marker_color=color,
            text=[f"{v:,}" for v in ys], textposition="inside")
    fig_tier.update_layout(
        barmode="stack",
        title=dict(text="Per-property coverage by reliability tier", y=0.98, yanchor="top"),
        yaxis_title="Molecules", height=440, margin=dict(t=100, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10)))

tier_prop_note = """<div class="audit-box info">
<strong>Why per-property tiers:</strong> the row-level <code>tier</code> tag is coarse —
a row tagged Tier A may have an experimentally measured density but a K-J-formula-derived
pressure (cross-source backfill). Per-property tiers are derived from the
<code>*_source_dataset</code> and <code>*_source_type</code> columns using the same rule set
as <code>assign_per_property_tiers.py</code>. These are the <em>rigorous</em> counts to use
when selecting training rows for a specific regression target.
</div>"""

tier_overview = pd.DataFrame([
    ("A — experimental / literature",
     f"{(lm['tier']=='A').sum():,}",
     "compiled_observed across all experimental literature sources; cm4c01978 density is XRD-measured",
     "High — peer-reviewed measurements"),
    ("B — quantum / physics simulation",
     f"{(lm['tier']=='B').sum():,}",
     "EXPLO5 thermochemistry (train_set/test_set D/P/Q, HOF). True DFT would live here too.",
     "Medium-high — physics-based, ≤5% typical error"),
    ("C — Empirical formulas (K-J / group-contribution)",
     f"{(lm['tier']=='C').sum():,}",
     "K-J formula for D/P (cm4c01978_si_001); Benson-style group-contribution for HOF; Girolami-style for density if ever used.",
     "Medium — ±10–20% D, ±20–30% P, ±5-10 kcal/mol HOF"),
    ("D — Data-driven ML / QSAR / 3DCNN",
     f"{(lm['tier']=='D').sum():,}",
     "3DCNN DFT-surrogate CNN; MDGNN + de-novo RL/TL predictors; q-RASPR QSAR; any generator's internal property head.",
     "Lower — bounded by training-set quality; can fail arbitrarily off-distribution"),
], columns=["Tier","Molecules (row-level)","Included sources","Reliability"])

tier_note = """<div class="audit-box info">
<strong>4-tier reliability model:</strong> labels are grouped by the physical trustworthiness
of the method that produced them, from most to least reliable.
</div>"""

tier_details_df = pd.DataFrame([
    ("A", "Experimental / literature",
     "Measured values: crystallography (XRD), calorimetry (DSC), detonation tests, pycnometry, drop-weight IS tests. Compiled from peer-reviewed publications.",
     "det_dataset_08-02-2022, Dm, combined_data, Huang_Massa, emdb_v21, 5039 (density), cm4c01978 (density only; it's XRD from CSD), train_set / test_set (density only — XRD), nist_cameo, Kroonblawd VOD/HOF (exp columns).",
     "<0.5% ρ, <1% HOF/D/P (reference quality)"),

    ("B", "Simulation (quantum / physics)",
     "Numerical solution of physics equations: DFT (B3LYP, M06, CCSD(T)), semi-empirical xTB, and thermochemistry codes that solve the Chapman-Jouguet equations (EXPLO5, CHEETAH, TIGER).",
     "train_set / test_set (D, P, Q, HOF from EXPLO5). No true DFT in master yet (would live here once batch xTB or Gaussian runs are ingested).",
     "3-5% D, 5-10% P, ±2-3 kcal/mol HOF (B3LYP)"),

    ("C", "Empirical formulas (K-J / group-contribution)",
     "Deterministic algebraic formulas. Physics-derived but approximate: Kamlet-Jacobs (D, P from ρ, HOF, formula); Benson group-contribution (HOF from atom/group increments); Girolami / Stine (density from atomic volumes). Not simulations — no numerical ODE/PDE solve.",
     "cm4c01978 (D, P from K-J; HOF from Benson-style group contribution — NOT from K-J which doesn't produce HOF). Any future kj_row_impute entries.",
     "10-20% D, 20-30% P, ±5-10 kcal/mol HOF, ~5% density"),

    ("D", "Data-driven ML / QSAR / 3DCNN",
     "Statistical regression fit to training data. Includes: neural ML surrogates (3DCNN trained on DFT targets, MDGNN, de-novo predictors), classical QSAR (q-RASPR quasi-SMILES regression), and any generator's integrated property head. No physics basis — trained to fit labels.",
     "3DCNN (26,254 DFT-surrogate rows), MDGNN, denovo_sampling_rl / tl, generation, q-RASPR.",
     "3-6% in-distribution; UNBOUNDED off-distribution"),
], columns=["Tier", "Name", "Method", "Sources in pipeline", "Typical accuracy"])

sec4 = (table_html(cov_df, "Per-property label coverage") +
        fig_to_html(fig_cov,     "fig_cov") +
        fig_to_html(fig_nlabels, "fig_nlabels") +
        f"<div class='audit-box info'><strong>K-J imputation potential:</strong> {_note_label}. "
        f"(Union by SMILES across sources, then D↔P filled from density via K-J.)</div>" +
        "<h3>Four-tier reliability labeling</h3>" +
        tier_note +
        table_html(tier_details_df, "Tier definitions — what each tier includes") +
        table_html(tier_overview, "Reliability tiers for the labeled master (row-level)") +
        "<h3>Per-property tiers (rigorous, from per-property provenance)</h3>" +
        tier_prop_note +
        table_html(tier_prop_df,  "Per-property coverage by reliability tier") +
        fig_to_html(fig_tier, "fig_tier"))

# ── 5. SOURCE DATASET DISTRIBUTION ───────────────────────────────────────────

SOURCE_NAME_MAP = {
    "3DCNN":                                    "EMDP — 3DCNN model predictions",
    "cm4c01978_si_001":                         "Chem.Mater. 2024 — crystal X-ray + K-J calcs",
    "generation":                               "RNNMGM — RNN generative sampling",
    "denovo_sampling_rl.predict.0_filtered":    "EMDP — de novo RL sampling",
    "denovo_sampling_tl.predict.0_filtered":    "EMDP — de novo TL sampling",
    "5039":                                     "EMDP — MDGNN predictions (5039 set)",
    "q-RASPR":                                  "EMDP — q-RASPR QSAR predictions",
    "train_set":                                "EMDP — upstream experimental train set",
    "det_dataset_08-02-2022":                   "RSC 2022 — detonation dataset",
    "Dm":                                       "RNNMGM — Dm reference set",
    "test_set":                                 "EMDP — upstream experimental test set",
    "emdb_v21_molecules_pubchem":               "EMDB v2.1 — experimental database",
    "combined_data":                            "ML-EM Notebooks — combined literature",
    "CHNOClF_dataset":                          "MultiTask EM — CHNO/ClF dataset",
    "Huang_Massa_data_with_all_SMILES":         "Huang & Massa — literature dataset",
}

src_counts = lm["source_dataset"].value_counts().reset_index()
src_counts.columns = ["source_dataset", "count"]
src_top = src_counts.head(15).copy()
src_top["label"] = src_top["source_dataset"].map(SOURCE_NAME_MAP).fillna(src_top["source_dataset"])

fig_src = px.bar(src_top, x="count", y="label", orientation="h",
                 color="count", color_continuous_scale="Blues",
                 title="Top 15 Source Datasets (labeled molecules)",
                 labels={"count": "Molecules", "label": "Source"},
                 custom_data=["source_dataset"])
fig_src.update_traces(
    hovertemplate="<b>%{y}</b><br>Raw key: %{customdata[0]}<br>Molecules: %{x:,}<extra></extra>")
fig_src.update_layout(height=520, margin=dict(t=50, l=340, b=40),
                      coloraxis_showscale=False, yaxis=dict(autorange="reversed"))

# legend table: raw key → readable name → label type
src_legend = src_top.merge(
    lm.groupby("source_dataset")["label_source_type"]
      .agg(lambda x: x.value_counts().index[0]).reset_index()
      .rename(columns={"label_source_type": "primary_label_type"}),
    on="source_dataset")
src_legend = src_legend[["label","count","primary_label_type","source_dataset"]].rename(columns={
    "label": "Readable name", "count": "Molecules",
    "primary_label_type": "Primary label type", "source_dataset": "Raw key"})
src_legend["Molecules"] = src_legend["Molecules"].map("{:,}".format)

# Unlabeled source distribution
um_src_counts = um_src["source_dataset"].value_counts().reset_index()
um_src_counts.columns = ["source_dataset", "count"]
um_top = um_src_counts.head(12).copy()
um_top["label"] = um_top["source_dataset"].map({
    "guacamol_v1_train": "GuacaMol v1 — drug-like benchmark",
    "CID-SMILES":        "PubChem CID-SMILES",
    "chembl_36":         "ChEMBL 36",
    "qm9":               "QM9 quantum chemistry",
}).fillna(um_top["source_dataset"])
fig_um = px.pie(um_top, names="label", values="count",
                title="Unlabeled Molecule Source Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3)
fig_um.update_layout(height=420, margin=dict(t=60))
fig_um.update_traces(textposition="inside", textinfo="percent+label")


lst_refined = lm["label_source_type_refined"].value_counts().reset_index()
lst_refined.columns = ["type", "count"]

REFINED_LABELS = {
    "compiled_observed": "Experimental (observed)",
    "model_predicted":   "ML model predicted",
    "kj_calculated":     "Kamlet-Jacobs calculated\n(crystal X-ray density)",
    "qsar_predicted":    "QSAR predicted",
    "unknown":           "Unknown (EMDB residual)",
}
REFINED_COLORS = {
    "compiled_observed": "#4CAF50",
    "model_predicted":   "#2196F3",
    "kj_calculated":     "#FF9800",
    "qsar_predicted":    "#9C27B0",
    "unknown":           "#9E9E9E",
}
lst_refined["label"] = lst_refined["type"].map(REFINED_LABELS).fillna(lst_refined["type"])

fig_lst = px.pie(lst_refined, names="label", values="count",
                 title="Label Source Type (refined)",
                 color="type",
                 color_discrete_map=REFINED_COLORS)
fig_lst.update_layout(height=420, margin=dict(t=60))
fig_lst.update_traces(textposition="inside", textinfo="percent+label",
                      textfont_size=11)

lst_table = lst_refined[["label","count"]].rename(
    columns={"label": "Label source type", "count": "Molecules"})
lst_table["Molecules"] = lst_table["Molecules"].map("{:,}".format)
lst_table["Description"] = lst_table["Label source type"].map({
    "Experimental (observed)":                  "Measured laboratory values",
    "ML model predicted":                       "EMDP 3DCNN/MDGNN/RNNMGM model outputs",
    "Kamlet-Jacobs calculated\n(crystal X-ray density)": "Chem.Mater.2024: crystal density from X-ray, detonation props via K-J equations",
    "QSAR predicted":                           "EMDP q-RASPR quantum-mechanical QSAR model",
    "Unknown (EMDB residual)":                  "109 EMDB entries with unresolved provenance",
})

sec5 = (fig_to_html(fig_src, "fig_src") +
        table_html(src_legend, "Source dataset legend") +
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>" +
        fig_to_html(fig_um,  "fig_um") +
        fig_to_html(fig_lst, "fig_lst") +
        "</div>" +
        table_html(lst_table, "Label source type breakdown"))

# ── 6. PROPERTY DISTRIBUTIONS ─────────────────────────────────────────────────

prop_units = {
    "density":             ("g/cm³",  (0.5, 3.5)),
    "heat_of_formation":   ("kJ/mol", (-4500, 2000)),
    "detonation_velocity": ("km/s",   (0, 12)),
    "detonation_pressure": ("GPa",    (0, 55)),
    "explosion_heat":      ("MJ/kg",  (0, 55)),
}

# --- combined overview (all sources, aggregate) ---
fig_props = make_subplots(rows=2, cols=3,
                          subplot_titles=[PROP_LABELS[p] for p in PROPS],
                          vertical_spacing=0.15, horizontal_spacing=0.08)

for i, p in enumerate(PROPS):
    row, col = divmod(i, 3)
    vals = lm[p].dropna()
    lo, hi = prop_units[p][1]
    vals_clip = vals[(vals >= lo) & (vals <= hi)]
    fig_props.add_trace(
        go.Histogram(x=vals_clip, nbinsx=60, name=PROP_LABELS[p],
                     marker_color=COLORS[i], showlegend=False,
                     hovertemplate=f"{prop_units[p][0]}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>"),
        row=row+1, col=col+1)

fig_props.update_xaxes(visible=False, row=2, col=3)
fig_props.update_yaxes(visible=False, row=2, col=3)
fig_props.update_layout(height=580, title_text="Property Value Distributions (all sources)",
                        margin=dict(t=80, b=40))

# --- per-TIER overlay (rigorous per-property provenance): ---
#     two figures per property, counts + probability density
TIER_COLORS = {"A": "#4CAF50", "B": "#2196F3", "C": "#FF9800", "D": "#9C27B0"}
TIER_LABELS = {
    "A": "Experimental / literature",
    "B": "Simulation (EXPLO5 / DFT)",
    "C": "Empirical formula (K-J, group-contribution)",
    "D": "ML / QSAR",
}

figs_by_src = []
for p in PROPS:
    lo, hi = prop_units[p][1]
    unit    = prop_units[p][0]
    tier_col = f"{p}_tier"
    fig_pair = make_subplots(rows=1, cols=2,
                              subplot_titles=[
                                  f"{PROP_LABELS[p]} — counts",
                                  f"{PROP_LABELS[p]} — normalized (probability density)"])
    for tcode in ["A", "B", "C", "D"]:
        sub = lm.loc[lm[tier_col] == tcode, p].dropna()
        sub = sub[(sub >= lo) & (sub <= hi)]
        if sub.empty:
            continue
        # left: raw counts
        fig_pair.add_trace(go.Histogram(
            x=sub, nbinsx=50, name=TIER_LABELS[tcode],
            marker_color=TIER_COLORS[tcode], opacity=0.65,
            legendgroup=tcode, showlegend=True), row=1, col=1)
        # right: probability density
        fig_pair.add_trace(go.Histogram(
            x=sub, nbinsx=50, name=TIER_LABELS[tcode],
            marker_color=TIER_COLORS[tcode], opacity=0.65,
            histnorm="probability density",
            legendgroup=tcode, showlegend=False), row=1, col=2)
    fig_pair.update_xaxes(title_text=unit, row=1, col=1)
    fig_pair.update_xaxes(title_text=unit, row=1, col=2)
    fig_pair.update_yaxes(title_text="Count",               row=1, col=1)
    fig_pair.update_yaxes(title_text="Probability density", row=1, col=2)
    fig_pair.update_layout(
        barmode="overlay",
        height=400, margin=dict(t=100, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=10)),
        title_text=f"{PROP_LABELS[p]} — counts (left) and normalized density (right) by per-property tier",
        title_y=0.99)
    figs_by_src.append(fig_pair)

# --- per-source stats table ---
stats_rows = []
for src in SOURCE_TYPES:
    sub = lm[lm["label_source_type_refined"] == src]
    for p in PROPS:
        vals = sub[p].dropna()
        if vals.empty:
            continue
        s = vals.describe()
        stats_rows.append({
            "Source type":   SOURCE_LABELS[src],
            "Property":      PROP_LABELS[p],
            "N":             f"{int(s['count']):,}",
            "Mean":          f"{s['mean']:.3f}",
            "Std":           f"{s['std']:.3f}",
            "Median":        f"{s['50%']:.3f}",
            "Min":           f"{s['min']:.3f}",
            "Max":           f"{s['max']:.3f}",
            "Unit":          prop_units[p][0],
        })
stats_df = pd.DataFrame(stats_rows)

# --- aggregate stats table ---
agg_rows = []
for p in PROPS:
    s = lm[p].dropna().describe()
    agg_rows.append({
        "Property": PROP_LABELS[p],
        "N":        f"{int(s['count']):,}",
        "Mean":     f"{s['mean']:.3f}",
        "Std":      f"{s['std']:.3f}",
        "Min":      f"{s['min']:.3f}",
        "25%":      f"{s['25%']:.3f}",
        "Median":   f"{s['50%']:.3f}",
        "75%":      f"{s['75%']:.3f}",
        "Max":      f"{s['max']:.3f}",
        "Unit":     prop_units[p][0],
    })
agg_df = pd.DataFrame(agg_rows)

src_type_note = """
<div class="audit-box info">
  <strong>Why do source-type distributions differ? — by design, not a bug</strong><br><br>
  <strong>ML model predicted (blue)</strong> consists entirely of de novo designed molecules from
  the EMDP paper's reinforcement-learning (RL) and transfer-learning (TL) sampling, plus the
  RNNMGM generation set. These were <em>specifically optimized to maximize detonation performance</em>,
  so their mean values are systematically higher: detonation velocity ~7.6 km/s vs. ~4.9 km/s for
  experimental, detonation pressure ~26 GPa vs. ~11 GPa, explosion heat ~5.0 vs. ~2.7 MJ/kg.
  This is <em>selection bias by design</em> — the de novo optimizer targets high-performance molecules.<br><br>
  <strong>Kamlet-Jacobs calculated (orange)</strong> comes from the cm4c01978_si_001 crystal-structure
  database (Cambridge Structural Database X-ray entries). This is a broad set of organic crystals
  <em>not selected for energetic performance</em>. Many are non-energetic (fatty acids, steroids,
  azo dyes), giving lower mean density (1.39 vs. 1.66 g/cm³) and consequently lower K-J detonation
  properties.<br><br>
  <strong>Compiled experimental (green)</strong> represents real measured data for known energetic
  materials — the most representative distribution for this application domain.
</div>"""

sec6 = (
    table_html(agg_df,   "Property descriptive statistics (all sources)") +
    fig_to_html(fig_props, "fig_props") +
    "<h3>Distributions broken down by label source type</h3>" +
    src_type_note +
    "".join(fig_to_html(f, f"fig_src_{i}") for i, f in enumerate(figs_by_src)) +
    table_html(stats_df, "Per-source-type descriptive statistics")
)

# ── 7. PROPERTY CORRELATIONS ──────────────────────────────────────────────────

corr_df = lm[PROPS].dropna(how="all")
corr_mat = corr_df.corr()
short_labels = ["Density","HOF","Det-Vel","Det-P"]

fig_corr = px.imshow(corr_mat,
                     x=short_labels, y=short_labels,
                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     text_auto=".2f",
                     title="Pairwise Property Correlations (Pearson)")
fig_corr.update_layout(height=460, margin=dict(t=70, b=60))

# pairwise scatter for density vs detonation_velocity (a meaningful pair)
dv = lm[["density","detonation_velocity","detonation_pressure",
         "label_source_type_refined"]].dropna()
fig_scatter = px.scatter(dv.sample(min(5000, len(dv)), random_state=42),
                         x="density", y="detonation_velocity",
                         color="label_source_type_refined",
                         color_discrete_map=SOURCE_COLORS,
                         opacity=0.5,
                         labels={"density": "Density (g/cm³)",
                                 "detonation_velocity": "Detonation Velocity (km/s)"},
                         title="Density vs Detonation Velocity (sample ≤5,000)")
fig_scatter.update_layout(height=440, margin=dict(t=60, b=50))

sec7 = (fig_to_html(fig_corr,   "fig_corr") +
        fig_to_html(fig_scatter, "fig_scatter"))

# ── 8. STRUCTURAL FEATURES ────────────────────────────────────────────────────

lm["has_nitro"] = lm["has_nitro"].astype(str).str.lower().isin(["true","1","yes"])
lm["has_azide"] = lm["has_azide"].astype(str).str.lower().isin(["true","1","yes"])

struct_summary = pd.DataFrame([
    ("N count (nitrogen atoms)", f"{lm['n_count'].mean():.2f}",
     f"{lm['n_count'].median():.0f}", f"0 – {lm['n_count'].max():.0f}"),
    ("O count (oxygen atoms)",   f"{lm['o_count'].mean():.2f}",
     f"{lm['o_count'].median():.0f}", f"0 – {lm['o_count'].max():.0f}"),
    ("Has nitro group",           f"{lm['has_nitro'].mean()*100:.1f}% positive", "–", "–"),
    ("Has azide group",           f"{lm['has_azide'].mean()*100:.1f}% positive", "–", "–"),
    ("Energetic proxy score",     f"{lm['energetic_proxy_score'].mean():.3f}",
     f"{lm['energetic_proxy_score'].median():.3f}",
     f"{lm['energetic_proxy_score'].min():.2f} – {lm['energetic_proxy_score'].max():.2f}"),
], columns=["Feature", "Mean / Prevalence", "Median", "Range"])

# N and O count distributions
fig_no = make_subplots(rows=1, cols=2,
                       subplot_titles=["Nitrogen Atom Count", "Oxygen Atom Count"])
fig_no.add_trace(go.Histogram(x=lm["n_count"], nbinsx=30,
                               marker_color="#3F51B5", name="N count"), row=1, col=1)
fig_no.add_trace(go.Histogram(x=lm["o_count"], nbinsx=30,
                               marker_color="#E91E63", name="O count"), row=1, col=2)
fig_no.update_layout(height=380, showlegend=False, margin=dict(t=60, b=40),
                     title_text="Heteroatom Count Distributions")

# nitro / azide donut
has_nitro_val = [lm["has_nitro"].sum(), (~lm["has_nitro"]).sum()]
has_azide_val = [lm["has_azide"].sum(), (~lm["has_azide"]).sum()]

fig_flags = make_subplots(rows=1, cols=2,
                          specs=[[{"type":"pie"},{"type":"pie"}]],
                          subplot_titles=["Has Nitro Group","Has Azide Group"])
fig_flags.add_trace(go.Pie(labels=["Yes","No"], values=has_nitro_val,
                            hole=0.4, marker_colors=["#FF5722","#E0E0E0"],
                            showlegend=True), row=1, col=1)
fig_flags.add_trace(go.Pie(labels=["Yes","No"], values=has_azide_val,
                            hole=0.4, marker_colors=["#9C27B0","#E0E0E0"],
                            showlegend=False), row=1, col=2)
fig_flags.update_layout(height=380, margin=dict(t=60, b=30),
                        title_text="Functional Group Flags (labeled molecules)")

# proxy score distribution
fig_proxy = px.histogram(lm, x="energetic_proxy_score", nbins=50,
                          color_discrete_sequence=["#FF9800"],
                          title="Energetic Proxy Score Distribution",
                          labels={"energetic_proxy_score": "Proxy Score"})
fig_proxy.update_layout(height=360, margin=dict(t=60, b=40))

sec8 = (table_html(struct_summary, "Structural feature summary") +
        fig_to_html(fig_no,    "fig_no") +
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>" +
        fig_to_html(fig_flags, "fig_flags") +
        fig_to_html(fig_proxy, "fig_proxy") +
        "</div>")

# ── 9. SMILES / SELFIES VALIDATION ───────────────────────────────────────────
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import selfies as sf

lm["smiles_len"]  = lm["smiles"].str.len()
lm["selfies_len"] = lm["selfies"].str.len()

def _validate_smiles(s):
    if not isinstance(s, str) or not s:
        return (False, False, 0, None)
    m = Chem.MolFromSmiles(s)
    if m is None:
        return (False, False, 0, None)
    canon = Chem.MolToSmiles(m)
    return (True, canon == s, m.GetNumHeavyAtoms(), canon)

def _validate_selfies(sm_smiles, sf_string):
    if not isinstance(sf_string, str) or not sf_string:
        return (False, False)
    try:
        decoded = sf.decoder(sf_string)
    except Exception:
        return (False, False)
    if not decoded:
        return (False, False)
    m = Chem.MolFromSmiles(decoded)
    if m is None:
        return (True, False)
    canon_dec = Chem.MolToSmiles(m)
    if not isinstance(sm_smiles, str):
        return (True, False)
    m2 = Chem.MolFromSmiles(sm_smiles)
    canon_src = Chem.MolToSmiles(m2) if m2 else None
    return (True, canon_dec == canon_src)

print("  validating SMILES / SELFIES (labeled master) …")
_val_rows = []
for s in lm["smiles"]:
    _val_rows.append(_validate_smiles(s))
lm["smi_valid"], lm["smi_canonical"], lm["heavy_atoms"], lm["smi_canon"] = zip(*_val_rows)

_sf_rows = []
for sm, sf_s in zip(lm["smiles"], lm["selfies"]):
    _sf_rows.append(_validate_selfies(sm, sf_s))
lm["sf_decodable"], lm["sf_roundtrip"] = zip(*_sf_rows)

# Unlabeled: stratified sample across sources for speed
print("  validating SMILES on unlabeled sample …")
um_sample = (um.groupby("source_dataset", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), 3000), random_state=42)))
_um_rows = [_validate_smiles(s) for s in um_sample["smiles"]]
um_sample["smi_valid"] = [r[0] for r in _um_rows]
um_sample["smi_canonical"] = [r[1] for r in _um_rows]

# ── Validation summary table ─────────────────────────────────────────────────
lm_valid_pct = 100 * lm["smi_valid"].mean()
lm_canon_pct = 100 * lm["smi_canonical"].mean()
lm_sf_dec_pct = 100 * lm["sf_decodable"].mean()
lm_sf_rt_pct = 100 * lm["sf_roundtrip"].mean()
um_valid_pct = 100 * um_sample["smi_valid"].mean()
um_canon_pct = 100 * um_sample["smi_canonical"].mean()

val_summary = pd.DataFrame([
    ("labeled_master (all)",   f"{len(lm):,}",
     f"{lm_valid_pct:.2f}%", f"{lm_canon_pct:.2f}%",
     f"{lm_sf_dec_pct:.2f}%", f"{lm_sf_rt_pct:.2f}%"),
    (f"unlabeled_master (sample, n={len(um_sample):,})", f"{len(um_sample):,}",
     f"{um_valid_pct:.2f}%", f"{um_canon_pct:.2f}%", "–", "–"),
], columns=["Dataset","N","SMILES parseable","SMILES canonical",
            "SELFIES decodable","SELFIES ↔ SMILES roundtrip"])

# per-source validity table (labeled)
vs_rows = []
for src, g in lm.groupby("source_dataset"):
    vs_rows.append({
        "Source":              src,
        "N":                   f"{len(g):,}",
        "Invalid SMILES":      f"{(~g['smi_valid']).sum():,}",
        "Non-canonical":       f"{(g['smi_valid'] & ~g['smi_canonical']).sum():,}",
        "SELFIES fail decode": f"{(~g['sf_decodable']).sum():,}",
        "SELFIES fail r-trip": f"{(g['sf_decodable'] & ~g['sf_roundtrip']).sum():,}",
    })
val_src_df = pd.DataFrame(vs_rows).sort_values("Source").reset_index(drop=True)

# per-source for unlabeled sample
ums_rows = []
for src, g in um_sample.groupby("source_dataset"):
    ums_rows.append({
        "Source":          src,
        "Sampled":         f"{len(g):,}",
        "Invalid":         f"{(~g['smi_valid']).sum():,}",
        "Invalid %":       f"{100*(~g['smi_valid']).mean():.2f}%",
        "Non-canonical":   f"{(g['smi_valid'] & ~g['smi_canonical']).sum():,}",
    })
um_val_df = pd.DataFrame(ums_rows).sort_values("Source").reset_index(drop=True)

# ── length distributions (labeled, log-y) ────────────────────────────────────
fig_slen = make_subplots(rows=1, cols=2,
                          subplot_titles=["SMILES String Length", "SELFIES String Length"])
fig_slen.add_trace(go.Histogram(x=lm["smiles_len"], nbinsx=60,
                                 marker_color="#00BCD4", name="SMILES"), row=1, col=1)
fig_slen.add_trace(go.Histogram(x=lm["selfies_len"], nbinsx=60,
                                 marker_color="#8BC34A", name="SELFIES"), row=1, col=2)
fig_slen.update_layout(height=380, showlegend=False, margin=dict(t=60, b=40),
                        title_text="String Length Distributions (labeled molecules)")
fig_slen.update_yaxes(type="log")

# heavy atoms vs SMILES length scatter
_haw = lm[lm["smi_valid"]].sample(min(5000, lm["smi_valid"].sum()), random_state=42)
fig_ha = px.scatter(_haw, x="heavy_atoms", y="smiles_len", opacity=0.4,
                     color_discrete_sequence=["#4f8ef7"],
                     title="Heavy atom count vs SMILES length (sanity check)",
                     labels={"heavy_atoms":"Heavy atoms", "smiles_len":"SMILES length"})
fig_ha.update_layout(height=360, margin=dict(t=60, b=50))

# top-20 longest SMILES
long_smi = lm[lm["smi_valid"]].nlargest(20, "smiles_len")[
    ["smiles", "smiles_len", "heavy_atoms", "source_dataset"]].copy()
long_smi["smiles"] = long_smi["smiles"].str[:80]
long_smi.columns = ["SMILES (truncated to 80)", "Length", "Heavy atoms", "Source"]

# length stats
len_stats = pd.DataFrame([
    ("SMILES length (labeled)",
     f"{lm['smiles_len'].mean():.1f}",
     f"{lm['smiles_len'].median():.0f}",
     f"{lm['smiles_len'].quantile(0.99):.0f}",
     f"{lm['smiles_len'].min():.0f} – {lm['smiles_len'].max():.0f}"),
    ("SELFIES length (labeled)",
     f"{lm['selfies_len'].mean():.1f}",
     f"{lm['selfies_len'].median():.0f}",
     f"{lm['selfies_len'].quantile(0.99):.0f}",
     f"{lm['selfies_len'].min():.0f} – {lm['selfies_len'].max():.0f}"),
    ("Heavy atoms (labeled, valid)",
     f"{lm.loc[lm['smi_valid'],'heavy_atoms'].mean():.1f}",
     f"{lm.loc[lm['smi_valid'],'heavy_atoms'].median():.0f}",
     f"{lm.loc[lm['smi_valid'],'heavy_atoms'].quantile(0.99):.0f}",
     f"{lm.loc[lm['smi_valid'],'heavy_atoms'].min():.0f} – {lm.loc[lm['smi_valid'],'heavy_atoms'].max():.0f}"),
], columns=["Metric","Mean","Median","99th percentile","Range"])

# ── canonicalization analysis ────────────────────────────────────────────────
n_diff_canon = int((lm["smi_valid"] & ~lm["smi_canonical"]).sum())
canon_examples = lm[lm["smi_valid"] & ~lm["smi_canonical"]][["smiles","smi_canon","source_dataset"]].head(10).copy()
canon_examples.columns = ["Stored SMILES","RDKit canonical","Source"]
canon_examples["Stored SMILES"]   = canon_examples["Stored SMILES"].str[:55]
canon_examples["RDKit canonical"] = canon_examples["RDKit canonical"].str[:55]

canon_note = f"""<div class="audit-box info">
<strong>Canonicalization audit:</strong> of {int(lm['smi_valid'].sum()):,} parseable SMILES
in labeled_master, {n_diff_canon:,} ({100*n_diff_canon/max(lm['smi_valid'].sum(),1):.2f}%)
are NOT in RDKit canonical form. A canonical SMILES is the unique, algorithm-chosen string
representation of a molecule (Weininger's Morgan-based tie-breaking extended by Daylight and
re-implemented in RDKit). Non-canonical SMILES still represent the correct molecule but break
string-based joins, caching, and deduplication — every pipeline stage should canonicalize at
ingestion. Stored vs canonical differences usually stem from: atom ordering, aromatic vs
Kekulé ring notation, and explicit vs implicit hydrogens.
</div>"""

# SELFIES roundtrip audit
n_sf_fail = int((~lm["sf_decodable"]).sum())
n_sf_mismatch = int((lm["sf_decodable"] & ~lm["sf_roundtrip"]).sum())

selfies_note = f"""<div class="audit-box info">
<strong>SELFIES encoding audit:</strong> SELFIES (SELF-referencing Embedded Strings; Krenn
et al. 2020) guarantee <em>100% syntactic validity</em> by construction — any random SELFIES
string decodes to a valid molecule. The roundtrip check here is:
<code>SMILES → (previously encoded) SELFIES → decode → RDKit-canonical SMILES → compare to
canonical(original)</code>. Expected failure modes: aromatic perception differences,
stereochemistry loss in v1 encodings, and charged/radical fragments that SELFIES normalises.
In labeled_master: {n_sf_fail:,} SELFIES fail to decode,
{n_sf_mismatch:,} decode but disagree with the source SMILES after canonicalization.
</div>"""

# SELFIES alphabet summary (top tokens)
_alphabet_counts = {}
for sf_s in lm["selfies"].dropna().sample(min(len(lm), 10000), random_state=42):
    try:
        toks = list(sf.split_selfies(sf_s))
    except Exception:
        continue
    for t in toks:
        _alphabet_counts[t] = _alphabet_counts.get(t, 0) + 1
alpha_df = (pd.DataFrame([(k, v) for k, v in _alphabet_counts.items()],
                          columns=["SELFIES token", "Count"])
              .sort_values("Count", ascending=False).head(20).reset_index(drop=True))
alpha_df["Count"] = alpha_df["Count"].map("{:,}".format)

sec9 = (table_html(val_summary, "SMILES / SELFIES validation overview") +
        table_html(val_src_df,  "Labeled master — validation by source") +
        table_html(um_val_df,   "Unlabeled master — validation by source (sampled)") +
        "<h3>SMILES canonicalization</h3>" +
        canon_note +
        table_html(canon_examples, "Examples of non-canonical stored SMILES") +
        "<h3>SELFIES encoding and roundtrip</h3>" +
        selfies_note +
        table_html(alpha_df, "Top-20 SELFIES tokens (10k sample)") +
        "<h3>String length distributions</h3>" +
        table_html(len_stats,   "String length descriptive stats") +
        fig_to_html(fig_slen, "fig_slen") +
        fig_to_html(fig_ha,   "fig_ha") +
        table_html(long_smi,  "Top 20 longest SMILES in labeled master"))

# ── 10. EXTERNAL DATASETS ────────────────────────────────────────────────────

# EMDB
emdb_cov = pd.DataFrame({
    "Property": ["Density (g/cm³)", "Heat of Formation (kJ/mol)", "HOF Gas"],
    "Count":    [emdb["density"].notna().sum(),
                 emdb["heat_of_formation"].notna().sum(),
                 emdb["heat_of_formation_gas"].notna().sum()],
    "Total":    [len(emdb)] * 3,
})
emdb_cov["Coverage %"] = (100 * emdb_cov["Count"] / emdb_cov["Total"]).map("{:.1f}%".format)

fig_emdb_density = px.histogram(emdb, x="density", nbins=30,
                                 color_discrete_sequence=["#009688"],
                                 title=f"EMDB Density Distribution (n={len(emdb):,})",
                                 labels={"density": "Density (g/cm³)"})
fig_emdb_density.update_layout(height=340, margin=dict(t=60, b=40))

fig_emdb_hof = px.histogram(emdb, x="heat_of_formation", nbins=30,
                              color_discrete_sequence=["#795548"],
                              title="EMDB Heat of Formation Distribution",
                              labels={"heat_of_formation": "HOF (kJ/mol)"})
fig_emdb_hof.update_layout(height=340, margin=dict(t=60, b=40))

# EMDP
emdp_all = pd.concat([emdp_tr.assign(split="train"),
                       emdp_te.assign(split="test")], ignore_index=True)
emdp_props = ["density","DetoD","DetoP","DetoQ","DetoT","DetoV","HOF_S","BDE"]
emdp_stats = emdp_all[emdp_props].describe().T.reset_index()
emdp_stats.columns = ["Property","Count","Mean","Std","Min","25%","Median","75%","Max"]
for col in ["Mean","Std","Min","25%","Median","75%","Max"]:
    emdp_stats[col] = emdp_stats[col].map("{:.3f}".format)
emdp_stats["Count"] = emdp_stats["Count"].astype(int).map("{:,}".format)

fig_emdp = px.scatter(emdp_all, x="density", y="DetoD",
                       color="split", opacity=0.7,
                       labels={"density": "Density (g/cm³)",
                               "DetoD":   "Detonation Velocity (km/s)"},
                       title="EMDP: Density vs Detonation Velocity (DetoD, km/s)",
                       color_discrete_sequence=["#2196F3","#FF5722"])
fig_emdp.update_layout(height=400, margin=dict(t=60, b=50))

sec10 = (
    f"<h3>EMDB (Energetic Materials Database) — {len(emdb):,} molecules</h3>" +
    table_html(emdb_cov, "EMDB property coverage") +
    "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>" +
    fig_to_html(fig_emdb_density, "fig_emdb_d") +
    fig_to_html(fig_emdb_hof,     "fig_emdb_h") +
    "</div>" +
    f"<h3>EMDP (Energetic Materials Design Pipeline) — {len(emdp_tr):,} train / {len(emdp_te):,} test</h3>" +
    table_html(emdp_stats, "EMDP property statistics") +
    fig_to_html(fig_emdp, "fig_emdp"))

# ── 11. PROPERTY NORMALIZATION ────────────────────────────────────────────────

norm_rows = []
props_dict = norm_json.get("properties", {})
for p, stats in props_dict.items():
    norm_rows.append({
        "Property": PROP_LABELS.get(p, p),
        "Mean (μ)":  f"{stats['property_mean']:.4f}",
        "Std (σ)":   f"{stats['property_std']:.4f}",
        "N (train)": f"{int(stats['count']):,}",
        "Missing":   f"{int(stats['missing']):,}",
    })
norm_df = pd.DataFrame(norm_rows)

sec11 = table_html(norm_df, "Property normalization statistics (computed from property_supervised train split)")


# ── 12. KAMLET-JACOBS CONSISTENCY AUDIT ──────────────────────────────────────
#
# K-J equations:  D (km/s) = 1.01 * sqrt(phi) * (1 + 1.3*rho)
#                 P (GPa)  = 1.558 * rho^2 * phi
# where phi = N * sqrt(M) * sqrt(Q)  (composite detonation parameter, cal^0.5/g)
#
# Eliminating phi: D_kj = 1.01*(1+1.3*rho)*sqrt(P/(1.558*rho^2))
# This cross-check is unit-free and requires no HOF.

_KJ_K1 = 1.01    # velocity constant
_KJ_K2 = 1.558   # pressure constant

kj = lm[["smiles", "density", "detonation_velocity", "detonation_pressure",
         "label_source_type_refined"]].copy()
kj.columns = ["smiles", "rho", "D", "P", "src"]

valid_rho = kj["rho"].notna() & (kj["rho"] > 0.3)

# implied composite phi from D+rho
has_D = kj["D"].notna() & valid_rho
kj.loc[has_D, "phi_D"] = (kj.loc[has_D, "D"] /
                           (_KJ_K1 * (1 + 1.3 * kj.loc[has_D, "rho"])))**2

# implied composite phi from P+rho
has_P = kj["P"].notna() & valid_rho
kj.loc[has_P, "phi_P"] = kj.loc[has_P, "P"] / (_KJ_K2 * kj.loc[has_P, "rho"]**2)

# cross-check: D predicted from P+rho, and residuals
cross = has_D & has_P
kj.loc[cross, "D_kj"] = (_KJ_K1 * (1 + 1.3 * kj.loc[cross, "rho"]) *
                          np.sqrt(np.maximum(kj.loc[cross, "P"] /
                                             (_KJ_K2 * kj.loc[cross, "rho"]**2), 0)))
kj.loc[cross, "delta_D"]   = kj.loc[cross, "D"]   - kj.loc[cross, "D_kj"]
kj.loc[cross, "rel_err_D"] = (np.abs(kj.loc[cross, "delta_D"]) /
                               kj.loc[cross, "D_kj"].replace(0, np.nan))

n_cross = cross.sum()

# ── Figure 12-a: D_actual vs D_kj scatter ─────────────────────────────────────
cross_df = kj[cross].dropna(subset=["D", "D_kj", "src"])
sample_cross = cross_df.sample(min(6000, len(cross_df)), random_state=42)

fig_kj_scatter = go.Figure()
for src in SOURCE_TYPES:
    sub = sample_cross[sample_cross["src"] == src]
    if sub.empty:
        continue
    fig_kj_scatter.add_trace(go.Scatter(
        x=sub["D_kj"], y=sub["D"],
        mode="markers",
        name=SOURCE_LABELS[src],
        marker=dict(color=SOURCE_COLORS[src], size=5, opacity=0.6),
    ))
# diagonal y = x
d_range = [cross_df[["D","D_kj"]].min().min() * 0.95,
           cross_df[["D","D_kj"]].max().max() * 1.02]
fig_kj_scatter.add_shape(type="line",
    x0=d_range[0], y0=d_range[0], x1=d_range[1], y1=d_range[1],
    line=dict(color="#ffffff", width=1, dash="dash"))
fig_kj_scatter.update_layout(
    title=dict(text=f"K-J Cross-Check: Actual vs K-J-Predicted Detonation Velocity (n={n_cross:,})",
               y=0.98, yanchor="top"),
    xaxis_title="D_kj from (P, ρ)  [km/s]",
    yaxis_title="D_actual  [km/s]",
    height=540, margin=dict(t=130, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=10)))

# ── Figure 12-b: residual ΔD histogram by source type ─────────────────────────
fig_kj_resid = go.Figure()
for src in SOURCE_TYPES:
    sub = kj[cross & (kj["src"] == src)]["delta_D"].dropna()
    if sub.empty:
        continue
    fig_kj_resid.add_trace(go.Histogram(
        x=sub, name=SOURCE_LABELS[src],
        marker_color=SOURCE_COLORS[src],
        opacity=0.7, nbinsx=60,
        histnorm="probability density"))
fig_kj_resid.update_layout(
    barmode="overlay",
    title="K-J Residual ΔD = D_actual − D_kj(P,ρ)  by source type",
    xaxis_title="ΔD  (km/s)",
    yaxis_title="Density",
    height=380, margin=dict(t=70, b=60))

# ── Figure 12-c: ρ vs D colored by source type ────────────────────────────────
rho_d = kj[has_D].dropna(subset=["rho","D","src"])
rho_d_samp = rho_d.sample(min(6000, len(rho_d)), random_state=42)
fig_rho_d = go.Figure()
for src in SOURCE_TYPES:
    sub = rho_d_samp[rho_d_samp["src"] == src]
    if sub.empty:
        continue
    fig_rho_d.add_trace(go.Scatter(
        x=sub["rho"], y=sub["D"],
        mode="markers",
        name=SOURCE_LABELS[src],
        marker=dict(color=SOURCE_COLORS[src], size=5, opacity=0.55)))
fig_rho_d.update_layout(
    title=dict(text="Density vs Detonation Velocity — by label source", y=0.98, yanchor="top"),
    xaxis_title="Density (g/cm³)", yaxis_title="Detonation Velocity (km/s)",
    height=500, margin=dict(t=130, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=10)))

# ── Figure 12-d: ρ vs P colored by source type ────────────────────────────────
rho_p = kj[has_P].dropna(subset=["rho","P","src"])
rho_p_samp = rho_p.sample(min(6000, len(rho_p)), random_state=42)
fig_rho_p = go.Figure()
for src in SOURCE_TYPES:
    sub = rho_p_samp[rho_p_samp["src"] == src]
    if sub.empty:
        continue
    fig_rho_p.add_trace(go.Scatter(
        x=sub["rho"], y=sub["P"],
        mode="markers",
        name=SOURCE_LABELS[src],
        marker=dict(color=SOURCE_COLORS[src], size=5, opacity=0.55)))
fig_rho_p.update_layout(
    title=dict(text="Density vs Detonation Pressure — by label source", y=0.98, yanchor="top"),
    xaxis_title="Density (g/cm³)", yaxis_title="Detonation Pressure (GPa)",
    height=500, margin=dict(t=130, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=10)))

# ── Figure 12-e: implied phi distributions ────────────────────────────────────
fig_phi = make_subplots(rows=1, cols=2,
                         subplot_titles=["φ implied by D and ρ", "φ implied by P and ρ"])
phi_clip_hi = 20   # clip for readability
for src in SOURCE_TYPES:
    col_src = SOURCE_COLORS[src]
    sub_d = kj[has_D & (kj["src"] == src)]["phi_D"].dropna()
    sub_d = sub_d[(sub_d > 0) & (sub_d < phi_clip_hi)]
    sub_p = kj[has_P & (kj["src"] == src)]["phi_P"].dropna()
    sub_p = sub_p[(sub_p > 0) & (sub_p < phi_clip_hi)]
    fig_phi.add_trace(go.Histogram(
        x=sub_d, name=SOURCE_LABELS[src],
        marker_color=col_src, opacity=0.65, nbinsx=50,
        showlegend=True), row=1, col=1)
    fig_phi.add_trace(go.Histogram(
        x=sub_p, name=SOURCE_LABELS[src],
        marker_color=col_src, opacity=0.65, nbinsx=50,
        showlegend=False), row=1, col=2)
fig_phi.update_layout(
    barmode="overlay", height=400, margin=dict(t=80, b=50),
    title_text="K-J Composite Parameter φ = N√(MQ)  [cal^0.5/g]  implied from labeled data")
fig_phi.update_xaxes(title_text="φ (0–20 range)", row=1, col=1)
fig_phi.update_xaxes(title_text="φ (0–20 range)", row=1, col=2)

# ── Table 12-a: cross-check stats by source type ──────────────────────────────
kj_stats_rows = []
cross_all = kj[cross].dropna(subset=["D","D_kj","delta_D"])
for src in SOURCE_TYPES:
    sub = cross_all[cross_all["src"] == src]
    if sub.empty:
        continue
    n      = len(sub)
    rmse   = np.sqrt((sub["delta_D"]**2).mean())
    mae    = sub["delta_D"].abs().mean()
    bias   = sub["delta_D"].mean()
    r2     = np.corrcoef(sub["D"], sub["D_kj"])[0, 1]**2 if n > 2 else np.nan
    pct_10 = (sub["rel_err_D"] < 0.10).mean() * 100 if n > 0 else 0
    pct_20 = (sub["rel_err_D"] < 0.20).mean() * 100 if n > 0 else 0
    kj_stats_rows.append({
        "Source Type":       SOURCE_LABELS[src],
        "N":                 f"{n:,}",
        "RMSE (km/s)":       f"{rmse:.3f}",
        "MAE (km/s)":        f"{mae:.3f}",
        "Mean Bias (km/s)":  f"{bias:+.3f}",
        "R²":                f"{r2:.4f}" if not np.isnan(r2) else "–",
        "Within 10%":        f"{pct_10:.1f}%",
        "Within 20%":        f"{pct_20:.1f}%",
    })
# add overall row
if len(cross_all):
    rmse  = np.sqrt((cross_all["delta_D"]**2).mean())
    mae   = cross_all["delta_D"].abs().mean()
    bias  = cross_all["delta_D"].mean()
    r2    = np.corrcoef(cross_all["D"], cross_all["D_kj"])[0, 1]**2
    p10   = (cross_all["rel_err_D"] < 0.10).mean() * 100
    p20   = (cross_all["rel_err_D"] < 0.20).mean() * 100
    kj_stats_rows.append({
        "Source Type":       "ALL",
        "N":                 f"{len(cross_all):,}",
        "RMSE (km/s)":       f"{rmse:.3f}",
        "MAE (km/s)":        f"{mae:.3f}",
        "Mean Bias (km/s)":  f"{bias:+.3f}",
        "R²":                f"{r2:.4f}",
        "Within 10%":        f"{p10:.1f}%",
        "Within 20%":        f"{p20:.1f}%",
    })
kj_stats_df = pd.DataFrame(kj_stats_rows)

# ── Table 12-b: phi coverage summary ─────────────────────────────────────────
phi_cov_rows = []
for src in SOURCE_TYPES:
    n_d = (has_D & (kj["src"] == src)).sum()
    n_p = (has_P & (kj["src"] == src)).sum()
    phi_d_med = kj.loc[has_D & (kj["src"] == src), "phi_D"].median()
    phi_p_med = kj.loc[has_P & (kj["src"] == src), "phi_P"].median()
    phi_cov_rows.append({
        "Source Type":      SOURCE_LABELS[src],
        "N (ρ+D)":          f"{n_d:,}",
        "Median φ_D":       f"{phi_d_med:.2f}" if not np.isnan(phi_d_med) else "–",
        "N (ρ+P)":          f"{n_p:,}",
        "Median φ_P":       f"{phi_p_med:.2f}" if not np.isnan(phi_p_med) else "–",
    })
phi_cov_df = pd.DataFrame(phi_cov_rows)

# ── Table 12-c: top outliers ──────────────────────────────────────────────────
outliers = cross_all.copy()
outliers["smiles"] = kj.loc[cross, "smiles"]
outliers = outliers.dropna(subset=["rel_err_D"])
outliers = outliers.nlargest(25, "rel_err_D")[
    ["smiles", "src", "rho", "D", "D_kj", "P", "delta_D", "rel_err_D"]]
outliers.columns = ["SMILES","Source","ρ","D_actual","D_kj","P_actual","ΔD","Rel.Err."]
outliers["ρ"]        = outliers["ρ"].map("{:.3f}".format)
outliers["D_actual"] = outliers["D_actual"].map("{:.3f}".format)
outliers["D_kj"]     = outliers["D_kj"].map("{:.3f}".format)
outliers["P_actual"] = outliers["P_actual"].map("{:.2f}".format)
outliers["ΔD"]       = outliers["ΔD"].map("{:+.3f}".format)
outliers["Rel.Err."] = outliers["Rel.Err."].map("{:.1%}".format)
outliers["SMILES"]   = outliers["SMILES"].str[:55]
outliers["Source"]   = outliers["Source"].map(lambda s: SOURCE_LABELS.get(s, s))

kj_note = f"""<div class="audit-box info">
<strong>Kamlet-Jacobs cross-validation:</strong> The K-J equations
D = 1.01·√φ·(1+1.3ρ) and P = 1.558·ρ²·φ share the same composite parameter φ.
For any molecule with measured density, detonation velocity <em>and</em> detonation pressure,
both equations must give the same φ — otherwise the labels are internally inconsistent.
Cross-check formula (no HOF needed):<br>
<code>D_kj = 1.01 · (1+1.3ρ) · √(P / (1.558·ρ²))</code><br>
{n_cross:,} molecules in labeled_master have all three values and are audited here.
A large |ΔD| (actual minus predicted) flags potential unit errors, transcription errors,
or molecules where K-J assumptions break down (e.g., very non-CHNO compounds).
</div>"""

sec12 = (
    kj_note +
    "<h3>D-P Internal Consistency Cross-Check</h3>" +
    fig_to_html(fig_kj_scatter, "fig_kj_scatter") +
    fig_to_html(fig_kj_resid,   "fig_kj_resid") +
    table_html(kj_stats_df, "K-J cross-check statistics by source type") +
    "<h3>Density vs Detonation Property Relationships</h3>" +
    "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>" +
    fig_to_html(fig_rho_d, "fig_rho_d") +
    fig_to_html(fig_rho_p, "fig_rho_p") +
    "</div>" +
    "<h3>Implied Composite Parameter φ = N√(MQ)</h3>" +
    fig_to_html(fig_phi, "fig_phi") +
    table_html(phi_cov_df, "φ coverage and median by source type") +
    "<h3>Top 25 K-J Outliers (largest relative error in D)</h3>" +
    table_html(outliers, "Outlier molecules: D_actual vs D_kj(P,ρ)")
)

# ── 13. EXPERIMENTAL COUNTER AUDIT ───────────────────────────────────────────
#
# Audit: for each experimental raw source, compare raw row count vs
# compiled_observed rows in labeled_master, and explain the delta.

RAW_EXP_SOURCES = {
    "3DCNN":                            ("data/raw/energetic_external/EMDP/Data/3DCNN.csv", "xlsx", "smiles"),
    "5039":                             ("data/raw/energetic_external/EMDP/Data/5039.csv", "csv", "smiles"),
    "train_set":                        ("data/raw/energetic_external/EMDP/Data/train_set.csv", "csv", "smiles"),
    "test_set":                         ("data/raw/energetic_external/EMDP/Data/test_set.csv", "csv", "smiles"),
    "det_dataset_08-02-2022":           ("data/raw/energetic_external/RSC_2022_public/det_dataset_08-02-2022.xlsx", "xlsx", "SMILES"),
    "Dm":                               ("data/raw/energetic_external/RNNMGM_upstream/data/Dm.csv", "csv", "SMILES"),
    "emdb_v21_molecules_pubchem":       ("data/raw/energetic_external/EMDB_public/emdb_v21_molecules_pubchem.csv", "csv", "smiles"),
    "combined_data":                    ("data/raw/energetic_external/Machine-Learning-Energetic-Molecules-Notebooks/datasets/combined_data.xlsx", "xlsx", None),
    "CHNOClF_dataset":                  ("data/raw/energetic_external/MultiTaskEM_upstream/data/CHNOClF_dataset.csv", "csv", "SMILES"),
    "Huang_Massa_data_with_all_SMILES": ("data/raw/energetic_external/Machine-Learning-Energetic-Molecules-Notebooks/datasets/Huang_Massa_data_with_all_SMILES.xlsx", "xlsx", None),
}

SOURCE_DESCRIPTIONS_EXTRA = {
    "cm4c01978_si_001 (K-J)": {
        "origin":    "Ma et al. 2022 (Chem. Mater., SI-001 CSD crystal database).",
        "nature":    "Kamlet-Jacobs FORMULA predictions. D = 1.01·√φ·(1+1.3ρ) and P = 1.558·ρ²·φ, where ρ is the crystallographic density from CSD and φ = N√(MQ) is computed from molecular formula assuming CHNO → N₂ + H₂O + CO₂ + C(s) product hierarchy plus HOF estimation.",
        "inclusion": "12,040 CSD refcodes with valid crystallographic density. Reliability: ±10–20% on D, ±20–30% on P for well-behaved CHNO; underestimates for metal-containing/F-rich or non-ideal detonation products. Not ML — purely deterministic formula.",
    },
    "denovo_sampling_rl": {
        "origin":    "EMDP de novo sampling with Reinforcement Learning agent.",
        "nature":    "Seq2Seq RNN generator fine-tuned with policy-gradient RL toward a multi-target reward (density, D, P). Property predictions come from the EMDP-trained predictor (deep MLP on ECFP + descriptor features, trained on 3DCNN+train_set+5039).",
        "inclusion": "6,097 novel SMILES accepted through filters (validity, drug-likeness, CHNO composition). Reliability: values are model predictions, not measurements. Prediction error ±0.3 km/s D, ±2–4 GPa P on in-distribution CHNO — worse on novel scaffolds generated out-of-distribution.",
    },
    "denovo_sampling_tl": {
        "origin":    "EMDP de novo sampling with Transfer-Learning fine-tune.",
        "nature":    "Same RNN generator as the RL variant but fine-tuned via teacher-forcing on known high-performance explosives. Same downstream EMDP predictor for property labels.",
        "inclusion": "4,922 generated SMILES passing validity filters. Reliability: same predictor error envelope as RL variant; TL samples tend to be closer to training scaffolds and therefore more reliably scored.",
    },
    "generation": {
        "origin":    "MDGNN / related generative sampling (internal 'generation' tag).",
        "nature":    "Neural generator outputs paired with property predictions; explosion_heat originally in cal/g, known unit-conversion bug was patched via fix_mdgnn_explosion_heat.py.",
        "inclusion": "7,072 accepted molecules. Reliability: generator-dependent. Explosion_heat audited and corrected ×0.004184 for rows above 12 MJ/kg threshold.",
    },
    "q-RASPR": {
        "origin":    "q-RASPR (QSAR for Reactivity / Structure-Property) — online QSAR server outputs.",
        "nature":    "QSAR regression models built on 2D/3D molecular descriptors (topological, electronic, CORAL-style SMILES-based indices). Typical training set: 200–500 experimental explosives. Outputs include density, D, P, impact sensitivity.",
        "inclusion": "2,438 molecules scored by the q-RASPR consensus ensemble. Reliability: lower than pure DFT or K-J — typical QSAR RMSE on blind test is ±0.08 g/cm³ density, ±0.5 km/s D, ±5 GPa P. Works best for scaffolds similar to training set.",
    },
}

SOURCE_DESCRIPTIONS = {
    "3DCNN": {
        "origin":    "EMDP (Energetic Materials Design Pipeline) companion dataset",
        "nature":    "DFT-calculated quantum-chemistry properties (NOT experimental). Columns include electronic_energy (Hartree), HOMO_LUMO_gap (eV), dipole_moment (Debye) — smoking-gun DFT outputs. Used to train a 3D CNN regressor.",
        "inclusion": "CHNO molecules used for 3D-CNN training; split into train/test/valid via set_designation. Currently mis-tagged as compiled_observed — should be reclassified as DFT-calculated/model_predicted.",
    },
    "5039": {
        "origin":    "EMDP Data/5039.csv — 5,039 molecules curated for BDE (bond dissociation energy) regression.",
        "nature":    "Mix of experimental density + computed BDE. Only SMILES, density, BDE columns present.",
        "inclusion": "All rows with valid SMILES; density retained as-is, BDE computed by upstream authors.",
    },
    "train_set": {
        "origin":    "EMDP Data/train_set.csv — Ma et al. 2022 training split (cm4c01978_si_001 crystal database).",
        "nature":    "Experimental crystallographic density (X-ray diffraction) paired with computed detonation velocity/pressure/heat (DetoD, DetoP, DetoQ) via EXPLO5 thermochemistry.",
        "inclusion": "1,800 CHNO molecules with full density+detonation property set (train portion of published train/test split).",
    },
    "test_set": {
        "origin":    "EMDP Data/test_set.csv — held-out test split of the same Ma et al. dataset as train_set.",
        "nature":    "Same property mix as train_set (experimental density + EXPLO5 detonation properties).",
        "inclusion": "200 CHNO molecules held out for model benchmarking.",
    },
    "det_dataset_08-02-2022": {
        "origin":    "RSC Adv. 2022 public detonation dataset (Casey et al.).",
        "nature":    "Experimental: measured maximum theoretical density (TMD) + experimentally characterized detonation properties for well-known military/industrial explosives.",
        "inclusion": "260 compiled CHNO(F/Cl) energetics with non-null SMILES and density.",
    },
    "Dm": {
        "origin":    "RNNMGM upstream reference table (Dm.csv) — energetic compendium with aliases.",
        "nature":    "Experimental curated values for named explosives (RDX, HMX, TNT, etc.) with density measurement.",
        "inclusion": "303 named compounds; only those with SMILES successfully resolved (≈247 after canonicalization).",
    },
    "emdb_v21_molecules_pubchem": {
        "origin":    "EMDB v2.1 (Energetic Materials Database, PubChem-resolved).",
        "nature":    "Experimental density and heat of formation (solid + gas phase) compiled from published literature.",
        "inclusion": "212 molecules where PubChem returned a resolvable SMILES; 109 retained after dedup against higher-priority sources.",
    },
    "combined_data": {
        "origin":    "Coulomb-Matrix Notebooks (combined_data.xlsx) — 416 CHNO energetics aggregated for ML benchmarking.",
        "nature":    "Experimental density and gas-phase formation enthalpy, with notes column flagging dubious entries.",
        "inclusion": "416 raw rows; after SMILES parsing + dedup against Huang-Massa parent and other sources, only 79 unique-to-this-source remain tagged compiled_observed.",
    },
    "CHNOClF_dataset": {
        "origin":    "MultiTaskEM upstream (CHNOClF_dataset.csv) — multi-task ML training dataset.",
        "nature":    "Long-format: each row is (SMILES, property_name, value). Mix of measured density and computed targets; most rows are predictions, not measurements.",
        "inclusion": "1,154 long-format rows cover ~250 unique molecules; only 3 rows survived dedup as genuinely experimental (rest are model predictions assigned elsewhere).",
    },
    "Huang_Massa_data_with_all_SMILES": {
        "origin":    "Huang & Massa classical energetics compilation (with added SMILES strings).",
        "nature":    "110 well-characterized explosives with experimental density and solid-phase heat of formation.",
        "inclusion": "Heavily overlaps with combined_data and Dm; only 2 unique-to-this-source rows remain after cross-source dedup.",
    },
}

def _load_raw(path, fmt):
    full = f"{BASE}/{path}"
    return pd.read_excel(full) if fmt == "xlsx" else pd.read_csv(full, low_memory=False)

def _canon(s):
    if not isinstance(s, str) or not s:
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None

print("  auditing experimental counters vs raw sources …")
audit_rows = []
lm_full = pd.read_csv(f"{BASE}/data/training/master/labeled_master.csv", low_memory=False,
                     usecols=["smiles","source_dataset","label_source_type"])
for src, (path, fmt, smi_col) in RAW_EXP_SOURCES.items():
    try:
        raw = _load_raw(path, fmt)
    except Exception as e:
        audit_rows.append({"Source": src, "Raw rows": "ERROR", "Raw unique SMILES": "–",
                           "Master rows (exp)": "–", "Master unique SMILES (exp)": "–",
                           "Retained %": "–", "SMILES-match %": "–"})
        continue
    # auto-detect SMILES column if missing
    if smi_col is None:
        for c in raw.columns:
            if "smiles" in c.lower():
                smi_col = c; break
    n_raw = len(raw)
    raw_smi = raw[smi_col].astype(str).map(_canon).dropna().unique() if smi_col else []
    n_raw_uniq = len(raw_smi)
    master_sub = lm_full[(lm_full["source_dataset"] == src) &
                         (lm_full["label_source_type"] == "compiled_observed")]
    n_master = len(master_sub)
    n_master_uniq = master_sub["smiles"].nunique()
    raw_set = set(raw_smi)
    master_set = set(master_sub["smiles"].dropna().unique())
    overlap = len(raw_set & master_set)
    match_pct = 100 * overlap / max(n_master_uniq, 1)
    ret_pct = 100 * n_master / max(n_raw, 1)
    audit_rows.append({
        "Source":                      src,
        "Raw rows":                    f"{n_raw:,}",
        "Raw unique SMILES":           f"{n_raw_uniq:,}",
        "Master rows (exp)":           f"{n_master:,}",
        "Master unique SMILES (exp)":  f"{n_master_uniq:,}",
        "Retained %":                  f"{ret_pct:.1f}%",
        "SMILES-match %":              f"{match_pct:.1f}%",
    })
audit_df = pd.DataFrame(audit_rows)

# Totals row
try:
    tot_raw = sum(int(r["Raw rows"].replace(",","")) for r in audit_rows if r["Raw rows"] != "ERROR")
    tot_master = sum(int(r["Master rows (exp)"].replace(",","")) for r in audit_rows if r["Master rows (exp)"] != "–")
    audit_df = pd.concat([audit_df, pd.DataFrame([{
        "Source": "TOTAL",
        "Raw rows": f"{tot_raw:,}",
        "Raw unique SMILES": "–",
        "Master rows (exp)": f"{tot_master:,}",
        "Master unique SMILES (exp)": "–",
        "Retained %": f"{100*tot_master/max(tot_raw,1):.1f}%",
        "SMILES-match %": "–",
    }])], ignore_index=True)
except Exception:
    pass

audit_note = """<div class="audit-box info">
<strong>Counter audit methodology:</strong> Each raw experimental source is compared against
<code>labeled_master</code> rows tagged <em>compiled_observed</em> from the same source.
<br><br>
<strong>Raw rows vs Master rows</strong> — differences arise from:
(1) rows without a parseable SMILES (dropped during ingestion);
(2) rows missing all target properties;
(3) deduplication when the same SMILES appears multiple times in the raw file;
(4) cross-source deduplication where the same canonical SMILES was already claimed by a
higher-priority source.
<br>
<strong>SMILES-match %</strong> = share of master experimental SMILES that can be re-matched
to the canonicalised raw file; values well below 100% indicate canonicalization drift
between ingestion and audit.
</div>"""

desc_rows = []
for src, d in {**SOURCE_DESCRIPTIONS, **SOURCE_DESCRIPTIONS_EXTRA}.items():
    desc_rows.append({
        "Source":              src,
        "Origin":              d["origin"],
        "Nature of values":    d["nature"],
        "Inclusion criteria":  d["inclusion"],
    })
desc_df = pd.DataFrame(desc_rows)

# Label audit findings
audit_findings = pd.DataFrame([
    ("3DCNN",                                 "compiled_observed",  "DFT-calculated (model_predicted)",
     "Raw file contains electronic_energy/HOMO-LUMO/dipole — DFT outputs. 26,254 rows. Fix: re-tag as model_predicted (or dft_calculated).",
     "HIGH"),
    ("5039",                                  "compiled_observed",  "Mixed (density experimental, BDE computed)",
     "Density from literature; BDE is a computed bond-dissociation-energy descriptor. Currently OK as compiled_observed for density only.",
     "LOW"),
    ("train_set / test_set (EMDP)",           "compiled_observed",  "Mixed (density experimental, D/P/Q from EXPLO5)",
     "ρ is XRD; DetoD/DetoP/DetoQ are EXPLO5 thermochemistry outputs — strictly model_predicted, not experimental.",
     "MEDIUM"),
    ("CHNOClF_dataset",                       "compiled_observed",  "Mostly model-predicted",
     "Long-format (SMILES, prop, value). Most rows are model outputs — only 3 survived dedup. Low impact but provenance should be verified.",
     "LOW"),
    ("cm4c01978_si_001",                      "kj_calculated",      "Correct",
     "Formula-derived K-J. Label matches data nature.",
     "–"),
    ("denovo_sampling_{rl,tl}",               "model_predicted",    "Correct",
     "ML generator + predictor. Label matches.",
     "–"),
    ("generation",                            "model_predicted",    "Correct (after explosion_heat unit fix)",
     "MDGNN outputs; unit bug patched.",
     "–"),
    ("q-RASPR",                               "qsar_predicted",     "Correct",
     "QSAR outputs. Label matches.",
     "–"),
], columns=["Source", "Current label", "Verdict", "Notes", "Severity"])

provenance_note = ""   # historical audit findings — all resolved; see tier column

method_overview = """<div class="audit-box info">
<strong>Prediction method primer:</strong>
<ul style="margin-top:6px;margin-left:20px">
<li><strong>DFT (density functional theory)</strong> — first-principles quantum chemistry.
Input: 3D molecular structure. Output: electronic energy, HOMO/LUMO, dipole, predicted density
via equation-of-state. Reliability: high for single-molecule properties (±1–2 kcal/mol HOF),
but crystal density requires additional lattice modeling (±3–5%).</li>
<li><strong>K-J formula (Kamlet-Jacobs)</strong> — deterministic algebraic formulas relating
density and composition to detonation velocity/pressure. Input: density + molecular formula
(+ HOF for Q). Output: D, P. Reliability: ±10–20% for CHNO with normal detonation products.</li>
<li><strong>3DCNN</strong> — 3D convolutional neural network trained on voxelized
molecular geometries to predict DFT-computed properties. Input: 3D grid of atomic densities.
Output: HOF, density. Reliability: cheap surrogate for DFT with ±3% MAE on in-distribution
molecules; degrades on out-of-distribution scaffolds.</li>
<li><strong>Generative models (MDGNN, RL/TL de novo)</strong> — seq2seq RNNs or graph-based
generators that propose novel SMILES, paired with downstream property predictors (usually
MLP/GNN on ECFP+descriptors). Input: learned latent space. Output: novel SMILES + predicted
properties. Reliability: predictions are only as good as the paired predictor; typical
RMSE ±0.3 km/s D, ±4 GPa P for in-distribution CHNO.</li>
<li><strong>QSAR (q-RASPR)</strong> — Quasi-SMILES Quantitative Structure-Property Reactivity
regression. Paper: Toropov &amp; Toropova (OCHEM / CORAL line). Method: a SMILES string is
encoded as a sequence of "SMILES attributes" (individual atoms, pairs, triples, connectors)
plus optional extended attributes (nanoparticle size, temperature, solvent). A Monte-Carlo
optimization fits a linear-regression on the correlation weights of those attributes to
training-set targets. Input: quasi-SMILES + 2D/3D descriptors. Output: density, D, P,
impact sensitivity. Training data: 200–500 experimentally characterized explosives from
Huang-Massa + RSC compilations. Reliability: RMSE ±0.08 g/cm³ ρ, ±0.5 km/s D, ±5 GPa P on
blind test; strong applicability-domain constraint — extrapolation to novel CHNO backbones
is unreliable.</li>
<li><strong>RNNMGM</strong> — Recurrent Neural Network Molecular Generator Model.
Paper: Kang &amp; Cho 2019 (extended for energetics by the Zenodo compendium authors). A
character-level LSTM RNN trained on canonical SMILES of known energetics then sampled with
temperature-controlled random generation. Input: learned token embeddings. Output: novel
SMILES strings (no property labels — RNNMGM is a generator, not a predictor). The raw
<code>rnnmgm_ds9</code> pool of 141,369 molecules populates the UNLABELED master. Reliability
of generation: ≈85–90% of samples parse as valid SMILES; proxy-score ranking is used as a
weak post-hoc energetic-relevance filter. Not used for property labels in labeled_master —
only as a pretraining pool. (Note: the <code>Dm.csv</code> file under
<code>RNNMGM_upstream/data/</code> is the reference table of known explosives bundled with
the RNNMGM repo, not RNNMGM output.)</li>
</ul>
</div>"""

# Method-oriented semantic classification
method_df = pd.DataFrame([
    ("3DCNN",
     "3D CNN regressor (after fix)", "dft_surrogate_ml",
     "Voxelised 3D molecular geometry (fixed grid of atomic densities)",
     "density, HOF (trained on DFT targets from GDB-9 / energetic set)",
     "~3% MAE in-distribution; degrades OOD",
     "26,254 rows"),
    ("5039 (density)",
     "Literature compilation", "experimental_density",
     "Canonical SMILES",
     "crystallographic density from open references",
     "high (<2% typical crystal error)",
     "4,582 density values"),
    ("5039 (BDE)",
     "Computed descriptor", "qm_derived_scalar",
     "SMILES → 3D optimization → homolytic bond dissociation energy",
     "Bond Dissociation Energy (kcal/mol)",
     "DFT-level ±1 kcal/mol",
     "bundled with 5039"),
    ("EMDP train_set / test_set (density)",
     "CSD crystallography (cm4c01978 SI)", "experimental_density",
     "CSD refcode → crystal unit cell",
     "ρ from XRD lattice",
     "very high (<1% typical)",
     "~2,000 rows"),
    ("EMDP train_set / test_set (DetoD/DetoP/DetoQ)",
     "EXPLO5 thermochemistry code", "thermochem_formula",
     "Molecular formula + HOF estimate + density + CJ detonation model",
     "D, P, Q",
     "~5% on well-behaved CHNO; worse OOD",
     "~2,000 rows"),
    ("det_dataset_08-02-2022",
     "Literature compilation (RSC Adv. 2022)", "experimental_full",
     "Published SMILES + measured TMD + experimental detonation",
     "ρ, D, P, Q from shock-tube / plate-dent",
     "experimental reference quality",
     "260 rows"),
    ("Dm",
     "Named-compound reference (RNNMGM repo)", "experimental_curated",
     "Explosive name → resolved SMILES",
     "ρ, HOF curated from handbooks",
     "handbook-level (2–5% typical)",
     "247 rows"),
    ("emdb_v21_molecules_pubchem",
     "EMDB v2.1 + PubChem resolver", "experimental_full",
     "Literature SMILES resolved via PubChem",
     "ρ, HOF (solid + gas)",
     "heterogeneous; literature-dependent",
     "109 rows"),
    ("combined_data",
     "Coulomb-Matrix notebook compilation", "experimental_curated",
     "Name + annotated SMILES",
     "ρ, gas-phase HOF, explosive energy, D, P",
     "literature curation; notes column flags dubious rows",
     "79 rows"),
    ("Huang_Massa_data_with_all_SMILES",
     "Huang & Massa classical compilation", "experimental_curated",
     "Name + manually added SMILES",
     "ρ, HOF_solid, shock/particle velocity, P",
     "reference quality for classical explosives",
     "2 rows (rest deduped into combined_data/det_dataset)"),
    ("CHNOClF_dataset",
     "MultiTaskEM long-format",      "mixed_predicted",
     "SMILES → DFT optimized descriptors",
     "target property per row",
     "varies per property",
     "3 rows"),
    ("cm4c01978_si_001 (K-J)",
     "Kamlet-Jacobs algebraic formula", "formula_deterministic",
     "Molecular formula (CHNO counts) + density + HOF estimate",
     "D, P, Q",
     "±10–20% D, ±20–30% P for ideal CHNO",
     "12,040 rows"),
    ("denovo_sampling_rl",
     "Seq2seq RNN generator + RL fine-tune + MLP property predictor",
     "ml_generator_plus_predictor",
     "Learned latent → SMILES; ECFP+descriptors for properties",
     "Novel SMILES + predicted ρ, D, P",
     "predictor ±0.3 km/s D, ±4 GPa P in-distribution",
     "6,097 rows"),
    ("denovo_sampling_tl",
     "Seq2seq RNN + transfer-learning fine-tune + same predictor",
     "ml_generator_plus_predictor",
     "Same as RL variant; TL seed from known energetics",
     "Novel SMILES + predicted ρ, D, P",
     "typically more reliable than RL (closer to training)",
     "4,922 rows"),
    ("generation (MDGNN)",
     "Graph-based generator + property head", "ml_generator_plus_predictor",
     "Latent → graph → SMILES",
     "ρ, D, P, explosion_heat (unit-fixed)",
     "generator-dependent",
     "7,072 rows"),
    ("q-RASPR",
     "QSAR regression on quasi-SMILES + descriptors",
     "qsar_regression",
     "Quasi-SMILES attributes + 2D/3D descriptors",
     "ρ, D, P, impact sensitivity",
     "±0.08 g/cm³ ρ, ±0.5 km/s D, ±5 GPa P; narrow applicability domain",
     "2,438 rows"),
    ("rnnmgm_ds9 (unlabeled)",
     "Character-level LSTM generator (RNNMGM)", "ml_generator_unlabeled",
     "Learned character embeddings",
     "Novel SMILES only (NO property labels)",
     "~85–90% syntactic validity; quality proxy score only",
     "141,369 rows (unlabeled pool)"),
    ("qm9, chembl_36, pubchem_cid_smiles, guacamol_v1_train (unlabeled)",
     "Public chemistry databases / benchmarks", "public_database_unlabeled",
     "Canonical SMILES only",
     "No properties",
     "varies — pretraining-quality structural diversity",
     "559k rows across these pools"),
], columns=["Source", "Method (family)", "Semantic label", "Input",
            "Output (what is predicted/stored)", "Reliability", "Rows in pipeline"])

semantic_note = """<div class="audit-box info">
<strong>Method-oriented semantic labels</strong> are more informative than the current
4-category <em>label_source_type</em> because they distinguish:
<em>experimental_full</em> (measured ρ+D+P), <em>experimental_density</em> (only ρ measured),
<em>experimental_curated</em> (literature reference quality), <em>formula_deterministic</em>
(K-J and similar algebraic models), <em>thermochem_formula</em> (EXPLO5-type CJ codes),
<em>qm_derived_scalar</em> / <em>dft_surrogate_ml</em> (DFT or DFT-trained NNs),
<em>qsar_regression</em> (descriptor-based QSAR), and <em>ml_generator_plus_predictor</em>
(generative models with their own property heads). The table below maps every source in the
pipeline to its method family. Downstream training could use these finer tags to weight
losses or segment validation.
</div>"""

sec13 = (audit_note +
         "<h3>Prediction method primer</h3>" +
         method_overview +
         "<h3>Method-oriented classification of every source</h3>" +
         semantic_note +
         table_html(method_df, "Every source mapped to method family, input, output, reliability") +
         "<h3>Source descriptions, origin, and inclusion criteria</h3>" +
         table_html(desc_df, "Per-source provenance (all label sources)") +
         "<h3>Counter reconciliation (experimental sources)</h3>" +
         table_html(audit_df, "Experimental counter audit: raw sources vs labeled_master"))

# ── 14. UNLABELED ↔ LABELED OVERLAP and ENERGETIC CONTENT ────────────────────

print("  analyzing labeled/unlabeled overlap …")
um_full = pd.read_csv(f"{BASE}/data/training/master/unlabeled_master.csv", low_memory=False,
                      usecols=["smiles","source_dataset","energetic_proxy_score",
                               "has_nitro","has_azide","n_count","o_count"])
um_full["has_nitro"] = um_full["has_nitro"].astype(str).str.lower().isin(["true","1"])
um_full["has_azide"] = um_full["has_azide"].astype(str).str.lower().isin(["true","1"])

_lm_set = set(lm["smiles"].dropna().unique())
_um_set = set(um_full["smiles"].dropna().unique())
_ovl    = _lm_set & _um_set

n_lm, n_um, n_ov = len(_lm_set), len(_um_set), len(_ovl)

um_only = um_full[~um_full["smiles"].isin(_lm_set)].copy()
n_only  = len(um_only)

n_nitro = int(um_only["has_nitro"].sum())
n_azide = int(um_only["has_azide"].sum())
n_chno  = int(((um_only["n_count"]>=3) & (um_only["o_count"]>=4)).sum())
n_p5    = int((um_only["energetic_proxy_score"]>0.5).sum())
n_p7    = int((um_only["energetic_proxy_score"]>0.7).sum())

# summary table
overlap_summary = pd.DataFrame([
    ("Labeled master — unique SMILES",               f"{n_lm:,}", "–"),
    ("Unlabeled master — unique SMILES",             f"{n_um:,}", "–"),
    ("Overlap (SMILES in both)",                     f"{n_ov:,}", f"{100*n_ov/n_um:.3f}% of unlabeled"),
    ("Unlabeled-only (disjoint from labeled)",       f"{n_only:,}", f"{100*n_only/n_um:.2f}% of unlabeled"),
], columns=["Set","Count","Share"])

# overlap breakdown by unlabeled source
ov_src = (um_full[um_full["smiles"].isin(_ovl)]["source_dataset"]
          .value_counts().reset_index())
ov_src.columns = ["Unlabeled source","Overlap count"]
ov_src["% of source"] = ov_src.apply(
    lambda r: f"{100*r['Overlap count']/(um_full['source_dataset']==r['Unlabeled source']).sum():.3f}%",
    axis=1)
ov_src["Overlap count"] = ov_src["Overlap count"].map("{:,}".format)

# energetic content in unlabeled-only
energ_df = pd.DataFrame([
    ("Has nitro group (-NO₂)",               f"{n_nitro:,}", f"{100*n_nitro/n_only:.2f}%"),
    ("Has azide group (-N₃)",                f"{n_azide:,}", f"{100*n_azide/n_only:.2f}%"),
    ("N≥3 AND O≥4 (CHNO backbone)",          f"{n_chno:,}",  f"{100*n_chno/n_only:.2f}%"),
    ("Energetic proxy score > 0.5",          f"{n_p5:,}",    f"{100*n_p5/n_only:.2f}%"),
    ("Energetic proxy score > 0.7",          f"{n_p7:,}",    f"{100*n_p7/n_only:.2f}%"),
], columns=["Signal","Unlabeled-only count","Share"])

# energetic content by source
en_by_src = (um_only.groupby("source_dataset")
              .agg(total=("smiles","count"),
                   nitro=("has_nitro","sum"),
                   azide=("has_azide","sum"),
                   chno_rich=("n_count", lambda s: ((s>=3) & (um_only.loc[s.index,"o_count"]>=4)).sum()),
                   mean_proxy=("energetic_proxy_score","mean"))
              .reset_index()
              .sort_values("total", ascending=False))
en_by_src["% nitro"] = (100 * en_by_src["nitro"] / en_by_src["total"]).map("{:.2f}%".format)
en_by_src["% CHNO-rich"] = (100 * en_by_src["chno_rich"] / en_by_src["total"]).map("{:.2f}%".format)
en_by_src["mean_proxy"] = en_by_src["mean_proxy"].map("{:.3f}".format)
for c in ("total","nitro","azide","chno_rich"):
    en_by_src[c] = en_by_src[c].map("{:,}".format)
en_by_src = en_by_src.rename(columns={
    "source_dataset":"Source","total":"Total","nitro":"Nitro",
    "azide":"Azide","chno_rich":"CHNO-rich","mean_proxy":"Mean proxy"})
en_by_src = en_by_src[["Source","Total","Nitro","% nitro","Azide",
                        "CHNO-rich","% CHNO-rich","Mean proxy"]]

# ── figures ───────────────────────────────────────────────────────────────────
fig_overlap = go.Figure(data=[go.Pie(
    labels=["Unlabeled-only", "Overlap with labeled"],
    values=[n_only, n_ov],
    hole=0.55,
    marker_colors=["#607D8B","#FF9800"],
    textinfo="label+value+percent")])
fig_overlap.update_layout(
    title=dict(text=f"Unlabeled master: overlap with labeled (total {n_um:,})",
               y=0.98, yanchor="top"),
    height=400, margin=dict(t=80, b=40),
    showlegend=False)

# energetic-signal bar chart
fig_energ = go.Figure()
fig_energ.add_trace(go.Bar(
    x=["Has nitro","Has azide","CHNO-rich (N≥3,O≥4)","Proxy>0.5","Proxy>0.7"],
    y=[n_nitro, n_azide, n_chno, n_p5, n_p7],
    text=[f"{v:,}" for v in [n_nitro, n_azide, n_chno, n_p5, n_p7]],
    textposition="outside",
    marker_color=["#FF5722","#9C27B0","#3F51B5","#FF9800","#FFC107"]))
fig_energ.update_layout(
    title=dict(text=f"Energetic-signal molecules in unlabeled-only pool (n={n_only:,})",
               y=0.98, yanchor="top"),
    yaxis_title="Molecules",
    height=400, margin=dict(t=80, b=60))

# proxy score histogram comparison
fig_proxy_cmp = go.Figure()
fig_proxy_cmp.add_trace(go.Histogram(
    x=um_only["energetic_proxy_score"].dropna(),
    nbinsx=40, histnorm="probability density",
    marker_color="#607D8B", opacity=0.65, name="Unlabeled-only"))
fig_proxy_cmp.add_trace(go.Histogram(
    x=lm["energetic_proxy_score"].dropna(),
    nbinsx=40, histnorm="probability density",
    marker_color="#4CAF50", opacity=0.65, name="Labeled"))
fig_proxy_cmp.update_layout(
    barmode="overlay",
    title=dict(text="Energetic proxy score: labeled vs unlabeled-only",
               y=0.98, yanchor="top"),
    xaxis_title="Proxy score", yaxis_title="Density",
    height=380, margin=dict(t=80, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))

overlap_note = f"""<div class="audit-box info">
<strong>Overlap & energetic coverage:</strong> labeled and unlabeled masters are effectively
<em>disjoint</em> — only {n_ov:,} SMILES ({100*n_ov/n_um:.3f}% of unlabeled) appear in both.
This means the unlabeled pool is a nearly pure source of pretraining signal with no label leakage.
<br><br>
The unlabeled-only pool contains <strong>{n_nitro:,} nitro-bearing molecules</strong>,
<strong>{n_azide:,} azides</strong>, and <strong>{n_chno:,} CHNO-rich molecules (N≥3, O≥4)</strong>
— strong candidates for novel energetic materials with completely unknown detonation properties.
The generative model will see these during pretraining and can propose variants for downstream
K-J / ML property screening.
</div>"""

sec14 = (
    overlap_note +
    table_html(overlap_summary, "Labeled vs unlabeled set overlap") +
    fig_to_html(fig_overlap, "fig_overlap") +
    "<h3>Overlap breakdown by unlabeled source</h3>" +
    table_html(ov_src, "SMILES shared with labeled_master, by unlabeled source") +
    "<h3>Energetic content in unlabeled-only pool</h3>" +
    table_html(energ_df, "Energetic-signal molecules (unlabeled-only)") +
    fig_to_html(fig_energ, "fig_energ") +
    fig_to_html(fig_proxy_cmp, "fig_proxy_cmp") +
    table_html(en_by_src, "Energetic content by unlabeled source (unlabeled-only subset)")
)

# ── 15. BUTINA CLUSTERING (ECFP4 + Tanimoto 0.4) ──────────────────────────────
import pickle as _pickle
_bc_path = f"{BASE}/data/training/metadata/butina_clusters.pkl"
try:
    with open(_bc_path, "rb") as _f:
        _bc = _pickle.load(_f)
    _has_clusters = True
except FileNotFoundError:
    _has_clusters = False

if _has_clusters:
    bsamp    = _bc["sample"]
    bclust   = _bc["clusters"]
    bcutoff  = _bc["cutoff"]
    n_total  = len(bsamp)
    n_clust  = len(bclust)
    n_single = sum(1 for c in bclust if len(c) == 1)

    # cluster sizes
    size_df = pd.Series([len(c) for c in bclust]).value_counts().sort_index()
    fig_csize = go.Figure(go.Bar(
        x=size_df.index, y=size_df.values,
        marker_color="#4f8ef7",
        text=[f"{v:,}" if v > max(size_df.values)*0.01 else "" for v in size_df.values],
        textposition="outside"))
    fig_csize.update_layout(
        title=dict(text=(f"Cluster size distribution "
                         f"(total {n_clust:,} clusters; "
                         f"{n_single:,} singletons at size=1, "
                         f"largest cluster size = {size_df.index.max()} molecules)"),
                   y=0.98, yanchor="top"),
        xaxis_title="Cluster size (molecules per cluster — log scale)",
        yaxis_title="Number of clusters with this size (log scale)",
        xaxis_type="log", yaxis_type="log",
        height=400, margin=dict(t=100, b=70))

    # singleton rate by bucket
    sing_ids = {c[0] for c in bclust if len(c) == 1}
    bsamp["is_singleton"] = bsamp.index.isin(sing_ids)
    sing_by_b = bsamp.groupby("bucket").agg(
        n=("smiles", "count"),
        singletons=("is_singleton", "sum")).reset_index()
    sing_by_b["Singleton rate"] = (100*sing_by_b["singletons"]/sing_by_b["n"]).map("{:.1f}%".format)
    sing_by_b.columns = ["Bucket","N in sample","Singletons","Singleton rate"]

    # bucket colors
    BUCKET_COLORS = {**SOURCE_COLORS, "unlabeled": "#607D8B"}
    BUCKET_LABELS = {**SOURCE_LABELS, "unlabeled": "Unlabeled"}

    fig_sing = go.Figure()
    for _, r in sing_by_b.iterrows():
        b = r["Bucket"]
        fig_sing.add_trace(go.Bar(
            x=[BUCKET_LABELS.get(b, b)],
            y=[100*int(r["Singletons"].__index__() if hasattr(r['Singletons'],'__index__') else r['Singletons']) /
               int(r['N in sample'])],
            marker_color=BUCKET_COLORS.get(b, "#888"),
            name=BUCKET_LABELS.get(b, b),
            text=[r["Singleton rate"]], textposition="outside"))
    fig_sing.update_layout(
        title=dict(text="Singleton rate by source bucket (higher → more structurally unique)",
                   y=0.98, yanchor="top"),
        yaxis_title="Singleton rate (%)",
        height=380, margin=dict(t=80, b=60), showlegend=False)

    # cluster composition: top 30 clusters, stacked-bar by bucket
    top_k = 30
    top_ids = sorted(range(len(bclust)), key=lambda i: -len(bclust[i]))[:top_k]
    comp_rows = []
    for rank, cid in enumerate(top_ids):
        members = bclust[cid]
        buckets = bsamp.iloc[members]["bucket"].value_counts().to_dict()
        row = {"Cluster": f"#{rank+1} (n={len(members)})"}
        row.update(buckets)
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows).fillna(0)
    fig_comp = go.Figure()
    for b in BUCKET_COLORS:
        if b in comp_df.columns:
            fig_comp.add_trace(go.Bar(
                x=comp_df["Cluster"], y=comp_df[b],
                name=BUCKET_LABELS.get(b, b),
                marker_color=BUCKET_COLORS[b]))
    fig_comp.update_layout(
        barmode="stack",
        title=dict(text="Top 30 largest clusters — composition by source bucket",
                   y=0.98, yanchor="top"),
        xaxis_title="Cluster (ranked by size)", yaxis_title="Molecules",
        height=440, margin=dict(t=100, b=80),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10)))

    # nearest-neighbor Tanimoto distribution (similarity)
    fig_nn = go.Figure()
    for b in BUCKET_COLORS:
        sub = bsamp[bsamp["bucket"] == b]["nn_tanimoto"].dropna()
        if sub.empty:
            continue
        fig_nn.add_trace(go.Histogram(
            x=sub, nbinsx=40, histnorm="probability density",
            name=BUCKET_LABELS.get(b, b),
            marker_color=BUCKET_COLORS[b], opacity=0.6))
    fig_nn.update_layout(
        barmode="overlay",
        title=dict(text=f"Within-cluster nearest-neighbor Tanimoto similarity (sample of {n_total:,})",
                   y=0.98, yanchor="top"),
        xaxis_title="Tanimoto similarity to cluster center",
        yaxis_title="Density",
        height=400, margin=dict(t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10)))

    # "novel chemical space" donut: clusters where ONLY unlabeled appears
    unlab_only = sum(1 for c in bclust
                     if set(bsamp.iloc[list(c)]["bucket"]) == {"unlabeled"})
    label_only = sum(1 for c in bclust
                     if "unlabeled" not in set(bsamp.iloc[list(c)]["bucket"]))
    mixed      = n_clust - unlab_only - label_only
    fig_novel = go.Figure(data=[go.Pie(
        labels=["Unlabeled-only (novel space)", "Mixed (overlap with labeled)",
                "Labeled-only (covered by measurements/predictions)"],
        values=[unlab_only, mixed, label_only],
        hole=0.5,
        marker_colors=["#607D8B", "#FF9800", "#4CAF50"])])
    fig_novel.update_layout(
        title=dict(text=f"Chemical-space coverage — {n_clust:,} clusters",
                   y=0.98, yanchor="top"),
        height=400, margin=dict(t=80, b=40),
        legend=dict(orientation="v", x=1.02, y=0.5))

    # Directional cluster-coverage matrix: cell[i,j] = share of source i's
    # clusters that are ALSO reached by source j. Row i reads as
    # "of the clusters containing at least one molecule from i, what % are
    # also populated by j". Diagonal is by definition 100%.
    bucket_clusters = {b: set(bsamp[bsamp["bucket"] == b]["cluster"])
                       for b in BUCKET_COLORS if b in bsamp["bucket"].unique()}
    blist = list(bucket_clusters.keys())
    cov_mat = np.zeros((len(blist), len(blist)))
    for i, a in enumerate(blist):
        base = bucket_clusters[a]
        if not base:
            continue
        for j, b in enumerate(blist):
            inter = base & bucket_clusters[b]
            cov_mat[i, j] = 100 * len(inter) / len(base)
    fig_overlap_mat = px.imshow(cov_mat,
        x=[BUCKET_LABELS.get(b, b) for b in blist],
        y=[BUCKET_LABELS.get(b, b) for b in blist],
        color_continuous_scale="Blues", zmin=0, zmax=100,
        text_auto=".0f",
        labels=dict(x="Also reached by →", y="Source clusters ↓", color="% of row's clusters"),
        title="Cluster reachability: % of row-source's clusters also hit by column-source")
    fig_overlap_mat.update_layout(height=480, margin=dict(t=120, b=80))

    # Companion figure: share of each source's molecules that live in
    # single-source (pure) vs mixed-source clusters — a direct "how embedded
    # is this source in the rest of chemical space" view
    pure_mix_rows = []
    for b in blist:
        sub = bsamp[bsamp["bucket"] == b]
        if sub.empty:
            continue
        cluster_to_buckets = (bsamp.groupby("cluster")["bucket"]
                                  .apply(lambda s: set(s)).to_dict())
        pure   = sum(1 for c in sub["cluster"] if cluster_to_buckets[c] == {b})
        mixed  = len(sub) - pure
        pure_mix_rows.append({
            "Source":        BUCKET_LABELS.get(b, b),
            "Pure-source":   pure,
            "Mixed-source":  mixed,
            "% mixed":       f"{100*mixed/len(sub):.1f}%" if len(sub) else "–",
        })
    pure_mix_df = pd.DataFrame(pure_mix_rows)
    fig_purity = go.Figure()
    fig_purity.add_bar(
        x=pure_mix_df["Source"], y=pure_mix_df["Pure-source"],
        name="Pure-source cluster", marker_color="#EF5350")
    fig_purity.add_bar(
        x=pure_mix_df["Source"], y=pure_mix_df["Mixed-source"],
        name="Shared with another source", marker_color="#66BB6A")
    fig_purity.update_layout(
        barmode="stack",
        title=dict(text="Molecules per source: clusters shared with other sources vs pure-source",
                   y=0.98, yanchor="top"),
        yaxis_title="Molecules", height=400, margin=dict(t=100, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10)))

    # top clusters table
    top_table_rows = []
    for rank, cid in enumerate(top_ids[:20]):
        members = bclust[cid]
        buckets_str = ", ".join(f"{k}:{v}" for k, v in
                                 bsamp.iloc[members]["bucket"].value_counts().items())
        top_table_rows.append({
            "Rank":       rank + 1,
            "Size":       len(members),
            "Centroid SMILES (truncated 55)": bsamp.iloc[members[0]]["smiles"][:55],
            "Source mix": buckets_str,
        })
    top_tbl = pd.DataFrame(top_table_rows)

    butina_note = f"""<div class="audit-box info">
    <strong>Butina clustering</strong> (Butina 1999): ECFP4 fingerprints (Morgan radius=2,
    2048 bits), pairwise Tanimoto distance, cutoff = <code>{bcutoff}</code>. Sample size
    <code>{n_total:,}</code> stratified across labeled buckets + unlabeled. Greedy algorithm:
    pick the molecule with the most neighbors within the cutoff, assign its neighbors to its
    cluster, remove, repeat. Singletons are molecules with zero neighbors below the threshold —
    a high singleton rate means that bucket contributes scaffolds not seen elsewhere in the
    sample (good for diversity, bad for generalization).
    <br><br>
    Per-source view: each cluster is categorised by which buckets contribute members.
    "Unlabeled-only" clusters are regions of chemical space the generative model will see in
    pretraining but where no labels (measured or predicted) exist — prime targets for novel
    energetic materials.
    </div>"""

    sec15 = (
        butina_note +
        "<h3>Cluster size distribution</h3>" +
        fig_to_html(fig_csize, "fig_csize") +
        "<h3>Singleton rate by source bucket</h3>" +
        table_html(sing_by_b, "Singleton counts and rate per bucket") +
        fig_to_html(fig_sing, "fig_sing") +
        "<h3>Top 30 clusters — composition by source bucket</h3>" +
        fig_to_html(fig_comp, "fig_comp") +
        "<h3>Within-cluster Tanimoto similarity</h3>" +
        fig_to_html(fig_nn, "fig_nn") +
        "<h3>Chemical-space coverage — novel vs overlapping</h3>" +
        fig_to_html(fig_novel, "fig_novel") +
        "<h3>Cross-source cluster reachability</h3>" +
        fig_to_html(fig_overlap_mat, "fig_overlap_mat") +
        table_html(pure_mix_df, "Pure-source vs shared molecules per bucket") +
        fig_to_html(fig_purity, "fig_purity") +
        "<h3>Top 20 clusters by size</h3>" +
        table_html(top_tbl, "Largest clusters and their source composition")
    )
else:
    sec15 = """<div class="audit-box warning">
    <strong>Butina clusters not pre-computed.</strong> Run
    <code>scripts/compute_butina_clusters.py</code> to generate
    <code>data/training/metadata/butina_clusters.pkl</code>, then re-run this EDA script.
    </div>"""

# ── 16. HYPOTHETICAL TRAINING-SET VARIANTS ───────────────────────────────────
#
# Three variants showing the quality-vs-quantity trade-off for regression
# training:
#
#   Variant A (gold): Tier A + B only    (measured + physics simulation)
#   Variant B (+ML):  A + B + Tier D for density and HOF only
#                     (adds 3DCNN and other ML predictors for these two,
#                     but NOT for detonation properties or explosion heat)
#   Variant C (+KJ):  Variant B, then K-J formula fills missing D/P/Q
#                     for any row with ρ + HOF available
#
# This is an ANALYTICAL projection — does not modify labeled_master.csv.

import numpy as np

def _tier_ok(row, prop, tiers):
    t = row.get(f"{prop}_tier")
    return (t in tiers) and pd.notna(row.get(prop))

rows_sim_cast = lm.copy()

# Variant A mask: property kept if tier in {A, B}
# Variant B mask: same for D/P/Q; density and HOF also accept Tier D (ML)
# Variant C mask: Variant B plus K-J-imputable D / P from (ρ+HOF)

def _count(df, prop, tiers):
    t = df[f"{prop}_tier"]
    return ((t.isin(list(tiers))) & df[prop].notna()).sum()

variant_rows = []
for prop in PROPS:
    a = int(_count(lm, prop, {"A","B"}))
    b = int(_count(lm, prop, {"A","B"})) + (
            int(((lm[f"{prop}_tier"]=="D") & lm[prop].notna()).sum())
            if prop in ("density", "heat_of_formation") else 0)
    # Variant C: for det_velocity / det_pressure, count rows where
    # both density and HOF are in Variant-B coverage.
    if prop in ("detonation_velocity", "detonation_pressure"):
        # start from B count for this prop
        c_base = b
        # add K-J-imputable: rows with rho (any A/B/D) AND HOF (any A/B/D)
        # AND currently missing this property at A/B/D tier
        has_rho = (lm["density_tier"].isin(["A","B","D"])) & lm["density"].notna()
        has_hof = (lm["heat_of_formation_tier"].isin(["A","B","D"])) & lm["heat_of_formation"].notna()
        has_this = (lm[f"{prop}_tier"].isin(["A","B","D"])) & lm[prop].notna()
        kj_addable = (has_rho & has_hof & ~has_this).sum()
        c = c_base + int(kj_addable)
    elif prop == "explosion_heat":
        c_base = b
        has_rho = (lm["density_tier"].isin(["A","B","D"])) & lm["density"].notna()
        has_hof = (lm["heat_of_formation_tier"].isin(["A","B","D"])) & lm["heat_of_formation"].notna()
        has_this = (lm[f"{prop}_tier"].isin(["A","B","D"])) & lm[prop].notna()
        kj_addable = (has_rho & has_hof & ~has_this).sum()
        c = c_base + int(kj_addable)
    else:
        # density / HOF: Variant C doesn't add anything new for these
        c = b
    variant_rows.append({
        "Property":                       PROP_LABELS[prop],
        "Variant A (A+B only)":           f"{a:,}",
        "Variant B (+ ML ρ and HOF)":     f"{b:,}",
        "Variant C (+ K-J D, P, Q)":      f"{c:,}",
    })
var_df = pd.DataFrame(variant_rows)

# Row-level: molecules with n-of-5 properties filled
def _rowfilled(df, prop_tier_allowed):
    """Return Series of number-of-properties-filled per row under a rule-of-tiers dict."""
    counts = pd.Series(0, index=df.index)
    for p, tiers in prop_tier_allowed.items():
        ok = (df[f"{p}_tier"].isin(list(tiers))) & df[p].notna()
        counts = counts + ok.astype(int)
    return counts

def _rowfilled_C(df):
    """For variant C: use A/B for all props, D for density+HOF, and
    K-J-imputable counts as 'filled' for D/P/Q where ρ + HOF exist."""
    counts = pd.Series(0, index=df.index)
    for p in PROPS:
        allowed = {"A","B"}
        if p in ("density","heat_of_formation"):
            allowed = {"A","B","D"}
        base_ok = df[f"{p}_tier"].isin(list(allowed)) & df[p].notna()
        if p in ("detonation_velocity","detonation_pressure","explosion_heat"):
            rho_ok = df["density_tier"].isin(["A","B","D"]) & df["density"].notna()
            hof_ok = df["heat_of_formation_tier"].isin(["A","B","D"]) & df["heat_of_formation"].notna()
            kj_ok = rho_ok & hof_ok & ~base_ok
            base_ok = base_ok | kj_ok
        counts = counts + base_ok.astype(int)
    return counts

rule_A = {p: {"A","B"} for p in PROPS}
rule_B = {p: ({"A","B","D"} if p in ("density","heat_of_formation") else {"A","B"}) for p in PROPS}

nA = _rowfilled(lm, rule_A)
nB = _rowfilled(lm, rule_B)
nC = _rowfilled_C(lm)

row_summary = pd.DataFrame([
    ("0 props", f"{(nA==0).sum():,}", f"{(nB==0).sum():,}", f"{(nC==0).sum():,}"),
    ("1 prop",  f"{(nA==1).sum():,}", f"{(nB==1).sum():,}", f"{(nC==1).sum():,}"),
    ("2 props", f"{(nA==2).sum():,}", f"{(nB==2).sum():,}", f"{(nC==2).sum():,}"),
    ("3 props", f"{(nA==3).sum():,}", f"{(nB==3).sum():,}", f"{(nC==3).sum():,}"),
    ("4 props", f"{(nA==4).sum():,}", f"{(nB==4).sum():,}", f"{(nC==4).sum():,}"),
    ("5 props", f"{(nA==5).sum():,}", f"{(nB==5).sum():,}", f"{(nC==5).sum():,}"),
    ("≥1 prop", f"{(nA>=1).sum():,}", f"{(nB>=1).sum():,}", f"{(nC>=1).sum():,}"),
    ("≥4 props",f"{(nA>=4).sum():,}", f"{(nB>=4).sum():,}", f"{(nC>=4).sum():,}"),
], columns=["Per-molecule coverage", "Variant A", "Variant B", "Variant C"])

# grouped bar: per-property coverage by variant
fig_var = go.Figure()
for name, col, values in [
    ("Variant A (A+B only)",       "#4CAF50",
     [int(_count(lm, p, {"A","B"})) for p in PROPS]),
    ("Variant B (+ML density,HOF)","#2196F3",
     [int(_count(lm, p, {"A","B"})) +
      (int(((lm[f"{p}_tier"]=="D") & lm[p].notna()).sum())
       if p in ("density","heat_of_formation") else 0) for p in PROPS]),
    ("Variant C (+K-J D,P,Q)",     "#FF9800",
     [int(
         int(_count(lm, p, {"A","B"})) +
         (int(((lm[f"{p}_tier"]=="D") & lm[p].notna()).sum())
          if p in ("density","heat_of_formation") else 0) +
         (((lm["density_tier"].isin(["A","B","D"])) & lm["density"].notna() &
           (lm["heat_of_formation_tier"].isin(["A","B","D"])) & lm["heat_of_formation"].notna() &
           ~((lm[f"{p}_tier"].isin(["A","B","D"])) & lm[p].notna())).sum()
          if p in ("detonation_velocity","detonation_pressure","explosion_heat") else 0)
      ) for p in PROPS]),
]:
    fig_var.add_bar(x=[PROP_LABELS[p] for p in PROPS], y=values, name=name,
                    marker_color=col, text=[f"{v:,}" for v in values],
                    textposition="outside")
fig_var.update_layout(
    barmode="group",
    title=dict(text="Hypothetical training-set variants — per-property row count",
               y=0.98, yanchor="top"),
    yaxis_title="Molecules with this property available",
    height=460, margin=dict(t=100, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=10)))

# row-level figure (n-of-5 distribution)
fig_var_row = go.Figure()
for name, col, s in [
    ("Variant A", "#4CAF50", nA),
    ("Variant B", "#2196F3", nB),
    ("Variant C", "#FF9800", nC),
]:
    vc = s.value_counts().sort_index()
    fig_var_row.add_bar(x=vc.index, y=vc.values, name=name,
                         marker_color=col, opacity=0.85,
                         text=[f"{v:,}" for v in vc.values], textposition="outside")
fig_var_row.update_layout(
    barmode="group",
    title=dict(text="Per-molecule property completeness — by variant",
               y=0.98, yanchor="top"),
    xaxis_title="Number of properties labeled (out of 5)",
    yaxis_title="Molecules", height=420, margin=dict(t=100, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=10)))

variant_note = """<div class="audit-box info">
<strong>Three hypothetical training-set variants</strong> — pick based on the
quality-quantity trade-off you want:
<ul style="margin-top:6px;margin-left:20px">
<li><strong>Variant A — gold standard:</strong> only Tier A (experimental) and
Tier B (physics simulation, EXPLO5) values. Small but very clean; typical error
&lt;5% on every property. Use for fine-tuning or benchmarking a model that was
pretrained on larger but noisier data.</li>
<li><strong>Variant B — add ML density and HOF only:</strong> A+B above, plus
Tier D values for density and heat of formation only (includes the 3DCNN DFT
surrogate's 26,254 molecules). Detonation properties and explosion heat still
come only from A or B sources. This is the "ML-augmented inputs to K-J"
dataset.</li>
<li><strong>Variant C — add K-J imputation for D, P, Q:</strong> Variant B
plus deterministic K-J-formula-computed detonation velocity, pressure, and
explosion heat on every row that has density + HOF available. Largest dataset
but D/P/Q come from an empirical formula (Tier C) rather than measurement or
thermochemistry code. Good for scale, use only if you accept ~15-25% error on
detonation properties and know your error propagation.</li>
</ul>
NOT modifying <code>labeled_master.csv</code> — this is an analytical
projection. To actually build Variant C, run a K-J imputation pass that adds
<code>kj_row_impute</code>-tagged cells.
</div>"""

sec16 = (
    variant_note +
    "<h3>Per-property row count by variant</h3>" +
    table_html(var_df, "Rows available per property under each variant") +
    fig_to_html(fig_var, "fig_var") +
    "<h3>Per-molecule completeness (out of 5 properties)</h3>" +
    table_html(row_summary, "Molecules with N properties filled, by variant") +
    fig_to_html(fig_var_row, "fig_var_row"))

# ── ASSEMBLE HTML ─────────────────────────────────────────────────────────────

NAV_ITEMS = [
    ("sec1",  "Executive Summary"),
    ("sec2",  "Raw Data Inventory"),
    ("sec3",  "Master Datasets"),
    ("sec4",  "Label Coverage"),
    ("sec5",  "Source Distributions"),
    ("sec6",  "Property Distributions"),
    ("sec7",  "Property Correlations"),
    ("sec8",  "Structural Features"),
    ("sec9",  "String Lengths"),
    ("sec10", "EMDB & EMDP reference sets"),
    ("sec11", "Normalization"),
    ("sec12", "K-J Consistency"),
    ("sec13", "Experimental Audit"),
    ("sec14", "Unlabeled Overlap"),
    ("sec15", "Butina Clustering"),
    ("sec16", "Training-set Variants"),
]

nav_html = "".join(f'<a href="#{sid}">{label}</a>' for sid, label in NAV_ITEMS)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EnergeticDiffusion2 — EDA Report</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3e;
    --text: #e0e4f0; --muted: #8892b0; --accent: #4f8ef7;
    --green: #4caf50; --orange: #ff9800; --red: #ef5350;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px; line-height: 1.6;
  }}
  header {{
    background: linear-gradient(135deg, #1a1d27 0%, #0d1b3e 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 20px;
  }}
  header h1 {{ font-size: 1.9rem; font-weight: 700; color: #fff; letter-spacing: -0.5px; }}
  header p  {{ color: var(--muted); margin-top: 6px; font-size: 0.9rem; }}
  nav {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 10px 40px; position: sticky; top: 0; z-index: 100;
    display: flex; flex-wrap: wrap; gap: 4px;
  }}
  nav a {{
    color: var(--muted); text-decoration: none; padding: 4px 10px;
    border-radius: 4px; font-size: 0.78rem; transition: all .2s;
  }}
  nav a:hover {{ background: var(--border); color: var(--text); }}
  main {{ max-width: 1300px; margin: 0 auto; padding: 32px 40px 80px; }}
  section {{ margin-bottom: 56px; }}
  section h2 {{
    font-size: 1.25rem; font-weight: 600; color: #fff;
    border-left: 3px solid var(--accent); padding-left: 12px;
    margin-bottom: 20px;
  }}
  section h3 {{ font-size: 1rem; color: var(--accent); margin: 28px 0 12px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 8px; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 22px; min-width: 160px; flex: 1;
  }}
  .card-val   {{ font-size: 1.7rem; font-weight: 700; color: var(--accent); }}
  .card-label {{ font-size: 0.82rem; color: var(--text); margin-top: 4px; font-weight: 600; }}
  .card-sub   {{ font-size: 0.72rem; color: var(--muted); margin-top: 2px; }}
  .tbl-title  {{ font-weight: 600; color: var(--muted); margin-bottom: 8px; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  .tbl-wrap   {{ overflow-x: auto; margin-bottom: 20px; border-radius: 8px; border: 1px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  thead tr {{ background: #1e2235; }}
  th {{
    text-align: left; padding: 9px 14px; color: var(--muted);
    font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
  }}
  td {{ padding: 7px 14px; border-bottom: 1px solid var(--border); color: var(--text); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #1e2235; }}
  .plotly-graph-div {{ border-radius: 10px; overflow: hidden;
    border: 1px solid var(--border); margin-bottom: 16px; background: var(--surface); }}
  .audit-box {{ border-radius: 8px; padding: 16px 20px; margin: 12px 0 20px;
    border-left: 4px solid; line-height: 1.7; font-size: 0.85rem; }}
  .audit-box.error   {{ background: #1f0f0f; border-color: #ef5350; }}
  .audit-box.warning {{ background: #1a1500; border-color: #ff9800; }}
  .audit-box.ok      {{ background: #0a1f0f; border-color: #4caf50; }}
  .audit-box.fixed   {{ background: #0a1f0f; border-color: #4caf50; }}
  .audit-box.info    {{ background: #0d1929; border-color: #64b5f6; }}
</style>
</head>
<body>
<header>
  <h1>EnergeticDiffusion2 — Exploratory Data Analysis</h1>
  <p>Generated 2026-04-24 | Source data exploration for energetic materials generation project</p>
</header>
<nav>{nav_html}</nav>
<main>
{section("1. Executive Summary", sec1, "sec1")}
{section("2. Raw Data Inventory", sec2, "sec2")}
{section("3. Master Datasets (pre-split)", sec3, "sec3")}
{section("4. Label Coverage Analysis", sec4, "sec4")}
{section("5. Source Dataset Distributions", sec5, "sec5")}
{section("6. Property Value Distributions", sec6, "sec6")}
{section("7. Property Correlations", sec7, "sec7")}
{section("8. Structural Features", sec8, "sec8")}
{section("9. SMILES and SELFIES String Lengths", sec9, "sec9")}
{section("10. EMDB and EMDP reference sets", sec10, "sec10")}
{section("11. Property Normalization Statistics", sec11, "sec11")}
{section("12. Kamlet-Jacobs Consistency Audit", sec12, "sec12")}
{section("13. Experimental Counter Audit vs Raw Sources", sec13, "sec13")}
{section("14. Labeled ↔ Unlabeled Overlap & Energetic Content", sec14, "sec14")}
{section("15. Butina Clustering (ECFP4, Tanimoto 0.4)", sec15, "sec15")}
</main>
</body>
</html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(OUT) / 1024
print(f"\nReport written to: {OUT}")
print(f"File size: {size_kb:.1f} KB")
