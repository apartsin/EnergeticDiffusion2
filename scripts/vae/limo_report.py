"""
Generate an HTML report from a fine-tuning experiment directory.

Reads:
    <exp>/config_snapshot.yaml
    <exp>/metadata.json
    <exp>/train.jsonl
    <exp>/eval_results.json  (if present)
    <exp>/external/LIMO/smoke_report.json (if present)

Outputs:
    <exp>/report.html

Usage:
    python scripts/vae/limo_report.py --exp experiments/limo_ft_energetic_<ts>
"""
from __future__ import annotations
import argparse
import json
import html
from pathlib import Path

import yaml


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def plot_curves(events: list[dict]) -> str:
    """Return inline Plotly HTML with training curves."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return "<p><em>Plotly not installed — curves unavailable</em></p>"

    train = [e for e in events if e.get("kind") == "train_step"]
    val   = [e for e in events if e.get("kind") == "val"]
    if not train:
        return "<p>No training steps logged.</p>"

    fig = make_subplots(rows=2, cols=2,
                         subplot_titles=["NLL (loss)", "KL divergence",
                                         "Token accuracy %", "Learning rate"])
    steps_t = [e["step"] for e in train]
    fig.add_trace(go.Scatter(x=steps_t, y=[e["nll"] for e in train],
                              name="train NLL", line=dict(color="#4f8ef7")),
                   row=1, col=1)
    if val:
        steps_v = [e["step"] for e in val]
        fig.add_trace(go.Scatter(x=steps_v, y=[e["val_nll"] for e in val],
                                  name="val NLL", mode="lines+markers",
                                  line=dict(color="#4CAF50")),
                       row=1, col=1)
    fig.add_trace(go.Scatter(x=steps_t, y=[e["kl"] for e in train],
                              name="train KL", line=dict(color="#FF9800")),
                   row=1, col=2)
    fig.add_trace(go.Scatter(x=steps_t, y=[100*e["acc"] for e in train],
                              name="train acc%", line=dict(color="#9C27B0")),
                   row=2, col=1)
    if val:
        fig.add_trace(go.Scatter(x=steps_v, y=[100*e["val_acc"] for e in val],
                                  name="val acc%", mode="lines+markers",
                                  line=dict(color="#4CAF50")),
                       row=2, col=1)
    fig.add_trace(go.Scatter(x=steps_t, y=[e["lr"] for e in train],
                              name="lr", line=dict(color="#EF5350")),
                   row=2, col=2)
    fig.update_layout(height=620, margin=dict(t=60, b=40),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       template="plotly_dark")
    fig.update_yaxes(type="log", row=2, col=2)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def esc(s) -> str:
    return html.escape(str(s))


def _row(label: str, value: str) -> str:
    return f"<tr><td>{esc(label)}</td><td>{esc(value)}</td></tr>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",  required=True)
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    args = ap.parse_args()

    base = Path(args.base)
    exp  = Path(args.exp)
    if not exp.is_absolute():
        exp = base / exp

    # Load artefacts
    cfg      = {}
    metadata = {}
    eval_res = {}
    smoke    = {}
    try:
        cfg = yaml.safe_load(open(exp / "config_snapshot.yaml"))
    except Exception as e:
        print(f"  warning: can't read config_snapshot.yaml: {e}")
    try:
        metadata = json.load(open(exp / "metadata.json"))
    except Exception as e:
        print(f"  warning: can't read metadata.json: {e}")
    try:
        eval_res = json.load(open(exp / "eval_results.json"))
    except Exception:
        pass
    try:
        smoke = json.load(open(base / "external/LIMO/smoke_report.json"))
    except Exception:
        pass

    events = read_jsonl(exp / "train.jsonl")

    # ── Sections ─────────────────────────────────────────────────────────────
    hdr_title = esc(cfg.get("run", {}).get("name", exp.name))
    notes     = esc(cfg.get("run", {}).get("notes", ""))

    # Metadata table
    meta_rows = [
        _row("Experiment",       exp.name),
        _row("Run name",         cfg.get("run", {}).get("name", "-")),
        _row("Seed",             cfg.get("run", {}).get("seed", "-")),
        _row("Start time",       metadata.get("start_time", "-")),
        _row("End time",         metadata.get("end_time", "-")),
        _row("Total minutes",    f"{metadata.get('total_minutes', 0):.1f}" if metadata else "-"),
        _row("Total steps",      metadata.get("total_steps", "-")),
        _row("Best val NLL",     f"{metadata.get('best_val_nll', 0):.4f}" if metadata else "-"),
        _row("Device",           metadata.get("device", "-")),
        _row("Resumed",          metadata.get("resumed", False)),
        _row("Notes",            notes),
    ]
    meta_table = f"""
    <table class="kv">
      <tbody>{''.join(meta_rows)}</tbody>
    </table>"""

    # Config table
    def _flatten(d, prefix=""):
        out = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.extend(_flatten(v, key))
            else:
                out.append((key, v))
        return out
    cfg_rows = "".join(_row(k, v) for k, v in _flatten(cfg))
    cfg_table = f"""
    <table class="kv">
      <tbody>{cfg_rows}</tbody>
    </table>"""

    # Smoke test section
    smoke_html = ""
    if smoke:
        smoke_rows = [
            _row("Tested N",                smoke.get("n_molecules_tested", "-")),
            _row("Weights load (missing)",  len(smoke.get("weights_load_missing", []))),
            _row("Token accuracy (zero-shot)", f"{smoke.get('token_accuracy_pct', 0):.2f}%"),
            _row("Valid decoded %",         f"{smoke.get('valid_decoded_pct', 0):.1f}%"),
            _row("Exact SMILES %",          f"{smoke.get('exact_smiles_pct', 0):.2f}%"),
            _row("OOV molecules",           smoke.get("oov_molecules", "-")),
            _row("Distinct OOV tokens",     smoke.get("distinct_oov_tokens", "-")),
            _row("Verdict",                 "PASS ✓" if smoke.get("verdict_pass") else "FAIL ✗"),
        ]
        smoke_html = f"""
        <section>
          <h2>Smoke test (pre-fine-tune baseline)</h2>
          <p>Zero-shot evaluation of the unmodified LIMO checkpoint on our energetic SMILES.</p>
          <table class="kv"><tbody>{''.join(smoke_rows)}</tbody></table>
        </section>"""

    # Evaluation section
    eval_html = ""
    if eval_res:
        rec = eval_res.get("reconstruction", {})
        smp = eval_res.get("sampling", {})
        prb = eval_res.get("density_probe", {})
        ls  = eval_res.get("latent_stats", {})
        eval_rows = [
            _row("Checkpoint step",   eval_res.get("step", "-")),
            _row("Best val NLL",      f"{eval_res.get('best_val', 0):.4f}"),
            _row("Recon token acc",   f"{rec.get('token_accuracy_pct', 0):.2f}%"),
            _row("Recon exact SMILES", f"{rec.get('exact_smiles_pct', 0):.2f}%"),
            _row("Recon valid decoded", f"{rec.get('valid_decoded_pct', 0):.2f}%"),
            _row("Prior sample valid", f"{smp.get('valid_pct', 0):.1f}%"),
            _row("Prior sample unique", f"{smp.get('unique_pct', 0):.1f}%"),
            _row("Density probe R² (test)", f"{prb.get('r2_test', 0):.3f}"),
            _row("Density probe MAE",  f"{prb.get('mae_test', 0):.4f}"),
            _row("Latent active dims", f"{ls.get('active_dims', 0)} / {ls.get('latent_dim', 0)}"),
            _row("KL total avg",       f"{ls.get('kl_total_avg', 0):.2f}"),
            _row("σ mean",             f"{ls.get('sigma_mean', 0):.3f}"),
        ]
        # sample gallery
        sample_smiles = smp.get("sample_smiles", [])[:16]
        smp_html = ""
        if sample_smiles:
            smp_html = ("<h3>Sample generations (z ~ N(0, I))</h3><ul class='smp'>"
                        + "".join(f"<li><code>{esc(s)}</code></li>" for s in sample_smiles)
                        + "</ul>")
        eval_html = f"""
        <section>
          <h2>Post-fine-tune evaluation</h2>
          <table class="kv"><tbody>{''.join(eval_rows)}</tbody></table>
          {smp_html}
        </section>"""

    # Training curves
    curves_html = plot_curves(events)

    # Final HTML
    html_out = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>LIMO Fine-tune — {hdr_title}</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3e;
    --text: #e0e4f0; --muted: #8892b0; --accent: #4f8ef7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif;
          font-size: 14px; line-height: 1.6; }}
  header {{ background: linear-gradient(135deg, #1a1d27 0%, #0d1b3e 100%);
            border-bottom: 1px solid var(--border); padding: 24px 36px; }}
  header h1 {{ font-size: 1.8rem; color: #fff; }}
  header p  {{ color: var(--muted); margin-top: 4px; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 32px 36px 80px; }}
  section {{ margin-bottom: 48px; }}
  section h2 {{ font-size: 1.25rem; color: #fff;
                border-left: 3px solid var(--accent); padding-left: 10px;
                margin-bottom: 16px; }}
  section h3 {{ font-size: 1rem; color: var(--accent); margin: 18px 0 8px; }}
  table.kv {{ width: 100%; border-collapse: collapse;
              background: var(--surface); border-radius: 8px; overflow: hidden;
              border: 1px solid var(--border); }}
  table.kv td {{ padding: 6px 14px; border-bottom: 1px solid var(--border);
                  font-size: 0.87rem; }}
  table.kv td:first-child {{ color: var(--muted); width: 40%; }}
  table.kv tr:last-child td {{ border-bottom: none; }}
  code {{ background: #111521; padding: 1px 6px; border-radius: 4px;
          font-family: Consolas, monospace; font-size: 0.82rem; }}
  ul.smp {{ list-style: none; padding: 10px;
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 8px; margin-top: 6px; }}
  ul.smp li {{ margin: 4px 0; }}
</style>
</head>
<body>
<header>
  <h1>LIMO Fine-tune — {hdr_title}</h1>
  <p>{notes}</p>
</header>
<main>

<section>
  <h2>Run overview</h2>
  {meta_table}
</section>

{smoke_html}

{eval_html}

<section>
  <h2>Training curves</h2>
  {curves_html}
</section>

<section>
  <h2>Configuration snapshot</h2>
  {cfg_table}
</section>

</main>
</body>
</html>
"""
    out = exp / "report.html"
    out.write_text(html_out, encoding="utf-8")
    print(f"Report → {out}")


if __name__ == "__main__":
    main()
