"""
HTML report for a diffusion training experiment.
Reads config_snapshot.yaml, metadata.json, train.jsonl, eval_results.json.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import yaml


def esc(s): return html.escape(str(s))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists(): return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except Exception: pass
    return out


def plot_curves(events: list[dict]) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return "<p><em>Plotly not available</em></p>"
    train = [e for e in events if e.get("kind") == "train_step"]
    val   = [e for e in events if e.get("kind") == "val"]
    if not train: return "<p>No training steps.</p>"
    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=["MSE loss", "Learning rate"])
    s_t = [e["step"] for e in train]
    fig.add_trace(go.Scatter(x=s_t, y=[e["loss"] for e in train],
                              name="train", line=dict(color="#4f8ef7")), 1, 1)
    if val:
        s_v = [e["step"] for e in val]
        fig.add_trace(go.Scatter(x=s_v, y=[e["val_loss"] for e in val],
                                  name="val", mode="lines+markers",
                                  line=dict(color="#4CAF50")), 1, 1)
    fig.add_trace(go.Scatter(x=s_t, y=[e["lr"] for e in train],
                              name="lr", line=dict(color="#EF5350")), 1, 2)
    fig.update_layout(height=420, template="plotly_dark",
                       margin=dict(t=60, b=40),
                       legend=dict(orientation="h", y=1.05))
    fig.update_yaxes(type="log", row=1, col=2)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _row(k, v): return f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    args = ap.parse_args()

    base = Path(args.base)
    exp = Path(args.exp)
    if not exp.is_absolute(): exp = base / exp

    cfg = {}
    metadata = {}
    eval_res = {}
    try: cfg = yaml.safe_load(open(exp / "config_snapshot.yaml"))
    except Exception: pass
    try: metadata = json.load(open(exp / "metadata.json"))
    except Exception: pass
    try: eval_res = json.load(open(exp / "eval_results.json"))
    except Exception: pass

    events = read_jsonl(exp / "train.jsonl")

    name = esc(cfg.get("run", {}).get("name", exp.name))
    notes = esc(cfg.get("run", {}).get("notes", ""))

    meta_rows = [
        _row("Experiment", exp.name),
        _row("Start", metadata.get("start_time", "-")),
        _row("End", metadata.get("end_time", "-")),
        _row("Total minutes", f"{metadata.get('total_minutes', 0):.1f}" if metadata else "-"),
        _row("Total steps", metadata.get("total_steps", "-")),
        _row("Best val loss", f"{metadata.get('best_val', 0):.4f}" if metadata else "-"),
        _row("Device", metadata.get("device", "-")),
        _row("Notes", notes),
    ]
    meta_table = f"<table class='kv'><tbody>{''.join(meta_rows)}</tbody></table>"

    def _flatten(d, prefix=""):
        out = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict): out.extend(_flatten(v, key))
            else: out.append((key, v))
        return out
    cfg_table = ("<table class='kv'><tbody>"
                  + "".join(_row(k, v) for k, v in _flatten(cfg))
                  + "</tbody></table>")

    eval_html = ""
    if eval_res:
        u = eval_res.get("unconditional", {})
        u_rows = [
            _row("Checkpoint step", eval_res.get("ckpt_step", "-")),
            _row("Best val", f"{eval_res.get('best_val', 0):.4f}"),
            _row("Guidance scale", eval_res.get("guidance", "-")),
            _row("DDIM steps", eval_res.get("n_steps", "-")),
            _row("EMA weights", eval_res.get("use_ema", False)),
            _row("Uncond generated", u.get("n_generated", "-")),
            _row("Uncond valid %", f"{u.get('valid_pct', 0):.2f}%"),
            _row("Uncond unique %", f"{u.get('unique_pct', 0):.2f}%"),
            _row("Uncond novel %", f"{u.get('novel_pct', 0):.2f}%"),
        ]
        u_table = f"<table class='kv'><tbody>{''.join(u_rows)}</tbody></table>"
        sample_smi = u.get("sample_smiles", [])[:16]
        smp_html = ""
        if sample_smi:
            smp_html = ("<h3>Unconditional samples</h3><ul class='smp'>"
                         + "".join(f"<li><code>{esc(s)}</code></li>"
                                    for s in sample_smi)
                         + "</ul>")
        # conditional
        cond_blocks = ""
        for p, per in eval_res.get("conditional", {}).items():
            rows = []
            for q, r in per.items():
                rows.append(_row(f"{q}  (target={r['target_raw']:.3f})",
                                   f"valid {r['n_valid']}/{r['n_generated']}, "
                                   f"unique {r['unique_pct']:.1f}%"))
            cond_blocks += (f"<h4>{esc(p)}</h4>"
                             f"<table class='kv'><tbody>{''.join(rows)}</tbody></table>")
        eval_html = f"""
        <section>
          <h2>Evaluation</h2>
          {u_table}
          {smp_html}
          <h3>Conditional sampling (per property, per quantile)</h3>
          {cond_blocks}
        </section>"""

    curves_html = plot_curves(events)

    out = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Diffusion — {name}</title><style>
 :root {{ --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3e;
          --text: #e0e4f0; --muted: #8892b0; --accent: #4f8ef7; }}
 * {{ box-sizing: border-box; margin: 0; padding: 0; }}
 body {{ background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6; }}
 header {{ background: linear-gradient(135deg,#1a1d27 0%,#0d1b3e 100%);
            padding: 22px 36px; border-bottom: 1px solid var(--border); }}
 header h1 {{ font-size: 1.8rem; color: #fff; }}
 header p {{ color: var(--muted); margin-top: 4px; }}
 main {{ max-width: 1200px; margin: 0 auto; padding: 32px 36px 80px; }}
 section {{ margin-bottom: 48px; }}
 section h2 {{ font-size: 1.2rem; color: #fff;
               border-left: 3px solid var(--accent); padding-left: 10px;
               margin-bottom: 14px; }}
 section h3 {{ font-size: 1rem; color: var(--accent); margin: 18px 0 8px; }}
 section h4 {{ font-size: 0.95rem; color: #d0d6e0; margin: 12px 0 6px; }}
 table.kv {{ width: 100%; border-collapse: collapse; background: var(--surface);
              border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
 table.kv td {{ padding: 6px 14px; border-bottom: 1px solid var(--border);
                 font-size: 0.86rem; }}
 table.kv td:first-child {{ color: var(--muted); width: 38%; }}
 table.kv tr:last-child td {{ border-bottom: none; }}
 code {{ background: #111521; padding: 1px 6px; border-radius: 4px;
          font-family: Consolas, monospace; font-size: 0.82rem; }}
 ul.smp {{ list-style: none; padding: 10px;
             background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; }}
 ul.smp li {{ margin: 4px 0; }}
</style></head><body>
<header>
  <h1>Diffusion (subset-conditional) — {name}</h1>
  <p>{notes}</p>
</header>
<main>
  <section><h2>Run overview</h2>{meta_table}</section>
  {eval_html}
  <section><h2>Training curves</h2>{curves_html}</section>
  <section><h2>Config snapshot</h2>{cfg_table}</section>
</main></body></html>
"""
    (exp / "report.html").write_text(out, encoding="utf-8")
    print(f"Report → {exp / 'report.html'}")


if __name__ == "__main__":
    main()
