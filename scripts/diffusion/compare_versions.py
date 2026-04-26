"""Build a unified comparison table from all denoiser experiments that
have a cfg_sweep.json. Best CFG is chosen per-cell (lowest rel_mae_pct).

Usage:
    python scripts/diffusion/compare_versions.py
    python scripts/diffusion/compare_versions.py --metric within_10_pct --reverse
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--metric", default="rel_mae_pct",
                    help="rel_mae_pct (lower better) or within_10_pct (higher better)")
    ap.add_argument("--reverse", action="store_true",
                    help="Treat metric as higher-is-better")
    ap.add_argument("--out", default="docs/version_comparison.md")
    args = ap.parse_args()

    root = Path(args.root)
    sweeps = sorted(root.glob("diffusion_subset_cond_expanded_*/cfg_sweep.json"))
    if not sweeps:
        print("no cfg_sweep.json files found"); return 1

    versions = []
    for p in sweeps:
        exp_name = p.parent.name
        # nickname: v1/v2/v3/v4 from the dir name
        if "_v2_" in exp_name: nick = "v2"
        elif "_v3_" in exp_name: nick = "v3"
        elif "_v4_nofilter_" in exp_name: nick = "v4-nf"
        elif "_v4_b_" in exp_name.lower() or "_v4b_" in exp_name.lower(): nick = "v4-B"
        elif "_v4_" in exp_name: nick = "v4"
        elif "_v5_" in exp_name: nick = "v5"
        else: nick = "v1"
        with open(p) as f:
            blob = json.load(f)
        scales = list(blob["sweep"].keys())
        # collect best per (prop, q)
        rows = {}
        prop_names = list(next(iter(blob["sweep"].values()))["results"].keys())
        for prop in prop_names:
            for q in ["q10", "q50", "q90"]:
                best_val = None; best_g = None
                for g in scales:
                    v = blob["sweep"][g]["results"][prop][q].get("validator_3dcnn", {})
                    m = v.get(args.metric)
                    if m is None: continue
                    if best_val is None or (
                        (m > best_val) if args.reverse else (m < best_val)
                    ):
                        best_val = m; best_g = g
                rows[(prop, q)] = (best_val, best_g)
        versions.append({"nick": nick, "exp": exp_name, "rows": rows,
                         "props": prop_names, "n": blob.get("n_per_target")})

    # build table
    nicks = [v["nick"] for v in versions]
    props = versions[0]["props"]

    md = ["# Version comparison (best CFG per cell)", "",
          f"metric: `{args.metric}`  (lower better)" if not args.reverse
          else f"metric: `{args.metric}`  (higher better)", ""]
    for v in versions:
        md.append(f"- **{v['nick']}**: `{v['exp']}` (n={v['n']})")
    md.append("")
    md.append("| Property | q | target | " + " | ".join(nicks) + " |")
    md.append("|---|---|---|" + "|".join(["---"]*len(nicks)) + "|")

    # target_raw is per-version but should be ~identical (assuming same stats);
    # use the first version's value as the representative.
    sweep0 = json.load(open(sweeps[0]))
    first_g = list(sweep0["sweep"].keys())[0]
    for prop in props:
        for q in ["q10", "q50", "q90"]:
            tgt = sweep0["sweep"][first_g]["results"][prop][q]["target_raw"]
            cells = []
            best_overall = None
            best_idx = None
            for i, v in enumerate(versions):
                val, g = v["rows"].get((prop, q), (None, None))
                if val is None:
                    cells.append("—")
                else:
                    cells.append(f"{val:.0f}@g{g}")
                    if best_overall is None or (
                        (val > best_overall) if args.reverse else (val < best_overall)
                    ):
                        best_overall = val; best_idx = i
            if best_idx is not None:
                cells[best_idx] = f"**{cells[best_idx]}**"
            md.append(f"| {prop} | {q} | {tgt:+.2f} | " + " | ".join(cells) + " |")
        md.append("|" + "|".join([" "]*(len(nicks)+4)) + "|")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved → {out}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    raise SystemExit(main())
