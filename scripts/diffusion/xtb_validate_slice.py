"""Variant of xtb_validate that takes a rank slice [start, end]."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "scripts/diffusion")
from xtb_validate import run_xtb, XTB

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end",   type=int, default=15)
    ap.add_argument("--workdir", default="logs/xtb_runs")
    args = ap.parse_args()

    md = Path(args.inp).read_text(encoding="utf-8")
    leads = []
    for line in md.split("\n"):
        if not line.startswith("| ") or "rank" in line or "---" in line: continue
        cells = [c.strip(" `") for c in line.split("|")]
        if len(cells) < 12: continue
        try:
            r = int(cells[1])
        except Exception: continue
        if args.start <= r <= args.end:
            leads.append({"rank": r, "smiles": cells[-2]})

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    if not Path(XTB).exists():
        raise SystemExit(f"xTB binary not found at {XTB}")

    results = []
    for L in leads:
        print(f"  [xtb] rank{L['rank']} {L['smiles'][:60]}", flush=True)
        r = run_xtb(L["smiles"], f"rank{L['rank']}", work)
        r["rank"] = L["rank"]
        results.append(r)
        print(f"    converged={r.get('converged')}  gap={r.get('gap_ev')}  graph={r.get('graph_survives')}")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n-> {args.out}")
    n_stable = sum(1 for r in results if r.get('gap_ev') and r['gap_ev'] >= 1.5)
    print(f"Gap >= 1.5 eV: {n_stable}/{len(results)}")

if __name__ == "__main__":
    main()
