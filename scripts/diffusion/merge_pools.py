"""Merge several rerank-v2 markdowns by canonical SMILES.
Keeps best composite per molecule, tags provenance, re-Pareto-front."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from rdkit import Chem


def parse_v2_md(path: Path):
    out = []
    if not path.exists(): return out
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.startswith("| ") or "rank" in line or "---" in line: continue
        cells = [c.strip(" `") for c in line.split("|")]
        if len(cells) < 17: continue
        try:
            row = {
                "rank":  int(cells[1]),
                "comp":  float(cells[2]),
                "perf":  float(cells[3]),
                "viab":  float(cells[4]),
                "sens":  float(cells[5]),
                "novel": float(cells[6]),
                "rho":   float(cells[7]),
                "hof":   float(cells[8].replace("+","")),
                "d":     float(cells[9]),
                "p":     float(cells[10]),
                "alerts": cells[11],
                "ob":    cells[12],
                "nNO2":  cells[13],
                "mw":    cells[14],
                "pareto": "*" in cells[15] or "★" in cells[15],
                "smiles": cells[16],
            }
            m = Chem.MolFromSmiles(row["smiles"])
            if m: row["canon"] = Chem.MolToSmiles(m)
            else: continue
            out.append(row)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="rerank_v2 markdowns; format: path:tag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=200)
    args = ap.parse_args()

    by_canon = {}
    for spec in args.inputs:
        if ":" in spec:
            path, tag = spec.split(":", 1)
        else:
            path, tag = spec, Path(spec).stem
        rows = parse_v2_md(Path(path))
        print(f"  {tag}: {len(rows)} candidates")
        for r in rows:
            c = r["canon"]
            if c not in by_canon or r["comp"] > by_canon[c]["comp"]:
                r["sources"] = {tag}
                by_canon[c] = r
            else:
                by_canon[c]["sources"] = by_canon[c].get("sources", set()) | {tag}
    merged = list(by_canon.values())
    print(f"\nMerged unique: {len(merged)}")
    merged.sort(key=lambda r: -r["comp"])
    keep = merged[:args.top]
    print(f"Top-{len(keep)} composite range: [{keep[-1]['comp']:.3f}, {keep[0]['comp']:.3f}]")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# Merged rerank: top-{len(keep)} unique by canonical SMILES\n\n")
        f.write(f"sources: {', '.join(args.inputs)}\n\n")
        f.write("| rank | comp | perf | viab | sens | novel | rho | HOF | D | P | MW | nNO2 | sources | SMILES |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(keep, 1):
            srcs = ",".join(sorted(r.get("sources", set())))
            f.write(f"| {i} | {r['comp']:.3f} | {r['perf']:.2f} | {r['viab']:.2f} | "
                    f"{r['sens']:.2f} | {r['novel']:.2f} | {r['rho']:.2f} | "
                    f"{r['hof']:+.0f} | {r['d']:.2f} | {r['p']:.2f} | "
                    f"{r['mw']} | {r['nNO2']} | {srcs} | `{r['canon']}` |\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
