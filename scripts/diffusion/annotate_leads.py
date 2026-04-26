"""Murcko-scaffold + chemistry-class annotation for breakthrough leads.

Reads SMILES from a rerank result markdown, computes:
  - Murcko scaffold (Bemis–Murcko ring system)
  - Atom composition (CHNO + halogens)
  - Functional group flags (NO2 count, ring N count, etc.)
  - SOTA proximity (Tanimoto vs known SOTA energetics CL-20, HMX, RDX, …)
  - Heuristic "feasibility tier" based on SA + complexity + composition

Outputs annotated markdown + CSV.
"""
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

BASE = Path("E:/Projects/EnergeticDiffusion2")

SOTA_REFS = {
    "TNT":   "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
    "RDX":   "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "HMX":   "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]",
    "PETN":  "O=[N+]([O-])OCC(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]",
    "CL-20": "O=[N+]([O-])N1[C@H]2[C@@H]3N([N+](=O)[O-])[C@@H]4[C@H](N2[N+](=O)[O-])N([N+](=O)[O-])[C@H]([C@H]1N3[N+](=O)[O-])N4[N+](=O)[O-]",
    "TATB":  "Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]",
    "FOX-7": "NC(=C([N+](=O)[O-])[N+](=O)[O-])N",
    "TKX-50": "O=[N+]([O-])C1=NN=C(N=N1)O[NH3+]",
}


def morgan(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def parse_md_table(md_path: Path) -> list[dict]:
    """Generic markdown-table parser that extracts the SMILES + numeric cols."""
    rows = []
    text = md_path.read_text(encoding="utf-8")
    # find lines starting with "| 1 |", "| 2 |", … (top-N rank rows)
    for line in text.splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|", line)
        if not m: continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        smi_match = re.search(r"`([^`]+)`", line)
        if not smi_match: continue
        rows.append({
            "rank": int(cells[0]),
            "cells": cells,
            "smi":  smi_match.group(1),
        })
    return rows


def murcko_scaffold(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None: return ""
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc, canonical=True) if sc else ""


def composition(mol):
    counts = {"C": 0, "H": 0, "N": 0, "O": 0, "F": 0, "Cl": 0,
              "Br": 0, "I": 0, "P": 0, "S": 0}
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        if s in counts: counts[s] += 1
        counts["H"] += a.GetTotalNumHs()
    formula = "".join(f"{e}{counts[e]}" for e in "CHNOFClBrIPS"
                       if counts.get(e, 0))
    return counts, formula


def feature_flags(mol):
    """Boolean flags for known motifs."""
    smarts_map = {
        "nitro":      "[N+](=O)[O-]",
        "nitrate_ester": "O[N+](=O)[O-]",
        "nitramine":  "[NX3][N+](=O)[O-]",
        "azide":      "[N-]=[N+]=N",
        "furazan":    "c1nonc1",
        "tetrazole":  "c1nnnn1",
        "triazole":   "c1nncn1",
        "triazine":   "c1ncncn1",
    }
    out = {}
    for k, sm in smarts_map.items():
        p = Chem.MolFromSmarts(sm)
        out[k] = bool(p is not None and mol.HasSubstructMatch(p))
    out["n_nitro"] = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N+](=O)[O-]")))
    out["heavy_atoms"] = mol.GetNumHeavyAtoms()
    return out


def feasibility_tier(sa, sc, n_heavy, has_unstable):
    """Coarse heuristic class for downstream triage."""
    if has_unstable: return "unsynth"
    if sa <= 4.0 and sc <= 2.5 and n_heavy <= 25: return "easy"
    if sa <= 5.0 and sc <= 3.5 and n_heavy <= 35: return "moderate"
    if sa <= 6.0 and sc <= 4.0: return "hard"
    return "very-hard"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_md", required=True,
                    help="Path to a rerank_results.md / joint_rerank.md / "
                         "rerank_multi.md file")
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    inp = Path(args.input_md)
    out_md = Path(args.out_md) if args.out_md else inp.parent / (inp.stem + "_annotated.md")
    out_csv = Path(args.out_csv) if args.out_csv else inp.parent / (inp.stem + "_annotated.csv")

    rows = parse_md_table(inp)
    print(f"parsed {len(rows)} candidate rows from {inp}")

    sota_fps = {n: morgan(s) for n, s in SOTA_REFS.items()}

    # Annotate each
    annotated = []
    md_lines = [
        f"# Annotated leads from `{inp.name}`", "",
        "Murcko scaffold, atom composition, motif flags, SOTA proximity, "
        "feasibility tier (heuristic).", "",
        "| rank | SMILES | scaffold (Murcko) | formula | n_nitro | motifs | nearest SOTA | feasibility tier |",
        "|" + "|".join(["---"] * 8) + "|",
    ]
    for r in rows:
        smi = r["smi"]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            md_lines.append(f"| {r['rank']} | `{smi}` | – | – | – | – | parse-fail | – |")
            continue
        sc = murcko_scaffold(smi)
        sc_short = sc[:40] + "…" if len(sc) > 40 else sc
        comp, formula = composition(m)
        feats = feature_flags(m)
        motif_tags = []
        for k in ("nitro", "nitramine", "nitrate_ester", "azide",
                   "furazan", "tetrazole", "triazole", "triazine"):
            if feats[k]: motif_tags.append(k)
        if not motif_tags: motif_tags = ["—"]
        # SOTA proximity
        fp = morgan(smi)
        sims = {n: DataStructs.TanimotoSimilarity(fp, sota_fps[n])
                for n in SOTA_REFS if sota_fps[n] is not None}
        nearest_sota = max(sims.items(), key=lambda kv: kv[1])
        # Feasibility tier — try to extract SA+SC from the row cells
        sa = sc_score = None
        try:
            cells = r["cells"]
            for c in cells:
                if re.match(r"^\d+\.\d+$", c):
                    pass
            # Heuristic: pull SA and SC from numeric cells if present in fixed positions
            # Many reports have ... SA | SC | ... before SMILES
            num_cells = [c for c in cells if re.match(r"^[+-]?\d+(\.\d+)?$", c)]
            # SA and SC are usually the last 2 numbers before tail tokens
            if len(num_cells) >= 2:
                sa = float(num_cells[-2])
                sc_score = float(num_cells[-1])
        except Exception:
            pass
        unstable = False  # could plug in chem_filter rules; keep heuristic-light
        tier = feasibility_tier(sa or 99, sc_score or 99,
                                  feats["heavy_atoms"], unstable)
        annotated.append({
            "rank":       r["rank"],
            "smi":        smi,
            "scaffold":   sc,
            "formula":    formula,
            "n_nitro":    feats["n_nitro"],
            "motifs":     ",".join(motif_tags),
            "nearest_sota": nearest_sota[0],
            "tan_sota":   nearest_sota[1],
            "sa":         sa,
            "sc":         sc_score,
            "tier":       tier,
        })
        md_lines.append(
            f"| {r['rank']} | `{smi}` | `{sc_short}` | {formula} | "
            f"{feats['n_nitro']} | {','.join(motif_tags)} | "
            f"{nearest_sota[0]} ({nearest_sota[1]:.2f}) | {tier} |"
        )
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    pd.DataFrame(annotated).to_csv(out_csv, index=False)
    print(f"saved {out_md}")
    print(f"saved {out_csv}")
    # Summary by tier
    if annotated:
        df = pd.DataFrame(annotated)
        print("\nFeasibility tier distribution:")
        print(df["tier"].value_counts())
        print("\nNearest SOTA distribution:")
        print(df["nearest_sota"].value_counts().head(8))


if __name__ == "__main__":
    sys.exit(main())
