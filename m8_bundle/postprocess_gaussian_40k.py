"""Local postprocessing for Gaussian latent 40k raw SMILES.

Applies the same pipeline as DGLD:
  1. RDKit canonicalization + dedup
  2. chem_filter (CHNO-only, red flags, oxygen balance)
  3. SA <= 6.5 and SC <= 4.5
  4. Tanimoto novelty window vs labelled_master [0.15, 0.65]
  5. Rank by N-fraction (proxy for energetic potential)

Outputs:
  m8_bundle/results/gaussian_latent_40k_postprocessed.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
SCRIPTS_DIFF = PROJECT_ROOT / "scripts" / "diffusion"
sys.path.insert(0, str(SCRIPTS_DIFF))

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from chem_filter import chem_filter
from feasibility_utils import real_sa, real_sc, SA_DROP_ABOVE, SC_DROP_ABOVE

RAW_PATH = HERE / "results" / "gaussian_latent_40k_raw.txt"
MASTER_PATH = PROJECT_ROOT / "m4_bundle" / "labelled_master.csv"
OUT_PATH = HERE / "results" / "gaussian_latent_40k_postprocessed.json"

TANIMOTO_MIN = 0.15  # too dissimilar -> OOD
TANIMOTO_MAX = 0.65  # too similar -> memorization
MIN_HEAVY_ATOMS = 10  # fragment filter


def morgan_fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def n_fraction(smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    if not m:
        return 0.0
    heavy = m.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N") / heavy


def main():
    raw = RAW_PATH.read_text(encoding="utf-8").splitlines()
    n_raw = len(raw)
    print(f"n_raw = {n_raw}")

    # 1. canonicalize + dedup
    seen: set[str] = set()
    valid: list[str] = []
    for s in raw:
        s = s.strip()
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        if c not in seen:
            seen.add(c)
            valid.append(c)
    print(f"n_valid (dedup) = {len(valid)}")

    # 2. minimum heavy atom count (fragment filter)
    valid = [s for s in valid
             if Chem.MolFromSmiles(s).GetNumHeavyAtoms() >= MIN_HEAVY_ATOMS]
    print(f"n_min_ha = {len(valid)}")

    # 3. chem_filter
    filtered = [s for s in valid if chem_filter(s, props=None)[0]]
    print(f"n_chem_pass = {len(filtered)}")

    # 4. SA + SC
    feasible = []
    for s in filtered:
        sa = real_sa(s)
        sc = real_sc(s)
        if sa <= SA_DROP_ABOVE and sc <= SC_DROP_ABOVE:
            feasible.append((s, sa, sc))
    print(f"n_feasible = {len(feasible)}")

    # 5. Tanimoto novelty window vs labelled_master
    print("Loading labelled_master for novelty filter ...")
    master_df = pd.read_csv(MASTER_PATH, usecols=["smiles"])
    master_fps = []
    for s in master_df["smiles"].dropna():
        fp = morgan_fp(s)
        if fp is not None:
            master_fps.append(fp)
    print(f"  master fingerprints: {len(master_fps)}")

    novel = []
    for s, sa, sc in feasible:
        fp = morgan_fp(s)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, master_fps)
        max_sim = max(sims) if sims else 0.0
        if TANIMOTO_MIN <= max_sim <= TANIMOTO_MAX:
            novel.append({"smiles": s, "sa": round(sa, 3), "sc": round(sc, 3),
                          "max_tanimoto": round(max_sim, 4),
                          "n_frac": round(n_fraction(s), 4)})
    print(f"n_novel (Tanimoto window) = {len(novel)}")

    # 6. rank
    scored = sorted(novel, key=lambda r: r["n_frac"] - 0.05 * r["sa"], reverse=True)
    top100 = scored[:100]

    keep_rate = len(novel) / n_raw if n_raw else 0.0
    result = {
        "method": "Gaussian-latent-40k",
        "n_raw": n_raw,
        "n_valid": len(valid),
        "n_chem_pass": len(filtered),
        "n_feasible": len(feasible),
        "n_novel": len(novel),
        "keep_rate_pct": round(100 * keep_rate, 3),
        "top1_smiles": top100[0]["smiles"] if top100 else None,
        "top1_n_frac": top100[0]["n_frac"] if top100 else None,
        "top_candidates": top100,
    }

    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== GAUSSIAN LATENT 40k SUMMARY ===")
    print(f"  n_raw        : {result['n_raw']}")
    print(f"  n_valid      : {result['n_valid']}")
    print(f"  n_chem_pass  : {result['n_chem_pass']}")
    print(f"  n_feasible   : {result['n_feasible']}")
    print(f"  n_novel      : {result['n_novel']}")
    print(f"  keep_rate    : {result['keep_rate_pct']:.3f}%")
    print(f"  top1_smiles  : {result['top1_smiles']}")
    print(f"  top1_n_frac  : {result['top1_n_frac']}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
