"""D15: motif analysis of top-ranked candidates vs Tier-A/B reference.

For each rerank result file, count which energetic motifs dominate top-N.
Compare against the Tier-A/B reference distribution. Identifies
under-represented motif families that the model isn't producing.
"""
import sys, re, json
from pathlib import Path
from collections import Counter
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

BASE = Path("E:/Projects/EnergeticDiffusion2")

MOTIFS = [
    ("nitro",          "[N+](=O)[O-]"),
    ("nitrate_ester",  "O[N+](=O)[O-]"),
    ("nitramine_NNO2", "[NX3][N+](=O)[O-]"),
    ("azide",          "N=[N+]=[N-]"),
    ("furazan",        "c1nonc1"),
    ("tetrazole",      "c1nnnn1"),
    ("triazole",       "c1nncn1"),
    ("triazine",       "c1ncncn1"),
    ("tetrazine",      "c1nnncn1"),
    ("nitroso",        "[N]=O"),
    ("dinitromethyl",  "C([N+](=O)[O-])[N+](=O)[O-]"),
    ("polynitro",      None),
]

def count_motifs(smiles_list):
    cnts = Counter()
    n = 0
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if not m: continue
        n += 1
        for name, smarts in MOTIFS:
            if name == "polynitro":
                pat = Chem.MolFromSmarts("[N+](=O)[O-]")
                if pat is not None and len(m.GetSubstructMatches(pat)) >= 3:
                    cnts[name] += 1
            else:
                pat = Chem.MolFromSmarts(smarts) if smarts else None
                if pat is not None and m.HasSubstructMatch(pat):
                    cnts[name] += 1
    return cnts, n

def main():
    blob = torch.load(BASE / "data/training/diffusion/latents_expanded.pt",
                       weights_only=False)
    smiles = blob["smiles"]
    cw = blob["cond_weight"]
    cv = blob["cond_valid"]
    raw = blob["values_raw"]
    pn = blob["property_names"]
    j_h = pn.index("heat_of_formation")

    # Tier-A/B reference
    trusted = (cv[:, j_h] & (cw[:, j_h] >= 0.99)).numpy()
    ref_smiles = [smiles[i] for i in range(len(smiles)) if trusted[i]]
    print(f"Tier-A/B HOF reference: {len(ref_smiles):,}")
    ref_cnts, ref_n = count_motifs(ref_smiles)
    # high-HOF subset
    hof = raw[trusted, j_h].numpy()
    high_mask = hof >= 200
    high_smiles = [s for s, k in zip(ref_smiles, high_mask) if k]
    print(f"  high-HOF (>+200 kcal/mol): {len(high_smiles)}")
    high_cnts, high_n = count_motifs(high_smiles)

    # Top candidates from latest rerank_multi.md (parse SMILES out of the table)
    rerank_file = BASE / "experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/rerank_multi.md"
    cand_smiles = []
    if rerank_file.exists():
        text = rerank_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            m = re.search(r"\| `([^`]+)` \|", line)
            if m: cand_smiles.append(m.group(1))
    print(f"  v6-multi top candidates: {len(cand_smiles)}")
    cand_cnts, cand_n = count_motifs(cand_smiles)

    md = ["# D15: motif distribution — Tier-A/B vs high-HOF subset vs top-ranked candidates",
           "",
           f"- Tier-A/B HOF rows scanned: {ref_n}",
           f"- Tier-A/B HOF >+200 kcal/mol: {high_n}",
           f"- v6-multi top candidates: {cand_n}",
           "",
           "| Motif | Tier-A/B % | High-HOF % | Top-cand % | Δ (top − A/B) |",
           "|---|---|---|---|---|"]
    for name, _ in MOTIFS:
        a = 100 * ref_cnts[name] / max(ref_n, 1)
        h = 100 * high_cnts[name] / max(high_n, 1)
        c = 100 * cand_cnts[name] / max(cand_n, 1)
        md.append(f"| {name} | {a:.1f} | {h:.1f} | {c:.1f} | {c-a:+.1f} |")
    md.append("")
    md.append("**Reading**: Δ < 0 means top candidates *under-produce* this motif "
              "vs trusted training data — a likely bottleneck for HOF.")
    out = BASE / "docs/diag_d15.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved {out}")

if __name__ == "__main__":
    sys.exit(main())
