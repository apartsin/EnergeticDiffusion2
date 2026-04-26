"""D16: feasibility (SA + SC) distribution of top-N candidates vs Tier-A/B.

Computes real SA + SC for:
  - top reranked candidates (from rerank_results.md and/or rerank_multi.md)
  - Tier-A/B reference subset
  - high-HOF subset (Tier-A/B with HOF > +200)

Reports the distribution shift. If candidates are systematically harder to
synthesise than Tier-A/B, raises a flag → tune up λ_SA / λ_SC or tighten
rerank thresholds.

This script does CPU-only work (no GPU); safe to run alongside training.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
import torch

BASE = Path("E:/Projects/EnergeticDiffusion2")
sys.path.insert(0, str(BASE / "scripts/diffusion"))


def parse_smiles_from_md(path: Path) -> list[str]:
    if not path.exists(): return []
    text = path.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        m = re.findall(r"`([^`]+)`", line)
        for s in m:
            if any(c in s for c in "()[]=#") and not s.endswith(".md"):
                out.append(s)
    seen = set(); ded = []
    for s in out:
        if s not in seen:
            seen.add(s); ded.append(s)
    return ded


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from feasibility_utils import real_sa, real_sc

    blob = torch.load(BASE / "data/training/diffusion/latents_expanded.pt",
                       weights_only=False)
    smiles = blob["smiles"]
    raw = blob["values_raw"]; cv = blob["cond_valid"]; cw = blob["cond_weight"]
    pn = blob["property_names"]
    j_h = pn.index("heat_of_formation")

    # Tier-A/B HOF subset
    trusted = (cv[:, j_h] & (cw[:, j_h] >= 0.99)).numpy()
    n_ref_max = 1500
    ref_idx = np.where(trusted)[0]
    if len(ref_idx) > n_ref_max:
        rng = np.random.default_rng(42)
        ref_idx = rng.choice(ref_idx, n_ref_max, replace=False)
    ref_smiles = [smiles[i] for i in ref_idx]

    # high-HOF (HOF > +200) subset
    hof = raw[trusted, j_h].numpy()
    high_mask = hof >= 200
    high_smiles = [s for s, k in zip([smiles[i] for i in np.where(trusted)[0]], high_mask) if k]
    if len(high_smiles) > 300:
        rng = np.random.default_rng(0)
        high_smiles = list(rng.choice(high_smiles, 300, replace=False))

    # candidate sets — auto-discover most recent rerank result files
    cand_files = []
    for exp_dir in sorted((BASE / "experiments").glob(
            "diffusion_subset_cond_expanded_*")):
        for fname in ("rerank_results.md", "rerank_multi.md"):
            f = exp_dir / fname
            if f.exists():
                cand_files.append(f)
    cand_smiles = []
    for f in cand_files:
        cand_smiles.extend(parse_smiles_from_md(f))
    seen = set(); cand_smiles = [s for s in cand_smiles
                                  if not (s in seen or seen.add(s))]
    cand_smiles = cand_smiles[:120]   # cap for cost

    def stats(label, smi_list):
        sa = np.array([real_sa(s) for s in smi_list], dtype=float)
        sc = np.array([real_sc(s) for s in smi_list], dtype=float)
        sa = sa[~np.isnan(sa)]; sc = sc[~np.isnan(sc)]
        return {
            "n": len(sa),
            "sa_mean": float(sa.mean()) if len(sa) else float("nan"),
            "sa_p90":  float(np.percentile(sa, 90)) if len(sa) else float("nan"),
            "sc_mean": float(sc.mean()) if len(sc) else float("nan"),
            "sc_p90":  float(np.percentile(sc, 90)) if len(sc) else float("nan"),
            "above_sa6.5": int((sa > 6.5).sum()) if len(sa) else 0,
            "above_sc4.5": int((sc > 4.5).sum()) if len(sc) else 0,
        }

    print(f"reference Tier-A/B: {len(ref_smiles)} SMILES")
    print(f"high-HOF (>+200): {len(high_smiles)}")
    print(f"top candidates: {len(cand_smiles)}")
    s_ref  = stats("ref",  ref_smiles)
    s_high = stats("high", high_smiles)
    s_cand = stats("cand", cand_smiles)

    md = ["# D16: feasibility distribution",
          "",
          "Real SA (Ertl-Schuffenhauer) and SC (Coley) computed on canonical SMILES.",
          "",
          "| Group | n | SA mean | SA p90 | SC mean | SC p90 | SA>6.5 | SC>4.5 |",
          "|---|---|---|---|---|---|---|---|"]
    for label, s in [("Tier-A/B reference", s_ref),
                      ("high-HOF (>+200 kcal/mol)", s_high),
                      ("top reranked candidates", s_cand)]:
        md.append(f"| {label} | {s['n']} | {s['sa_mean']:.2f} | {s['sa_p90']:.2f} | "
                  f"{s['sc_mean']:.2f} | {s['sc_p90']:.2f} | "
                  f"{s['above_sa6.5']} | {s['above_sc4.5']} |")

    delta_sa = s_cand["sa_mean"] - s_ref["sa_mean"]
    delta_sc = s_cand["sc_mean"] - s_ref["sc_mean"]
    md += ["",
           f"**Δ SA mean** (top − reference): {delta_sa:+.2f}",
           f"**Δ SC mean** (top − reference): {delta_sc:+.2f}",
           "",
           "Pass / fail:",
           f"- ΔSA ≤ +0.3 → {'PASS' if delta_sa <= 0.3 else 'FAIL'} (top candidates not "
           f"meaningfully harder to synthesise)",
           f"- ΔSC ≤ +0.2 → {'PASS' if delta_sc <= 0.2 else 'FAIL'}",
           "",
           "If FAIL: raise `--w_sa` / `--w_sc` in rerank, or enable feasibility "
           "guidance during sampling (`feasibility_sampler.py`)."]
    out = BASE / "docs/diag_d16.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    sys.exit(main())
