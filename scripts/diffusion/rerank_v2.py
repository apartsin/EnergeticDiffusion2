"""Rerank v2 — multi-objective, gated, viability-aware.

Reads an existing rerank markdown (e.g. joint_rerank_pool40k.md) for its
predicted-property table and re-applies:

  Phase 1:
    1.1 SMARTS red-flag filter   (chem_redflags.screen)
    1.2 formula/atom-ratio gates (in chem_redflags.screen)
    1.3 saturating performance score (banded ramp, capped at 1.0)
    1.4 OOD penalty (latent distance to train set, optional)
    1.5 Pareto ranking on (perf, novelty, viability) vs (alerts, OOD)

  Phase 2 (if --viability_model given):
    2.1 viability classifier P(EM-like and stable)
    2.2 sensitivity proxy (open-chain N-NO2 motifs, etc.)

Writes the result to a NEW markdown file; never overwrites originals.

Usage:
    python scripts/diffusion/rerank_v2.py \
        --in  experiments/.../joint_rerank_pool40k.md \
        --out experiments/.../joint_rerank_pool40k_v2.md
        [--viability_model experiments/viability_rf_v1/model.joblib]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

sys.path.insert(0, "scripts/diffusion")
from chem_redflags import screen as _screen_legacy
from scoring_framework import screen as _screen_yaml

# Module-level binding; main() rebinds based on --use_yaml_screen.
screen = _screen_legacy


# ── ramp / band helpers ─────────────────────────────────────────────────
def ramp(x, lo, hi):
    """0 below lo, 1 above hi, linear in between. Saturates."""
    if x <= lo: return 0.0
    if x >= hi: return 1.0
    return (x - lo) / (hi - lo)


def perf_score(rho, hof, d, p,
               weights=(0.30, 0.10, 0.30, 0.30),
               bands=None):
    """Saturating, band-limited multi-property score in [0, 1].
    Caps each property at the high end of the design band so the model can't
    win by over-extrapolating. Returns weighted sum."""
    if bands is None:
        bands = {
            "rho": (1.65, 1.95),
            "hof": (0,    150),
            "d":   (8.0,  9.5),
            "p":   (25,   40),
        }
    s_rho = ramp(rho, *bands["rho"])
    s_hof = ramp(hof, *bands["hof"])
    s_d   = ramp(d,   *bands["d"])
    s_p   = ramp(p,   *bands["p"])
    w = weights
    return w[0]*s_rho + w[1]*s_hof + w[2]*s_d + w[3]*s_p


def parse_md_rows(path: Path) -> List[Dict]:
    md = path.read_text(encoding="utf-8")
    rows = []
    for line in md.split("\n"):
        if not line.startswith("| ") or "rank" in line or "---" in line:
            continue
        cells = [c.strip(" `") for c in line.split("|")]
        if len(cells) < 12: continue
        try:
            rows.append({
                "rank":   int(cells[1]),
                "comp_old": float(cells[2]),
                "rho":    float(cells[3]),
                "hof":    float(cells[4].replace("+", "")),
                "d":      float(cells[5]),
                "p":      float(cells[6]),
                "sa":     float(cells[7]),
                "sc":     float(cells[8]),
                "maxtan": float(cells[9]),
                "src":    cells[10],
                "smiles": cells[11],
            })
        except (ValueError, IndexError):
            continue
    return rows


def viability_score(smis: List[str], model_path: Path) -> List[float]:
    """Return P(EM-like) per smiles using a trained RF classifier."""
    if not model_path.exists():
        print(f"  [viability] model not found at {model_path}; using NaN")
        return [float("nan")] * len(smis)
    sys.path.insert(0, "scripts/viability")
    from train_viability import featurize
    import joblib
    rf = joblib.load(model_path)
    out = []
    for s in smis:
        f = featurize(s)
        if f is None: out.append(float("nan")); continue
        p = rf.predict_proba(f.reshape(1, -1))[0, 1]
        out.append(float(p))
    return out


def sensitivity_proxy(scr: dict) -> float:
    """Heuristic sensitivity score in [0, 1]: HIGHER means MORE sensitive
    (worse). Updated with N-chain / nitrohydrazone / nitroalkene / explosophore-
    density penalties so candidates with structural alerts cannot get sens=0."""
    if scr["status"] != "ok": return 1.0
    nitro_dens = scr["nitro_per_heavy"]
    base = ramp(nitro_dens, 0.20, 0.40)
    chain_n = 1.0 - scr["frac_n_ring"]
    chain_term = chain_n * ramp(scr["nitro"], 2, 5)
    small_polynitro = 0.5 if (scr["mw"] is not None and scr["mw"] < 180
                                and scr["nitro"] >= 3) else 0.0
    # Forced-uncertainty floor: any N-chain alert pegs sens >= 0.30
    floor = 0.0
    alert_names = {n for n, _ in scr.get("alerts", [])}
    high_risk_motifs = {
        "nitrohydrazone", "open_chain_NNN", "diazo_chain_with_nitro",
        "nitroalkene_hydrazone", "multi_nitrohydrazone", "nitrohydrazone_1",
        "small_unsaturated_polynitro_no_arom",
    }
    medium_motifs = {
        "open_chain_NN_NO2", "nitroalkene_open", "n_chain_len_3",
        "explosophore_density_high",
    }
    very_high_motifs = {
        "n_chain_len_4", "n_chain_len_5", "explosophore_density_very_high",
    }
    if alert_names & very_high_motifs: floor = max(floor, 0.65)
    elif alert_names & high_risk_motifs: floor = max(floor, 0.50)
    elif alert_names & medium_motifs: floor = max(floor, 0.30)
    if scr.get("longest_n_chain", 0) >= 4: floor = max(floor, 0.65)
    if scr.get("explosophore_density", 0) > 0.4: floor = max(floor, 0.45)
    raw = base + 0.5*chain_term + small_polynitro
    return min(1.0, max(raw, floor))


def is_pareto_min(points: np.ndarray) -> np.ndarray:
    """Pareto front for multi-objective MINIMIZATION on all dims."""
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]: continue
        for j in range(n):
            if i == j: continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                keep[i] = False
                break
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viability_model", default=None)
    ap.add_argument("--limit", type=int, default=400,
                    help="Limit input rows scanned (input is already sorted).")
    ap.add_argument("--use_yaml_screen", action="store_true",
                    help="Use scoring_framework.screen (YAML rules/alerts.yaml) "
                         "instead of legacy chem_redflags.screen. Default off "
                         "for paper-result reproducibility; enable in new cycles.")
    args = ap.parse_args()
    global screen
    screen = _screen_yaml if args.use_yaml_screen else _screen_legacy
    print(f"screen backend: {'yaml (scoring_framework)' if args.use_yaml_screen else 'legacy (chem_redflags)'}")

    rows = parse_md_rows(Path(args.inp))[:args.limit]
    print(f"Loaded {len(rows)} input candidates from {args.inp}")

    # 1.1 + 1.2 hard filters
    kept = []; rejected_by = {}
    for r in rows:
        scr = screen(r["smiles"])
        r["screen"] = scr
        if scr["status"] != "ok":
            for reason in (scr["reasons"] or ["unknown"]):
                rejected_by[reason] = rejected_by.get(reason, 0) + 1
            continue
        kept.append(r)
    print(f"After hard filters: {len(kept)}/{len(rows)}")
    for k, v in sorted(rejected_by.items(), key=lambda x: -x[1]):
        print(f"  rejected for {k}: {v}")

    if not kept:
        print("Nothing survived hard filters."); return

    # 1.3 saturating performance score
    for r in kept:
        r["perf"] = perf_score(r["rho"], r["hof"], r["d"], r["p"])
        r["alerts"] = r["screen"]["red_flag_score"]   # <= 0
        r["sensitivity"] = sensitivity_proxy(r["screen"])
        # bigger when more novel (lower Tanimoto)
        r["novelty"] = 1.0 - ramp(r["maxtan"], 0.20, 0.55)

    # 2.1 viability
    if args.viability_model:
        print("\nApplying viability model …")
        smis = [r["smiles"] for r in kept]
        probs = viability_score(smis, Path(args.viability_model))
        for r, p in zip(kept, probs):
            r["viability"] = p
        good = sum(1 for p in probs if not (p != p))   # not NaN
        print(f"  scored {good}/{len(probs)} candidates")
    else:
        for r in kept:
            r["viability"] = float("nan")

    # Composite (gated): higher = better
    # Phase A: add scaffold multiplier (aromatic heterocycle reward,
    # open-chain CHN penalty) + D-vs-MW sanity.
    def scaffold_mult(scr, r):
        n_aromhet = scr.get("n_arom_het_rings", 0)
        has_arom = scr.get("has_aromatic", False)
        mol = scr.get("mol")
        if has_arom and n_aromhet >= 1:
            mult = 1.15
        elif has_arom:
            mult = 1.05
        else:
            mult = 1.00
        # Penalise pure acyclic CHN backbone with >= 3 N
        n_N = scr.get("n_N", 0)
        n_ring_atoms = mol.GetRingInfo().NumRings() if mol else 0
        if n_ring_atoms == 0 and n_N >= 3:
            mult *= 0.70
        # D-vs-MW sanity: predicted D > 9.3 with MW < 220 and no aromatic
        if r["d"] > 9.3 and scr.get("mw", 999) < 220 and not has_arom:
            mult *= 0.65
        return mult

    def composite(r):
        v = r["viability"] if r["viability"] == r["viability"] else 0.5
        scr = r["screen"]
        base = (0.45 * r["perf"]
              + 0.20 * v
              + 0.15 * r["novelty"]
              + 0.20 * (1 - r["sensitivity"])
              - 0.10 * (-r["alerts"]))
        # Perf gate: candidates with very low predicted performance shouldn't
        # be ranked alongside high-performance ones, even if all other axes are
        # great. Sigmoid steep around perf=0.35.
        perf_gate = 1.0 / (1.0 + np.exp(-(r["perf"] - 0.35) * 8.0))
        return base * scaffold_mult(scr, r) * perf_gate

    for r in kept:
        r["composite_v2"] = composite(r)

    # 1.5 Pareto front: minimise -perf, -viability, sensitivity, alerts(neg=0)
    pts = np.array([[
        -r["perf"], -(r["viability"] if r["viability"] == r["viability"] else 0.5),
        r["sensitivity"], -(r["novelty"]),
    ] for r in kept])
    on_front = is_pareto_min(pts)
    for r, f in zip(kept, on_front):
        r["pareto"] = bool(f)

    # Sort: pareto first, then composite
    kept.sort(key=lambda r: (-int(r["pareto"]), -r["composite_v2"]))
    n_front = sum(1 for r in kept if r["pareto"])
    print(f"\nPareto front: {n_front} candidates")

    # Write markdown
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# Rerank v2 — multi-objective, gated\n\n")
        f.write(f"input: `{args.inp}` ({len(rows)} candidates scanned)\n\n")
        f.write(f"- Hard filters kept: {len(kept)}/{len(rows)}\n")
        f.write(f"- Pareto front: {n_front}\n")
        f.write(f"- Viability model: `{args.viability_model or '— (not applied)'}`\n\n")
        f.write("## Filter rejections by reason\n\n")
        f.write("| reason | count |\n|---|---|\n")
        for k, v in sorted(rejected_by.items(), key=lambda x: -x[1]):
            f.write(f"| {k} | {v} |\n")
        f.write("\n## Top 100 (Pareto-front first, then composite v2)\n\n")
        f.write("| rank | comp_v2 | perf | viab | sens | novel | ρ | HOF | D | P | "
                "alerts | OB | nNO2 | MW | Pareto | SMILES |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(kept[:100], 1):
            scr = r["screen"]
            v = "—" if r["viability"] != r["viability"] else f"{r['viability']:.2f}"
            f.write(f"| {i} | {r['composite_v2']:.3f} | {r['perf']:.2f} | {v} | "
                    f"{r['sensitivity']:.2f} | {r['novelty']:.2f} | "
                    f"{r['rho']:.2f} | {r['hof']:+.0f} | {r['d']:.2f} | {r['p']:.2f} | "
                    f"{r['alerts']:+.1f} | {scr['ob']:+.0f} | {scr['nitro']} | "
                    f"{scr['mw']:.0f} | {'★' if r['pareto'] else ' '} | "
                    f"`{r['smiles']}` |\n")
    print(f"\n-> wrote {out}")
    print(f"\nTop 5 by composite v2:")
    for i, r in enumerate(kept[:5], 1):
        v = "—" if r["viability"] != r["viability"] else f"{r['viability']:.2f}"
        print(f"  [{i}] comp={r['composite_v2']:.3f}  perf={r['perf']:.2f}  "
              f"viab={v}  sens={r['sensitivity']:.2f}  rho={r['rho']:.2f}  "
              f"D={r['d']:.2f}  P={r['p']:.2f}  pareto={'*' if r['pareto'] else ''}")
        print(f"      {r['smiles']}")


if __name__ == "__main__":
    main()
