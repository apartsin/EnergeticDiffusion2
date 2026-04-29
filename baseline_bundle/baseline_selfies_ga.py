"""SELFIES-GA baseline for comparison against DGLD.

Genetic algorithm that directly optimises the same Pareto composite used in
rerank_v2.py, starting from the energetic training corpus.

Scoring pipeline mirrors rerank_v2.composite():
    composite = 0.45 * perf_score(rho, hof, d, p)
              + 0.20 * viability (hardcoded 0.5 — no viability model at
                                  inference time; same as rerank_v2 default)
              + 0.15 * novelty  (1 - ramp(maxTan, 0.20, 0.55))
              + 0.20 * (1 - sensitivity_proxy)
              - 0.10 * |red_flag_score|
              * scaffold_mult * perf_gate

The UniMol (3DCNN) predictor is used for rho / HOF / D / P, exactly as in
the DGLD pipeline.

Usage (local, CPU or GPU):
    /c/Python314/python baseline_bundle/baseline_selfies_ga.py \
        --corpus  baseline_bundle/corpus.csv \
        --out     results/selfies_ga_top100.json \
        --n_pool  1000 --n_gen 20 --n_top 100

For the Modal run (larger scale), see modal_baseline_selfies_ga.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors

RDLogger.DisableLog("rdApp.*")

try:
    import selfies as sf
except ImportError:
    raise ImportError("selfies not installed. Run: pip install selfies")


# ---------------------------------------------------------------------------
# Paths (relative to project root; override via --base)
# ---------------------------------------------------------------------------
DEFAULT_BASE = Path(__file__).parent.parent
SMOKE_MODEL_RELPATH = "data/raw/energetic_external/EMDP/Data/smoke_model"
CORPUS_RELPATH = "baseline_bundle/corpus.csv"


# ---------------------------------------------------------------------------
# Scoring helpers (mirrored from scripts/diffusion/rerank_v2.py)
# ---------------------------------------------------------------------------

def _ramp(x: float, lo: float, hi: float) -> float:
    if x <= lo: return 0.0
    if x >= hi: return 1.0
    return (x - lo) / (hi - lo)


def perf_score(rho: float, hof: float, d: float, p: float) -> float:
    """Saturating band-limited performance score in [0, 1]."""
    bands = {"rho": (1.65, 1.95), "hof": (0.0, 150.0),
             "d": (8.0, 9.5),    "p": (25.0, 40.0)}
    w = (0.30, 0.10, 0.30, 0.30)
    return (w[0] * _ramp(rho,  *bands["rho"])
          + w[1] * _ramp(hof,  *bands["hof"])
          + w[2] * _ramp(d,    *bands["d"])
          + w[3] * _ramp(p,    *bands["p"]))


def sensitivity_proxy_simple(smi: str) -> float:
    """Fast heuristic sensitivity score in [0, 1] without importing rerank_v2.
    Higher = more sensitive (worse). Mirrors the numeric part of
    rerank_v2.sensitivity_proxy; omits the SMARTS-alert floor for speed.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 1.0
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0:
        return 1.0
    mw = Descriptors.MolWt(mol)
    n_nitro = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N+](=O)[O-]")))
    n_N = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    n_ring_N = sum(1 for a in mol.GetAtoms()
                   if a.GetSymbol() == "N" and a.IsInRing())
    nitro_dens = n_nitro / max(n_heavy, 1)
    frac_n_ring = n_ring_N / max(n_N, 1)
    base = _ramp(nitro_dens, 0.20, 0.40)
    chain_n = 1.0 - frac_n_ring
    chain_term = chain_n * _ramp(n_nitro, 2, 5)
    small_polynitro = 0.5 if mw < 180 and n_nitro >= 3 else 0.0
    raw = base + 0.5 * chain_term + small_polynitro
    return min(1.0, raw)


def red_flag_score_simple(smi: str) -> float:
    """Returns a non-positive red-flag score. -1.0 = rejected, 0.0 = clean.
    Uses the legacy chem_redflags if available; otherwise a light built-in
    check that catches the most common structural issues.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return -1.0
    # Hard rejects: net charge, peroxides, gem-tetranitro
    if Chem.GetFormalCharge(mol) != 0:
        return -1.0
    bad_atoms = {a.GetSymbol() for a in mol.GetAtoms()} - {"C", "H", "N", "O", "F"}
    if bad_atoms:
        return -1.0
    reject_smarts = [
        "[O][O]",                                              # peroxide
        "[CX4]([N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",  # gem-tetranitro
        "[#6]1=[#6]=[#6]1",                                   # cumulated cyclopropene
    ]
    for sma in reject_smarts:
        pat = Chem.MolFromSmarts(sma)
        if pat and mol.HasSubstructMatch(pat):
            return -1.0
    # Soft demerits
    demerit_smarts = [
        ("[CX4]([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", 0.60),   # trinitromethyl
        ("[#6]=[#7;!R][#7;!R][N+](=O)[O-]", 0.50),                  # nitrohydrazone
        ("[#7;!R]=[#7;!R][#7;!R]", 0.50),                            # open-chain NNN
    ]
    total_demerit = 0.0
    for sma, weight in demerit_smarts:
        pat = Chem.MolFromSmarts(sma)
        if pat:
            n = len(mol.GetSubstructMatches(pat))
            total_demerit += weight * n
    return -min(total_demerit, 1.0)


def scaffold_mult(smi: str) -> float:
    """Scaffold multiplier: rewards aromatic heterocycles, penalises
    acyclic CHN backbone. Mirrors rerank_v2.scaffold_mult."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 1.0
    has_arom = any(a.GetIsAromatic() for a in mol.GetAtoms())
    n_arom_het = sum(
        1 for r in mol.GetRingInfo().AtomRings()
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() and
               mol.GetAtomWithIdx(i).GetSymbol() in ("N", "O") for i in r))
    if has_arom and n_arom_het >= 1:
        mult = 1.15
    elif has_arom:
        mult = 1.05
    else:
        mult = 1.00
    n_N = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    n_ring_atoms = mol.GetRingInfo().NumRings()
    if n_ring_atoms == 0 and n_N >= 3:
        mult *= 0.70
    return mult


def composite_score(smi: str, rho: float, hof: float, d: float, p: float,
                    maxtan: float = 0.0) -> float:
    """Full composite score. Returns -inf for hard-rejected molecules.
    Higher is better.
    """
    rfs = red_flag_score_simple(smi)
    if rfs <= -1.0:
        return float("-inf")

    perf = perf_score(rho, hof, d, p)
    viability = 0.5           # no viability model at inference; same as default
    novelty = 1.0 - _ramp(maxtan, 0.20, 0.55)
    sens = sensitivity_proxy_simple(smi)

    base = (0.45 * perf
          + 0.20 * viability
          + 0.15 * novelty
          + 0.20 * (1.0 - sens)
          - 0.10 * (-rfs))
    perf_gate = 1.0 / (1.0 + np.exp(-(perf - 0.35) * 8.0))
    return base * scaffold_mult(smi) * perf_gate


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus_smiles(corpus_path: Path, max_n: int = 50_000) -> list[str]:
    """Load canonical SMILES from the energetic corpus CSV.
    Filters to CHNO-only, neutral, RDKit-valid molecules.
    Returns up to max_n shuffled entries.
    """
    print(f"[corpus] loading {corpus_path} ...")
    df = pd.read_csv(corpus_path, usecols=["smiles"], low_memory=False)
    smiles_raw = df["smiles"].dropna().tolist()
    print(f"[corpus]   {len(smiles_raw)} rows")

    allowed = {"C", "H", "N", "O", "F"}
    valid = []
    for smi in smiles_raw:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if Chem.GetFormalCharge(mol) != 0:
            continue
        atoms = {a.GetSymbol() for a in mol.GetAtoms()}
        if atoms - allowed:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        valid.append(canon)

    # Deduplicate
    valid = list(dict.fromkeys(valid))
    print(f"[corpus]   {len(valid)} valid unique CHNO-neutral SMILES")

    if len(valid) > max_n:
        random.shuffle(valid)
        valid = valid[:max_n]
    return valid


# ---------------------------------------------------------------------------
# SELFIES mutation helpers
# ---------------------------------------------------------------------------

def smiles_to_selfies(smi: str) -> Optional[str]:
    try:
        return sf.encoder(smi)
    except Exception:
        return None


def selfies_to_smiles(sel: str) -> Optional[str]:
    try:
        smi = sf.decoder(sel)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _get_alphabet(corpus_selfies: list[str]) -> list[str]:
    alphabet = sf.get_alphabet_from_selfies(corpus_selfies)
    # Add common energetic tokens that may be sparse in the corpus
    extra = {"[N+1expl]", "[N-1expl]", "[O+0]", "[=N]", "[#N]",
             "[=O]", "[N]", "[O]", "[C]", "[Ring1]", "[Ring2]",
             "[Branch1]", "[Branch2]"}
    return sorted(alphabet | extra)


def mutate_selfies(sel: str, alphabet: list[str], rng: random.Random,
                   max_mutations: int = 3) -> Optional[str]:
    """Apply 1..max_mutations random token-level mutations to a SELFIES string.
    Operations: substitution, insertion, deletion.
    Returns mutated SMILES (canonical) or None if decode fails.
    """
    tokens = list(sf.split_selfies(sel))
    if not tokens:
        return None

    n_mut = rng.randint(1, max_mutations)
    for _ in range(n_mut):
        if not tokens:
            break
        op = rng.choice(["sub", "ins", "del"])
        idx = rng.randrange(len(tokens))
        if op == "sub":
            tokens[idx] = rng.choice(alphabet)
        elif op == "ins":
            tokens.insert(idx, rng.choice(alphabet))
        elif op == "del" and len(tokens) > 2:
            tokens.pop(idx)

    new_sel = "".join(tokens)
    return selfies_to_smiles(new_sel)


def crossover_selfies(sel_a: str, sel_b: str,
                      rng: random.Random) -> tuple[Optional[str], Optional[str]]:
    """Single-point crossover on two SELFIES token lists."""
    toks_a = list(sf.split_selfies(sel_a))
    toks_b = list(sf.split_selfies(sel_b))
    if len(toks_a) < 2 or len(toks_b) < 2:
        return None, None
    cut_a = rng.randint(1, len(toks_a) - 1)
    cut_b = rng.randint(1, len(toks_b) - 1)
    child_a = "".join(toks_a[:cut_a] + toks_b[cut_b:])
    child_b = "".join(toks_b[:cut_b] + toks_a[cut_a:])
    return selfies_to_smiles(child_a), selfies_to_smiles(child_b)


# ---------------------------------------------------------------------------
# 3DCNN scoring batch
# ---------------------------------------------------------------------------

def score_pool(smis: list[str], validator,
               ref_fps: list | None = None) -> list[dict]:
    """Score a list of SMILES with the UniMol 3DCNN validator.
    Returns list of dicts with keys: smiles, rho, hof, d, p, composite.
    Invalid predictions get composite = -inf.
    """
    if not smis:
        return []

    print(f"  [3dcnn] scoring {len(smis)} molecules ...", flush=True)
    preds = validator.predict(smis)

    rho_arr = np.asarray(preds["density"],            dtype=float)
    hof_arr = np.asarray(preds["HOF_S"],              dtype=float)
    d_arr   = np.asarray(preds["DetoD"],              dtype=float)
    p_arr   = np.asarray(preds["DetoP"],              dtype=float)

    results = []
    for i, smi in enumerate(smis):
        rho = float(rho_arr[i])
        hof = float(hof_arr[i])
        d   = float(d_arr[i])
        p   = float(p_arr[i])

        if any(np.isnan(v) for v in (rho, hof, d, p)):
            results.append({"smiles": smi, "rho": None, "hof": None,
                             "d": None, "p": None, "composite": float("-inf")})
            continue

        # Novelty via Tanimoto to reference set
        maxtan = 0.0
        if ref_fps is not None:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
                maxtan = float(max(sims)) if sims else 0.0

        comp = composite_score(smi, rho, hof, d, p, maxtan)
        results.append({"smiles": smi, "rho": rho, "hof": hof,
                         "d": d, "p": p, "composite": comp,
                         "maxtan": maxtan})
    return results


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

def run_ga(
    corpus_smiles: list[str],
    validator,
    n_pool: int = 1000,
    n_gen: int = 20,
    n_top: int = 100,
    elite_frac: float = 0.50,
    new_frac: float = 0.20,
    crossover_frac: float = 0.15,
    max_mutations: int = 3,
    seed: int = 42,
    ref_fps: list | None = None,
) -> list[dict]:
    """Run the SELFIES-GA for n_gen generations.

    Each generation:
      1. Score all molecules with 3DCNN + composite
      2. Keep top elite_frac by composite (parents)
      3. Fill remainder by:
         a. Mutation of random parents
         b. Single-point crossover of random parent pairs
         c. new_frac random molecules from corpus (diversity injection)
      4. Deduplicate

    Returns sorted list of top n_top dicts after the final generation.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Initial pool: sample from corpus
    pool_size = min(n_pool, len(corpus_smiles))
    pool = rng.sample(corpus_smiles, pool_size)
    print(f"[GA] initial pool size: {len(pool)}")

    # Precompute SELFIES for corpus subset (for random injection)
    corpus_selfies_map: dict[str, str] = {}  # smi -> selfies
    for smi in corpus_smiles[:5000]:
        sel = smiles_to_selfies(smi)
        if sel:
            corpus_selfies_map[smi] = sel
    corpus_smi_list = list(corpus_selfies_map.keys())

    # Build SELFIES alphabet from the initial pool
    init_selfies = []
    for smi in pool[:500]:
        sel = smiles_to_selfies(smi)
        if sel:
            init_selfies.append(sel)
    if not init_selfies:
        raise RuntimeError("No valid SELFIES could be encoded from initial pool.")
    alphabet = _get_alphabet(init_selfies)
    print(f"[GA] SELFIES alphabet size: {len(alphabet)}")

    best_overall: list[dict] = []
    stagnation_count = 0
    prev_best_comp = float("-inf")

    for gen in range(1, n_gen + 1):
        t0 = time.time()
        print(f"\n[GA] === Generation {gen}/{n_gen}  (pool={len(pool)}) ===")

        # 1. Score
        scored = score_pool(pool, validator, ref_fps=ref_fps)
        scored = [r for r in scored if r["composite"] > float("-inf")]
        scored.sort(key=lambda r: -r["composite"])

        # Track best
        if scored:
            best_comp = scored[0]["composite"]
            print(f"  best composite: {best_comp:.4f}  "
                  f"(rho={scored[0]['rho']:.3f}, "
                  f"D={scored[0]['d']:.2f}, "
                  f"P={scored[0]['p']:.2f})")
            if best_comp > prev_best_comp + 1e-4:
                stagnation_count = 0
                prev_best_comp = best_comp
            else:
                stagnation_count += 1
                if stagnation_count >= 5:
                    print(f"  [GA] convergence: no improvement for 5 gens, stopping.")
                    break

        # Accumulate unique best candidates across all generations
        seen_smis = {r["smiles"] for r in best_overall}
        for r in scored[:n_top]:
            if r["smiles"] not in seen_smis:
                best_overall.append(r)
                seen_smis.add(r["smiles"])

        n_elite = max(int(len(scored) * elite_frac), 10)
        parents = [r["smiles"] for r in scored[:n_elite]]
        if not parents:
            print("  [GA] no valid parents, re-seeding from corpus.")
            parents = rng.sample(corpus_smi_list, min(50, len(corpus_smi_list)))

        # Convert parents to SELFIES
        parents_selfies = []
        for smi in parents:
            sel = smiles_to_selfies(smi)
            if sel:
                parents_selfies.append((smi, sel))

        # 2. Build next generation
        next_pool_set: set[str] = set(parents)   # elites always survive

        # Crossover offspring
        n_cross = int(n_pool * crossover_frac)
        cross_attempts = 0
        while len(next_pool_set) < len(parents) + n_cross and cross_attempts < n_cross * 5:
            cross_attempts += 1
            if len(parents_selfies) < 2:
                break
            (_, sel_a), (_, sel_b) = rng.sample(parents_selfies, 2)
            c1, c2 = crossover_selfies(sel_a, sel_b, rng)
            for c in (c1, c2):
                if c and c not in next_pool_set:
                    next_pool_set.add(c)

        # Mutation offspring
        n_mutants = n_pool - int(n_pool * new_frac) - len(next_pool_set)
        mut_attempts = 0
        while len(next_pool_set) < len(next_pool_set) + n_mutants and mut_attempts < n_mutants * 5:
            if len(next_pool_set) >= n_pool - int(n_pool * new_frac):
                break
            mut_attempts += 1
            if not parents_selfies:
                break
            _, sel = rng.choice(parents_selfies)
            child = mutate_selfies(sel, alphabet, rng, max_mutations)
            if child and child not in next_pool_set:
                next_pool_set.add(child)

        # Random injection from corpus
        n_new = n_pool - len(next_pool_set)
        if n_new > 0 and corpus_smi_list:
            injected = rng.sample(corpus_smi_list,
                                   min(n_new, len(corpus_smi_list)))
            next_pool_set.update(injected)

        pool = list(next_pool_set)[:n_pool]
        elapsed = time.time() - t0
        print(f"  next pool: {len(pool)}  ({elapsed:.1f}s)")

    # Final scoring pass on accumulated best_overall
    print(f"\n[GA] Final scoring of {len(best_overall)} accumulated candidates ...")
    if best_overall:
        final_smis = [r["smiles"] for r in best_overall]
        final_scored = score_pool(final_smis, validator, ref_fps=ref_fps)
        final_scored = [r for r in final_scored if r["composite"] > float("-inf")]
        final_scored.sort(key=lambda r: -r["composite"])
        return final_scored[:n_top]
    return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="SELFIES-GA baseline: directly optimise the DGLD Pareto composite.")
    ap.add_argument("--base",    default=str(DEFAULT_BASE),
                    help="Project root (default: auto-detected parent of this file)")
    ap.add_argument("--corpus",  default=None,
                    help="Path to energetic corpus CSV with a 'smiles' column. "
                         "Defaults to <base>/baseline_bundle/corpus.csv")
    ap.add_argument("--model_dir", default=None,
                    help="Path to UniMol smoke_model directory. "
                         "Defaults to <base>/data/raw/energetic_external/EMDP/Data/smoke_model")
    ap.add_argument("--out",     default=None,
                    help="Output JSON path. Defaults to <base>/results/selfies_ga_top100.json")
    ap.add_argument("--n_pool",  type=int, default=1000,
                    help="Population size per generation (default 1000)")
    ap.add_argument("--n_gen",   type=int, default=20,
                    help="Max generations (default 20)")
    ap.add_argument("--n_top",   type=int, default=100,
                    help="Number of top candidates to output (default 100)")
    ap.add_argument("--elite_frac",  type=float, default=0.50)
    ap.add_argument("--new_frac",    type=float, default=0.20)
    ap.add_argument("--max_mut",     type=int,   default=3)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--with_novelty", action="store_true",
                    help="Compute Tanimoto to training set for novelty term "
                         "(slower; uses 5k ref fps)")
    ap.add_argument("--max_corpus",  type=int, default=50_000,
                    help="Maximum corpus entries to load (default 50000)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    base = Path(args.base)
    corpus_path = Path(args.corpus) if args.corpus else base / CORPUS_RELPATH
    model_dir   = Path(args.model_dir) if args.model_dir else base / SMOKE_MODEL_RELPATH
    out_path    = Path(args.out) if args.out else base / "results/selfies_ga_top100.json"

    # Verify model exists
    if not (model_dir / "model_0.pth").exists():
        print(f"[ERROR] 3DCNN model not found at {model_dir}")
        print("        Expected model_0.pth in that directory.")
        sys.exit(1)

    # Load UniMol validator
    sys.path.insert(0, str(base / "scripts/diffusion"))
    from unimol_validator import UniMolValidator
    print(f"[init] Loading UniMol validator from {model_dir} ...")
    validator = UniMolValidator(model_dir)

    # Load corpus
    corpus_smiles = load_corpus_smiles(corpus_path, max_n=args.max_corpus)
    if not corpus_smiles:
        print("[ERROR] No valid SMILES loaded from corpus.")
        sys.exit(1)

    # Optional reference fingerprints for novelty computation
    ref_fps = None
    if args.with_novelty:
        print("[init] Building reference fingerprints from corpus (up to 5k) ...")
        ref_fps = []
        for smi in corpus_smiles[:5000]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                ref_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        print(f"[init]   {len(ref_fps)} reference fps")

    # Run GA
    t_start = time.time()
    top_candidates = run_ga(
        corpus_smiles=corpus_smiles,
        validator=validator,
        n_pool=args.n_pool,
        n_gen=args.n_gen,
        n_top=args.n_top,
        elite_frac=args.elite_frac,
        new_frac=args.new_frac,
        max_mutations=args.max_mut,
        seed=args.seed,
        ref_fps=ref_fps,
    )
    elapsed = time.time() - t_start

    # Serialize
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method":           "SELFIES-GA",
        "n_pool":           args.n_pool,
        "n_gen":            args.n_gen,
        "n_top":            len(top_candidates),
        "seed":             args.seed,
        "elapsed_s":        round(elapsed, 1),
        "corpus":           str(corpus_path),
        "model_dir":        str(model_dir),
        "top_candidates":   top_candidates,
        "summary": {
            "top1_composite":   top_candidates[0]["composite"] if top_candidates else None,
            "top1_D_kms":       top_candidates[0]["d"]         if top_candidates else None,
            "top1_P_GPa":       top_candidates[0]["p"]         if top_candidates else None,
            "top1_rho":         top_candidates[0]["rho"]        if top_candidates else None,
            "topN_mean_composite": float(np.mean([r["composite"] for r in top_candidates]))
                                   if top_candidates else None,
            "topN_mean_D":         float(np.mean([r["d"] for r in top_candidates
                                                   if r["d"] is not None]))
                                   if top_candidates else None,
            "topN_max_D":          float(max(r["d"] for r in top_candidates
                                              if r["d"] is not None))
                                   if top_candidates else None,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[GA] Done in {elapsed:.0f}s. Top {len(top_candidates)} -> {out_path}")
    if top_candidates:
        r0 = top_candidates[0]
        print(f"  #1: comp={r0['composite']:.4f}  rho={r0['rho']:.3f}  "
              f"D={r0['d']:.2f}  P={r0['p']:.2f}  HOF={r0['hof']:.1f}")
        print(f"      {r0['smiles']}")


if __name__ == "__main__":
    main()
