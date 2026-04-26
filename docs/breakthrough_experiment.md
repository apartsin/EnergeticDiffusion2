# Breakthrough discovery experiment

Goal: produce candidate SMILES that **beat current SOTA energetic materials**
(CL-20 / ICM-101 / TKX-50) on at least one of {ρ, D, P, HOF} **without**
regressing on synthesisability.

Two independent paths, both should be executed and compared:

| Path | Strategy | Best for |
|---|---|---|
| **A — De novo (Path A)** | Joint v3 + v4-B pool + rerank with feasibility composite + chem_filter + hard caps | scaffold-novel candidates; insensitive to LIMO roundtrip weakness |
| **B — Compound-to-compound (Path B)** | SDEdit on known SOTA seeds, property-targeted | analog improvements — explicit synthesis-route inheritance |

Path A is the **primary** strategy now (Path B's effectiveness is bottlenecked
by LIMO v1's poor scaffold preservation; will dominate after LIMO v2).

---

## 1. Targets

| Metric | Threshold for "breakthrough" | SOTA reference |
|---|---|---|
| ρ | ≥ 1.95 g/cm³ | CL-20 = 2.04 |
| D | ≥ 9.5 km/s | TKX-50 = 9.7 |
| P | ≥ 40 GPa | CL-20 = 42 |
| HOF | ≥ +200 kcal/mol | ICM-101 = +204 |
| SA | ≤ 5.0 (ideally ≤ 4.5) | CL-20 = 5.5 |
| SC | ≤ 3.5 | CL-20 = 3.6 |
| Tanimoto vs nearest known | ∈ [0.25, 0.45] (Path A) or [0.30, 0.60] (Path B) | — |

A candidate clearing **all** of these is a publishable lead.

---

## 2. Filters applied per candidate (in order)

1. RDKit parse + canonicalise.
2. `--require_neutral` — formal charge = 0, no radicals.
3. `chem_filter` — composition (CHNO + N), property bounds, unstable
   motifs (peroxide, polynitrogen chains, halogens, P/S), oxygen balance
   −250 % to +60 %.
4. 3DCNN ensemble validation: predicted ρ, D, P, HOF.
5. Real SA + SC scoring on canonical SMILES.
6. Hard caps: SA ≤ 5.0, SC ≤ 3.5.
7. Property-target box: ρ ≥ 1.95, D ≥ 9.5, P ≥ 40, HOF ≥ 200.
8. Novelty: not in 382 k internal training set, not in PubChem.

Composite ranking applies after these gates. Surviving candidates are the
breakthrough leads.

---

## 3. Path A — joint v3 + v4-B pool

```bash
python scripts/diffusion/joint_rerank.py \
    --exp_v4b experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z \
    --exp_v3  experiments/diffusion_subset_cond_expanded_v3_20260425T140941Z \
    --cfg 7 --n_pool_each 3000 --n_keep 80 \
    --target_density 2.0 --target_d 9.5 --target_p 40 --target_hof 220 \
    --hard_sa 5.0 --hard_sc 3.5 \
    --require_neutral --with_chem_filter
```

Why both v3 and v4-B:
- v4-B excels at ρ / D / P targeting (rerank pool=1500 → 0.2-1.0 % rel_MAE).
- v3 excels at HOF tail (max generated +341 kcal/mol, 18 % within-10 %).
- Joint pool dedupes canonical SMILES, ranks by single composite, keeps the
  union of strengths.

Expected funnel:
- 6,000 raw → ~3,000 RDKit-valid → ~1,500 neutral → ~600 chem_filter pass
  → ~150 in target box → ~30 below SA/SC caps → top-20 by composite.

---

## 4. Path B — c2c from SOTA seeds

```bash
python scripts/diffusion/c2c_pipeline.py \
    --exp experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z \
    --seeds_csv data/c2c/seeds.csv \
    --strengths 0.3 0.5 0.7 --n_variants 200 \
    --target_density 2.05 --target_d 9.6 --target_p 42 --target_hof 200 \
    --require_neutral --with_chem_filter
```

Per-seed gating:

A variant `v` of seed `s` is a **Path-B success** if **all** hold:

| Criterion | Threshold |
|---|---|
| Tanimoto(v, s) | ∈ [0.30, 0.60] |
| ΔD vs s | ≥ +0.2 km/s |
| Δρ vs s | ≥ −0.05 g/cm³ |
| ΔP vs s | ≥ 0 GPa |
| ΔHOF vs s | ≥ 0 (or within ±50) |
| ΔSA vs s | ≤ +0.5 |
| chem_filter + neutral + novelty | required |

---

## 5. Three breakthrough classes (combined output)

| Class | Definition | Provenance |
|---|---|---|
| **A — Analog improvement** | Path-B success: variant beats seed on ≥1 metric, Tanimoto 0.3–0.6 to seed | c2c |
| **B — De novo Pareto** | Path-A candidate clearing the full target box AND novel scaffold (Tanimoto < 0.3 to *every* SOTA) | unconditional |
| **C — Feasibility-gain analog** | Path-B success with ΔSA ≤ −0.5 (much easier to make than seed) | c2c |

---

## 6. Acceptance — when do we stop and call it a result?

Run is a real breakthrough if **any** of:

1. **≥ 3 Class-B candidates** simultaneously clear ρ ≥ 1.95, D ≥ 9.5, P ≥ 40,
   HOF ≥ 200, SA ≤ 5.0, SC ≤ 3.5, AND Tanimoto < 0.30 to every known SOTA.
2. **≥ 1 Class-A candidate** beats CL-20 *jointly* on D ≥ 9.5 AND ρ ≥ 2.0 AND SA ≤ 5.5.
3. **≥ 1 Class-C candidate** matches CL-20 properties (within 5 %) at SA ≤ 4.0.

---

## 7. Reporting structure

`<exp>/breakthrough_results.md`:

```
| class | seed | candidate SMILES | ρ | D | P | HOF | SA | SC | Tan(seed) | Tan(SOTA) | DFT-flag |
| ----- | ---- | ---------------- | - | - | - | --- | -- | -- | --------- | --------- | -------- |
```

For each candidate row, attach:
- predicted properties (3DCNN ensemble)
- nearest neighbour in training set (Tanimoto + SMILES)
- proposed synthesis route (high level — from seed similarity, if Path B)
- DFT-validation TODO flag (always TODO; Psi4 spot-check is the next step)

`<exp>/breakthrough_summary.md`:

- counts per class
- pareto plot (Tanimoto-to-SOTA × ΔD vs CL-20 × SA)
- top-20 leads regardless of class
- interesting failure modes (e.g. "high D + high SA cluster" suggests
  unsynthesisable polynitros) for chemistry triage

---

## 8. Honest pre-registered hypotheses

| Hypothesis | Likely outcome with current LIMO v1 |
|---|---|
| Path A finds Class-B candidates | **likely** — joint pool gives ~10–30 candidates clearing ρ/D/P box, ~1–5 also clearing HOF and SA caps |
| Path B finds Class-A analog improvements | **unlikely** — c2c Tanimoto-to-seed ~0.1–0.2 means most variants fall outside the [0.3, 0.6] band |
| Path B finds Class-C feasibility analogs | **unlikely** — same Tanimoto problem |
| LIMO v2 motif-rich fine-tune unlocks Path B | **expected** — D2 + D15 directly identified the LIMO-side bottleneck |

So the breakthrough we should expect *today* is Path-A / Class-B; Path-B
becomes powerful after LIMO v2.

---

## 9. Required code (status)

| Component | Status |
|---|---|
| `joint_rerank.py` (Path A) | **TODO** — straightforward extension of `rerank_multi.py` with two checkpoints |
| `c2c_pipeline.py` (Path B) | done; recently produced the c2c index showing Tanimoto weakness |
| `chem_filter.py` | done; integrated into `rerank_sweep` and `rerank_multi` |
| Composite scoring with feasibility | done; `--with_feasibility --w_sa 0.5 --w_sc 0.25` |
| Novelty lookup | done; `match_candidates.py` checks PubChem + internal |
| LIMO v2 fine-tune | plan written, not run; gating Path B effectiveness |

---

## 10. Single-command production wrapper (after `joint_rerank.py` lands)

```bash
bash scripts/run_breakthrough.sh
```

Will execute:
1. Path A joint pool (~5 min)
2. Path B c2c on top SOTA seeds (~10 min)
3. Merge + classify candidates by class
4. Run novelty lookup on all leads
5. Produce `breakthrough_results.md` + `breakthrough_summary.md`

Total wall-clock: ~20 minutes on RTX 2060.
