# Feasibility guidance (SA + SC) — integration plan

> **Validation result (2026-04-26)**: rerank-weight composite works (`(w_sa, w_sc) = (0.5, 0.25)` lowers top-40 SA mean by 0.22 with no property regression). Sampling-time gradient guidance shelved — the existing DDIM schedule's `α̅[T-1]=0` edge case forces a `z0_pred` norm clip that masks the gradient signal. A full sampler rewrite is needed to revisit; not pursued for now.


Add **synthetic-accessibility** signals into the production generation /
ranking / evaluation pipeline. Uses the already-trained SA + SC latent
predictors (no retraining required).

---

## 1. Background

| Score | What it measures | Lower = | Source |
|---|---|---|---|
| **SA** (Ertl-Schuffenhauer) | molecule synthetic-accessibility heuristic | easier to make | `external/LIMO/sascorer.py` (RDKit-based) |
| **SC** (Coley) | machine-learned synthetic complexity | easier to make | `external/scscore/` |

Both are computed on canonical SMILES, so they're not directly differentiable
in latent space. We've already trained two **latent-side surrogate
predictors** (`ScorePredictor` MLPs) on `(z_mu, score)` pairs covering all
382 k SMILES. They are differentiable w.r.t. z — the prerequisite for
classifier guidance.

| Predictor | r² | MAE (standardised) | Checkpoint |
|---|---|---|---|
| SA latent surrogate | 0.72 | 0.40 | `experiments/guidance_sa_sc_20260424T172820Z/checkpoints/sa_best.pt` |
| SC latent surrogate | 0.67 | 0.44 | `experiments/guidance_sa_sc_20260424T172820Z/checkpoints/sc_best.pt` |

These are already in the repo.

---

## 2. Where it slots into the production pipeline

```
Diffusion (DDIM sampling)
        │
        ▼
   raw latents z*  ────► [optional] feasibility-guided DDIM
        │                  • gradient term −λ_SA ∇_z f_SA(z)
        │                  • gradient term −λ_SC ∇_z f_SC(z)
        ▼
 LIMO decoder ──► SMILES
        │
        ▼
   3DCNN validator ──► (ρ̂, D̂, P̂, ĤOF)
        │
        ▼
   composite reranker  ◄─── SA, SC computed on canonical SMILES
        │                   (real, not surrogate; for ranking precision)
        ▼
   top-N candidates with
   {ρ, D, P, HOF, SA, SC} scored
```

Two integration points:

| Stage | What we add | Why this stage |
|---|---|---|
| **Sampling (DDIM)** | gradient term −λ_SA ∇ f_SA − λ_SC ∇ f_SC at each step | shifts the *latent distribution* toward feasible regions |
| **Reranking (post-validation)** | SA + SC contribute to the composite rank | filters out remaining infeasible outliers; uses *real* SA/SC, not the differentiable surrogate |

Both are cheap (the surrogate is one MLP forward; real SA/SC takes ~1 ms each per molecule) and complementary — sampling-time guidance widens the candidate pool in the right direction, post-hoc reranking enforces the hard constraint.

---

## 3. Training procedure

**No new training is required.** The SA + SC latent surrogates are already
trained (43 s per head, on 363 k Tier-A/B + Tier-D latents, R² > 0.65).

If we ever need to refresh:

| Setting | Value | Reason |
|---|---|---|
| Code | `scripts/guidance/train.py` (existing) | unchanged |
| Config | `configs/guidance.yaml` | unchanged |
| Wall-clock | ~90 s for both heads | trivial |
| Trigger | only if LIMO is retrained (latents change → surrogate stale) | re-run is part of every LIMO retrain checklist |

The surrogates are **tied to LIMO's latent space**. If we swap LIMO for
MolMIM / ChemFormer / CDDD / HierVAE, the SA + SC surrogates must be
retrained from scratch on the new latents.

---

## 4. Generation: feasibility-guided DDIM

New file: `scripts/diffusion/feasibility_sampler.py` — extends the existing
classifier-guidance sampler with SA + SC gradient terms.

### 4.1 Math

Classifier guidance with ε-prediction (Dhariwal & Nichol 2021):

```
ε_guided = ε − σ_t · λ · ∇_z log p(c | z_t)
```

For our case we minimise SA + SC (lower = easier to synthesise), so we use
the *negative* of the surrogate's standardised output as `log p(c | z)`:

```
ε_guided = ε + σ_t · (λ_SA · ∇_z f_SA(z_t) + λ_SC · ∇_z f_SC(z_t))
```

Reminder: `f_SA(z_t)` returns a *standardised* SA score; positive sign
means the surrogate predicts "harder to synthesise". Subtracting the
gradient pushes z toward predicted-low SA, i.e. easier-to-synthesise
chemistry. (Note the sign flip vs property guidance, where we wanted to
*increase* the property toward target.)

### 4.2 Important detail — surrogate vs. noisy z

The SA / SC surrogates were trained on **clean** `z_mu`, not noisy `z_t`.
Empirical lesson from the prior property-classifier guidance run: applying
the gradient at high noise levels (early DDIM steps) hurts more than it
helps. The `feasibility_sampler.py` mirrors that by exposing
`feasibility_warmup_steps` (default = 25 of 40) so the gradient only fires
in the late, mostly-clean steps.

### 4.3 Hyperparameters and defaults

| Knob | Default | Range to sweep | Rationale |
|---|---|---|---|
| `λ_SA` | 0.5 | {0.0, 0.3, 0.5, 1.0} | small — we want a soft pull, not an override |
| `λ_SC` | 0.3 | {0.0, 0.2, 0.3, 0.5} | smaller than λ_SA — SC has lower r² |
| `feasibility_warmup_steps` | 25 (of 40) | {0, 15, 25, 30} | match property-head behaviour; surrogate is on clean z |
| Gradient clip per step | 1.0 in latent norm | – | prevents single-step overshoot |
| Combine with property guidance | yes (additive) | – | both gradients sum into the ε correction |

### 4.4 CLI

```bash
python scripts/diffusion/feasibility_sampler.py \
    --exp experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z \
    --cfg 7 --n_pool 400 --n_keep 40 \
    --lambda_sa 0.5 --lambda_sc 0.3 \
    --feasibility_warmup_steps 25 \
    --require_neutral
```

(Falls back to vanilla rerank when `λ_SA = λ_SC = 0`.)

### 4.5 Falls-back gracefully

If either checkpoint is missing or fails to load, the sampler logs a
warning and proceeds without that gradient term — production stays
working.

---

## 5. Reranking: composite with SA + SC penalty

Both `rerank_sweep.py` and `rerank_multi.py` get a new flag:

```bash
--with_feasibility \
--w_sa 1.0  --w_sc 0.5
```

The composite score becomes:

```
composite =
    Σ_p w_p · |pred_p − target_p| / std_p           (existing)
  + w_sa · max(0, SA(smi) − SA_threshold)           (new)
  + w_sc · max(0, SC(smi) − SC_threshold)           (new)
```

Hard cap (drop, don't penalise) above an absolute ceiling:
- `SA > 6.5` → drop (very hard to synthesise)
- `SC > 4.5` → drop (very high complexity)

These thresholds match the literature defaults from Ertl & Schuffenhauer
and the SCScore paper.

### 5.1 SA / SC computation

Use the **real** scorers (not the latent surrogate) on the canonical
SMILES. Already imported in earlier scripts:

```python
from sascorer import calculateScore as sa_score
from scscore.standalone_model_numpy import SCScorer
```

Cost: ~1 ms per molecule for SA, ~3 ms per molecule for SC. On a
pool of 1500: ~1.5 s SA + ~4.5 s SC = trivial.

### 5.2 Output additions

The markdown report gains two columns:

```
| rank | composite | density | HOF | D | P | SA | SC | SMILES |
```

`candidate_matches.md` likewise gains SA / SC columns so that
final-report readers can sort/filter by feasibility.

---

## 6. Evaluation — feasibility coverage diagnostic

New diagnostic D16:

| | What it measures |
|---|---|
| **D16** | SA / SC distribution of top-N candidates vs Tier-A/B reference |
| Pass | top-40 mean SA ≤ Tier-A/B mean SA + 0.3; SC ≤ Tier-A/B mean SC + 0.2 |
| Fail | top candidates are systematically harder to make than the training distribution |

Failure mode triggers raising λ_SA / λ_SC or tightening the rerank
thresholds.

---

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Latent SA / SC surrogate has only r² ≈ 0.7, so its gradient is noisy | medium | small λ defaults (0.3 / 0.5); warmup so it only fires when z_t is near-clean |
| Pulling toward "feasible" trades off against property targets | high | default λ deliberately small; sweep first to find sweet spot |
| Real SA / SC computation slows reranks | low | ~5 s on 1500 molecules; negligible vs 3DCNN cost |
| Threshold cutoffs too aggressive → empty candidate set | medium | hard caps configurable; default tuned against existing v6 candidate distribution |

---

## 8. Plan to validate (when run, later)

1. Sweep `λ_SA × λ_SC` ∈ {(0,0), (0.3,0.3), (0.5,0.3), (1.0,0.5)} at
   pool=400 multi-property rerank, q90, all 4 properties.
2. For each, report:
   - top-40 mean composite (lower better)
   - top-40 mean SA / SC
   - top-40 mean per-property rel_MAE
   - candidate uniqueness (n_unique top-40)
3. Pick the λ pair that doesn't regress per-property metrics by > 5 % AND
   reduces top-40 mean SA by ≥ 0.3.

Acceptance: if no λ pair improves SA without regressing properties, the
guidance is dropped from production and the rerank-only flag retained as
post-hoc filter.

---

## 9. Production switchover

After validation passes:
- `rerank_sweep.py --with_feasibility` becomes default in
  `scripts/diffusion/rerank_sweep.py` invocation.
- `rerank_multi.py --with_feasibility` likewise.
- `feasibility_sampler.py` exposed as v6.5 — *optional* upgrade above v6
  rerank. Not enabled by default until sweep confirms gain.
- Production overview ([`docs/production_overview.md`](production_overview.md)) updated.

---

## 10. Files to add / modify

| File | Action |
|---|---|
| `scripts/diffusion/feasibility_sampler.py` | **new** — DDIM with SA + SC gradient guidance |
| `scripts/diffusion/feasibility_utils.py` | **new** — load surrogate ckpts; wrap real SA/SC scorers |
| `scripts/diffusion/rerank_sweep.py` | **modify** — add `--with_feasibility`, `--w_sa`, `--w_sc`, optional gradient route |
| `scripts/diffusion/rerank_multi.py` | **modify** — same |
| `scripts/diffusion/match_candidates.py` | **modify** — add SA / SC columns to the lookup report |
| `scripts/diagnostics/d16_feasibility_distribution.py` | **new** — D16 feasibility coverage |
| `docs/production_overview.md` | **update** after validation |

No code is *executed* in this plan — implementation is committed but
disabled-by-default until the validation sweep is run.
