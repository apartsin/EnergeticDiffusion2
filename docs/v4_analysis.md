# Pre-v4 Deep Analysis

What v1–v3 have taught us, what's actually broken, and the ranked
improvements to try in v4. Written before v3 sweep is complete; will be
revised once v3 numbers land.

---

## 1. What we actually know

### 1.1 Quantitative recap (best CFG per cell)

| Property × q | v1 (g=7) | v2 (best g) | v3 (pending) |
|---|---|---|---|
| density q90 rel-MAE | 16 % | 13 % | ? |
| HOF q50 rel-MAE | 169 % | **272 %** | ? |
| HOF q90 rel-MAE | 127 % | 162 % | ? |
| det. velocity q90 | 30 % | 22 % | ? |
| det. pressure q90 | 51 % | 48 % | ? |
| best val_loss | 0.0515 | 0.0480 | 0.0482 (so far) |

### 1.2 Symptoms

| # | Symptom | v1 | v2 | Likely cause |
|---|---|---|---|---|
| S1 | q50 generations drift toward inert chemistry | mild | severe | Symmetric oversampling pulled mass toward 382k low tail |
| S2 | q90 generations cannot reach high targets | yes | yes | Training data has very few rows with high D / P / HOF |
| S3 | Unconditional samples include long alkane chains | yes | yes | Latent prior is too broad; LIMO was fine-tuned on energetics but the 382k pool isn't all energetic |
| S4 | CFG saturates above g≈5 | yes | yes | Pure CFG cannot manufacture distribution support that doesn't exist |
| S5 | Bias near dataset mean regardless of target | yes | partial | Conditioning signal weak; generator defaults to mean when uncertain |

### 1.3 Things v3 *will* fix

- S1 (asymmetric oversampling restricted to Tier-A/B high-tail).
- *Maybe* part of S2, since high-end Tier-A/B is now boosted.

### 1.4 Things v3 will *not* fix

- S3 (prior is upstream of any oversampling change)
- S4 (CFG ceiling is structural, not a hyperparameter)
- The fundamental data scarcity: only **2,366 rows** have Tier-A/B
  detonation pressure values in 382,604 — even 5× boost is 11k effective
  exposures per epoch, which is not a lot to learn the high-P manifold from.

---

## 2. Root-cause analysis

### 2.1 Distribution coverage is the bottleneck, not capacity

44.6 M params on a 1024-d latent with a 382k dataset is *over-parameterised*.
Val loss has plateaued around 0.048 across v1/v2/v3 — that's not a learning
problem, it's an information-content limit. The model has memorised what's
present and cannot extrapolate to what isn't.

Concrete evidence:

- Generated max D ≈ 5.5 km/s across all CFG scales and all versions.
- Generated max HOF ≈ −65 kcal/mol; we never see HOF > 0.
- Tier-A/B contains ~40 rows with HOF > +200 kcal/mol — that's the entire
  high-energy tail the model has to extrapolate from. Five copies via
  oversampling is still 200 exposures per epoch out of 363k. That's
  ~0.05% sampling probability — easily drowned out.

### 2.2 The prior is contaminated

LIMO fine-tuning ran 8,500 steps on energetic SMILES, then we encoded all
382,604. **But not all 382k are energetics** — the 3DCNN-derived rows include
generic chemistry. Inert molecules sit close to the energetic manifold in
latent space (LIMO didn't have time to push them apart in 8.5k steps), and
the diffusion prior averages over both populations.

Sample S3 evidence: uncond samples include `CCCCCCCCCCC...` strings — these
aren't in the energetic curated set; they're in the broader pool that LIMO
saw during fine-tune extension to all 382k.

### 2.3 Conditioning is weakly grounded

Property normalisation uses Tier-A+B stats but is applied to all 382k. So
when the 3DCNN smoke model predicts HOF = −300 kcal/mol for an inert ZINC
molecule, that gets z-scored against the *energetic* mean (-97) and std
(284), producing z ≈ −0.71. The conditioning signal "z = −0.71" then becomes
an inert-molecule indicator rather than "low-energy energetic." The denoiser
is correctly learning the joint distribution (z_lat, c) it sees — but that
joint is wrong for our purpose.

### 2.4 CFG cannot manufacture support

Classifier-free guidance amplifies the conditional score over the
unconditional score. If the conditional probability density is zero in some
region (because no training point lives there), no CFG scale can put mass
there. Bumping g just makes the model more confident about its existing
ceiling.

---

## 3. v4 candidate improvements

Each row gives expected impact (H/M/L), implementation cost, and what it
mitigates.

### 3.1 Data-side (highest expected impact)

| # | Change | Mitigates | Cost | Expected Δ |
|---|---|---|---|---|
| D1 | **Energetic-motif filter on training set**: keep rows with ≥1 SMARTS in {`[N+](=O)[O-]`, `N=N=N`, `O[N+](=O)[O-]`, furazan/tetrazole/triazine ring, polynitro CHNO} OR Tanimoto ≥ 0.3 to a 100-mol energetic seed list. Re-encode latents on filtered set (~80–120k rows). | S3, partial S2 | 30 min | **H** — fixes the prior |
| D2 | **Tier-aware z-score normalisation**: compute per-property mean/std on the *whole* 382k pop, OR per-tier separately, to avoid the "z=−0.71 means inert" pathology described in §2.3 | S5 | 10 min | **M** |
| D3 | **Renormalise targets at sample time** to the population the model actually saw, not Tier-A+B stats | S5 | 5 min | **M** |
| D4 | **Larger curated-energetic set**: pull additional Tier-A/B rows from the LANL CHIRALMolecules db, ICT, NIST WebBook (~5–10k more high-D/high-P labels) | S2 | 1–2 days work | **H** if achieved |

### 3.2 Training-side

| # | Change | Mitigates | Cost | Expected Δ |
|---|---|---|---|---|
| T1 | **Two-stage curriculum** (per [`denoiser_training_strategy.md`](denoiser_training_strategy.md)): stage A = unconditional pretrain on filtered set (30 ep) → stage B = weighted conditional (60 ep) → stage C = high-tier fine-tune (10 ep) | S3, S5 | 2–3 h | **H** |
| T2 | **Per-property loss weighting** (not just mask weighting): weight `MSE(eps_pred, eps_true)` by `mean(cond_weight × mask)` per row so Tier-A/B rows contribute more to the gradient | S5 | 15 min code | **M** |
| T3 | **Discrete categorical conditioning**: replace continuous z with a 5-bin one-hot ("very-low / low / mid / high / very-high"). Easier to learn and to specify at sample time. | S2 (high-bin) | 1 h code | **M** |
| T4 | **Bigger oversample factor (10–20×) on TOP-5 % of Tier-A/B** — push the rare high-energy tail harder than v3's 5× on top-10 % | S2 | 5 min config | **M** |
| T5 | **EMA decay sweep** (0.999 vs 0.9999 vs 0.99): current EMA may be too tight for 53k-step runs | none of S1–S5 | small | **L** |
| T6 | **Longer training**: val loss hit floor ~0.048 in v2/v3; if data-side changes raise the ceiling, training will need 100k+ steps | S2 | 2× wall-clock | **M** |

### 3.3 Sampling-side

| # | Change | Mitigates | Cost | Expected Δ |
|---|---|---|---|---|
| Sa1 | **Classifier guidance** from a property predictor `g(z) → (D, P, HOF)` trained directly in latent space. Add `λ ∇_z g` to the score. SA + SC are already trained; need a property head. | S2, S5 | 1–2 h to train + wire | **H** |
| Sa2 | **CFG rescale (Karras et al.)** to break the high-CFG saturation: `eps_scaled = eps_null + g·(eps_cond − eps_null) · σ_data / σ_pred` | S4 | 30 min | **M** |
| Sa3 | **Generate-and-rerank**: sample 10× more candidates than needed, 3DCNN-validate, keep best by composite score. Already partially in `evaluate_candidates.py`. | S2, S5 | already exists | **H** at inference cost |
| Sa4 | **Targeted SDEdit from known high-D seeds**: instead of pure cond gen, SDEdit-perturb known-high-D molecules to explore the high-end neighbourhood | S2 | 10 min | **M** |
| Sa5 | **DDIM steps 40 → 100**: more steps may help when CFG is high (less variance) | S4 | 2.5× sample time | **L** |
| Sa6 | **Per-property guidance scale** at sample time: g_density=3, g_HOF=8, g_D=6, g_P=8 | S4 | 30 min | **L** |

---

## 4. Recommended v4 plan

**Bundle the highest-impact, lowest-cost items:**

| Step | Item | Cost |
|---|---|---|
| 1 | D1 — Energetic-motif filter, re-encode latents | 30 min |
| 2 | D2 + D3 — Renormalise against the filtered set's stats | 10 min |
| 3 | T2 — Per-row loss weighting by `cond_weight × mask` | 15 min |
| 4 | T4 — Bigger oversample factor on top-5 % Tier-A/B | 5 min |
| 5 | T1 stage C — Final 10-epoch fine-tune on Tier-A/B-only | included in train.py |
| 6 | Train v4 | 90 min |
| 7 | Sa1 — Train latent property head; wire classifier guidance into `cfg_sweep.py` | 90 min train + 30 min wire |
| 8 | Sa3 — Wrap cfg_sweep with 10× generate + rerank | 30 min |
| 9 | Final eval | 20 min |

**Total wall-clock**: ~5 h. Most cost is in the 90-min training and the
property-head training.

### Items deliberately skipped for v4

- D4 (more curated data) — too long-horizon for this sprint.
- T3 (discrete bins) — would invalidate `cfg_sweep.py`'s z-score path; revisit if v4 still misses high-q90.
- T5 (EMA sweep) — low expected gain.
- Sa5 (more DDIM steps) — only worth it if guidance is strong; defer.

### Acceptance criteria for v4

| Metric | v3 expected | v4 target |
|---|---|---|
| density q90 within-10 % | ~15 % | **≥ 30 %** |
| det. velocity q90 within-10 % | ~10 % | **≥ 25 %** |
| det. pressure q90 within-10 % | ~10 % | **≥ 20 %** |
| HOF q50 rel-MAE | < v2's 272 % | **< 100 %** |
| Unconditional samples with ≥1 energetic SMARTS | currently ~25 % | **≥ 80 %** |

If any q90 within-10 % stays below v3 levels after v4 → that means the
data-side fix wasn't enough, and **D4 (data acquisition) becomes the only
remaining lever**.

---

## 5. Things to revisit, not in v4

- **LIMO architecture**: SELFIES-MLP is shallow; a transformer SMILES VAE
  (Moses-Char-RNN or BART-MolFormer) would give a more structured latent
  but requires re-pretraining, re-encoding, and discarding all v1–v3
  checkpoints. Defer until v5 if needed.
- **Latent geometry diagnostics**: PCA / UMAP of Tier-A/B vs broader pool.
  Cheap to run; will validate the "prior contamination" hypothesis.
- **Active learning loop**: train predictor → sample → DFT-validate top-K →
  add to training set. Already partially set up via `psi4_hof.py`, but the
  loop is not closed. Closing it would be the long-term path past v4.
