# Root-cause diagnostics for the q90 ceiling

We've trained 5 versions; q90 plateau persists. Before chasing more
hyperparameters, run targeted diagnostics that *isolate where the failure is*.
Each diagnostic answers a binary question — pass / fail — and points to a
specific intervention.

---

## D1. Validator self-consistency  (cost: 5 min, **must run first**)

**Question**: Is the 3DCNN smoke validator itself the noise floor of our
metric? If it can't predict known properties on Tier-A/B SMILES, no model
will look good.

**How**: take all Tier-A/B rows with experimentally measured D, P, density.
Run them through `UniMolValidator`. Compute per-property MAE between
prediction and ground truth.

**Pass**: validator MAE ≤ 10 % on density; ≤ 15 % on D, P, HOF.
**Fail**: any property MAE >25 % → our q90 metrics partly measure validator
noise. Replace metric with: 3DCNN-smoke + K-J consensus (higher confidence
when both agree).

---

## D2. Encoder–decoder roundtrip on high-property molecules  (cost: 10 min)

**Question**: Can LIMO even reproduce the chemistry that lives at q90?
If high-D molecules don't survive the encode→decode roundtrip, the
diffusion denoiser is being asked to generate things its decoder can't
output.

**How**: pick 50 known high-D Tier-A molecules (D > 7 km/s). Encode →
decode (no diffusion) → canonicalise. Measure:
- Roundtrip recovery rate (canonical SMILES exact match)
- Mean Tanimoto similarity (if not exact)
- Of the round-tripped SMILES, what fraction still carry an energetic
  SMARTS?

**Pass**: ≥ 60 % exact recovery, ≥ 0.85 mean Tanimoto.
**Fail**: < 30 % recovery → bottleneck is LIMO. Fix: more LIMO fine-tune
steps (currently only 8.5k) or switch to a transformer SMILES VAE.

---

## D3. Property predictability from latents  (cost: 30 min)

**Question**: Does LIMO's latent space *encode* the properties at all?
If a small predictor `f(z) → property` doesn't work on Tier-A/B latents,
the conditioning signal can never be recovered no matter how cleverly the
denoiser is conditioned.

**How**: train a 3-layer MLP `1024 → 512 → 1` per property on Tier-A/B
latents (already partially done for SA + SC). 80/20 split. Report Pearson
r and MAE.

**Pass**: r ≥ 0.85 for density / D / P; r ≥ 0.7 for HOF.
**Fail**: r < 0.5 → latent space doesn't encode the property → re-train
LIMO with auxiliary property heads (item F2 in `improvements_deep_think`).

---

## D4. Conditional generation at training-distribution targets  (cost: 30 min)

**Question**: Given a Tier-A/B row's *real* property values, does the
denoiser regenerate something close to that row's actual latent?

**How**: pick 100 Tier-A/B rows. For each:
1. Use their actual property values as conditioning targets.
2. Generate 5 samples each.
3. Compute distance from generated z to original z.
4. Decode generated samples; how often is the original SMILES recovered?

**Pass**: median distance < 1.0; ≥ 5 % of generated samples reproduce
original SMILES exactly.
**Fail**: median distance > 2.0 → the denoiser is ignoring the
conditioning entirely; the failure is in the FiLM / mask logic, not data.

---

## D5. Out-of-range conditioning  (cost: 5 min)

**Question**: Does the model saturate at q90 because the data ends there,
or does it just *cap* its output regardless of how high you ask for?

**How**: at sample time, request `target_z = +3.0` (= ~3-σ above mean,
beyond q99.9). Compare generated property mean & spread to q90 result.

**Pass**: q99 generations are noticeably higher than q90 (model extrapolates).
**Fail**: q99 ≈ q90 → model has saturated; only fix is data acquisition or
classifier guidance with strong gradient pushing past the support.

---

## D6. Latent-space coverage of high-property region  (cost: 1 h)

**Question**: Where do high-D / high-P / high-HOF molecules live in latent
space? Are they isolated points the denoiser can't reach, or scattered
across the manifold?

**How**: PCA all 382k z_mu to 2D. Color by property z-score (per property).
Run also a UMAP for non-linear structure. Generate uncond samples from
v4-nf, project to same PCA, overlay.

**Pass**: high-property region is a contiguous cluster, and uncond samples
cover at least its outer envelope.
**Fail**: high-property points are scattered or far from where uncond
samples land → diffusion prior never visits the high-property region.
Fix: classifier guidance, latent gradient ascent, or active-learning data.

---

## D7. Validator self-disagreement = uncertainty proxy  (cost: 30 min)

**Question**: For each generated sample, do `model_0.pth` and `model_1.pth`
agree? Wide disagreement = uncertain; we should rerank by certainty, not
just by point estimate.

**How**: modify cfg_sweep to record both fold predictions. Compute
disagreement σ_models per row.

**Pass**: disagreement σ correlates with prediction error (high σ →
prediction is unreliable).
**Fail**: σ uncorrelated with error → ensemble doesn't help reranking.

---

## D8. Tier-D label noise quantification  (cost: 15 min)

**Question**: How noisy is the 3DCNN smoke prediction *as a label*?
~2,500 rows have BOTH a Tier-A/B value and a 3DCNN prediction — quantify
that noise directly.

**How**: take rows where both Tier-A/B and 3DCNN predictions exist for the
same property. Compute MAE between them. This is the upper bound on how
well our denoiser can learn from Tier-D conditioning.

**Pass**: MAE ≤ 20 % per property → label noise tolerable.
**Fail**: MAE > 50 % → most of the 360k Tier-D rows are actively misleading.
Re-weight Tier-D in the loss to ~0 (effectively v4-B), or hard-mask them.

---

## D9. Sample-quality vs DDIM step count  (cost: 30 min)

**Question**: Are 40 DDIM steps enough at high CFG?

**How**: re-run cfg_sweep at g=7 with [20, 40, 80, 160] DDIM steps on v4-nf.
Plot mean predicted q90 value per step count.

**Pass**: 40 steps already converged.
**Fail**: 80–160 steps reach higher q90 values → cheap inference-time
improvement available; switch to DPM-Solver++ for fewer steps at same
quality.

---

## D10. Conditioning signal correlation  (cost: 15 min)

**Question**: Does the model's eps prediction actually depend on the
conditioning input, or is it ignoring c?

**How**: encode same z_t at same t. Run the model with
- conditioning A (q90 of D)
- conditioning B (q10 of D)
- conditioning all-zero (uncond)
Measure mean cosine distance between (A − uncond) and (B − uncond).

**Pass**: large negative cosine (eps shifts in opposite directions for
opposite targets).
**Fail**: cosine ≈ 0 → conditioning is being ignored at high noise levels;
need a different conditioning architecture (B1/B2).

---

## Suggested execution order (time budget)

| Order | Diagnostic | Why first | Time |
|---|---|---|---|
| 1 | D1 | sets the ceiling on what any model can score | 5 min |
| 2 | D8 | tells us whether Tier-D should even be used | 15 min |
| 3 | D5 | one config flag away from running; high info | 5 min |
| 4 | D2 | LIMO is the foundation; if broken nothing else helps | 10 min |
| 5 | D3 | if latent doesn't encode property, conditioning is hopeless | 30 min |
| 6 | D10 | quick check of FiLM signal | 15 min |
| 7 | D6 | answers the "is the high tail reachable" question | 1 h |
| 8 | D4 | confirms whether the model can use conditioning at training points | 30 min |
| 9 | D9 | only worth running if sampling is the bottleneck | 30 min |
| 10 | D7 | nice-to-have for production reranking | 30 min |

**Total**: ~4 hours, mostly CPU.

---

## What we'll learn (decision tree)

```
D1 fails  → metric is broken; need a better validator
D2 fails  → bottleneck is LIMO; fix it before anything else
D3 fails  → latent geometry doesn't encode properties; switch to property-aware LIMO
D5 saturates → no extrapolation possible; data acquisition or classifier guidance
D6 isolated → high tail unreachable; latent gradient ascent or active learning
D8 high noise → Tier-D is poison; switch to v4-B style
D10 weak  → conditioning architecture needs upgrade; B1 (DiT) or B2 (cross-attn)
```

If D1 / D2 / D3 / D5 all pass cleanly → the model & data are fine and the
problem is purely sampling-time. Then G1 (classifier guidance) + G2
(rerank) become the right next step.

If D2 or D3 fails → invest in LIMO before touching the denoiser again.

If D5 / D6 fails → this is a data problem; no amount of architecture or
training tweaks will help. Active learning loop becomes the only path.
