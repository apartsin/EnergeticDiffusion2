# Deep-think: ways to push the denoiser past the q90 ceiling

A broader survey of approaches than the v4 plan, organized by *what kind
of lever they pull*. Items already exhausted (CFG sweep, motif filter,
oversampling, mask weighting) are noted but not re-listed.

The hard ceilings to break:
- D q90: model max ≈ 5.7 km/s, target 7.86
- P q90: model max ≈ 14 GPa, target 26.6
- HOF q90: model max ≈ −30 kcal/mol, target +268
- Unconditional samples drift to long alkanes (~ ¾ are non-energetic)

---

## A. Diffusion-process changes (cheap, high impact)

These are 1–10 line changes to `train.py` / `model.py` that often unlock
real gains without dataset work.

| # | Change | Cost | Expected impact | Why it might help |
|---|---|---|---|---|
| A1 | **Min-SNR loss weighting** (Hang et al. 2023): `loss *= min(SNR(t), 5)/SNR(t)` | 5 min | **H** | Stops the loss from being dominated by easy noise levels; reliably +5–15 % across diffusion benchmarks |
| A2 | **v-prediction reparameterization**: predict `v = α·ε − σ·z₀` instead of `ε` | 30 min | **M** | Numerically stabler at high SNR; better at sharp conditional targets |
| A3 | **EDM σ-noise schedule** (Karras et al. 2022) + **DPM-Solver++** sampler | 2 h | **H** | Replaces linear β with continuous σ; needs ~half the steps for same quality, more stable at high CFG |
| A4 | **Stochastic SDE sampling instead of DDIM** | 30 min | **M** | DDIM (deterministic) tends to mode-collapse at high CFG; SDE adds noise that lets samples explore the high-property tail |
| A5 | **CFG rescale (Lin et al. 2024)**: cap output norm to avoid CFG-induced over-saturation | 30 min | **M** | Frees us to push CFG to 10–15 without artifacts |
| A6 | **Heun's 2nd-order corrector** in DDIM | 30 min | **L–M** | Cheap accuracy improvement; complementary to A3 |

**Bundle suggestion**: A1 + A4 in one config flag. Costs 1 hour, often gives v3-equivalent gains for free.

---

## B. Architecture changes (medium cost)

The current FiLM ResNet on 1024-d treats the latent as a flat vector. There
is structure in LIMO's latent that we're throwing away.

| # | Change | Cost | Expected impact | Notes |
|---|---|---|---|---|
| B1 | **DiT-style** (Peebles & Xie 2023): chunk z into 64×16 tokens, transformer with AdaLN-Zero conditioning | 1 day | **H** | Standard for latent diffusion; sample quality usually +20 % |
| B2 | **Cross-attention conditioning** instead of FiLM: each property is a token, latent attends over them | 4 h | **M** | Lets model attend more to "the property that's masked in" |
| B3 | **U-Net-on-1024d**: down to 256→64→256→1024, multi-scale skips | 4 h | **M–L** | Multi-scale denoising; may overfit on small data |
| B4 | **Property-token concatenation**: append [SMARTS-count features, oxygen balance, atom counts] as an extra conditioning slot | 1 h | **M** | Deterministic from SMILES; gives the model anchors that aren't model-predicted |

---

## C. Conditioning re-design

Current: continuous z-score per property + binary mask. Some failure modes
trace directly to this choice.

| # | Change | Cost | Expected impact | Notes |
|---|---|---|---|---|
| C1 | **Discrete bins** (very-low / low / mid / high / very-high) instead of continuous z | 2 h | **M** | Easier loss surface; q90 generations stop drifting toward mean. Caveat: invalidates current eval |
| C2 | **Conditioning augmentation**: jitter `values[mask]` by ε~N(0, 0.1·std) at train time | 15 min | **L–M** | Robustness to off-center sample-time targets |
| C3 | **Multi-property co-conditioning emphasis**: bias `subset_size_probs` toward 3–4 simultaneous properties | 5 min | **M** | The "high D + high P + high HOF" combination is exactly what we want; current schedule samples it only 35 % of the time |
| C4 | **Auxiliary multi-task head**: predict properties from `z_t` *during training*, side-loss; shapes the latent geometry | 1 day | **H** | Gradients from property-prediction sharpen the conditioning manifold; complementary to classifier guidance at sample time |

---

## D. Training-strategy / objective changes

| # | Change | Cost | Expected impact | Notes |
|---|---|---|---|---|
| D1 | **RL / reward fine-tune** (DDPO, DRaFT-style) using 3DCNN as the reward model | 1–2 days | **H** | Direct optimisation toward the metric we evaluate on; can post-train v3 weights without re-running pretraining |
| D2 | **Self-consistency distillation**: sample n times, validate, fine-tune on validated samples (chain-of-validation) | 3–4 h | **M** | Cheap form of the RL approach; fewer hyperparams |
| D3 | **Two-stage curriculum (T1 from v4_analysis)**: uncond pretrain → conditional → tier-A/B fine-tune | already planned | **M** | v4 didn't include this; v4-B could |
| D4 | **Adversarial regulariser**: small discriminator distinguishing real (z, c) from generated; gradients flow back into denoiser | 1 day | **M** | Pushes generations onto the real (z,c) manifold; tricky to stabilise |
| D5 | **Per-property weighted loss**: weight latent MSE by which property is being conditioned, not just by `cond_weight` | 1 h | **L–M** | Currently MSE is property-agnostic; could give D/P 2× weight given they're the bottleneck |

---

## E. Data leverage (without acquiring new labels)

| # | Change | Cost | Expected impact | Notes |
|---|---|---|---|---|
| E1 | **Pseudo-label via 3DCNN ensemble + K-J consensus**: keep only rows where the two label sources agree within X %; treat as semi-trusted (cond_weight 0.85) | 4 h | **M** | Cleaner Tier-D signal; small gain on q50 |
| E2 | **Latent-space interpolation augmentation**: synthesise training pairs by interpolating between high-D Tier-A/B latents and labelling with interpolated properties | 1 day | **M** | More density in the high-property region of latent space; may produce off-manifold artifacts |
| E3 | **3DCNN-uncertainty-aware weighting**: rerun 3DCNN with both `model_0` and `model_1` separately; rows where the two disagree get downweighted | 2 h + GPU | **M** | Currently we use the average; the disagreement *is* the uncertainty signal |
| E4 | **Filter Tier-D rows by 3DCNN confidence**: reject rows where the 3DCNN ensemble disagrees beyond threshold; reduces noise without acquiring new data | 1 h | **M** | Same data, less noise |

---

## F. LIMO-side changes (re-encoding latents costs ~30 min, all denoiser checkpoints become invalid)

These reset the v1–v4 lineage but may be worth it.

| # | Change | Cost | Expected impact | Notes |
|---|---|---|---|---|
| F1 | **More LIMO fine-tune steps** (current 8.5k → 30k+) | 6–8 h | **M** | The LIMO latent geometry isn't fully aligned to energetic chemistry yet; another 3× of fine-tune compute may tighten it |
| F2 | **Property-aware LIMO**: add property prediction head during LIMO fine-tune; latent gets organised by D/P/HOF | 1 day | **H** | Property gradient flows through encoder → latent space becomes structured around what we care about. Diffusion conditioning then has an easier job |
| F3 | **Switch base VAE** to MolFormer-XL or BART-Smiles | 2–3 days | **H** | LIMO's MLP encoder is shallow; transformer SMILES VAEs give richer latents. Big change |

---

## G. Sampling-side post-hoc (no retraining)

| # | Change | Cost | Expected impact | Notes |
|---|---|---|---|---|
| G1 | **Classifier guidance** from a small property predictor `g(z) → (D, P, HOF, ρ)` trained on Tier-A/B latents only. Add `λ ∇_z g(z)` to the diffusion score | 4 h | **H** | The single most-likely-to-help post-hoc lever; SA + SC predictors are already trained, can replicate the recipe |
| G2 | **Generate-and-rerank**: sample 10–20× more candidates per target, validate with 3DCNN, keep top by composite score | 30 min wrap | **H** | Already planned; trivially additive |
| G3 | **Per-property guidance scale + per-quantile schedule**: g_density=3, g_HOF=8 is plausible; can also schedule g across DDIM steps | 1 h | **M** | Current single-g is suboptimal |
| G4 | **SDEdit from high-D seeds + classifier-guided refinement**: take known-high-D molecules, perturb partially, denoise with high CFG and classifier guidance; finds neighbours of the high-property region | 2 h | **H** | We already have SDEdit in evaluate.py; just need to combine with G1 |
| G5 | **Latent-space gradient ascent under prior constraint**: directly optimise `z*` to maximise predicted properties while staying within the diffusion prior support (`||z* − z_prior||` constraint). Like guided protein design | 1 day | **H** | Departs from generative sampling but very effective when extremes are scarce |
| G6 | **Ensemble denoising**: average ε predictions from v1+v3+v5 at each DDIM step | 30 min | **L–M** | Cheap diversity boost; small gain |

---

## H. Strategic alternatives (bigger structural changes)

| # | Approach | Why consider | Cost |
|---|---|---|---|
| H1 | **GFlowNet in latent space** | Naturally reward-conditioned; explores high-reward regions by construction; better at extremes than diffusion in many domains | 1–2 weeks |
| H2 | **Active-learning loop**: generate → DFT-validate top-K → add to Tier-B → retrain → repeat | The only known way to break the data-coverage ceiling for q90 truly | 2–4 weeks compute, but offline |
| H3 | **Switch to property-guided MCMC** in latent space | Forget the generative prior, just optimise directly under the 3DCNN+SA+SC composite reward, constrain to LIMO support | 3–4 days |
| H4 | **Hybrid: diffusion prior + latent gradient ascent at sample time** | Combines generative diversity with explicit optimisation toward extremes — the best of both | 1 week |

---

## Recommended next moves (priority-ordered)

If we have **1 day**:
1. A1 (min-SNR) + A4 (SDE sampling) bundled into v5 config — train + sweep
2. G1 (classifier guidance via latent property head) — train head on v3 latents, wire into cfg_sweep
3. G2 (rerank wrapper) — already trivial
4. Re-evaluate v3 with classifier guidance; expected to beat v3-vanilla

If we have **3 days**:
- All of above
- B1 (DiT-style transformer denoiser) — biggest architecture upgrade
- D1 or D2 (RL / self-consistency fine-tune of v3)
- F2 (property-aware LIMO retrain) — restart of latent lineage

If we have **1 week**:
- All of above
- H4 (hybrid: diffusion + latent gradient ascent at sample time)
- Begin H2 (active learning) by DFT-validating top-50 v3 candidates

If we have **2+ weeks**:
- F3 (switch base VAE) and start fresh
- Full H2 active-learning loop with weekly retrain cycles

---

## What I would NOT do next

- Another oversampling tweak. v2/v3/v4 covered the search space.
- Another SMARTS filter variant. v4 showed filtering hurts the prior.
- Bigger model. We already overfit a 44.6 M param model on this data.
- More EMA decay sweeps. Marginal.
- More CFG sweeps without changing the model. Saturated.
- Continuing to add features without measuring the *contamination* effect — every addition compounds the chance of v4-style regressions.

---

## A single experiment that would clarify a lot

Train v5 = v3 recipe + **A1 (min-SNR)** + **A4 (SDE sampling)** + **C3
(co-conditioning emphasis on 3–4 props)** only. Three minimal changes, all
1-line. If it beats v3 → bundle of A+C is the right next direction.
If it doesn't → the model side is exhausted and the next move must be
G1+G2 (sampling-side classifier guidance) or H2/H4 (data + optimisation).
