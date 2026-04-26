# Denoiser Training Strategy (post-3DCNN expansion)

Conservative plan for re-training the subset-conditional diffusion denoiser on
the expanded 382k-row latent set, given that conditioning labels span four
quality tiers (A experimental, B physics-sim, C Kamlet-Jacobs, D ML/3DCNN
ensemble).

## Core principles
1. **Don't let Tier-D drown out Tier-A/B/C.** Smoke-model predictions are
   plentiful but noisy; weight them down per property.
2. **Mask-aware loss.** Only condition on properties that are actually present
   for that row; unmasked properties contribute nothing to the FiLM signal.
3. **Two-stage curriculum.** Pretrain unconditional, then fine-tune
   conditional, so the prior over chemistry isn't corrupted by label noise.
4. **Validate against held-out experimental rows only.** Never let Tier-D
   appear in the eval set.

## Per-property conditioning weight table

`cond_weight[i, p]` enters the loss as a multiplier on the FiLM-conditioned
score-matching term for property `p` of row `i`. Values below are *defaults*;
override per row when a higher-tier source is available.

| Property | Tier A (exp.) | Tier B (DFT/sim) | Tier C (K-J) | Tier D (3DCNN smoke) | Notes |
|---|---|---|---|---|---|
| **density (ρ)**   | 1.0 | 0.9  | 0.6 | 0.5 | EMDP density is Girolami-empirical, not DFT — treat as Tier-C even when shipped in 3DCNN.csv |
| **HOF**           | 1.0 | 1.0  | 0.4 | 0.5 | Tier-B = B3LYP atomization; K-J doesn't predict HOF, so Tier-C N/A in practice |
| **D (det. vel.)** | 1.0 | 0.95 | 0.7 | 0.5 | K-J D is well-calibrated when ρ + HOF are good; smoke-model D is the weakest target |
| **P (det. press.)** | 1.0 | 0.95 | 0.7 | 0.4 | Same as D, but smoke-model P is noisier |
| **mask weight (overall)** | 1.0 | 1.0 | 1.0 | 0.7 | Multiply all four columns when entire row is Tier-D-only |

Implementation: in `expand_conditioning.py`, write `cond_weight: (N, 4)`
alongside `cond` and `cond_mask`. In `train.py`, the per-property loss term
becomes `w_ip * mask_ip * mse(eps_pred, eps_true | film(c_ip))`.

## Training schedule

| Stage | Epochs | Data | Loss | CFG drop-prob | LR | Purpose |
|---|---|---|---|---|---|---|
| 1. Unconditional pretrain | 30 | all 382k, mask=0 | plain MSE | 1.0 | 2e-4 | Learn the latent prior over molecule space; no label noise can leak in |
| 2. Mixed conditional | 60 | all 382k, real masks | weighted MSE (table above) | 0.15 | 1e-4 | Main run; high-quality rows dominate the conditional signal |
| 3. High-tier fine-tune | 10 | only rows with ≥1 Tier-A/B label (~28k) | weighted MSE, weights ×1.0 | 0.10 | 3e-5 | Sharpen on clean labels without forgetting prior |

EMA (0.9999) on throughout. Save every 2 epochs; keep best-3 by held-out
NLL on Tier-A rows.

## Anti-overfitting guards

| Guard | Setting | Why |
|---|---|---|
| Weight decay | 1e-3 on FiLM, 0 on backbone | FiLM is the small parameter set most prone to memorising labels |
| Cond-dropout (CFG) | 0.15 in stage 2 | Forces denoiser to keep working when labels are absent at sample time |
| Property-dropout | 0.30 per property indep. | Even when row has all 4 labels, randomly hide some so model handles arbitrary subsets |
| Label noise injection | σ=0.05·std for Tier-D, 0 for A/B | Smooths the mismatch between smoke-model labels and ground truth |
| Grad clip | 1.0 | Standard |
| fp16 + loss scaling | dynamic | RTX-2060 budget |

## Validation protocol

| Check | Source | Frequency | Pass threshold |
|---|---|---|---|
| Recon NLL on held-out Tier-A | 200-row stratified split | every 2 epochs | non-increasing for 6 epochs ⇒ early-stop |
| Conditional sample fidelity | 3DCNN ensemble re-predicts ρ/D/P/HOF on 80 generated samples per condition bin | every 5 epochs | mean abs rel-err on ρ < 6%, D < 8% |
| Novelty (NN-Tanimoto vs train) | 300 unconditional samples | every 5 epochs | median < 0.55 |
| SA + SC distribution | latent-space scorers | every 5 epochs | median SA ≤ train+0.3, SC ≤ train+0.2 |
| Top-K candidate audit | Psi4 B3LYP HOF on top-20 high-D candidates | end-of-run only | absolute HOF error < 30 kcal/mol vs predicted |

## Don'ts

- Don't drop Tier-D from training. Removing 90% of the data hurts the prior more than the label noise hurts the conditioner. Down-weight, don't delete.
- Don't share FiLM heads across properties. Per-property heads let the model express asymmetric label trust.
- Don't evaluate the denoiser with the same 3DCNN smoke model that produced its training labels. Use Tier-A/B held-out + Psi4 spot-checks.
- Don't resume stage-2 from a stage-3 checkpoint. Stage-3 narrows the distribution; stage-2 must restart from stage-2's own best.
