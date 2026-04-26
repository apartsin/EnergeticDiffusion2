# Denoiser v3 — Training Protocol

Subset-conditional latent diffusion for **energetic-material generation**.
v3 is the current best practice; v1/v2 are deprecated experiments retained
only for ablation comparison.

Config: [`configs/diffusion_expanded_v3.yaml`](../configs/diffusion_expanded_v3.yaml)
Code:   [`scripts/diffusion/train.py`](../scripts/diffusion/train.py)

---

## 1. Data selection

### Source
- LIMO VAE latents `z_mu` ∈ ℝ¹⁰²⁴, encoded from 382,604 SMILES.
- Conditioning targets: `[density, HOF, det. velocity, det. pressure]`.

### Tier provenance per property

| Tier | Source | Weight | Typical count per property |
|---|---|---|---|
| A | Experimental (literature)              | 1.0 | ~10–15% of curated set |
| B | DFT (B3LYP/6-31G(d) atomization, etc.)  | 1.0 | rest of curated set |
| C | Kamlet-Jacobs (D, P only)              | 0.7 | currently unused (3DCNN preempts) |
| D | 3DCNN smoke-model (Uni-Mol v1, 2-fold ensemble) | 0.7 | bulk of 382k |

Curated (A+B) row counts per property: density 19,176; HOF 2,625;
det. velocity 2,539; det. pressure 2,366.

### Stored arrays in `latents_expanded.pt`

| Key | Shape | Notes |
|---|---|---|
| `z_mu`            | (N, 1024) float32 | LIMO encoder output |
| `values_norm`     | (N, 4) float32    | z-scored against Tier-A+B stats |
| `values_raw`      | (N, 4) float32    | original physical units |
| `cond_valid`      | (N, 4) bool       | True if any source provided a value |
| `cond_weight`     | (N, 4) float32    | 1.0 (A/B) or 0.7 (D); 0.0 if invalid |
| `tiers`           | (N, 4) int8       | 0=A 1=B 2=C 3=D |
| `stats`           | dict              | per-property `{mean, std, count}` from Tier-A+B only |

---

## 2. Architecture

`ConditionalDenoiser` (FiLM-conditioned ResNet on the 1024-d latent):

| Hyperparameter | Value |
|---|---|
| latent dim | 1024 |
| hidden | 2048 |
| n_blocks | 8 (FiLM-modulated ResNet) |
| time embedding dim | 256 |
| per-property embed dim | 64 |
| total parameters | 44.6 M |
| dropout | 0.0 |

Conditioning input per row: `(values_norm, mask)` where mask ∈ {0,1}⁴ — the
denoiser learns to predict the noise given any subset of the 4 properties.

---

## 3. Training schedule

| Setting | Value | Reason |
|---|---|---|
| optimizer | AdamW (β₁=0.9, β₂=0.98, wd=1e-4) | stable for diffusion |
| lr | 1e-4 with 500-step warmup, cosine to 0.1× | standard |
| batch size | 128 | fits 6 GB VRAM with fp16 |
| epochs (cap) | 20 | early-stop usually fires first |
| wall-clock budget | 90 min | RTX 2060 reality |
| precision | fp16 + dynamic loss-scale | |
| EMA decay | 0.999 | sampling uses EMA weights |
| diffusion steps T | 1000 | linear β |
| grad clip | 1.0 | |
| early stop | patience 12 evals, min_delta 1e-4 | |

### Subset-mask sampling

Per row, draw a mask size from the categorical distribution and randomly
choose that many *valid* properties (weighted by `cond_weight`):

| size | prob |
|---|---|
| 0 | 0.10 |
| 1 | 0.25 |
| 2 | 0.30 |
| 3 | 0.20 |
| 4 | 0.15 |

After choosing, apply **property-dropout** with rate 0.30 (each chosen entry
zeroed independently). Then with `cfg_dropout_rate = 0.10` the entire mask is
zeroed (full unconditional pass) — this is what classifier-free guidance
needs at sample time.

### Energetic-bias oversampling (v3 distinguishing feature)

Goal: keep the prior on the **energetic manifold**. Without intervention the
382k pop is dominated by inert ZINC/PubChem chemistry that drags the diffusion
prior off-target.

Mechanism (in `train.py` row-weight builder):

```text
For each property j:
  trusted_j = rows where cond_weight[:,j] >= 0.9          # Tier A/B only
  hi_j      = quantile(values_norm[trusted_j, j], 0.90)
  boost rows where (trusted_j  AND  values_norm[:,j] >= hi_j)  by 5×
```

Effect: rows with experimentally / DFT-confirmed *high* density / HOF / D / P
are sampled 5× more often. Inert-molecule lows (which dominate the 3DCNN
backfill tail) are deliberately **not** boosted — earlier v2 ablation showed
that symmetric oversampling pulled the q50 generations toward the inert-low
tail. About 0.5 % of training rows are boosted; effective sampling weight on
energetic high-tail goes up ~10–20×.

### Loss

Plain ε-prediction MSE on the latent: `loss = MSE(eps_pred, eps_true)`.
Conditioning enters through FiLM, not through the loss; trust differences
between tiers are expressed via the mask sampling weights, not loss weights.

---

## 4. Validation

### During training (`val_every_steps = 500`)

- Random 5 % of rows held out (stratified by random seed).
- Same subset-mask procedure as training (so val_loss is comparable to
  train_loss but not a fidelity metric).
- Best checkpoint = lowest val_loss with min_delta 1e-4.

### Post-training (`scripts/diffusion/cfg_sweep.py`)

Run on `best.pt` with EMA weights, multiple guidance scales (default
`[2, 5, 7]`), 50 samples per (property × quantile).

For each property `p` and target quantile `q ∈ {q10, q50, q90}`:

1. Set `mask = onehot(p)`, `values[p] = q_z`, all other entries 0.
2. DDIM-sample 40 steps at the chosen guidance scale.
3. Decode to SMILES via LIMO; canonicalise; drop invalids.
4. Run the **3DCNN smoke ensemble** as an *independent property predictor*
   on the generated SMILES. Predictions are averaged across `model_0.pth` /
   `model_1.pth` (2-fold scaffold-CV ensemble).
5. Compute `MAE`, `rel_MAE_pct`, `within_10_pct` against `target_raw`.

Outputs: `<exp>/cfg_sweep.json` and `<exp>/cfg_sweep.md`.

### Acceptance bands (energetic-material focus)

q90 is the metric that matters; q10 just measures whether the model can
generate inert chemistry.

| Property | q90 within-10 % target | rel-MAE goal |
|---|---|---|
| density           | ≥ 25 % | ≤ 10 % |
| heat_of_formation | ≥ 10 % | ≤ 50 % |
| det. velocity     | ≥ 20 % | ≤ 20 % |
| det. pressure     | ≥ 15 % | ≤ 30 % |

These are *not yet met* by v3 alone. Sampling-time classifier guidance from
the SA + SC predictors (`scripts/guidance/`) and a candidate-rerank step
(`evaluate_candidates.py`) are intended to close the remaining gap.

### Out-of-band checks

| Check | Where | Cadence |
|---|---|---|
| Unconditional novelty (NN-Tanimoto) | `evaluate.py --n_uncond 300` | end-of-run |
| Compound-to-compound (SDEdit) variants | `evaluate.py --compound2compound` | end-of-run |
| Top-K Psi4 B3LYP HOF spot-checks | `scripts/simulation/psi4_hof.py` | manual, top candidates only |

---

## 5. Pitfalls observed across v1 → v3

1. **CFG too weak (v1)**: guidance=2 collapsed to dataset mean; sweeping
   3/5/7 nudges q10 closer but cannot rescue q90. CFG alone is not enough.
2. **Symmetric oversampling (v2)**: 5× boost on top *and* bottom 10 %
   produced a model that under-predicts q50 by ~300 kcal/mol HOF, ~2 km/s D.
   Cause: the "low tail" of the 382k pop is inert chemistry, not low-energy
   energetics. Asymmetric high-only oversampling fixed it (v3).
3. **Unconditional drift to long alkane/aromatic strings**: visible in v1
   sample dumps. Indicates the prior is too broad. Future fix: filter training
   set to molecules with explicit energetic motifs OR ≥ 0.3 Tanimoto to a
   known-energetic seed list (planned for v4).

---

## 6. Reproducing v3

```bash
# (one-time prerequisites)
#   - latents.pt produced by encode_latents.py
#   - preds_3dcnn.pt produced by run_3dcnn_all.py over all 382k SMILES
#   - latents_expanded.pt produced by expand_conditioning.py

/c/Python314/python scripts/diffusion/train.py \
    --config configs/diffusion_expanded_v3.yaml

# evaluate
EXP=$(ls -1dt experiments/diffusion_subset_cond_expanded_v3_* | head -1)
/c/Python314/python scripts/diffusion/cfg_sweep.py \
    --exp "$EXP" --scales 2 5 7 --n_per_target 50

# full eval (uncond + sdedit + 3DCNN-validated cond)
/c/Python314/python scripts/diffusion/evaluate.py \
    --exp "$EXP" --n_uncond 300 --n_cond 80 --guidance 5
/c/Python314/python scripts/diffusion/report.py --exp "$EXP"
```
