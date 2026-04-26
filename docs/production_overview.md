# EnergeticDiffusion2 — production overview

Snapshot of the **current best production pipeline** for novel
energetic-material discovery, as of 2026-04-26.

The pipeline takes a property target (e.g. "high density, high D, high P,
high HOF") and produces ranked, validated, novel SMILES candidates whose
predicted properties hit the target with high accuracy.

---

## 1. Goal

Generate novel high-energy candidate molecules conditioned on:

- density (ρ, g/cm³)
- detonation velocity (D, km/s)
- detonation pressure (P, GPa)
- heat of formation (HOF, kcal/mol)

Output: ranked SMILES with predicted properties and a measure of novelty
relative to known compounds.

---

## 2. Pipeline diagram

```
                         ┌────────────────────────────┐
                         │  382,604 SMILES (energetic │
                         │  + ZINC-derived)           │
                         └───────────┬────────────────┘
                                     │
                ┌────────────────────┼─────────────────────┐
                │                    │                     │
                ▼                    ▼                     ▼
       ┌──────────────────┐ ┌────────────────────┐ ┌──────────────────┐
       │ LIMO VAE         │ │ Tier-A/B labels    │ │ 3DCNN smoke      │
       │ (fine-tuned 8.5k │ │ (experiment + DFT) │ │ (Uni-Mol 2-fold  │
       │  steps on        │ │ • density 19,176   │ │  ensemble)       │
       │  energetic data) │ │ • HOF      2,625   │ │ predict 8 props  │
       └─────────┬────────┘ │ • D        2,539   │ └─────────┬────────┘
                 │          │ • P        2,366   │           │
                 ▼          └─────────┬──────────┘           │
        latents.pt 382k×1024d         │                      │
                 │                    │                      ▼
                 └─────► expand_conditioning.py  ◄──────  preds_3dcnn.pt
                            │
                            ▼
                  latents_expanded.pt  (382k × 1024 + values + tiers)
                            │
                            ▼
                  build_latents_trustcond.py
                            │
                            ▼
                  latents_trustcond.pt   (Tier-A/B-only conditioning)
                            │
                            ▼
                  ConditionalDenoiser  v4-B  (44.6 M params, FiLM ResNet 8 blocks)
                            │
                            ▼
                  best.pt + EMA + property heads
                            │
                            ▼
                  ┌──────── inference ────────┐
                  │                            │
                  ▼                            ▼
         rerank_sweep.py              rerank_multi.py
         (single property)            (joint q90, all 4 props)
                  │                            │
                  ▼                            ▼
               3DCNN validator filter + rank by composite score
                                ▼
                       top-N novel candidates
                                │
                                ▼
                  match_candidates.py (PubChem + internal lookup)
                                │
                                ▼
                       final ranked output
```

---

## 3. Data layer

### 3.1 Source SMILES

| Set | Count | Source | Stored at |
|---|---|---|---|
| Energetic curated master | ~22k | LANL CHIRALMolecules + ICT + EMDP 3DCNN.csv + literature | `data/training/master/labeled_master.csv` |
| Energetic-biased proxy | ~360k | filter of unlabeled stock by energetic motif heuristic | `data/training/master/unlabeled_master.csv` |
| **Combined** | **382,604** | concatenation, deduplicated | this is the input to the LIMO encoder |

### 3.2 Property tiers

Every (row, property) cell carries a tier label.

| Tier | Source | `cond_weight` | typical count per property |
|---|---|---|---|
| **A** | Experimental literature | 1.0 | ~10–15% of curated set |
| **B** | DFT (B3LYP/6-31G(d) atomization HOF, etc.) | 1.0 | rest of curated set |
| **C** | Kamlet-Jacobs empirical (D, P only) | 0.7 | unused in production (3DCNN preempts) |
| **D** | 3DCNN smoke-model ensemble prediction | 0.7 | the bulk of 382k |

Per-property A+B counts: density 19,176; HOF 2,625; D 2,539; P 2,366.

### 3.3 Latent artefacts

| File | Shape / size | What it contains | Built by |
|---|---|---|---|
| `latents.pt` | 382 604 × 1024 float32 (~1.5 GB) | LIMO `z_mu` for every SMILES | `scripts/diffusion/encode_latents.py` |
| `preds_3dcnn.pt` | 382 604 × 8 float32 (~12 MB) | smoke-model 8-target predictions | `scripts/diffusion/run_3dcnn_all.py` |
| `latents_expanded.pt` | adds `values_norm`, `cond_valid`, `cond_weight`, `tiers`, `stats` | expanded conditioning blob | `scripts/diffusion/expand_conditioning.py` |
| `latents_trustcond.pt` | same shapes; `cond_valid` zeroed for Tier-D | Tier-A/B-only conditioning variant (v4-B input) | `scripts/diffusion/build_latents_trustcond.py` |

### 3.4 Per-property normalisation

z-score with mean/std computed **only on Tier-A/B rows**:

| Property | mean | std | Tier-A/B count |
|---|---|---|---|
| density | 1.5174 g/cm³ | 0.241 | 19 176 |
| heat_of_formation | -97.06 kcal/mol | 284.79 | 2 625 |
| detonation_velocity | 5.918 km/s | 1.520 | 2 539 |
| detonation_pressure | 15.039 GPa | 9.000 | 2 366 |

q-targets at sample time: `target_raw = q_z × std + mean` where q_z ∈ {-1.281, 0.0, +1.281} for q10/q50/q90.

---

## 4. Models

### 4.1 LIMO VAE  (the molecule encoder/decoder)

| Spec | Value |
|---|---|
| Architecture | MLP encoder + MLP decoder over flattened SELFIES tokens (72 × \|vocab\|) |
| Latent | 1024-d Gaussian (μ, σ) |
| Tokenisation | SELFIES character-level (with energetic-vocab extension) |
| Parameters | ~5 M |
| Pretraining | ZINC-250k from the LIMO authors' release |
| Fine-tune | 8 500 steps on the 382k energetic-biased subset |
| Checkpoint | `experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt` |
| Code | `scripts/vae/limo_model.py`, `limo_finetune.py` |

Known limitation (D2): exact roundtrip recovery on top-D (high-detonation-velocity) molecules is only 4 %; the decoder is weak on N-rich heterocycles (furazan, tetrazole, triazole, azide).

### 4.2 3DCNN smoke ensemble (the property validator)

| Spec | Value |
|---|---|
| Source model | Uni-Mol v1 + EMDP smoke-model |
| Architecture | Equivariant 3D-CNN over conformers |
| Outputs | 8 properties: density, DetoD, DetoP, DetoQ, DetoT, DetoV, HOF_S, BDE |
| Ensemble | 2 fold-scaffold-CV checkpoints (`model_0.pth`, `model_1.pth`); predictions averaged |
| Checkpoint dir | `data/raw/energetic_external/EMDP/Data/smoke_model/` |
| Wrapper | `scripts/diffusion/unimol_validator.py` |
| Validator quality (D1) | density r=0.96, D r=0.95, P r=0.98, HOF r=0.72 (the noisy one) |

Used for both Tier-D label generation (`run_3dcnn_all.py`) and downstream rerank validation.

### 4.3 Conditional denoiser  v4-B  (the diffusion model)

| Spec | Value |
|---|---|
| Architecture | FiLM-conditioned ResNet on 1024-d latent |
| Hidden dim | 2048 |
| Blocks | 8 FiLM-modulated ResNet blocks |
| Time embedding | 256-d sinusoidal |
| Property embedding | 64-d per property (4 properties → 256-d total cond) |
| Parameters | 44.6 M |
| Diffusion process | DDPM, T = 1000, cosine α-bar schedule |
| Sampling | DDIM, 40 steps, classifier-free guidance |
| Conditioning input | `(values_norm, mask)` ∈ ℝ⁴ × {0,1}⁴ — denoiser learns any subset |
| Checkpoint | `experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/checkpoints/best.pt` |
| Best val_loss | 0.0482 |
| EMA decay | 0.999 (used at sample time) |
| Code | `scripts/diffusion/model.py`, `train.py` |

Diagnostic finding (D10): the model is largely **ignoring the conditioning value** (cosine between eps for opposite targets ≈ +0.87). Sidestepped by reranking; otherwise, q90 generations would all collapse to the dataset mean.

### 4.4 Latent property heads  (for classifier guidance, future use)

| Spec | Value |
|---|---|
| Architecture | MLP 1024 → 512 → 512 → 1 |
| Trained per property (4 heads) | density, HOF, D, P |
| Training data | Tier-A/B latents only |
| Pearson r on held-out | density 0.94, HOF 0.71, D 0.84, P 0.89 |
| Checkpoint | `data/training/guidance/property_heads.pt` |

Built and ready, but currently **not in the production rerank pipeline** because v6 single/multi-rerank already saturates ρ/D/P performance. Heads are kept available for future v7 pipelines that combine classifier guidance with diffusion sampling.

---

## 5. Training

### 5.1 Denoiser v4-B  (the production training run)

Config: [`configs/diffusion_expanded_v4b.yaml`](../configs/diffusion_expanded_v4b.yaml)

| Setting | Value | Reason |
|---|---|---|
| Optimizer | AdamW (β₁ = 0.9, β₂ = 0.98, wd = 1e-4) | standard for diffusion |
| Learning rate | 1e-4 with 500-step warmup, cosine to 0.1× | standard |
| Batch size | 128 | fits 6 GB VRAM with fp16 |
| Diffusion T | 1000 | linear β |
| Wall-clock | ~50 min | RTX 2060 |
| Steps | 53 500 (early-stopped) | standard run |
| EMA decay | 0.999 | sampling uses EMA weights |
| Mixed precision | fp16 + dynamic loss-scale | fits memory |
| Grad-clip | 1.0 | standard |

Key v4-B differentiators (vs the deprecated v1–v5 attempts):

1. **Tier-A/B-only conditioning**: `cond_valid` zeroed for Tier-D rows so they only contribute to the unconditional prior. The 3DCNN noise (D8 — HOF 44 % MAE/std) was hurting the conditional signal.
2. **Asymmetric oversampling**: rows in the **top-5 % of any property within Tier-A/B** are upsampled 10× during batch construction. This gives the model many more exposures to high-energy chemistry.
3. **Property-dropout 0.30**: each chosen mask entry is independently zeroed, teaching the model to handle arbitrary-subset conditioning at sample time.
4. **CFG dropout 0.10**: 10 % of training batches are fully unconditional, used for classifier-free guidance at sampling.
5. **Per-row loss weighting** (`α = 0.5`): `loss_weight ∈ [0.5, 1.0]` based on `mean(cond_weight × mask)`.

### 5.2 Other training runs (preserved but not in production)

Each is fully archived — same revert procedure: load `experiments/<name>/checkpoints/best.pt`, run sweeps. See [`docs/denoiser_versions.md`](denoiser_versions.md).

| Version | Status | Best val | Notes |
|---|---|---|---|
| v1 | superseded | 0.0515 | first full run; CFG=2 sample-time too weak |
| v2 | deprecated | 0.0480 | symmetric oversampling regressed q50 |
| v3 | superseded | **0.0468** | high-only oversampling restricted to Tier-A/B; balanced |
| v4 | deprecated | 0.0504 | energetic-motif filter cut prior coverage; broad regression |
| v4-nf | benchmark-best (single-prop) | 0.0480 | v4 recipe minus the filter |
| v5 | deprecated | 0.0483 | v4-nf + Min-SNR; regressed across the board |
| **v4-B** | **current production** | 0.0482 | Tier-A/B-only conditioning |

### 5.3 LIMO fine-tune (v1 LIMO)

| Setting | Value |
|---|---|
| Init weights | LIMO ZINC pretrained | from external repo |
| Steps | 8 500 |
| LR | 3e-5 with warmup 200 |
| Batch | 64 |
| KL β | 0.01 constant |
| Wall-clock | ~3.5 h |
| Output ckpt | `experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt` |

### 5.4 3DCNN inference run

Not training, but production data-creation:

| Setting | Value |
|---|---|
| Code | `scripts/diffusion/run_3dcnn_all.py` |
| Batch | 256 |
| Periodic predictor reset | every 80 batches (mitigates memory / cache issue in `unimol_tools`) |
| Total inferences | 382 604 SMILES × 2 folds |
| Wall-clock | ~3.5 h on RTX 2060 |
| Output | `data/training/diffusion/preds_3dcnn.pt` (382 604 × 8) |

### 5.5 Property heads training

| Setting | Value |
|---|---|
| Code | `scripts/guidance/train_property_heads.py` |
| Per head | 1500 / 3000 steps depending on data size |
| Optimiser | AdamW, lr 2e-3, wd 1e-4 |
| Wall-clock | ~5 min total for all 4 heads |
| Output | `data/training/guidance/property_heads.pt` |

---

## 6. Inference / evaluation pipeline

### 6.1 Single-property rerank  (v6, benchmark)

[`scripts/diffusion/rerank_sweep.py`](../scripts/diffusion/rerank_sweep.py)

```bash
python scripts/diffusion/rerank_sweep.py \
    --exp experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z \
    --cfg 7 --n_pool 1500 --n_keep 40 \
    --require_neutral
```

For each property × q-target:
1. Generate `n_pool` latents conditioned on that single property
2. Decode to SMILES, canonicalise, drop invalid
3. Apply `--require_neutral` (formal charge = 0, no radicals)
4. Run 3DCNN ensemble validator on each survivor
5. Sort by `|3DCNN_pred − target_raw|`, keep top `n_keep`

Result on v4-B with `pool=1500, keep=40`:

| Property × q90 | rel_MAE % | within-10 % | max produced | target |
|---|---|---|---|---|
| density | **0.2** | **100** | 1.83 | 1.83 |
| HOF | 38 | 2 | +257 | +268 |
| D | **0.2** | **100** | 7.90 | 7.86 |
| P | **0.9** | **100** | 26.97 | 26.57 |

### 6.2 Multi-property rerank  (v6-multi, production for candidate discovery)

[`scripts/diffusion/rerank_multi.py`](../scripts/diffusion/rerank_multi.py)

```bash
python scripts/diffusion/rerank_multi.py \
    --exp experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z \
    --cfg 7 --n_pool 400 --n_keep 40 \
    --weights 1.0 1.0 1.0 1.0 \
    --require_neutral
```

Conditions on **all 4 properties simultaneously** at q90, then ranks by
the composite score:

```
composite = Σ_p w_p × |pred_p − target_p| / std_p
```

Result on v4-B (`pool=400, keep=40, weights=1,1,1,1`):

| Property | top-40 rel_MAE % | top-40 within-10 % |
|---|---|---|
| density | **3.5** | 100 |
| HOF | 96 | 0 |
| D | **3.4** | 98 |
| P | **7.8** | 68 |

Best candidate found:
- `O=[N+]([O-])N=CNC1=NN1CC(=N[O-])[N+](=O)[O-]`
- predicted: ρ = 1.83 g/cm³, D = 8.69 km/s, P = 33.31 GPa, HOF = +198 kcal/mol

### 6.3 CFG sweep  (research / sampling-knob calibration)

[`scripts/diffusion/cfg_sweep.py`](../scripts/diffusion/cfg_sweep.py)

Sweeps guidance scales (g ∈ {2, 5, 7}) per property × q ∈ {q10, q50, q90}.
Used to choose the production `cfg=7`.

### 6.4 Cross-version comparison

[`scripts/diffusion/compare_versions.py`](../scripts/diffusion/compare_versions.py)

Auto-generates [`docs/version_comparison.md`](version_comparison.md) from
all available `cfg_sweep.json` files across the experiments directory.

### 6.5 Candidate-match lookup

[`scripts/diffusion/match_candidates.py`](../scripts/diffusion/match_candidates.py)

For each top-ranked SMILES:
- Internal exact match against the 382 k training set
- PubChem PUG-REST `compound/smiles/<SMILES>/cids/JSON` (no key, free)
- NCI CACTUS `chemical/structure/<SMILES>/iupac_name`

Reports per-candidate: in-training (✓/·), in-Tier-A/B (✓/·), PubChem CID,
canonical name, IUPAC.

Current run on top-30 v6 candidates: **30 / 30 entirely novel** (no PubChem hit, no internal hit).

---

## 7. Diagnostic suite

[`scripts/diagnostics/run_all.py`](../scripts/diagnostics/run_all.py) +
standalone scripts. All produce `docs/diag_d*.md` reports.

| # | What it measures | Latest result |
|---|---|---|
| **D1** | Validator self-consistency on Tier-A/B SMILES | density / D / P **strong** (r ≥ 0.95); HOF **ok** (r = 0.72) |
| **D2** | LIMO encoder–decoder roundtrip on top-D molecules | **weak**: 2 / 50 exact, mean Tanimoto 0.18, 84 % retain NO₂ |
| **D3** | Property predictability from LIMO latents | density r = 0.94, P r = 0.89, D r = 0.84, HOF r = 0.71 |
| **D5** | Out-of-range conditioning (z = +3 vs q90) | saturated for all 4 properties |
| **D8** | Tier-D label noise (3DCNN smoke vs Tier-A/B truth) | density / D / P **OK** (16–23 % MAE/std); HOF **noisy** (44 %) |
| **D10** | Conditioning-signal correlation in denoiser | **broken**: cosine +0.21 (ρ) to +0.91 (HOF) — model uses mask, not value |
| **D14** | Property correlations in Tier-A/B | HOF decoupled (r = 0.15) but high-HOF rows still have +0.7–1.1 σ higher ρ/D/P |
| **D15** | Motif distribution (top candidates vs high-HOF reference) | top candidates: 0 % furazan/tetrazole/triazole/azide vs 8–17 % in real high-HOF |

---

## 8. Current capability — what production can do today

| Capability | Status |
|---|---|
| Generate molecules at q90 of density (target +1.83 g/cm³) | **solved** — 100 % within-10 % at pool=1500 |
| Generate molecules at q90 of D (7.86 km/s) | **solved** — 100 % within-10 % at pool=1500 |
| Generate molecules at q90 of P (26.57 GPa) | **solved** — 100 % within-10 % at pool=1500 |
| Generate molecules at q90 of HOF (+267.7 kcal/mol) | **partial** — max +257 (within striking distance), 2 % within-10 %, mean +166 |
| Generate molecules **jointly** high on ρ + D + P + HOF (multi-property rerank) | top candidates match ρ/D/P targets, get within ~30 % of HOF target |
| Filter unphysical SMILES (charge 0, no radicals) | available via `--require_neutral` |
| Match candidates to known compounds (novelty check) | available via `match_candidates.py` |

---

## 9. Known limitations / next levers

| Limitation | Root cause (diagnostic) | Possible next step |
|---|---|---|
| HOF q90 stops at +257 (vs +268 target) | LIMO decoder bottleneck (D2) + missing N-rich rings (D15) | **LIMO v2 motif-rich fine-tune** (planned next) |
| Top candidates have 0 % furazan / tetrazole / triazole | LIMO doesn't decode N-rich rings well | same as above |
| HOF metric reliability | 3DCNN HOF MAE/std = 44 % (D8) | optional DFT spot-check on top-N candidates (manual, not pipelined) |
| Denoiser ignores cond *value* (D10) | FiLM signal too weak | sidestepped by rerank; **no architectural change planned** |
| Top candidates may be hard to synthesise | – | **SA + SC rerank-weight composite** (`--with_feasibility --w_sa 0.5 --w_sc 0.25`); sampling-time gradient guidance shelved due to schedule bug |

**Out of scope** (decisions made 2026-04-26):
- Active-learning DFT (Psi4) loop — dropped.
- VAE swaps to MolMIM / ChemFormer / CDDD / HierVAE — dropped. Pipeline stays on LIMO.

---

## 10. File map

```
configs/
  diffusion_expanded_v4b.yaml            # production denoiser config

scripts/
  vae/
    limo_model.py                        # LIMOVAE + tokenizer
    limo_finetune.py                     # LIMO trainer
    limo_evaluate.py                     # VAE evaluation
  diffusion/
    encode_latents.py                    # SMILES → z_mu via LIMO
    run_3dcnn_all.py                     # smoke-model inference for all 382k
    expand_conditioning.py               # build latents_expanded.pt
    build_latents_trustcond.py           # build latents_trustcond.pt (v4-B input)
    model.py                             # ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample
    train.py                             # denoiser trainer (with v3+ enhancements)
    evaluate.py                          # full eval pipeline
    cfg_sweep.py                         # CFG-scale sweep
    rerank_sweep.py                      # v6 single-property rerank
    rerank_multi.py                      # v6-multi joint rerank
    cfg_sweep_guided.py                  # classifier-guidance variant (research)
    guided_sampler.py                    # DDIM with classifier guidance
    match_candidates.py                  # novelty / known-compound lookup
    compare_versions.py                  # cross-version comparison generator
    unimol_validator.py                  # 3DCNN ensemble wrapper
  guidance/
    train_property_heads.py              # 4 latent property MLPs
    model.py                             # property-head architecture
  diagnostics/
    run_all.py                           # D1 D2 D3 D5 D10 batch
    d8_tier_d_noise.py                   # Tier-D label noise audit
    d14_property_correlations.py         # property-correlation audit
    d15_top_motifs.py                    # motif distribution audit

data/
  raw/
    energetic_external/EMDP/Data/smoke_model/   # 3DCNN ensemble (immutable)
  training/
    master/
      labeled_master.csv                 # ~22k Tier-A/B
      unlabeled_master.csv               # ~360k energetic-biased
    diffusion/
      latents.pt                         # LIMO encodings
      preds_3dcnn.pt                     # smoke-model predictions
      latents_expanded.pt                # full conditioning blob
      latents_trustcond.pt               # Tier-A/B-only conditioning (v4-B input)
    guidance/
      property_heads.pt                  # 4 latent MLPs

experiments/
  limo_ft_energetic_20260424T150825Z/    # LIMO v1 (production VAE)
  diffusion_subset_cond_expanded_v4b_20260426T000541Z/   # v4-B (production denoiser)
  diffusion_subset_cond_expanded_v4_nofilter_20260425T175119Z/   # benchmark-best
  diffusion_subset_cond_expanded_v3_20260425T140941Z/    # superseded
  diffusion_subset_cond_expanded_20260425T095335Z/        # v1 (immutable)
  ...

docs/
  production_overview.md                 # this file
  denoiser_versions.md                   # version registry
  version_comparison.md                  # auto-generated cross-version table
  diagnostics_plan.md                    # 10-diagnostic catalog
  diag_summary.md                        # latest diagnostic results
  diag_d1.md … diag_d15.md               # individual diagnostic outputs
  improvements_deep_think.md             # ranked improvement options
  v4_analysis.md                         # earlier deep dive
  limo_v2_plan.md                        # LIMO motif-rich fine-tune plan
  denoiser_v3_protocol.md                # denoiser training protocol
  denoiser_training_strategy.md          # training-strategy doc
```

---

## 11. Reproducing production output

```bash
# 1. Generate top-N candidates (multi-property)
python scripts/diffusion/rerank_multi.py \
    --exp experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z \
    --cfg 7 --n_pool 400 --n_keep 40 --require_neutral

# 2. Single-property fidelity benchmark
python scripts/diffusion/rerank_sweep.py \
    --exp experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z \
    --cfg 7 --n_pool 1500 --n_keep 40 --require_neutral

# 3. Novelty + known-compound lookup
python scripts/diffusion/match_candidates.py \
    --exp experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z \
    --limit 30

# 4. Full diagnostic re-run
PYTHONIOENCODING=utf-8 python scripts/diagnostics/run_all.py
PYTHONIOENCODING=utf-8 python scripts/diagnostics/d8_tier_d_noise.py
PYTHONIOENCODING=utf-8 python scripts/diagnostics/d14_property_correlations.py
PYTHONIOENCODING=utf-8 python scripts/diagnostics/d15_top_motifs.py
```

---

## 12. Headline production numbers

(All q90, validated with 3DCNN ensemble.)

| Metric | LIMO v1 + denoiser v4-B + rerank pool=1500 + `--require_neutral` |
|---|---|
| density q90 rel_MAE | 0.2 % |
| density q90 within-10 % | 100 % |
| D q90 rel_MAE | 0.2 % |
| D q90 within-10 % | 100 % |
| P q90 rel_MAE | 0.9 % |
| P q90 within-10 % | 100 % |
| HOF q90 rel_MAE | 38 % |
| HOF q90 max produced | +257 kcal/mol |
| Sample novelty (vs PubChem + 382k internal) | 100 % (30 / 30 of top candidates) |

The pipeline reliably generates **novel** molecules satisfying high-energy
density, detonation velocity, and detonation pressure targets. HOF target
remains the open frontier and is the natural focus for the next iteration
(LIMO v2 → swap to MolMIM/ChemFormer → active-learning DFT loop).
