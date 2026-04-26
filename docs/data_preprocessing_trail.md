# Data preprocessing + transformations — full audit trail

For paper Methods section. Every transformation applied to the raw data,
every artefact generated, every dataset name used. Read top-to-bottom for
the chronological pipeline; each row links to the script that produced it.

---

## 1. Raw inputs

| Source | What it provides | Provenance |
|---|---|---|
| LANL CHIRALMolecules | curated energetic molecules with experimental properties | external lab |
| ICT-DB | additional energetic compounds | external |
| EMDP `3DCNN.csv` | 26,265 molecules with DFT-quality electronic / thermochem labels + Girolami-empirical density | EMDP repo |
| ZINC-250 k | LIMO author's pretraining corpus | upstream LIMO release |
| 3DCNN smoke-model checkpoints | `model_0.pth`, `model_1.pth` (2-fold scaffold-CV ensemble, 84 M params each, Uni-Mol v1 architecture) | EMDP / Uni-Mol |

Raw artefacts at `data/raw/energetic_external/`.

---

## 2. Preprocessing — labelled-master construction

Pipeline in `scripts/` (multiple scripts, evolved over the project):

| Step | Script | Input → Output | Notes |
|---|---|---|---|
| Initial labelled merge | `scripts/audit_fix_labels.py` and earlier scripts | several CSVs → first-cut `labeled_master.csv` | – |
| Fix CM4-C density provenance bug | `scripts/fix_cm4c_density_provenance.py` | – | mislabelled CM4-C density as DFT |
| 3DCNN HOF unit fix | `scripts/fix_3dcnn_hof_units.py` | kJ/mol → kcal/mol (× 4.184) | EMDP shipped HOF in kJ/mol but labelled "kcal/mol" |
| 3DCNN label-source fix | `scripts/fix_3dcnn_label.py` | re-tag rows: actual EMDP rows are *DFT*, not 3DCNN-predicted, despite filename | forensic finding (atom-count correlation r = -0.99 was the smoking gun) |
| MDGNN explosion-heat fix | `scripts/fix_mdgnn_explosion_heat.py` | repaired sign convention | – |
| 4-tier system | `scripts/apply_4tier_system.py` | adds `tier` column ∈ {A, B, C, D} | A=experimental, B=DFT, C=K-J empirical, D=3DCNN smoke |
| Per-property tier assignment | `scripts/assign_per_property_tiers.py` | each (row, property) cell tagged independently | a row may be Tier-B for HOF but Tier-A for density |
| K-J imputation | `scripts/impute_kj_general.py`, `scripts/impute_kj_from_explo5_hof.py` | adds Kamlet-Jacobs D, P estimates as Tier-C | for rows with ρ + HOF + formula |
| Energetic-name resolution | `scripts/resolve_energetic_names_v2.py` | adds canonical names | for trace-back to literature |
| Kroonblawd Tier-A merge | `scripts/merge_kroonblawd_tierA.py` | adds external Tier-A rows | – |
| Pre-conversion backups | `*_pre_*.csv.bak` files | every step preserved a backup | for rollback |

**Output**: `data/training/master/labeled_master.csv` — 50 MB, ~22 k rows with
per-property Tier-A/B/C/D labels, formal source attribution, name where known.

Companion: `data/training/master/unlabeled_master.csv` — 248 MB, ~360 k rows of
energetic-biased SMILES from rnnmgm_ds9 + nitro / azide proxy matches.

---

## 3. Motif-augmentation pass (LIMO v3 / v3.1 / MolMIM-FT input)

| Step | Script | Detail |
|---|---|---|
| SMARTS scanner | `scripts/vae/build_motif_augmented.py` | 9 patterns: furazan, tetrazole (×2), triazole (×2), azide (×2), tetrazine, furoxan |
| Per-row replication | same | rare-motif rows × 5; polynitro (≥3 NO₂) × 2; rest × 1 |
| Combined output | `data/training/master/labeled_motif_aug.csv` | **1,220,590** rows (vs 765,018 unique inputs) |
| Composition (uniques) | – | 93,348 rare-motif × 5 = 466 740 lines; 82,180 polynitro × 2 = 164 360; 589,490 baseline × 1 = 589,490 |

---

## 4. SMILES → SELFIES → token tensors (LIMO encoder input)

| Step | Script | Detail |
|---|---|---|
| SELFIES encode | `scripts/vae/limo_finetune.py:encode_dataset` | uses `selfies` library; alphabet built from ZINC-250 k + extension tokens |
| Token cache | `data/training/vae_tokens/energetic_ft_<hash>.pt` (v1) and `vae_tokens_v2_1/energetic_ft_<hash>.pt` (motif-aug) | content-hashed; same hash across runs to allow reuse |
| Vocab | `external/LIMO/vocab_cache.json` | 108 tokens including BOS/EOS/PAD |
| max_len | 72 | drops longer SMILES |
| `drop_oov` flag | True | drops rows with OOV tokens vs the cached vocab |

Cache file sizes: 190 MB (v1, 326,016 unique SMILES) and 630 MB (motif-aug, 1,079,878 rows after token-validity filter).

---

## 5. LIMO encoding → diffusion latents

| Step | Script | Output | Notes |
|---|---|---|---|
| Forward through V1 LIMO encoder | `scripts/diffusion/encode_latents.py` | `data/training/diffusion/latents.pt` (382,604 × 1024 + smiles + meta) | 1.5 GB |
| 3DCNN smoke ensemble inference (8 outputs/molecule) | `scripts/diffusion/run_3dcnn_all.py` | `data/training/diffusion/preds_3dcnn.pt` (382,604 × 8) | 12 MB; ran with `--reset_every_batches 80` to mitigate `unimol_tools` cache decay |
| Conditioning expansion (Tier-A/B + smoke backfill) | `scripts/diffusion/expand_conditioning.py` | `data/training/diffusion/latents_expanded.pt` | adds `values_norm`, `cond_valid`, `cond_weight`, `tiers`, `stats`. Tier-A/B weight 1.0, Tier-D weight 0.7 |
| Trustcond filter (Tier-A/B-only conditioning) | `scripts/diffusion/build_latents_trustcond.py` | `data/training/diffusion/latents_trustcond.pt` | sets `cond_valid=False` where `cond_weight < 0.99`; v4-B input |
| Energetic-motif filter (deprecated; v4 only) | `scripts/diffusion/filter_energetic.py` | `data/training/diffusion/latents_v4_filtered.pt` | 228 k rows; **regressed** vs full set; v4 deprecated |

z-score statistics computed only from Tier-A/B trusted rows (per property),
not from the broader pool. This was deliberate so target z-scores at sample
time refer to the energetic-chemistry distribution.

---

## 6. Latent normalization (planned, vast.ai job D)

Currently **not applied** in production. L5 diagnostic finding: LIMO V1
training latents have mean ‖z‖ ≈ 8 vs expected ~32 (sqrt(1024)) for a
unit-Gaussian prior. Future job D plan:

```
mu_global  = z_mu.mean(dim=0)               # (1024,)
std_global = z_mu.std(dim=0).clamp(min=1e-6)
z_normalised = (z_mu - mu_global) / std_global    # ~ N(0, I) per dim
```

Save alongside as `latent_stats = {mu_global, std_global}` for the inverse
transform at decode time.

---

## 7. Auxiliary scoring data

| Step | Script | Output | Notes |
|---|---|---|---|
| Compute SA score (Ertl-Schuffenhauer, RDKit) | `scripts/guidance/compute_sa_sc.py` | `data/training/diffusion/latents_with_scores.pt` adds `sa_score` field (382,604 × 1) | mean 3.92, std 1.15 |
| Compute SC score (Coley) | same | adds `sc_score` field | mean 3.53, std 0.83 |
| 4 latent property heads (clean z) | `scripts/guidance/train_property_heads.py` | `data/training/guidance/property_heads.pt` | r 0.71-0.94 per property |
| 2 latent SA + SC surrogates (clean z) | `scripts/guidance/train.py` | `experiments/guidance_sa_sc_<ts>/checkpoints/{sa,sc}_best.pt` | r² 0.72 SA, 0.67 SC |
| 2 time-aware SA + SC surrogates | `scripts/guidance/train_t_aware_surrogates.py` | `data/training/guidance/property_heads_t.pt` | r² 0.31 / 0.40 — lower because trained on noisy z_t for classifier guidance under DDPM |

---

## 8. Slim bundles for vast.ai upload

| Bundle | Contains | Size | Built by |
|---|---|---|---|
| `vast_jobs/smiles_cond_bundle.pt` | SMILES + values_raw + values_norm + cond_valid + cond_weight + tiers + stats + property_names — **no latents** | 51 MB | inline `python -c` script |

Used to avoid uploading the 1.5 GB `latents.pt` when vast.ai jobs only need
the SMILES list + conditioning labels (latents are recomputed by the new
encoder on the remote).

---

## 9. Tokenizer / vocabulary

- SELFIES alphabet built from ZINC-250 k SMILES via `selfies.get_alphabet_from_selfies`
- Cached to `external/LIMO/vocab_cache.json` (108 tokens including PAD=0)
- Single tokenizer used for V1, V2.1b, V3, V3.1 (all share frozen V1 encoder)
- MolMIM uses its own SentencePiece tokenizer (vocab 640) — different alphabet, must use MolMIM's tokenizer with MolMIM's encoder

---

## 10. Naming conventions

| Category | Prefix |
|---|---|
| LIMO experiments | `experiments/limo_ft_<descriptor>_<timestamp>/` |
| Denoiser experiments | `experiments/diffusion_subset_cond_expanded_<variant>_<timestamp>/` |
| Guidance heads | `experiments/guidance_<descriptor>_<timestamp>/` |
| Latent files | `data/training/diffusion/latents{_expanded,_trustcond,_v4_filtered,_v2,_molmim}.pt` |
| Token caches | `data/training/vae_tokens{,_v2_1,_v3}/energetic_ft_<hash>.pt` |
| Sweep / rerank outputs | `<exp_dir>/{cfg_sweep,rerank_results,rerank_multi,joint_rerank}{,_hof,_annotated}.{md,json}` |

Timestamps: `YYYYMMDDTHHmmSSZ` UTC.

---

## 11. Architecture changes summary

| Variant | Encoder | Bottleneck | Decoder | Param change |
|---|---|---|---|---|
| LIMO V0 (ZINC pretrained) | MLP, 64-d emb × 72 → 1000 → 2000 → 2 × 1024 | Gaussian (μ, log σ²) | MLP 1024 → 1000 → 2000 → 72 × 108 | baseline |
| LIMO V1 (energetic FT) | same | same | same | weights only |
| LIMO V2.1b (motif-aug) | same | same | same | weights only |
| LIMO V3 (failed) | same FROZEN | same | replaced: 4-layer transformer, parallel non-AR, 384-d, 6 heads, 16 memory tokens | -decoder ~3 M, +9.6 M trainable; total 23.9 M |
| **LIMO V3.1** (success) | same FROZEN | same | replaced: 4-layer **causal AR** transformer + teacher forcing + skip-connection from encoder embeddings | total 23.9 M, 9.6 M trainable |
| Denoiser V1–V6 (all) | – | – | FiLM-conditioned ResNet on 1024-d latent: 8 blocks, 2048 hidden, 256-d time emb, per-property emb 64 | 44.6 M |
| Planned DiT denoiser (vast.ai job B) | – | – | 64 tokens × 16-d, 12 transformer blocks, 512 hidden, 8 heads, AdaLN-Zero conditioning | ~50 M |
| Planned MolMIM hybrid | Perceiver enc 6 layers 512-d | 512-d Gaussian (MIM-trained) | transformer dec 6 layers 512-d | ~70 M; needs new denoiser at 512-d |

---

## 12. Training procedure changes summary

| Variant | LR | β (KL) | Schedule | Other |
|---|---|---|---|---|
| LIMO V1 | 3e-5 | 0.01 const | warmup 200 steps, cosine to 0.1× | 8.5 k steps; init=ZINC pretrained |
| LIMO V2.1b | 1e-5 | 0.01 const | warmup 100 | init=V1; aug data 1.22 M; same arch |
| LIMO V3 | 3e-4 | 0.01 const | warmup 500 | init=V1 (frozen); train new transformer dec; failed |
| LIMO V3.1 | 3e-4 | 0.01 const | warmup 1000 | init=V1 (frozen); causal AR + skip; **succeeded** |
| Denoiser V1 | 1e-4 | – | warmup 500, cosine 0.1× | EMA 0.999; T=1000 cosine α-bar |
| V2 | + 5× sym oversample top/bottom 10 % | | | failed at q50 |
| V3 | + 5× **high-only** oversample, Tier-A/B trusted | | | recovers q50; HOF best |
| V4 | + energetic-motif filter (228 k filtered set) | | | failed |
| V4-nf | V4 minus filter | | | best ρ q90 |
| V5 | V4-nf + Min-SNR γ=5 | | | failed |
| V4-B | V4-nf + Tier-A/B-only conditioning | | | best D / P q90 |

---

## 13. What gets reported in the paper Methods section

In order:
1. Section 2 above (raw data + preprocessing pipeline including bug fixes — important for reproducibility).
2. Section 3 (motif augmentation rationale).
3. Section 4 (tokenization).
4. Section 5 (LIMO encoding + 3DCNN labelling + tier system).
5. Section 11 (architecture variants).
6. Section 12 (training procedure variants).
7. Section 6 (latent normalization, if vast.ai job D delivers improvement).
8. Section 7 (auxiliary heads).

The all-attempts registry `docs/all_attempts_registry.md` is the supplementary material covering all configurations tried.
