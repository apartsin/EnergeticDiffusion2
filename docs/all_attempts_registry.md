# All-attempts registry — for paper reporting

Every model variant, denoiser config, sampling strategy, and integration
attempt tried in this project. Status, key metrics, failure modes, and
links to per-experiment artefacts. Updated as experiments complete.

Generated 2026-04-26. Maintain by appending; never delete rows.

---

## Section 1 — VAE / encoder variants

| # | Variant | Architecture | Pretrain | FT data | Result | Verdict | Artefacts |
|---|---|---|---|---|---|---|---|
| V0 | LIMO ZINC pretrained | MLP enc/dec, 1024-d Gaussian | 250 k ZINC-250k | – | baseline | as-shipped | `external/LIMO/vae.pt` |
| V1 | LIMO v1 fine-tune (energetic) | same | + ZINC | 326 k energetic-biased, 8 500 steps | val_acc 64.5 %, RDX self-consistency 0.50 | **production VAE for v1–v6 denoisers** | `experiments/limo_ft_energetic_20260424T150825Z/` |
| V2 | LIMO v2 (motif-aug, pre-bug) | same | + V1 init | meant 1.22 M motif-aug; **cache bug** loaded same 326 k | converged in 2.6 min, no improvement | **buggy run, ignored** | `experiments/limo_ft_motif_rich_20260426T053753Z/` |
| V2.1 | LIMO v2.1 (motif-aug, retried) | same | + V1 init, lr 3e-5 | second cache bug; loaded old 326 k via cached file | self-cons regressed RDX 0.50 → 0.11 | **buggy, ignored** | `experiments/limo_ft_motif_rich_v2_1_20260426T054702Z/` |
| V2.1b | LIMO v2.1b (motif-aug, fixed) | same | + V1 init, augmented_source flag | 1.08 M rows after token filter, 16 k steps | val_loss 1.23 (vs V1 ~0.96), motif AUCs unchanged vs V1, self-cons regressed slightly | **data-aug exhausted, architecture is bottleneck** | `experiments/limo_ft_motif_rich_v2_1_20260426T055459Z/` |
| V3 | LIMO v3 (frozen V1 enc + non-AR transformer dec) | 4-layer transformer decoder, parallel decode, 23.9 M params (9.6 M trainable) | + V1 frozen | 1.08 M motif-aug, 32 904 steps | val_acc collapsed to **31.81 %** (= pad baseline); decodes to `CCCCCCC…` for any input | **catastrophic failure of non-AR decoder** | `experiments/limo_v3_transformer_20260426T062604Z/` |
| V3.1 | LIMO v3.1 (frozen V1 enc + AR causal transformer dec + skip-conn) | 4-layer causal transformer decoder, teacher forcing, encoder-embed skip | + V1 frozen | same | **in flight (b g5rnl7s0)** | TBD | `experiments/limo_v3_1_AR_*/` |
| V3.2 (planned) | LIMO v3.2 (8-layer 512-d AR + EMA) | larger v3.1 | – | – | not yet run | – | – |
| **MolMIM v1.3** | NVIDIA pretrained MIM-VAE (Perceiver enc + transformer dec, 70 M params, 512-d Gaussian) | 1 B SMILES (ZINC15 + PubChem) | none | downloaded; encoder/decoder verified via state-dict inspection | – | **weights-on-disk; integration in vast.ai job A queued** | `models/molmim/extracted/` |
| ChemBERTa-2 (planned) | DeepChem 77 M-param BERT encoder | 77 M SMILES | – | – | not run | shelved (MolMIM is more native) | – |
| MoLFormer-XL (planned) | IBM 100 M-param encoder-only | 1.1 B SMILES | – | – | hit `transformers.onnx` import error on Python 3.14 | shelved pending fix | – |

---

## Section 2 — Denoiser variants (all v1–v6 use V1 LIMO latents)

| # | Variant | Recipe | best val_loss | Best CFG | Diagnostic notes | Production status | Artefacts |
|---|---|---|---|---|---|---|---|
| D1 | v1 | first full subset-conditional run, no special tricks | 0.0515 | g=7 | CFG=2 too weak at sample time | superseded | `…_20260425T095335Z/` |
| D2 | v2 | symmetric oversampling top/bottom-10 % | 0.0480 | g=2 | q50 regressed (272 % HOF rel-MAE) | deprecated | `…_v2_20260425T121727Z/` |
| D3 | v3 | high-only oversampling (Tier-A/B), factor=5 | **0.0468** | g=5–7 | best HOF q90 (111 % rel-MAE, max +341 with rerank) | **current for HOF** | `…_v3_20260425T140941Z/` |
| D4 | v4 | + energetic-motif filter (228 k filtered) | 0.0504 | – | filter cut prior coverage; broad regression | deprecated | `…_v4_20260425T160108Z/` |
| D4-nf | v4-nf | v4 minus filter (back to 382 k) | 0.0480 | g=7 | best ρ q90 (11 % rel-MAE), HOF regress vs v3 | benchmark-best | `…_v4_nofilter_20260425T175119Z/` |
| D5 | v5 | v4-nf + Min-SNR γ=5 | 0.0483 | – | regressed on most q-cells; Min-SNR didn't help | deprecated | `…_v5_20260425T224932Z/` |
| D6 | v4-B | Tier-A/B-only conditioning | 0.0482 | g=5–7 | best D q90 (12 %), P q90 (26 %); HOF q90 hard | **current for ρ/D/P** | `…_v4b_20260426T000541Z/` |
| D7 (planned) | v9 (MolMIM 512-d) | retrain at 512-d on MolMIM latents | – | – | – | vast.ai job A | – |
| D8 (planned) | DiT denoiser | replace FiLM-MLP with 50 M-param transformer + AdaLN-Zero | – | – | – | vast.ai job B | – |
| D9 (planned) | v4-B norm | latents normalised to N(0,I), retrain v4-B | – | – | – | vast.ai job D | – |

---

## Section 3 — Sampling / inference strategies

| # | Strategy | Description | Top-property gain | Production status | Notes |
|---|---|---|---|---|---|
| S1 | Vanilla DDIM, fixed CFG=2 | original | baseline | superseded | – |
| S2 | CFG sweep g ∈ {2, 5, 7} | per (prop, q) optimal CFG | ~5 % rel-MAE improvement | always sweep | – |
| S3 | Generate-and-rerank, single-prop, pool=200 | sample 200, validate, top-40 | ρ/D/P q90 → 100 % within-10 % | superseded by pool=1500 | – |
| S4 | Single-prop rerank, pool=1500 | larger pool | density q90 0.2 % rel-MAE, D q90 0.2 %, P q90 0.9 %, **HOF q90 38 % (max +257)** | **production for single-prop** | `rerank_sweep.py` |
| S5 | Multi-property rerank (joint q90, composite) | condition on all 4, rank by composite | 100 % within-10 % (ρ/D/P), 0 % (HOF) | **production for joint** | `rerank_multi.py` |
| S6 | Joint v3 + v4-B rerank pool=1500 | merge pools from both denoisers | top: ρ=1.91, HOF=+163, D=9.20, P=36.7, SA=4.55, MaxTan=0.38 (5 leads found) | **production for breakthrough discovery** | `joint_rerank.py` |
| S7 | + chem_filter (composition + unstable motifs + property bounds + oxygen balance) | drop physical false positives | small but stable | always-on flag | `chem_filter.py` |
| S8 | + `--require_neutral` | drop charge ≠ 0 | small | always-on flag | – |
| S9 | + `--with_feasibility` (SA + SC composite) | rerank-time penalty + hard caps | top SA mean 5.05 → 4.29 (-0.22) | always-on flag | – |
| S10 | Sampling-time SA + SC gradient guidance (clean-z surrogate) | classifier guidance | failed: clean-z surrogate, gradient eaten by clip | shelved | – |
| S11 | Sampling-time guidance + t-aware surrogates + schedule-fixed sampler | round-5 fixed | -0.10 SA at λ=(3, 1.5); modest gain | available not default | `feasibility_sampler.py` |
| S12 | C2c (SDEdit) on energetic seeds | partial-noise + denoise from seed | bounded by LIMO self-consistency ceiling (0.04–0.50 per seed) | needs LIMO v3.x to be useful | `c2c_pipeline.py` |
| S13 | C2c + latent anchor (α=0.3 toward seed) | preserves scaffold | only marginal vs S12 | shelved pending LIMO v3 | – |
| S14 | HOF-prioritised joint rerank | composite weighted toward HOF | 80 candidates produced; PETN×40, FOX-7×18, RDX×18, CL-20×3, TNT×1; all `easy` SA tier | available | `joint_rerank_hof.md` |
| S15 (planned) | Massive rerank pool=20 000 | 20 k each from v3 + v4-B | – | vast.ai job C | – |

---

## Section 4 — Auxiliary models / heads

| # | Item | Architecture | Purpose | Result | Status |
|---|---|---|---|---|---|
| A1 | SA latent surrogate (clean-z) | MLP 1024 → 512 → 512 → 256 → 1 | classifier guidance for SA | r²=0.72 on Tier-A/B, 5 min train | available (`property_heads.pt`) |
| A2 | SC latent surrogate (clean-z) | same | same for SC | r²=0.67 | available |
| A3 | Per-property heads (4 heads) | same | classifier guidance for ρ/HOF/D/P | r 0.71–0.94 | available |
| A4 | Time-aware SA + SC surrogates | MLP + sinusoidal time embedding | classifier guidance under noisy z | SA r²=0.31, SC r²=0.40 (lower because of harder task) | available (`property_heads_t.pt`) |

---

## Section 5 — Data variants

| # | Dataset | Size | Composition | Use | Status |
|---|---|---|---|---|---|
| DAT1 | 382k energetic SMILES | 382 604 | curated + ZINC-derived | input to LIMO encoder | original |
| DAT2 | labeled_master.csv | ~22 k Tier-A/B | experimental + DFT labels | conditioning ground truth | original |
| DAT3 | unlabeled_master.csv | ~360 k | energetic-biased proxy | LIMO fine-tune corpus | original |
| DAT4 | labeled_motif_aug.csv | 1.22 M (after dedup) | rare motifs ×5, polynitro ×2 | LIMO v3 / v3.1 / MolMIM-FT corpus | original |
| DAT5 | latents.pt | 382 604 × 1024-d float32 | LIMO v1 encodings | input to expand_conditioning | original |
| DAT6 | latents_expanded.pt | 382 604 × 1024 + cond | full conditioning blob | input to denoiser train | original |
| DAT7 | latents_trustcond.pt | same shapes | Tier-A/B-only cond_valid | v4-B input | original |
| DAT8 | latents_v4_filtered.pt | 228 310 (motif filter) | filtered by SMARTS | v4 (deprecated) | preserved |
| DAT9 | preds_3dcnn.pt | 382 604 × 8 | smoke-model predictions | Tier-D conditioning labels | original |
| DAT10 | smiles_cond_bundle.pt | 51 MB slim | SMILES + cond + stats only (no latents) | vast.ai job uploads | new |
| DAT11 (planned) | latents_molmim.pt | 382 604 × 512 | MolMIM encodings | vast.ai job A output | TBD |

---

## Section 6 — Diagnostic outputs

| # | Diagnostic | Latest result | Verdict | Path |
|---|---|---|---|---|
| D1 | Validator self-consistency on Tier-A/B | density / D / P r ≥ 0.95; HOF r=0.72 | reliable for ρ/D/P | `docs/diag_d1.md` |
| D2 | LIMO encoder–decoder roundtrip on top-D | 2/50 exact, mean Tanimoto 0.18, 84 % retain NO₂ | weak — LIMO bottleneck | `docs/diag_d2.md` |
| D3 | Property predictability from z | ρ r=0.94, P r=0.89, D r=0.84, HOF r=0.71 | strong — latents encode properties | `docs/diag_d3.md` |
| D5 | Out-of-range conditioning | saturated for all 4 properties at z=+3 | data ceiling, not model | `docs/diag_d5.md` |
| D8 | Tier-D label noise (3DCNN smoke vs Tier-A/B truth) | density / D / P MAE/std 16–23 %; HOF 44 % | HOF labels noisy | `docs/diag_d8.md` |
| D10 | Conditioning signal correlation in denoiser | broken: cosine +0.21 (ρ) to +0.91 (HOF) — uses mask, not value | sampling guidance / rerank required | `docs/diag_d10.md` |
| D14 | Property correlations on Tier-A/B | HOF decoupled from ρ/D/P (r=0.15) | joint optimization possible | `docs/diag_d14.md` |
| D15 | Motif distribution top candidates vs reference | top cand 0 % furazan/tetrazole/triazole/azide | LIMO + data gap | `docs/diag_d15.md` |
| D16 | SA / SC distribution shift after rerank | top SA mean 4.29 vs Tier-A/B 4.51 | -0.22 SA improvement | `docs/diag_d16.md` |
| L5 | LIMO latent norm vs N(0,I) prior | mean ‖z‖ = 8 vs expected 32 | scale mismatch — explains DDIM blow-up | `docs/diag_limo_v2_1b.md` |
| L10 | Motif linear probe (logistic on z) | nitro 0.96, polynitro 0.97, triazole 0.84, **furazan 0.61, tetrazole 0.61** | encoder partially loses N-rich rings | `docs/diag_limo_v2_1b.md` |
| L13 | Self-consistency on energetic seeds | RDX 0.50, PETN 0.46, DAAF 0.44, NTO 0.31, FOX-7 0.04, others ≤ 0.13 | varies wildly; LIMO bottleneck per-seed | `docs/diag_limo_v2_1b.md` |

---

## Section 7 — Top breakthrough leads (for paper)

From joint v3+v4-B rerank pool=1500, q90 targets, all 4 properties + chem_filter + neutrality:

| Rank | SMILES | ρ | HOF | D (km/s) | P (GPa) | SA | SC | MaxTan(train) | Source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `O=[N+]([O-])C=NO[N+](=O)[O-]` | 1.91 | +163 | 9.20 | 36.7 | 4.55 | 1.57 | 0.38 | v4-B |
| 2 | `O=[N+]([O-])N=CO[N+](=O)[O-]` | 1.90 | +114 | 9.33 | 37.9 | 4.61 | 1.60 | 0.43 | v4-B |
| 3 | `O=[N+]([O-])N=CN[N+](=O)[O-]` | 1.89 | +83 | 9.42 | 38.2 | 4.61 | 1.62 | 0.31 | v4-B |
| 4 | `O=[N+]([O-])C=C=N[N+](=O)[O-]` | 1.89 | **+185** | 9.04 | 36.5 | 4.80 | 2.02 | 0.33 | v4-B |
| 9 | `NC(C=N[N+](=O)[O-])=NC(=C[N+](=O)[O-])[N+](=O)[O-]` | 1.86 | **+172** | 8.82 | 33.3 | 4.47 | 2.12 | 0.30 | v4-B |

All 30 top candidates checked: 0/30 in PubChem, 0/30 in 382k internal — fully novel.

---

## Section 8 — Compute log

| Run | Hardware | Wall-clock | Cost |
|---|---|---|---|
| LIMO v1 fine-tune | RTX 2060 6 GB | 3.5 h | local |
| 3DCNN inference 382k | RTX 2060 | 3.5 h | local |
| Each denoiser variant (v1–v6) | RTX 2060 | 50–70 min | local |
| Each cfg_sweep | RTX 2060 | 5–10 min | local |
| Each rerank (pool=1500) | RTX 2060 | ~10 min | local |
| LIMO v2.1b motif-aug retrain | RTX 2060 | 6.7 min (early-stopped) | local |
| LIMO v3 transformer-decoder | RTX 2060 | 28.1 min (early-stopped, failed) | local |
| LIMO v3.1 (in flight) | RTX 2060 | TBD | local |
| Vast.ai jobs A–D (planned) | A100 40 GB / RTX 4090 | ~3–4 h each | ~$5 total batch |

---

## How to maintain this registry

After any new experiment:

1. Append a row to the relevant section (don't delete failed attempts — they
   inform the paper's "what we tried" narrative).
2. If a new artefact directory is created, link it.
3. If an attempt invalidated an earlier finding, mark the older row with
   "(see #N for revision)" — keep both.
4. For paper supplementary: every row above corresponds to a configuration
   the paper should report.

Result for paper: this is the single source of truth for "what configurations
were tried, what worked, what didn't, why".
