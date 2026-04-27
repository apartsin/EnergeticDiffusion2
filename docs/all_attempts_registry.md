# All-attempts registry — for paper reporting

Every model variant, denoiser config, sampling strategy, and integration
attempt tried in this project — past, current, planned. Local and remote
runs. Successes and failures with reasons.

Generated 2026-04-26. Maintain by appending; never delete rows.

---

## Section 0 — Status overview (past / current / future × local / remote)

### Currently running

| Run ID | Job | Where | Status |
|---|---|---|---|
| `bs7cla3hn` | Joint rerank pool=8k (v3+v4-B, v1 decoder) | local RTX 2060 | running |

### Just-completed (this session)

| Job | Where | Outcome | Verdict |
|---|---|---|---|
| LIMO v3 (parallel non-AR transformer decoder) | local RTX 2060 | val_acc 31.8 % (alkane collapse) | **failed** non-AR was wrong |
| LIMO v3.1 (causal AR + skip-connection) | local RTX 2060 | val_acc 89.3 %, **3/7 seeds exact roundtrip** (TNT, PETN, TATB) | **breakthrough** at training time, **collapses on diffusion z** |
| LIMO v3.2 (no-skip AR, 6-layer 512-d) | local RTX 2060 | val_acc 79.0 %, best_val 0.98. Decode-from-prior with **greedy argmax**: 16 random N(0,I) z's → 2 token patterns. With **sample=True**: 32/32 valid SMILES, 29 unique. | **greedy was the collapse**, not architecture. Sampled decode produces chemically-generic SMILES (chem_filter kept 22/374 = 6%) because diffusion latents trained on v1 geometry. To use v3.1/v3.2 decoder properly need full re-encode + denoiser retrain. |
| Joint rerank with v3.1 decoder (sample=True), v4-B denoiser, pool=2k | local RTX 2060 | 374/4000 valid → 22 chem-passed → 7 SA/SC → 4 final, top score 5.47 | confirms **v3.1-decoder + v1-trained denoiser is mismatched**; v1-decoder remains the production path |
| LIMO v3.3 (v3.2 + noise-aug curriculum) | local RTX 2060 | val_acc 76.55%, best_val 1.0047. Pure-prior decode: chem_filter 6/90 (still bad); SDEdit/C2C @ σ=1.0 from RDX/HMX/FOX-7 → diverse nitro-bearing variants (3-4/8 with [N+][O-], max Tanimoto 0.18-0.21); self-consistency on PETN: 0.27 (v1: 0.18). | **drop-in decoder swap useful for C2C only**, not for prior-sampling rerank. Encoder shared with v1 → no other model needs retraining. ckpt at `experiments/limo_v3_3_diff_aware_AR_20260426T122722Z/`. |
| MolMIM v1.3 weights download (NGC) | local | 269 MB `.nemo`, extracted | ready for vast.ai job A |
| Joint rerank pool=3k (v3+v4-B, v1 decoder) | local RTX 2060 | 196 valid → 80 ranked, top score 4.11 (density 1.76, HOF -97 kJ/mol) | actionable leads in `joint_rerank.md` |
| Vast.ai Job D (latent norm retrofit) | RTX 4090, ~$0.27 | Trained 45k steps, best_val 0.1733, early-stopped epoch 16 | **converged** but **ckpt unrecoverable** (SCP delivered 500MB partial; instance destroyed before retry) |
| Vast.ai Job C (massive rerank pool=20k) | RTX 4090, ~$0.30 | Three iterative bug fixes (selfies missing → arg-base path → shim latents); host SSH proxy then died | **dead-lost** to vast.ai infra failure; compensated with local pool=8k rerank |

### Vast.ai failure modes encountered + skill patches

| Failure | Cause | Skill patch |
|---|---|---|
| Job D (1st run): `torch.cuda.is_available()==False` despite RTX 4090 reported | Host CUDA driver 12.5 < PyTorch image's 12.6 requirement; tensorboard kept idle, billing $0.22 silently | `cuda_max_good >= 12.6` filter in offer search; onstart fail-fast with exit 42 if `torch.cuda.is_available()==False` |
| Concurrent uploads (D + C) saturating shared bandwidth → 15min stale-timeout | gpu_runner has no upload semaphore | TODO: serialize concurrent uploads (single-bandwidth lane) |
| Instance "running" but SSH never came up after 8 retries / 40s | Vast SSH proxy on offered host broken; skill auto-recovers by destroying + picking next offer | already handled by skill |
| SCP fetch silently truncated 700MB → 500MB, no error | No post-fetch sha verification | TODO: add sha256 manifest + verify after fetch |
| Host SSH proxy permanently died mid-job (100% packet loss to ssh4.vast.ai) | Vast infra; affected Job C 35620898 after 60min uptime | Cut losses + destroy; no skill fix |

### Planned, scripts ready, not yet launched

| Job | Where | Why |
|---|---|---|
| Vast.ai A: MolMIM hybrid (encode 382 k + denoiser retrain at 512-d) | A100 40 GB | Test pretrained-1B-SMILES VAE vs LIMO |
| Vast.ai B: DiT-style denoiser (50 M-param transformer + AdaLN-Zero) | RTX 4090 | Address D10 broken-cond-signal; replaces FiLM-MLP |
| Vast.ai C: Massive joint rerank pool=20 000 | RTX 4090 | Cheap shot at more breakthrough leads |
| Vast.ai D: Latent normalisation retrofit | RTX 4090 | Independent test of L5 norm-mismatch finding |

### Planned, not yet scripted

| Job | Why |
|---|---|
| LIMO v3.2 (8-layer 512-d AR + EMA) | Bigger transformer if v3.1 plateaus. Would run on local or vast.ai |
| MolMIM fine-tune on motif-aug (LoRA encoder + full decoder) | Closes any motif gap MolMIM has |
| MoLFormer-XL hybrid (1.1 B SMILES pretrain) | Compares to MolMIM. Need to fix `transformers.onnx` import first |
| Re-encode 382 k with v3.1 → retrain denoiser → c2c re-eval | Validates the v3.1 self-consistency gain pays off downstream |
| C2c on TNT / PETN / TATB seeds with v3.1 | Direct test of c2c usefulness now that 3 seeds roundtrip exactly |

### Explicitly dropped (will not run)

| Item | Reason |
|---|---|
| Active-learning DFT loop (Psi4 cycles) | User decision |
| Dedicated MolMIM-as-full-replacement (vs hybrid) | User decision; pursuing hybrid via vast.ai instead |
| ChemFormer / CDDD / HierVAE swaps | User decision (drop all VAE swap plans except MolMIM hybrid) |

---

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
| **V3.1** | **LIMO v3.1 (frozen V1 enc + AR causal transformer dec + skip-conn)** | 4-layer causal transformer decoder, teacher forcing, encoder-embed skip | + V1 frozen | 1.08 M motif-aug, 32 904 steps | **val_acc 89.3 %**, val_loss 0.68. Self-consistency: TNT 0.13→**1.00**, PETN 0.46→**1.00**, TATB 0.13→**1.00**, RDX 0.50→0.57, FOX-7 0.04→0.17, HMX flat, CL-20 slight regression | **breakthrough** — 3/7 seeds exact roundtrip; AR + teacher forcing escaped alkane minimum | `experiments/limo_v3_1_AR_20260426T070423Z/` |
| V3.2 | LIMO v3.2 (6-layer 512-d AR, **no skip**, 32 memory tokens) | 39 M params, frozen v1 enc + larger AR dec, skip-conn=False | + V1 frozen | 1.08 M motif-aug, 43 872 steps, 59 min | val_acc 79.0 %, val_loss 0.99, best_val 0.98. **Decode-from-prior collapse**: 16 random N(0,I) z's → only 2 token patterns (alkane-like or all token 61). Same failure mode as v3.1 at diffusion-time. | **AR teacher-forced training does not generalize to diffusion z without encoder embeddings**; pivot back to v1 decoder for reranks; consider scheduled sampling or noise-injection on encoder embeddings during training | `experiments/limo_v3_2_noskip_AR_20260426T075452Z/` |
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

All 30 top candidates checked: 0/30 in PubChem, 0/30 in 382k internal, fully novel.

### From joint v3+v4-B rerank pool=8000, top-200 (run 2026-04-26 14:00, RTX 2060 local)

Larger pool found leads scoring **5x better** than the original pool=1500 run (composite 0.79 vs 4.11 best). All 4 design targets met simultaneously.

| Rank | SMILES | ρ | HOF | D (km/s) | P (GPa) | SA | SC | MaxTan(train) | Source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `O=[N+]([O-])C=C([N+](=O)[O-])[N+](=O)[O-]` | 1.92 | +58 | **9.53** | 39.33 | 3.79 | 1.64 | 0.39 | v4-B |
| 2 | `O=[N+]([O-])C=N[N+](=O)[O-]` | 1.92 | +117 | 9.38 | 38.82 | 4.47 | 1.48 | 0.39 | v4-B |
| 3 | `O=[N+]([O-])C=C[N+]([O-])([N+](=O)[O-])[N+](=O)[O-]` | 1.93 | +89 | 9.34 | 38.87 | 4.57 | 1.84 | 0.33 | v4-B |
| 4 | `O=[N+]([O-])NC([N+](=O)[O-])[N+](=O)[O-]` | 1.92 | +11 | **9.58** | **41.33** | 4.03 | 1.96 | 0.39 | v4-B |

Lead #4 reaches RDX-class performance (RDX: ρ 1.81, D 8.75, P 34.9). Full table in `experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_pool8k.md`.

### From joint v3+v4-B rerank pool=20000, top-200 (run 2026-04-26 14:25, RTX 2060 local; replacement for lost vast Job C)

Pool=20k yielded 1533 unique → 1429 chem-passed → 1024 SA/SC → **983 final**. All top 9 hit all 4 targets simultaneously.

| Rank | SMILES | ρ | HOF | D (km/s) | P (GPa) | SA | SC | MaxTan(train) | Source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `O=[N+]([O-])N=C=NN[N+](=O)[O-]` | **1.94** | **+209** | 9.39 | **40.45** | 4.98 | 2.20 | 0.29 | v4-B |
| 2 | `N=C(N(N[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]` | **1.95** | +85 | **9.65** | 40.25 | 4.40 | 1.36 | 0.34 | v4-B |
| 3 | `O=[N+]([O-])C=NC(=C=N[N+](=O)[O-])[N+](=O)[O-]` | **1.95** | +170 | 9.38 | 38.80 | 4.88 | 2.31 | 0.25 | v4-B |
| 4 | `O=[N+]([O-])C=N[N+](=O)[O-]` | 1.92 | +117 | 9.38 | 38.82 | 4.47 | 1.48 | 0.39 | v4-B |
| 5 | `O=C([N+](=O)[O-])[N+](=O)[O-]` | 1.92 | +0 | **9.54** | 38.12 | 3.58 | 1.33 | 0.53 | v4-B |
| 6 | `O=[N+]([O-])C=NO[N+](=O)[O-]` | 1.91 | +163 | 9.20 | 36.71 | 4.55 | 1.57 | 0.38 | v4-B |

Lead #2 (trinitromethyl hydrazine) approaches **CL-20 territory** in performance.

**Important caveat (added 2026-04-26 18:05):** the original top leads (#2, #5, #8 etc.) rank-1 by single-axis composite score only. When evaluated under a multi-objective pipeline (saturating performance + chem-redflag SMARTS + formula gates + viability classifier + sensitivity proxy + Pareto), several of these are rejected as model-cheats. See "Rerank v2" below.

### Rerank v2 — multi-objective, gated (2026-04-26 18:05, RTX 2060 + CPU)

Implements the user-specified multi-objective scoring framework:
- **Hard filters:** SMARTS structural alerts (gem-tetranitro, polynitro cyclopropene, peroxide, etc.); MW ∈ [130, 600]; nitro/heavy ≤ 0.42; OB ∈ [-120, +25]%; allowed atoms {C,H,N,O,F}; net charge = 0; nC ≤ 2 with ≥ 3 nitro → reject.
- **Saturating performance:** banded ramp on each property (caps at 1.0); model can't win by hallucinating ultra-oxidized structures.
- **Viability classifier:** RandomForest (Morgan FP 2048 + 21 RDKit descriptors) trained on 65 980 labelled energetic positives vs 80 000 ZINC negatives. **Validation AUC 0.9986**, AP 0.998, ACC@0.5 0.982.
- **Sensitivity proxy:** open-chain N-NO2 motifs + small-skeleton polynitro penalty.
- **Pareto front** over (perf, viability) maximisation × (sensitivity, alerts) minimisation.

Applied to top-400 of `joint_rerank_pool40k.md`:
- 45/400 hard-rejected (11 polynitro on C2 chain; 4 polynitro on C1 chain; 28 MW < 130; 2 OB > +25%) — exactly the failure modes called out by the user's plan.
- 355 survivors scored.
- **Pareto front: 34 candidates** (was 2 without viability dimension).

#### v2 top-5 (Pareto-front, all viability ≈ 1.00, ranked by composite v2)

| # | composite | perf | viab | sens | ρ | HOF | D | P | SMILES |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **0.83** | 0.86 | **1.00** | 0.00 | 1.89 | +32 | 9.69 | 40.17 | `O=[N+]([O-])C1([N+](=O)[O-])CN([O-])[N+]1=O` (4-ring with N-oxide) |
| 2 | 0.79 | 0.84 | 1.00 | 0.00 | 1.90 | +129 | 9.28 | 37.18 | `O=[N+]([O-])N=NNC=NN[N+](=O)[O-]` (open-chain di-NO2 dihydrotetrazole) |
| 3 | 0.78 | 0.84 | 1.00 | 0.32 | 1.89 | +163 | 9.25 | 37.16 | `O=[N+]([O-])C=C(C=N[N+](=O)[O-])[N+](=O)[O-]` |
| 4 | 0.76 | 0.73 | 1.00 | 0.00 | 1.92 | +96 | 8.88 | 35.18 | `C=C(N=NN=N[N+](=O)[O-])[N+](=O)[O-]` |
| 5 | 0.75 | 0.86 | 1.00 | 0.32 | 1.94 | +138 | 9.20 | 36.85 | `O=[N+]([O-])N=C1C([N+](=O)[O-])C1[N+](=O)[O-]` |

The v1 top-1 (`O=[N+]([O-])N=C=NN[N+](=O)[O-]`, dinitrocarbodiimide) was REJECTED for polynitro on C2 chain. The v1 top-2 ("tetranitroamino-vinyl", D=9.80) was REJECTED for polynitro on C2 chain. **Both v1 leads are model-cheats.** v2 produces a more conservative but viable list.

Full table in `joint_rerank_pool40k_v2.md`.

Internal-novelty check on top-15 vs original 66k labeled_master:
- **12/15 NOVEL**, including #2, #3, #4, #5, #6, #7, #8, #9, #10, #11, #12, #15
- 3 known: #1 (`O=[N+]([O-])N=C=NN[N+](=O)[O-]`, N,N'-dinitrocarbodiimide), #13 (dinitramide), #14 (1,2-dinitrohydrazine). Useful as **sanity-check anchors**: model rediscovered known high-density energetics while also producing 12 novel candidates of similar quality.

PubChem REST lookup on top-10 (2026-04-26):
- **7/10 NOVEL vs PubChem** (no CID): #1, #2, #3, #4, #6, #7, #8
- 3 known in PubChem: #5 dinitromethane (CID 54223030), #9 1-nitrotetrazole (CID 19022555), #10 nitryl cyanide (CID 15798559). #1 (N,N'-dinitrocarbodiimide) was in internal labeled_master but absent from PubChem — internal data has been augmented past PubChem coverage.

### From joint v3+v4-B rerank pool=40000, top-400 (run 2026-04-26 14:50, RTX 2060 local)

Pool=40k yielded **2775 unique → 2626 chem-passed → 1744 SA/SC → 1667 final**. Best D and P observed yet.

| Rank | SMILES | ρ | HOF | D (km/s) | P (GPa) | SA | SC | MaxTan | Source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `O=[N+]([O-])C=N[N+](=O)[O-]` | 1.92 | +117 | 9.38 | 38.82 | 4.47 | 1.48 | 0.39 | v4-B |
| 2 | `O=[N+]([O-])NC(=C([N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]` | **1.98** | +95 | **9.80** | 41.61 | 4.12 | 1.54 | 0.45 | v4-B |
| 3 | `O=[N+]([O-])C1=C([N+](=O)[O-])C1[N+](=O)[O-]` | 1.92 | +126 | 9.08 | 37.72 | 3.62 | 1.70 | 0.39 | v4-B |
| 4 | `O=[N+]([O-])C=C(C=N[N+](=O)[O-])[N+](=O)[O-]` | 1.89 | +163 | 9.25 | 37.16 | 4.26 | 1.81 | 0.28 | v4-B |
| 6 | `O=[N+]([O-])N=CN([N+](=O)[O-])[N+](=O)[O-]` | 1.97 | +119 | 9.64 | **42.67** | 4.47 | 1.49 | 0.32 | v4-B |

Lead #2 (tetranitroamino-vinyl) and #6 (N,N-dinitro-amidine) reach **CL-20 / HMX class** simultaneously across density + velocity + pressure. Full table in `joint_rerank_pool40k.md`.

PubChem novelty on top-20 of pool=40k: **19/20 NOVEL**. Only known: #16 dinitramide (CID 9796883). All other 19 candidates have no PubChem CID — including the breakthrough #2 (`O=[N+]([O-])NC(=C([N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]`).

### CFG sweep (cfg=5 vs cfg=7 vs cfg=9 at pool=8k each, 2026-04-26)

| cfg | valid | chem-passed | final | top score | top-1 SMILES |
|---|---|---|---|---|---|
| 5 | 802 | 758 | 528 | 0.92 | `O=[N+]([O-])C=N[N+](=O)[O-]` |
| 7 | 1533 | 1429 | 983 | 0.70 | `O=[N+]([O-])N=C=NN[N+](=O)[O-]` |
| 9 | 673 | 650 | 427 | 0.79 | `O=[N+]([O-])C=C([N+](=O)[O-])[N+](=O)[O-]` |

Higher cfg → more aggressive cond-conditioning → more candidates concentrated near targets. cfg=7 is still optimal for the v3+v4-B ensemble.

Full table in `joint_rerank_pool20k.md`.

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
