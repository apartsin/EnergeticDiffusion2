# LIMO v2 — motif-rich fine-tune plan

Continue fine-tuning LIMO VAE from the v1 best checkpoint
(`experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt`,
step 8,500) with a re-balanced dataset that **emphasises** the N-rich
ring chemistry our v1 LIMO is bad at decoding.

D2 + D15 are the diagnostics that motivate this:
- D2 — only 4 % exact roundtrip recovery on top-D molecules;
- D15 — 0 % furazan / tetrazole / triazole / azide in top reranked candidates
  vs 8–17 % in real high-HOF training rows.

Goal: lift roundtrip recovery on the rare-motif tail without breaking the
already-good decode of common nitramine / polynitro chemistry.

---

## 1. Version preservation

| Item | Path | Status |
|---|---|---|
| LIMO ZINC pretrained | `external/LIMO/vae.pt` | **immutable** |
| LIMO v1 fine-tune    | `experiments/limo_ft_energetic_20260424T150825Z/` | **immutable** |
| LIMO v1 latents      | `data/training/diffusion/latents.pt` (382k rows) | **immutable** |
| Expanded latents     | `data/training/diffusion/latents_expanded.pt` | **immutable** |
| Trustcond latents    | `data/training/diffusion/latents_trustcond.pt` | **immutable** |
| All v1–v6 denoiser experiments + sweeps + reranks | unchanged | **immutable** |
| **v2 LIMO (new)** | `experiments/limo_ft_motif_rich_<timestamp>/` | new |
| **v2 latents (new)** | `data/training/diffusion/latents_v2.pt` (after re-encode) | new |

Roll-back path: if v2 fine-tune regresses on overall reconstruction, we
keep using v1; nothing existing depends on v2.

---

## 2. Data acquisition

### 2.1 Targets — motifs we want LIMO to learn

Per D15 these are the gaps:

| Motif | SMARTS | High-HOF prevalence | v1 candidate prevalence |
|---|---|---|---|
| furazan / 1,2,5-oxadiazole | `c1nonc1` | 16.9 % | 0 % |
| triazole | `c1nncn1`, `c1cnnn1` | 12.1 % | 0 % |
| tetrazole | `c1nnnn1` | 11.3 % | 0 % |
| azide | `[N-]=[N+]=N`, `N=[N+]=[N-]` | 8.2 % | 0 % |
| triazine | `c1ncncn1` | 3.0 % | 0 % |
| polynitro (≥3 NO₂) | composite | 42.0 % | 10 % |
| furoxan (furazan-N-oxide) | `c1nonc1=O` | (subset of furazan) | – |

### 2.2 Sources

1. **PubChem PUG-REST substructure search** (no key, free, ~5 req/s safe).
   For each SMARTS query, fetch up to 5,000 CIDs → resolve to canonical
   SMILES → keep only those with ≤ LIMO_MAX_LEN SELFIES tokens and only
   atoms in LIMO's vocab.
2. **Internal Tier-A/B rows already containing the motif** — included
   automatically since the existing master CSV is reused.
3. **No external paid sources** (Reaxys/SciFinder excluded by scope).

Expected pool size (after token-validity filter): **15–30 k unique SMILES**.

### 2.3 Quality controls before training

- Strip salts, keep largest fragment (`Chem.GetMolFrags(asMols=True)` →
  largest by atom count).
- Reject if RDKit `SanitizeMol` raises.
- Reject if SELFIES encode fails or length > 72 (LIMO_MAX_LEN).
- Reject if any token is OOV in LIMO's vocab.
- Reject duplicates (canonical-SMILES dedup against existing 382k).

### 2.4 Cached output

`data/raw/motif_rich/pubchem_<motif>.smi` — one file per motif query.
Combined into `data/training/master/motif_rich_extra.csv` with columns
`smiles, motif, source`.

---

## 3. Training-set re-balancing

### 3.1 Combined dataset

```
existing_subset  =  energetic_biased  (220k after limo_finetune.py filters)
motif_extra      =  pubchem motif-rich pool (~20k after filter)
combined         =  existing + motif_extra              (~240k)
```

Tag each row with a binary `is_motif_rich` flag (computed from SMARTS
match against the 6 target motif patterns; True if ≥ 1 hit).

### 3.2 Sampling weights (per-row)

```
row_weight = 1.0                            (default)
row_weight ×= 5  if has any rare motif      (furazan / tetrazole / triazole / azide / triazine)
row_weight ×= 2  if has polynitro (≥3 NO₂)
row_weight ×= 0.5 if row contains only saturated-C / no nitro at all
                  (rarely in training but downweight to free budget)
```

Effect: in expectation, a training batch of 64 contains ~25–35 motif-rich
examples (vs 1–2 currently).

### 3.3 Loss weighting

Per-row scalar weight applied to NLL only (KL term unchanged):

```
loss_per_row = nll_weight_per_row * nll(row) + β * kl(row)
nll_weight_per_row = 1.0 + (3.0 if has rare motif else 0.0)
```

Rationale: KL is per-dim Gaussian; multiplying it muddies the latent
prior. Reconstructing a tetrazole correctly is what we care about, so
amplify only NLL.

### 3.4 No per-token loss weighting

Simpler and more robust. Per-token amplification (boost cross-entropy on
SELFIES tokens that participate in target rings) requires SMILES → SELFIES
→ ring-atom-token mapping which is brittle. Defer; row-level weighting
should already shift behaviour decisively.

---

## 4. Hyperparameters

Continue from v1 best.pt; do **not** restart from scratch.

| Parameter | v1 value | **v2 value** | Rationale |
|---|---|---|---|
| init weights | `external/LIMO/vae.pt` | **v1 best.pt** | resume, don't restart |
| lr | 3e-5 | **1e-5** | lower so we don't unlearn v1 |
| warmup_steps | 200 | **100** | resuming, less warm-up needed |
| min_lr_ratio | 0.1 | **0.1** | unchanged |
| batch_size | 64 | **64** | unchanged (RTX 2060 6 GB) |
| epochs (cap) | 6 | **8** | total wall-clock 8 h budget |
| total_time_minutes | 220 | **480** | overnight run |
| KL β | 0.01 (const) | **0.01** | preserve latent geometry |
| weight_decay | 0.01 | **0.01** | unchanged |
| precision | fp16 | fp16 | unchanged |
| grad_clip | 1.0 | 1.0 | unchanged |
| dropout | none | none | unchanged |
| **NEW**: row sample weight | – | rare ×5, polynitro ×2 | §3.2 |
| **NEW**: per-row NLL multiplier | – | rare ×4 | §3.3 |
| **NEW**: motif-extra append | – | ~20 k PubChem rows | §2 |

Total expected steps: ~8 epochs × ~3,750 batches/epoch ≈ **30 k more steps**
on top of v1's 8.5 k → v2 lands around step 38 k.

Early stop unchanged (patience 8 evals, min Δ 1e-3). If KL collapses
(per-dim KL < 0.05) we'll abort and lower the rare-motif weight.

---

## 5. Diagnostics protocol

Run after v2 training completes; compare against v1 on identical splits.

### 5.1 Reconstruction accuracy

| Test | Population | Pass threshold |
|---|---|---|
| **R1** Roundtrip on all Tier-A/B  | 1,000 sampled rows | ≥ 60 % exact, mean Tanimoto ≥ 0.85, no regression vs v1 |
| **R2** Roundtrip on top-D Tier-A/B | 50 highest-D rows | **≥ 30 % exact** (v1: 4 %) |
| **R3** Per-motif roundtrip      | up to 200 SMILES per motif | exact ≥ 50 % each motif |

### 5.2 Latent geometry (sanity)

| Test | What | Pass threshold |
|---|---|---|
| **R4** Per-dim KL distribution | over 5 k val rows | ≤ 5 % of dims with KL < 0.05 (no collapse) |
| **R5** Latent ‖z‖ distribution  | over 5 k val rows | mean ‖z‖ within ±10 % of v1 mean |
| **R6** Cosine similarity preserved | 200 Tanimoto-matched pairs | mean cos in latent-space > 0.5 (v1 similar) |

### 5.3 Property-encoding preservation

| Test | What | Pass threshold |
|---|---|---|
| **R7** Property-MLP r vs v1 | 1024 → 512 → 1 head per property, trained on Tier-A/B latents | r within ±0.05 of v1 (D3 baseline) |

If R7 *improves* (especially HOF r), even better.

### 5.4 Sampling-time validity

| Test | What | Pass threshold |
|---|---|---|
| **R8** Sample-from-prior | 1 000 latents from N(0, I), decode | valid SMILES rate ≥ v1 + 5 % |
| **R9** Motif emergence in unconditional samples | SMARTS check | ≥ 5 % furazan + tetrazole + triazole combined (v1: 0 %) |

### 5.5 Failure-mode log

For every diagnostic that fails, log 5 example SMILES (truth → decoded)
to `experiments/<v2>/diag_failures.md` for manual inspection.

### 5.6 Acceptance gates

- **Pass**: R1 + R2 + R3 + R4 all OK → proceed to re-encode latents (out of scope here, will require user approval).
- **Soft fail (R2 only)**: if R2 < 30 % but R1 ≥ 50 %, escalate to per-token loss weighting (deferred enhancement) before adopting.
- **Hard fail**: R1 regresses below 50 % → keep v1, mark v2 deprecated.

---

## 6. Wall-clock budget

| Step | Time |
|---|---|
| 1. Fetch motif-rich SMILES from PubChem | 30–60 min |
| 2. Build augmented dataset + cache tokens | 15 min |
| 3. Patch + smoke-test trainer | 30 min |
| 4. Train v2 (target ~30 k steps) | **6–8 h** |
| 5. Diagnostics | 30 min |
| 6. Report | 15 min |
| **Total** | **~9 h** |

Single overnight run on RTX 2060.

---

## 7. Out-of-scope (per user instruction)

- Re-encoding `latents.pt` (will run later, separately, with approval).
- Retraining the denoiser (v8). The user explicitly said "stop after VAE
  fine-tuning result".
- Switching base VAE (MolFormer etc.) — separate decision.
- Active learning DFT loop — separate decision.

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Catastrophic forgetting of common chemistry | medium | low LR (1e-5), KL β unchanged, R1 acceptance gate |
| Latent posterior collapses on rare-motif rows | low | KL β unchanged + R4 monitor |
| PubChem fetch is unreliable | medium | cache aggressively, accept whatever fetches; min 3 motifs returned ≥ 1 k |
| Token vocab incomplete for fetched SMILES | low | drop OOV rows (existing code path) |
| 6–8 h GPU budget overrun | low | hard wall-clock budget in trainer |

---

## 9. Acceptance criteria (single-line summary)

LIMO v2 passes if **R2 (top-D roundtrip) ≥ 30 %**, **R3 (per-motif
roundtrip) ≥ 50 %** for furazan/tetrazole/triazole/azide, and **R1 + R4
+ R7** show no regression vs v1.
