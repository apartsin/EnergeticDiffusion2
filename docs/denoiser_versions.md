# Denoiser Version Registry

Each row = a fully self-contained, revertible checkpoint.

## How revert works

Every experiment dir is **immutable** and contains everything needed to
reproduce or sample from that version:

```
experiments/<name>/
  ├── checkpoints/best.pt        ← model + EMA + optimizer + scaler + config
  ├── checkpoints/last.pt
  ├── config_snapshot.yaml       ← exact config that produced these weights
  ├── train.log / train.jsonl    ← full training trace
  ├── cfg_sweep.json / .md       ← post-training CFG sweep results
  └── (eval_results.json, report.html) when evaluate.py + report.py were run
```

To revert / sample from any version:

```bash
EXP=experiments/diffusion_subset_cond_expanded_v2_20260425T121727Z
python scripts/diffusion/cfg_sweep.py --exp "$EXP" --scales 5 --n_per_target 100
python scripts/diffusion/evaluate.py  --exp "$EXP" --guidance 5
```

No retraining required.

## Versions

| Version | Status | Experiment dir | Config | best_val | Best CFG | Notes |
|---|---|---|---|---|---|---|
| v1 | superseded | `diffusion_subset_cond_expanded_20260425T095335Z` | [`diffusion_expanded_v1.yaml`](../configs/diffusion_expanded_v1.yaml) | 0.0515 | g=7 | First full expanded-conditioning run. CFG=2 default at sample time was too weak. |
| v2 | deprecated | `diffusion_subset_cond_expanded_v2_20260425T121727Z` | [`diffusion_expanded_v2.yaml`](../configs/diffusion_expanded_v2.yaml) | 0.0480 | g=2 | Added cond_weight-mask + symmetric extremes oversampling + property-dropout. Q90 improved but **Q50 regressed badly** (HOF rel-MAE 169% → 272%) because low-tail of 382k pop is inert chemistry. Do not use. |
| v3 | **current (HOF generation)** | `diffusion_subset_cond_expanded_v3_20260425T140941Z` | [`diffusion_expanded_v3.yaml`](../configs/diffusion_expanded_v3.yaml) | 0.0468 | g=5–7 | With rerank pool=1500: HOF q90 rel_MAE **20.4 %**, within-10 % **18 %**, max generated **+341 kcal/mol** (vs target +268). Best HOF performer; weaker than v4-B on D / P. |
| v4 | deprecated | `diffusion_subset_cond_expanded_v4_20260425T160108Z` | [`diffusion_expanded_v4.yaml`](../configs/diffusion_expanded_v4.yaml) | 0.0504 | g=5–7 | Energetic-motif filter (228k rows) + 10× top-5% oversample + per-row loss weighting. Filter hurt the prior; broad regression across all metrics. |
| v4-nf | current (density) | `diffusion_subset_cond_expanded_v4_nofilter_20260425T175119Z` | [`diffusion_expanded_v4_nofilter.yaml`](../configs/diffusion_expanded_v4_nofilter.yaml) | 0.0480 | g=7 | Best on density q90 (11 %) and density q50 (11 %). |
| v5 | deprecated | `diffusion_subset_cond_expanded_v5_20260425T224932Z` | [`diffusion_expanded_v5.yaml`](../configs/diffusion_expanded_v5.yaml) | 0.0483 | — | v4-nf + Min-SNR γ=5. Regressed on most cells; Min-SNR didn't help here. |
| v4-B | superseded by v6 | `diffusion_subset_cond_expanded_v4b_20260426T000541Z` | [`diffusion_expanded_v4b.yaml`](../configs/diffusion_expanded_v4b.yaml) | 0.0482 | g=5–7 | Tier-A/B-only conditioning. Best raw-sampling result (no rerank): D q90 12 %, P q90 26 %, P q50 36 %, HOF q50 140 %. |
| v6 | benchmark-best (single-prop) | v4-B checkpoint + [`rerank_sweep.py`](../scripts/diffusion/rerank_sweep.py) | sampling-time only | — | g=7, pool=200, keep=40 | Single-property rerank. density q90 **2.2 %** (100 %), D q90 **2.1 %** (100 %), P q90 **4.8 %** (100 %), HOF q90 94 % (0 %). |
| v6-multi | **current (joint candidates)** | v4-B checkpoint + [`rerank_multi.py`](../scripts/diffusion/rerank_multi.py) | sampling-time only | — | g=7, pool=400, keep=40 | Joint q90 conditioning + composite reranking. Top candidates jointly satisfy density / D / P (rank-16 hits ρ=1.83, D=8.69, P=33.31 with HOF=+198). Per-property top-40 within-10 %: density 100 %, D 98 %, P 68 %, HOF 0 %. **Use for downstream candidate selection.** |

## Per-version cfg_sweep summaries

- v1: [`cfg_sweep.md`](../experiments/diffusion_subset_cond_expanded_20260425T095335Z/cfg_sweep.md)
- v2: [`cfg_sweep.md`](../experiments/diffusion_subset_cond_expanded_v2_20260425T121727Z/cfg_sweep.md)
- v3: pending

## Common pitfalls when adding a new version

- **Do not delete or rename old experiment dirs.** They are the only true
  archive; the configs in `configs/` are convenient duplicates.
- **Do not edit `configs/diffusion_expanded_vN.yaml` after a run starts.**
  Either copy to `_vN+1.yaml` or use the snapshot in the experiment dir as
  the source of truth.
- **`scripts/diffusion/run_expanded_pipeline.sh` may auto-resume the latest
  expanded run.** Inspect `LATEST_EXP` before re-running, or pass `--resume`
  explicitly. For controlled experimentation prefer running `train.py`
  directly with `--config configs/diffusion_expanded_vN.yaml`.
- **Always copy the previous config when starting a new version.** Diff
  against the previous version so changes are explicit.

## Shared upstream artifacts (do not modify)

These are inputs to all denoiser versions. Re-encoding latents would
invalidate all stored checkpoints because z-space differs.

| Artifact | Path | Source |
|---|---|---|
| LIMO fine-tuned weights | `experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt` | step 8500, 2026-04-24 |
| Latents (raw)   | `data/training/diffusion/latents.pt`          | encode_latents.py against the LIMO ckpt above |
| 3DCNN preds     | `data/training/diffusion/preds_3dcnn.pt`      | run_3dcnn_all.py (smoke 2-fold ensemble) |
| Expanded latents| `data/training/diffusion/latents_expanded.pt` | expand_conditioning.py — used by v1/v2/v3 |

If the LIMO checkpoint or latents are ever re-generated, **bump a major
version** of the denoiser registry and start v4+ against the new latents;
older v1–v3 checkpoints remain valid only against the old latents.
