# Run Status (2026-04-24)

## Request

Run the entire experiment end-to-end.

## Result

Not fully completed for the MDD spec in `docs/energetic_materials_generation_project_mdd.md`.

Reason:
- The repository does not include a native implementation of the spec pipeline stages (SELFIES VAE, latent diffusion training, generation/filtering/ranking/evaluation runner).
- Available executable code is mostly upstream external projects under `data/raw/energetic_external/*`.

## What Was Executed Successfully

A complete **train -> predict** smoke cycle was run using upstream EMDP + `unimol_tools`:

- Installed dependency:
  - `python -m pip install unimol-tools`
- Ran a smoke training + prediction script (1 epoch, 2-fold scaffold split) in:
  - `data/raw/energetic_external/EMDP/Data`
- Completed successfully and produced model + prediction artifacts.

### Key outputs

- `data/raw/energetic_external/EMDP/Data/smoke_model/config.yaml`
- `data/raw/energetic_external/EMDP/Data/smoke_model/model_0.pth`
- `data/raw/energetic_external/EMDP/Data/smoke_model/model_1.pth`
- `data/raw/energetic_external/EMDP/Data/smoke_predictions/test_set.predict.0.csv`
- `data/raw/energetic_external/EMDP/Data/smoke_predictions/test_metric.json`

## What Failed Before Fixes

- `EMDP/run.py` initially failed because `unimol_tools` was missing.
- After install, upstream `run.py` failed on unsupported metric `rmse` (in this `unimol_tools` version for multilabel regression).
- Path expectation issue (`train_set.csv`) required running from `EMDP/Data`.

## Remaining Blockers for Full Spec E2E

To run the **full** MDD experiment, the following are still missing in this repo:

- VAE training code (`SELFIES -> z -> SELFIES`)
- Latent export pipeline (`latent_dataset.parquet` from trained VAE)
- Property predictor training code aligned to MDD artifacts
- Diffusion training stages A/B/C
- Generation, filtering, novelty/diversity scoring, ranking, and final evaluation report pipeline

Expected spec artifacts not produced yet:
- `vae_checkpoint.pt`
- `vae_metrics.csv`
- `latent_dataset.parquet`
- `property_predictor_checkpoints/`
- `property_predictor_metrics.csv`
- `diffusion_checkpoint.pt`
- `generated_candidates.csv`
- `filtered_candidates.csv`
- `ranked_candidates.csv`
- `evaluation_report.md`

