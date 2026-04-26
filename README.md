# EnergeticDiffusion2

Latent-diffusion pipeline for **novel energetic-material discovery**.

Combines a fine-tuned LIMO VAE, a tier-aware conditional diffusion
denoiser (FiLM-ResNet on 1024-d latents), 3DCNN-ensemble validation,
multi-property reranking, SA + SC feasibility filters, and a chemistry
sanity layer to produce ranked SMILES candidates whose predicted ρ, D,
P, HOF reach or beat current SOTA energetic materials (CL-20 / TKX-50 /
ICM-101 class).

## Quickstart

```bash
# clone + setup
git clone https://github.com/apartsin/EnergeticDiffusion2.git
cd EnergeticDiffusion2
# Heavy artefacts (checkpoints, latents) are NOT in the repo.
# After cloning, fetch the release-hosted production checkpoints:
bash scripts/download_artifacts.sh

# run the joint-rerank breakthrough discovery driver
python scripts/diffusion/joint_rerank.py \
    --exp_v4b experiments/diffusion_subset_cond_expanded_v4b_<ts> \
    --exp_v3  experiments/diffusion_subset_cond_expanded_v3_<ts> \
    --cfg 7 --n_pool_each 1500 --n_keep 80 \
    --target_density 1.95 --target_d 9.5 --target_p 40 --target_hof 220 \
    --hard_sa 5.0 --hard_sc 3.5 \
    --tanimoto_min 0.20 --tanimoto_max 0.55 \
    --require_neutral --with_chem_filter --with_feasibility
```

## Project layout

```
configs/   YAML training configs for VAE, denoisers (v1 → v6), guidance
scripts/
  vae/         LIMO model + finetune trainer + motif-augmented dataset builder
  diffusion/   denoiser, samplers (vanilla + feasibility-guided), reranks
  guidance/    SA/SC predictor (clean & time-conditional), property heads
  diagnostics/ D1–D16 root-cause + distribution diagnostics
  simulation/  Psi4 HOF wrapper for spot-checks
docs/      architecture, training, diagnostics, breakthrough plan
data/      gitignored (382 k SMILES, latents, smoke-model preds)
experiments/  gitignored (training runs, sweeps, rerank outputs)
external/  gitignored (LIMO + scscore upstream code)
```

## Key documents

- [`docs/production_overview.md`](docs/production_overview.md) — full pipeline
- [`docs/denoiser_versions.md`](docs/denoiser_versions.md) — v1 → v6 registry, what each version is current for
- [`docs/diagnostics_plan.md`](docs/diagnostics_plan.md) — 10 diagnostics catalog
- [`docs/diag_summary.md`](docs/diag_summary.md) — latest D1–D15 results
- [`docs/breakthrough_experiment.md`](docs/breakthrough_experiment.md) — Path A / Path B / class A-B-C
- [`docs/feasibility_guidance_plan.md`](docs/feasibility_guidance_plan.md) — SA + SC integration
- [`docs/c2c_diagnostic_rounds.md`](docs/c2c_diagnostic_rounds.md) — c2c bounded by LIMO ceiling
- [`docs/limo_v2_plan.md`](docs/limo_v2_plan.md) — motif-rich LIMO continuation
- [`docs/improvements_deep_think.md`](docs/improvements_deep_think.md) — ranked options for future work

## Production at a glance

Two-version setup (rerank pool=1500 + chem_filter + feasibility caps):

| Goal | Use |
|---|---|
| Joint high (ρ, D, P) candidates | **v4-B** (`diffusion_subset_cond_expanded_v4b_*`) |
| HOF-targeted generation | **v3** (`diffusion_subset_cond_expanded_v3_*`) |
| Multi-property breakthrough discovery | **`joint_rerank.py`** (combines both) |

Headline numbers (q90, validated by 3DCNN ensemble, rerank pool=1500):

| Metric | v4-B | v3 |
|---|---|---|
| density q90 within-10 % | **100 %** | 82 % |
| D q90 within-10 % | **100 %** | 57 % |
| P q90 within-10 % | **100 %** | 22 % |
| HOF q90 within-10 % | 2 % | **18 %** |
| HOF q90 max produced | +257 | **+341 kcal/mol** |
| novelty (PubChem + 382 k internal) | 100 % | 100 % |

## Top breakthrough leads (joint rerank, 2026-04-26)

| ρ (g/cm³) | HOF (kcal/mol) | D (km/s) | P (GPa) | SA | SMILES |
|---|---|---|---|---|---|
| 1.91 | +163 | 9.20 | 36.7 | 4.55 | `O=[N+]([O-])C=NO[N+](=O)[O-]` |
| 1.90 | +114 | 9.33 | 37.9 | 4.61 | `O=[N+]([O-])N=CO[N+](=O)[O-]` |
| 1.89 | +185 | 9.04 | 36.5 | 4.80 | `O=[N+]([O-])C=C=N[N+](=O)[O-]` |
| 1.86 | +172 | 8.82 | 33.3 | 4.47 | `NC(C=N[N+](=O)[O-])=NC(=C[N+](=O)[O-])[N+](=O)[O-]` |

All novel (Tanimoto 0.30–0.55 to nearest training row), no PubChem hit.

## Hardware

Training + inference designed for an **RTX 2060 (6 GB VRAM)** with fp16.
Diffusion / VAE training uses `torch.cuda.amp` for fit. 3DCNN smoke-model
inference and Psi4 HOF spot-checks are CPU-bound.

## License

This is research code. Do not use without a clear authorisation context
(academic project, defensive security research, pen-test). Generation of
energetic-material candidates is an active research area; downstream
synthesis would require appropriate institutional review.

## Status

This repo is a **research-pipeline snapshot**, not a polished library.
Heavy artefacts (LIMO + denoiser checkpoints) live in GitHub Releases.
See [`docs/backups/todo_snapshot_<date>.md`](docs/backups/) for the live
state of pending work.
