# vast.ai jobs — parallel batch

Four independent training jobs written to vast.ai conventions ([train]
prefix, TensorBoard, GPU assert, etc.). Designed to run concurrently after
the smoke test passes.

| Job | Script | GPU | ETA | Cost | Tier |
|---|---|---|---|---|---|
| A | `job_a_molmim_hybrid.py` | A100 40 GB | ~3 h | ~$1.50 | 1 |
| B | `job_b_dit_denoiser.py` | RTX 4090 | ~4 h | ~$1.60 | 1 |
| C | `job_c_massive_rerank.py` | RTX 4090 | ~1.5 h | ~$0.60 | 1 |
| D | `job_d_norm_retrofit.py` | RTX 4090 | ~2 h | ~$0.80 | 3 |

## Files needed per job (uploaded to R2 by gpu_runner.py)

### A — MolMIM hybrid
- `models/molmim/molmim_v1.3/molmim_70m_24_3.nemo` (269 MB)
- `vast_jobs/smiles_cond_bundle.pt` (51 MB)
- `scripts/diffusion/train.py` → renamed `denoiser_train.py`
- `scripts/diffusion/model.py`
- a 512-d denoiser config: `configs/diffusion_expanded_v9_512d.yaml` (TODO: write)

### B — DiT denoiser
- `data/training/diffusion/latents_trustcond.pt` (1.6 GB)

### C — massive rerank
- `experiments/.../v3/checkpoints/best.pt` (700 MB)
- `experiments/.../v4b/checkpoints/best.pt` (700 MB)
- `experiments/limo_ft_*/checkpoints/best.pt` (140 MB)
- `data/raw/energetic_external/EMDP/Data/smoke_model/` tarred (700 MB)
- `data/training/diffusion/latents_expanded.pt` (1.6 GB)
- our scripts dir tarred

### D — norm retrofit
- `data/training/diffusion/latents_trustcond.pt` (1.6 GB)
- denoiser train.py + model.py
- `configs/diffusion_expanded_v4b.yaml` adapted to point at normalised latents

## Launch commands

```bash
RUNNER="/c/Users/apart/Projects/claude-skills/gpu2vast/gpu_runner.py"

# Job A
PYTHONIOENCODING=utf-8 /c/Python314/python "$RUNNER" run \
  --script "python job_a_molmim_hybrid.py" \
  --data vast_jobs/job_a_molmim_hybrid.py \
         vast_jobs/smiles_cond_bundle.pt \
         models/molmim/molmim_v1.3/molmim_70m_24_3.nemo \
         scripts/diffusion/train.py \
         scripts/diffusion/model.py \
         configs/diffusion_expanded_v9_512d.yaml \
  --gpu A100 --max-price 2.00 --max-hours 4

# Job B
PYTHONIOENCODING=utf-8 /c/Python314/python "$RUNNER" run \
  --script "python job_b_dit_denoiser.py" \
  --data vast_jobs/job_b_dit_denoiser.py \
         data/training/diffusion/latents_trustcond.pt \
  --gpu RTX_4090 --max-price 0.80 --max-hours 5

# Job C — needs tarballs prepared first
tar -czf rerank_code.tar.gz scripts/diffusion scripts/vae external/LIMO/sascorer.py external/scscore
tar -cf smoke_model_dir.tar -C data/raw/energetic_external/EMDP/Data smoke_model
PYTHONIOENCODING=utf-8 /c/Python314/python "$RUNNER" run \
  --script "python job_c_massive_rerank.py" \
  --data vast_jobs/job_c_massive_rerank.py \
         experiments/diffusion_subset_cond_expanded_v3_*/checkpoints/best.pt \
         experiments/diffusion_subset_cond_expanded_v4b_*/checkpoints/best.pt \
         data/training/diffusion/latents_expanded.pt \
         experiments/limo_ft_energetic_*/checkpoints/best.pt \
         smoke_model_dir.tar rerank_code.tar.gz \
  --gpu RTX_4090 --max-price 0.80 --max-hours 2

# Job D
PYTHONIOENCODING=utf-8 /c/Python314/python "$RUNNER" run \
  --script "python job_d_norm_retrofit.py" \
  --data vast_jobs/job_d_norm_retrofit.py \
         data/training/diffusion/latents_trustcond.pt \
         scripts/diffusion/train.py \
         scripts/diffusion/model.py \
         configs/diffusion_expanded_v4b.yaml \
  --gpu RTX_4090 --max-price 0.80 --max-hours 3
```

## Status

- Smoke test (skill validation): in flight
- Job A-D: scripts written, NOT launched. Awaiting smoke verification + final approval.
