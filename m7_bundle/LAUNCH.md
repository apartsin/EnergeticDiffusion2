# M7 100k-pool launch guide

## What this runs

5 new sampling lanes (seeds 3-4 + new guidance configs) producing 100k raw SMILES total,
fused and ranked by the same Pareto pipeline as M6 but with a wider Tanimoto novelty
window [0.15, 0.65] to surface more structurally diverse leads.

Lane summary:
| Lane | Config | CFG w | Seed | Notes |
|------|--------|-------|------|-------|
| L0 | C0 unguided | 7.0 | 3 | Extend M6 seed coverage |
| L1 | C0 unguided | 7.0 | 4 | Extend M6 seed coverage |
| L2 | C4 high-w | 12.0 | 3 | Stronger property conditioning |
| L3 | C5 viab=4.0 | 7.0 | 3 | Strong viability push |
| L4 | C6 hazard=3.0+viab=0.5 | 7.0 | 3 | Hazard-dominant steering |

## Step 1: Upload to vast.ai

Use the gpu2vast skill or upload manually:

```bash
# From project root
tar czf m7_bundle.tar.gz m7_bundle/ m1_bundle/
# Upload tar to vast.ai instance, extract at /workspace/
```

The script reuses all model files from the M6 run — copy from the existing vast.ai instance
or re-upload the checkpoints. Required files (same as M6):
- v4b checkpoint (DGLD-H)
- v3 checkpoint (DGLD-P)
- LIMO checkpoint
- score model (v3f)
- meta.json + vocab.json

## Step 2: Run sampling on RTX 4090

```bash
cd /workspace
python m7_bundle/m7_100k.py \
  --v4b_ckpt checkpoints/v4b_ckpt.pt \
  --v3_ckpt checkpoints/v3_ckpt.pt \
  --limo_ckpt checkpoints/limo_ft.pt \
  --score_model checkpoints/score_model_v3f.pt \
  --meta_json checkpoints/meta.json \
  --vocab_json checkpoints/vocab.json \
  --pool_per_run 10000 \
  --results_dir results/m7
```

Expected wall time: ~30 min on RTX 4090 (5 lanes x 2 denoisers x 10k samples).

## Step 3: Download results

```bash
# Download results/m7/*.txt and results/m7/m7_summary.json
```

## Step 4: Postprocess locally

```bash
# From project root, with RTX 2060 (CPU-only scoring is fine)
python m7_bundle/m7_post.py \
  --results_dir results/m7 \
  --meta_json checkpoints/meta.json \
  --labelled_master data/raw/energetic_external/EMDP/Data/labelled_master.csv \
  --smoke_model data/raw/energetic_external/EMDP/Data/smoke_model \
  --tanimoto_min 0.15 \
  --tanimoto_max 0.65 \
  --top_n 200 \
  --out_json experiments/m7_100k_postprocess.json \
  --out_md experiments/m7_100k_postprocess.md
```

## What to look for in results

1. New scaffold families not in M6 top-100 (the wider Tanimoto window [0.15,0.65] captures
   more borderline-novel structures that [0.20,0.55] rejected)
2. D > 9.5 km/s from seeds 3/4 (independent latent-space regions vs seeds 0-2)
3. Hazard-steered leads from L4 may show lower h50_BDE score indicating reduced sensitivity
4. Compare L2 (high-w=12) vs L0/L1 (w=7) to see if stronger CFG improves top-1 D
