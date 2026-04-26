#!/usr/bin/env bash
# Run the full expanded-conditioning pipeline once 3DCNN has completed.
#
# Stages (each only runs if the previous artefact doesn't already exist):
#   1. expand_conditioning.py   → latents_expanded.pt
#   2. train.py (config expanded) → new experiment dir
#   3. evaluate.py on the new experiment
#   4. report.py on the new experiment
#
# Resumable: stop & restart is safe; each step checks for its output.

set -u
cd "$(dirname "$0")/../.."

BASE=$(pwd)
PY="/c/Python314/python"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

LATENTS_ORIG="$BASE/data/training/diffusion/latents.pt"
PREDS_3DCNN="$BASE/data/training/diffusion/preds_3dcnn.pt"
LATENTS_EXPANDED="$BASE/data/training/diffusion/latents_expanded.pt"

if [ ! -f "$PREDS_3DCNN" ]; then
  echo "ERROR: $PREDS_3DCNN missing. Run scripts/diffusion/run_3dcnn_all.py first."
  exit 1
fi

echo "============================================================"
echo "Stage 1: expand conditioning"
echo "============================================================"
if [ ! -f "$LATENTS_EXPANDED" ]; then
  # K-J imputation disabled: 3DCNN already predicts D and P for every
  # molecule, making K-J imputation mostly redundant (fires only on 3DCNN
  # failures, ~1-3% of rows). Drop it for simplicity.
  "$PY" scripts/diffusion/expand_conditioning.py \
    --latents_in "$LATENTS_ORIG" \
    --preds_3dcnn "$PREDS_3DCNN" \
    --out "$LATENTS_EXPANDED" \
    --tier_d_weight 0.7
else
  echo "  [skip] $LATENTS_EXPANDED exists"
fi

echo ""
echo "============================================================"
echo "Stage 2: train denoiser (expanded conditioning)"
echo "============================================================"
# Find whether a prior expanded-run is present; if so, resume
LATEST_EXP=$(ls -1dt experiments/diffusion_subset_cond_expanded_* 2>/dev/null | head -1)
if [ -n "$LATEST_EXP" ] && [ -f "$LATEST_EXP/checkpoints/last.pt" ]; then
  echo "  Resuming from $LATEST_EXP"
  "$PY" scripts/diffusion/train.py --config configs/diffusion_expanded.yaml \
        --resume "$LATEST_EXP"
else
  "$PY" scripts/diffusion/train.py --config configs/diffusion_expanded.yaml
fi

# Find the experiment dir after training
EXP=$(ls -1dt experiments/diffusion_subset_cond_expanded_* | head -1)
echo "  experiment: $EXP"

echo ""
echo "============================================================"
echo "Stage 3: evaluate"
echo "============================================================"
if [ ! -f "$EXP/eval_results.json" ]; then
  "$PY" scripts/diffusion/evaluate.py --exp "$EXP" --n_uncond 300 --n_cond 80
else
  echo "  [skip] $EXP/eval_results.json exists"
fi

echo ""
echo "============================================================"
echo "Stage 4: HTML report"
echo "============================================================"
"$PY" scripts/diffusion/report.py --exp "$EXP"

echo ""
echo "============================================================"
echo "DONE."
echo "  expanded latents: $LATENTS_EXPANDED"
echo "  experiment:       $EXP"
echo "  report:           $EXP/report.html"
echo "============================================================"
