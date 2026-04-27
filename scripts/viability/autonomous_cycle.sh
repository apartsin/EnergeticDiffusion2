#!/usr/bin/env bash
# Autonomous training/sampling cycle. Run unattended.
# Each cycle:
#   1. wait for previous step's output
#   2. train / sample
#   3. diagnose
#   4. log + iterate
#
# This is a controller invoked via wakeup checks; each call advances the state
# by reading what's complete from disk and running the next step.
set -e
PY=/c/Python314/python
ROOT=E:/Projects/EnergeticDiffusion2
cd "$ROOT"

state_file="logs/cycle_state.txt"
mkdir -p logs experiments

current_state=$(cat "$state_file" 2>/dev/null || echo "init")
echo "[$(date '+%H:%M:%S')] state=$current_state"

advance_to() {
  echo "$1" > "$state_file"
  echo "[$(date '+%H:%M:%S')] -> $1"
}

case "$current_state" in
  init)
    # Wait for labels to be ready
    if [ -f experiments/latent_labels_v1.pt ]; then
      advance_to train_pending
    else
      echo "[wait] labels not yet ready"
    fi
    ;;
  train_pending)
    # Check GPU is free
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q -E "[0-9]+,"; then
      echo "[start] training multi-head score model"
      $PY scripts/viability/train_multihead_latent.py \
          --labels experiments/latent_labels_v1.pt \
          --out experiments/score_model_v1 \
          --epochs 12 --batch 512 \
          > logs/multihead_train.log 2>&1 &
      echo $! > logs/cycle_train.pid
      advance_to train_running
    else
      echo "[wait] GPU busy"
    fi
    ;;
  train_running)
    pid=$(cat logs/cycle_train.pid 2>/dev/null || echo "0")
    if ! kill -0 "$pid" 2>/dev/null && [ -f experiments/score_model_v1/model.pt ]; then
      echo "[done] training complete"
      advance_to diagnose_pending
    else
      echo "[wait] training in progress (pid=$pid)"
    fi
    ;;
  diagnose_pending)
    echo "[start] diagnosing score model"
    $PY scripts/viability/diagnose_score_model.py \
        --model experiments/score_model_v1/model.pt \
        > logs/score_diagnose.log 2>&1
    advance_to diagnose_done
    ;;
  diagnose_done)
    echo "[done] all autonomous-cycle steps complete; check experiments/score_model_v1/"
    ;;
esac
