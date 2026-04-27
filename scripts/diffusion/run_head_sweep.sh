#!/usr/bin/env bash
# Head-weight ablation sweep for multi-head classifier guidance.
# Each condition is pool=5k (2.5k per denoiser), cfg=7. Outputs an MD per
# condition. We post-rerank with v2 strict + the latest perf-gated/scaffold-aware
# composite to compare.
set -e
PY=/c/Python314/python
EXP_V3=experiments/diffusion_subset_cond_expanded_v3_20260425T140941Z
EXP_V4B=experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z
SCORE=experiments/score_model_v2/model.pt
RF=experiments/viability_rf_v2_hardneg/model.joblib
OUT_DIR=$EXP_V4B
mkdir -p logs

SWEEP=(
  "ungUNG:0:0:0"
  "low:0.5:0.2:0"
  "default:1.0:0.3:0"
  "viabhi:1.5:0.3:0"
  "senshi:1.0:0.6:0"
  "withSA:1.0:0.3:0.3"
)

for entry in "${SWEEP[@]}"; do
  name=$(echo "$entry" | cut -d: -f1)
  v=$(echo "$entry" | cut -d: -f2)
  s=$(echo "$entry" | cut -d: -f3)
  a=$(echo "$entry" | cut -d: -f4)
  out_md="$OUT_DIR/sweep_head_${name}.md"
  log="logs/sweep_head_${name}.log"
  echo "[$(date '+%H:%M:%S')] === $name (viab=$v sens=$s sa=$a) ==="
  if [ "$v" = "0" ] && [ "$s" = "0" ] && [ "$a" = "0" ]; then
    SCORE_ARGS=""
  else
    SCORE_ARGS="--score_model $SCORE --guide_viab $v --guide_sens $s --guide_sa $a"
  fi
  $PY scripts/diffusion/joint_rerank.py \
      --exp_v3 "$EXP_V3" --exp_v4b "$EXP_V4B" \
      $SCORE_ARGS \
      --cfg 7 --n_pool_each 2500 --n_keep 400 \
      --target_density 1.95 --target_d 9.5 --target_p 40 --target_hof 220 \
      --hard_sa 5.0 --hard_sc 3.5 --tanimoto_min 0.20 --tanimoto_max 0.55 \
      --require_neutral --with_chem_filter --with_feasibility \
      --out "$out_md" \
      > "$log" 2>&1
  # v2 strict
  $PY scripts/diffusion/rerank_v2.py \
      --in "$out_md" --out "${out_md%.md}_v2.md" \
      --viability_model "$RF" --limit 400 \
      >> "$log" 2>&1
  echo "[$(date '+%H:%M:%S')] $name DONE -> ${out_md%.md}_v2.md"
done

echo "[$(date '+%H:%M:%S')] all sweep conditions complete"
