#!/usr/bin/env bash
# Sweep s_SC scale with v3e (SC head working). Keep SA at 0.
set -e
PY=/c/Python314/python
EXP_V3=experiments/diffusion_subset_cond_expanded_v3_20260425T140941Z
EXP_V4B=experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z
SCORE=experiments/score_model_v3e/model.pt
RF=experiments/viability_rf_v2_hardneg/model.joblib
OUT=$EXP_V4B
mkdir -p logs

SWEEP=(
  "v3e_sc0:1.0:0.3:0:0"
  "v3e_sc15:1.0:0.3:0:0.15"
  "v3e_sc30:1.0:0.3:0:0.30"
)
for entry in "${SWEEP[@]}"; do
  IFS=: read -r name v s a sc <<< "$entry"
  out_md="$OUT/sweep_${name}.md"
  log="logs/sweep_${name}.log"
  echo "[$(date '+%H:%M:%S')] === $name (viab=$v sens=$s sa=$a sc=$sc) ==="
  $PY scripts/diffusion/joint_rerank.py \
      --exp_v3 "$EXP_V3" --exp_v4b "$EXP_V4B" \
      --score_model "$SCORE" --guide_viab "$v" --guide_sens "$s" --guide_sa "$a" --guide_sc "$sc" \
      --cfg 7 --n_pool_each 2500 --n_keep 400 \
      --target_density 1.95 --target_d 9.5 --target_p 40 --target_hof 220 \
      --hard_sa 5.0 --hard_sc 3.5 --tanimoto_min 0.20 --tanimoto_max 0.55 \
      --require_neutral --with_chem_filter --with_feasibility \
      --out "$out_md" > "$log" 2>&1
  $PY scripts/diffusion/rerank_v2.py \
      --in "$out_md" --out "${out_md%.md}_v2.md" \
      --viability_model "$RF" --limit 400 >> "$log" 2>&1
  echo "[$(date '+%H:%M:%S')] $name -> ${out_md%.md}_v2.md"
done
