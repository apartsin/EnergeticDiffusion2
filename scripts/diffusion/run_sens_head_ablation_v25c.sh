#!/usr/bin/env bash
# 2.5c: matched-compute ablation comparing v3e (heuristic sens head)
# against v3e_h50 (literature-grounded sens head from 306 (SMILES, h50) pairs).
#
# Three conditions, identical pool size, identical CFG, identical reranker:
#   A. unguided           (no score_model)
#   B. v3e         sens=0.3, viab=1.0
#   C. v3e_h50     sens=0.3, viab=1.0
#
# Compares top-1 composite + top-1 D + scaffold class.
set -e
PY=/c/Python314/python
EXP_V3=experiments/diffusion_subset_cond_expanded_v3_20260425T140941Z
EXP_V4B=experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z
RF=experiments/viability_rf_v2_hardneg/model.joblib
OUT=$EXP_V4B
mkdir -p logs

POOL=1500   # 2 pools = 3000 samples per condition
KEEP=200

declare -a SWEEP=(
  "A_unguided::1.0:0.3:0:0"
  "B_v3e:experiments/score_model_v3e/model.pt:1.0:0.3:0:0"
  "C_v3e_h50:experiments/score_model_v3e_h50/model.pt:1.0:0.3:0:0"
)

for entry in "${SWEEP[@]}"; do
  IFS=: read -r name score v s a sc <<< "$entry"
  out_md="$OUT/sens_ablation_25c_${name}.md"
  log="logs/sens_ablation_25c_${name}.log"
  echo "[$(date '+%H:%M:%S')] === $name (score=${score:-none} v=$v s=$s) ==="
  if [ -z "$score" ]; then
    $PY scripts/diffusion/joint_rerank.py \
        --exp_v3 "$EXP_V3" --exp_v4b "$EXP_V4B" \
        --cfg 7 --n_pool_each $POOL --n_keep $KEEP \
        --target_density 1.95 --target_d 9.5 --target_p 40 --target_hof 220 \
        --hard_sa 5.0 --hard_sc 3.5 --tanimoto_min 0.20 --tanimoto_max 0.55 \
        --require_neutral --with_chem_filter --with_feasibility \
        --out "$out_md" > "$log" 2>&1
  else
    $PY scripts/diffusion/joint_rerank.py \
        --exp_v3 "$EXP_V3" --exp_v4b "$EXP_V4B" \
        --score_model "$score" --guide_viab "$v" --guide_sens "$s" --guide_sa "$a" --guide_sc "$sc" \
        --cfg 7 --n_pool_each $POOL --n_keep $KEEP \
        --target_density 1.95 --target_d 9.5 --target_p 40 --target_hof 220 \
        --hard_sa 5.0 --hard_sc 3.5 --tanimoto_min 0.20 --tanimoto_max 0.55 \
        --require_neutral --with_chem_filter --with_feasibility \
        --out "$out_md" > "$log" 2>&1
  fi
  $PY scripts/diffusion/rerank_v2.py \
      --in "$out_md" --out "${out_md%.md}_v2.md" \
      --viability_model "$RF" --limit $KEEP >> "$log" 2>&1
  echo "[$(date '+%H:%M:%S')] $name -> ${out_md%.md}_v2.md"
done

echo
echo "=== Summary ==="
for entry in "${SWEEP[@]}"; do
  IFS=: read -r name _ _ _ _ _ <<< "$entry"
  out_v2="$OUT/sens_ablation_25c_${name}_v2.md"
  if [ -f "$out_v2" ]; then
    head -8 "$out_v2"
    echo "---"
  fi
done
