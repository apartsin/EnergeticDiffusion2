#!/bin/bash
# Combined E1-pool40k + AiZynth-deep job. Each step writes to results/
# even if a later step fails (set -u, no -e at the top).
set -u

cd /workspace/data
mkdir -p results

echo "===================================="
echo "[STEP 1/2] E1 anneal-clamp pool=40k"
echo "===================================="
T0=$SECONDS

pip install --quiet rdkit-pypi selfies 2>&1 | tail -3 || true

set +e
python3 -u m1_anneal_clamp.py \
    --v4b_ckpt v4b_best.pt --v3_ckpt v3_best.pt --limo_ckpt limo_best.pt \
    --score_model score_model_v3e.pt --meta_json meta.json --vocab_json vocab.json \
    --pool_per_run 40000 --results_dir results 2>&1 | tee results/e1_pool40k_log.txt
E1_EXIT=$?
set -e

echo "[STEP 1 done in $((SECONDS - T0))s, exit=$E1_EXIT]"
ls -la results/m1_anneal_clamp_*.txt 2>&1 | tail -5 || echo "  (no E1 output files)"

echo
echo "===================================="
echo "[STEP 2/2] AiZynth L4+L5 deep budget"
echo "===================================="
T1=$SECONDS

set +e
apt-get update -qq 2>&1 | tail -2
apt-get install -y --no-install-recommends libxrender1 libxext6 libsm6 libgl1 2>&1 | tail -3
pip install --quiet aizynthfinder 2>&1 | tail -3
mkdir -p aizynth_data && cd aizynth_data && download_public_data . 2>&1 | tail -5; cd ..

python3 -u aizynth_run.py \
    --targets smiles_targets_L4L5.json \
    --config aizynth_data/config.yml \
    --out results/aizynth_L4L5_deep.json \
    --max_iterations 1000 --time_limit 1800 2>&1 | tee results/aizynth_deep_log.txt
AIZ_EXIT=$?
set -e

echo "[STEP 2 done in $((SECONDS - T1))s, exit=$AIZ_EXIT]"

# Final summary
{
  echo "{\"step1_e1\":{\"exit\":$E1_EXIT,\"elapsed_s\":$((SECONDS - T0))},\"step2_aizynth\":{\"exit\":$AIZ_EXIT}}"
} > results/combo_status.json

echo "[train] === DONE ==="
