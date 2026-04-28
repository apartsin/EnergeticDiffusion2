#!/bin/bash
set -eu
echo "[setup] Installing system deps ..."
apt-get update -qq && apt-get install -y --no-install-recommends libxrender1 libxext6 libsm6 libgl1 2>&1 | tail -3
echo "[setup] Installing AiZynthFinder ..."
pip install --quiet aizynthfinder rdkit-pypi pyyaml
echo "[setup] Downloading public USPTO model ..."
mkdir -p aizynth_data && cd aizynth_data
download_public_data .
cd ..
echo "[setup] Patching config to relax expansion-policy cutoffs ..."
python3 relax_config.py aizynth_data/config.yml
echo "[setup] Running retrosynthesis (relaxed: 11 leads, cutoff_cumulative=1.0, cutoff_number=500) ..."
mkdir -p results
python3 -u aizynth_run.py --targets smiles_targets_relaxed.json --config aizynth_data/config.yml --out results/aizynth_results_relaxed.json
echo "[train] === DONE ==="
