#!/bin/bash
set -eu
echo "[setup] Installing system deps ..."
apt-get update -qq && apt-get install -y --no-install-recommends libxrender1 libxext6 libsm6 libgl1 2>&1 | tail -3
echo "[setup] Installing AiZynthFinder (core, no [all]) ..."
pip install --quiet aizynthfinder rdkit-pypi
echo "[setup] Downloading public USPTO model ..."
mkdir -p aizynth_data && cd aizynth_data
download_public_data .
cd ..
echo "[setup] Running retrosynthesis ..."
mkdir -p results
python3 -u aizynth_run.py --config aizynth_data/config.yml --out results/aizynth_results.json
echo "[train] === DONE ==="
