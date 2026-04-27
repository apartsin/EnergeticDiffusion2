#!/bin/bash
set -eu
echo "[setup] Installing deps ..."
pip install --quiet rdkit-pypi pandas selfies
pip install --quiet unimol-tools 2>&1 || pip install --quiet unimol-tools --no-deps
echo "[setup] Extracting smoke_model ..."
tar xzf smoke_model.tar.gz --no-same-owner --no-same-permissions
ls -la smoke_model/ 2>&1 | head -3
echo "[setup] Running postprocess ..."
mkdir -p results
python3 -u m6_post.py --results_dir . --meta_json meta.json --labelled_master labelled_master.csv --smoke_model smoke_model --top_n 100 --limit_per_file 20000 --out_json results/m6_post.json --out_md results/m6_post.md
echo "[train] === DONE ==="
