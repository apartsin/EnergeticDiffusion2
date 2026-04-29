#!/bin/bash
set -eu
pip install --quiet rdkit-pypi pandas
python3 -u m4a_moses_metrics.py --results_dir . --labelled_master labelled_master.csv --out_dir results --limit 10000
echo "[train] === DONE ==="
