#!/bin/bash
set -eu
pip install --quiet rdkit-pypi pandas
mkdir -p results
python3 -u smiles_lstm_baseline.py --corpus corpus.csv --epochs 5 --batch 128 --max_len 120 --hidden 512 --layers 2 --n_samples 10000 --temperature 1.0
echo "[train] === DONE ==="
