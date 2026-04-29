#!/bin/bash
set -eu
pip install --quiet rdkit-pypi selfies
python3 -u m1_grid.py --v4b_ckpt v4b_best_fp16.pt --v3_ckpt v3_best_fp16.pt --limo_ckpt limo_best_fp16.pt --score_model score_model_v3f.pt --meta_json meta.json --vocab_json vocab.json --pool_per_run 10000 --seed 0
echo "[train] === DONE ==="
