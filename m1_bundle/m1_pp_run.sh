#!/bin/bash
set -eu
pip install --quiet rdkit-pypi selfies
python3 -u m1_pp.py --v4b_ckpt v4b_best.pt --v3_ckpt v3_best.pt --limo_ckpt limo_best.pt --score_model score_model_v3f.pt --meta_json meta.json --vocab_json vocab.json --pool_per_run 10000 --seed 0 --target_density 1.95 --target_hof 220 --target_d 9.5 --target_p 40
echo "[train] === DONE ==="
