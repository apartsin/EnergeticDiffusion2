#!/bin/bash
set -eu
pip install --quiet rdkit-pypi selfies
python3 -u m1_sweep.py --v4b_ckpt v4b_best.pt --v3_ckpt v3_best.pt --limo_ckpt limo_best.pt --score_model score_model_v3e.pt --meta_json meta.json --vocab_json vocab.json --pool_per_run ${POOL:-10000} --n_steps 40 --cfg_scale 7.0 --seeds ${SEEDS:-0 1 2} --target_density 1.95 --target_hof 220 --target_d 9.5 --target_p 40
echo "[train] === DONE ==="
