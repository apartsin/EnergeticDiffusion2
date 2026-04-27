#!/bin/bash
set -eu
echo "[setup] Python $(python3 --version)"
echo "[setup] Pinning huggingface_hub<0.20 to fix NeMo 1.22 ModelFilter import ..."
pip install -q "huggingface_hub<0.20" 2>&1 | tail -3
echo "[setup] Verifying NeMo can import after the pin ..."
python3 -c "from huggingface_hub import ModelFilter; print('  ModelFilter OK')" 2>&1 | tail -3
python3 -c "import nemo; print(f'  nemo {nemo.__version__}')" 2>&1 | tail -3
echo "[setup] Running smoke_molmim.py ..."
mkdir -p results
python3 -u smoke_molmim.py 2>&1 | tee results/smoke_stdout.log
echo "[train] === DONE ==="
