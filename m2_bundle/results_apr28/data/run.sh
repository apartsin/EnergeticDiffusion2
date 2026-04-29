#!/bin/bash
set -eu
echo "[setup] installing deps"
pip install --quiet rdkit-pypi pyscf gpu4pyscf-cuda12x geometric
python3 -c "import torch, pyscf; from gpu4pyscf import dft; print('[setup] OK', 'torch', torch.__version__, 'pyscf', pyscf.__version__)"
echo "[setup] === RUN ==="
mkdir -p results
python3 -u dft_pipeline.py --smiles dft_5mol_targets.json --results results
echo "[train] === DONE ==="
