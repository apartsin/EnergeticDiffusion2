#!/bin/bash
set -eu
echo "[setup] installing deps"
pip install --quiet rdkit-pypi pyscf gpu4pyscf-cuda12x geometric
python3 -c "import torch, pyscf; print('[setup] torch', torch.__version__, 'pyscf', pyscf.__version__)"
python3 -c "from gpu4pyscf import dft; print('[setup] gpu4pyscf available')"
echo "[setup] === RUN ==="
python3 -u m2_dft_pipeline.py --smiles m2_smiles.json --results results --anchors
echo "[train] === DONE ==="
