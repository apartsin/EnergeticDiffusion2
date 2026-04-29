#!/bin/bash
set -eu
pip install --quiet rdkit-pypi pyscf gpu4pyscf-cuda12x geometric
python3 -u m2_dft_pipeline.py --smiles m5_smiles.json --results results --anchors
echo "[train] === DONE ==="
