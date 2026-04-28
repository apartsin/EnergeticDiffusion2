#!/bin/bash
set -eu
echo "[setup] pod-2 hedge for 6-anchor calibration; HMX-first; results_pod2/ namespace"
python3 -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install --quiet rdkit-pypi pyscf gpu4pyscf-cuda12x geometric
python3 -c "from gpu4pyscf import dft; print('[setup] gpu4pyscf imported OK')"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE - abort'"
echo "[setup] === RUN ==="
mkdir -p results_pod2
[ -f m2_atom_refs.json ] && cp m2_atom_refs.json results_pod2/m2_atom_refs.json
python3 -u m2_anchors_pod2.py --results results_pod2
echo "[train] === DONE ==="
