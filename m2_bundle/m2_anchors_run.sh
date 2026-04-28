#!/bin/bash
set -eu
echo "[setup] installing deps (pin torch BEFORE gpu4pyscf to avoid ABI mismatch)"
# Image is pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime; torch already present.
python3 -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install --quiet rdkit-pypi pyscf gpu4pyscf-cuda12x geometric
python3 -c "from gpu4pyscf import dft; print('[setup] gpu4pyscf imported OK')"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE — abort'"
echo "[setup] === RUN ==="
mkdir -p results
# Move uploaded RDX/TATB and atom_refs into results/ so the driver can reuse them
[ -f m2_atom_refs.json ] && cp m2_atom_refs.json results/m2_atom_refs.json
[ -f m2_lead_RDX.json ] && cp m2_lead_RDX.json results/m2_lead_RDX.json
[ -f m2_lead_TATB.json ] && cp m2_lead_TATB.json results/m2_lead_TATB.json
[ -f m2_summary.json ] && cp m2_summary.json results/m2_summary.json
# Also copy each per-lead JSON if present (needed for K-J formula lookups)
for f in m2_lead_*.json; do [ -f "$f" ] && cp "$f" results/"$f" || true; done
python3 -u m2_anchors_extension.py --results results
echo "[train] === DONE ==="
