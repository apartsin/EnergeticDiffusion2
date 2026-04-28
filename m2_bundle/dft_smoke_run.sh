#!/bin/bash
# DFT smoke test runner.
# Pin torch BEFORE installing gpu4pyscf-cuda12x so its CUDA kernels match.
# Exits non-zero with the failing probe number from dft_smoke.py.
set -eu

echo "========================================================="
echo "[smoke-env] $(date -u +%FT%TZ)"
echo "[smoke-env] hostname: $(hostname)"
echo "[smoke-env] uname: $(uname -a)"
if [ -f /etc/os-release ]; then
  . /etc/os-release
  echo "[smoke-env] OS: $PRETTY_NAME"
fi
echo "[smoke-env] DOCKER_IMAGE=${DOCKER_IMAGE:-<unset>}"
echo "[smoke-env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[smoke-env] nvidia-smi:"
nvidia-smi || { echo "[smoke-env] FATAL: nvidia-smi not found / no GPU"; exit 99; }

echo "========================================================="
echo "[smoke-env] python: $(python3 -V 2>&1)"
echo "[smoke-env] pip: $(python3 -m pip --version 2>&1)"

echo "========================================================="
echo "[smoke-pip] pinning torch 2.4.1 + cu124 BEFORE other installs"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

echo "[smoke-pip] installing pyscf, gpu4pyscf-cuda12x, rdkit-pypi, geometric"
python3 -m pip install --quiet pyscf gpu4pyscf-cuda12x rdkit-pypi geometric

echo "[smoke-pip] installed versions:"
python3 -m pip show torch pyscf gpu4pyscf-cuda12x rdkit-pypi geometric 2>/dev/null \
  | grep -E "^(Name|Version|Location):" || true

echo "========================================================="
echo "[smoke-pre] early-fail probe: from gpu4pyscf import dft"
python3 -c "from gpu4pyscf import dft; print('[smoke-pre] gpu4pyscf import OK:', dft.__file__)" \
  || { echo "[smoke-pre] FATAL: gpu4pyscf import failed"; exit 2; }

echo "========================================================="
echo "[smoke-run] launching dft_smoke.py"
mkdir -p results
set +e
python3 -u dft_smoke.py 2>&1 | tee results/dft_smoke.log
RC=${PIPESTATUS[0]}
set -e
echo "[smoke-run] dft_smoke.py exit code: $RC"
if [ "$RC" -ne 0 ]; then
  echo "[smoke-run] FAIL: dft_smoke.py exited $RC (probe number)"
  exit "$RC"
fi
echo "[smoke-run] === ALL PROBES PASSED ==="
ls -la results/
