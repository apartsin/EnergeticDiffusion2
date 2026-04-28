#!/bin/bash
# m2_anchors_run_vast.sh — entry point for m2-anchors-6cal on vast.ai.
# Invoked by gpu2vast onstart.sh (which has already cd'd to /workspace/data
# and exported R2_* + JOB_ID env vars).
set -u

echo "[setup] === BOOT ==="
date -u
echo "[setup] env: bucket=${R2_BUCKET:-MISSING} job=${JOB_ID:-MISSING}"

# Hard CUDA gate
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[setup] GPU:', torch.cuda.get_device_name(0), 'torch', torch.__version__)" || { echo "[setup] FATAL: no CUDA"; exit 2; }

echo "[setup] installing DFT deps"
pip install --quiet rdkit-pypi pyscf gpu4pyscf-cuda12x geometric boto3 2>&1 | tail -5
python3 -c "from gpu4pyscf import dft; print('[setup] gpu4pyscf imported OK')" || { echo "[setup] FATAL: gpu4pyscf import failed"; exit 3; }

# Stage cached inputs into results/
mkdir -p results
[ -f m2_atom_refs.json ] && cp m2_atom_refs.json results/m2_atom_refs.json
[ -f m2_lead_RDX.json ]  && cp m2_lead_RDX.json  results/m2_lead_RDX.json
[ -f m2_lead_TATB.json ] && cp m2_lead_TATB.json results/m2_lead_TATB.json
for f in m2_lead_*.json m2_summary.json; do [ -f "$f" ] && cp "$f" results/"$f" || true; done

# Daemon
chmod +x monitor_daemon_vast.sh || true
bash monitor_daemon_vast.sh > /tmp/monitor.log 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > /tmp/monitor.pid
echo "[setup] monitor_daemon_vast started pid=$MONITOR_PID"

echo "[setup] === RUN ==="
set +e
python3 -u m2_anchors_extension.py --results results 2>&1 | tee -a /tmp/job.log
RC=${PIPESTATUS[0]}
set -e

echo "[ext] main exited rc=$RC"
kill -TERM "$MONITOR_PID" 2>/dev/null || true
sleep 5
echo "[train] === DONE ==="
exit $RC
