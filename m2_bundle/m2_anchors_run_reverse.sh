#!/bin/bash
# m2_anchors_run_reverse.sh; entry point for m2-anchors-6cal-reverse.
# Same shape as m2_anchors_run_v2.sh but writes to results_reverse/ and runs
# m2_anchors_reverse.py (NTO -> FOX7 -> PETN -> HMX).
set -u

echo "[setup] === BOOT (reverse) ==="
date -u
echo "[setup] env: ep=${RUNPOD_STORAGE_ENDPOINT:-MISSING} vol=${RUNPOD_STORAGE_VOLUME_ID:-MISSING} pfx=${RUNPOD_STORAGE_JOB_PREFIX:-MISSING}"

# Hard CUDA gate
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[setup] GPU:', torch.cuda.get_device_name(0), 'torch', torch.__version__)" || { echo "[setup] FATAL: no CUDA"; exit 2; }

echo "[setup] installing deps"
pip install --quiet rdkit-pypi pyscf gpu4pyscf-cuda12x geometric boto3 2>&1 | tail -5
python3 -c "from gpu4pyscf import dft; print('[setup] gpu4pyscf imported OK')" || { echo "[setup] FATAL: gpu4pyscf import failed"; exit 3; }

# Stage cached inputs into results_reverse/
mkdir -p results_reverse
[ -f m2_atom_refs.json ] && cp m2_atom_refs.json results_reverse/m2_atom_refs.json
[ -f m2_lead_RDX.json ]  && cp m2_lead_RDX.json  results_reverse/m2_lead_RDX.json
[ -f m2_lead_TATB.json ] && cp m2_lead_TATB.json results_reverse/m2_lead_TATB.json

# Make daemon executable + start it in background.
chmod +x monitor_daemon_reverse.sh || true
bash monitor_daemon_reverse.sh > /tmp/monitor.log 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > /tmp/monitor.pid
echo "[setup] monitor_daemon_reverse started pid=$MONITOR_PID"

echo "[setup] === RUN (reverse: NTO,FOX7,PETN,HMX) ==="
set +e
python3 -u m2_anchors_reverse.py --results results_reverse 2>&1 | tee -a /tmp/job.log
RC=${PIPESTATUS[0]}
set -e

echo "[ext] main exited rc=$RC"
kill -TERM "$MONITOR_PID" 2>/dev/null || true
sleep 5
echo "[train] === DONE ==="
exit $RC
