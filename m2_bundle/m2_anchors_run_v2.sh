#!/bin/bash
# m2_anchors_run_v2.sh — entry point for m2-anchors-6cal-v2.
# Launches monitor_daemon.sh in background for live S3 diagnostics,
# then runs the 6-anchor DFT calibration.
set -u

echo "[setup] === BOOT ==="
date -u
echo "[setup] env: ep=${RUNPOD_STORAGE_ENDPOINT:-MISSING} vol=${RUNPOD_STORAGE_VOLUME_ID:-MISSING} pfx=${RUNPOD_STORAGE_JOB_PREFIX:-MISSING}"

# Hard CUDA gate
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[setup] GPU:', torch.cuda.get_device_name(0), 'torch', torch.__version__)" || { echo "[setup] FATAL: no CUDA"; exit 2; }

echo "[setup] installing deps"
pip install --quiet rdkit-pypi pyscf gpu4pyscf-cuda12x geometric boto3 2>&1 | tail -5
python3 -c "from gpu4pyscf import dft; print('[setup] gpu4pyscf imported OK')" || { echo "[setup] FATAL: gpu4pyscf import failed"; exit 3; }

# Stage cached inputs into results/
mkdir -p results
[ -f m2_atom_refs.json ] && cp m2_atom_refs.json results/m2_atom_refs.json
[ -f m2_lead_RDX.json ]  && cp m2_lead_RDX.json  results/m2_lead_RDX.json
[ -f m2_lead_TATB.json ] && cp m2_lead_TATB.json results/m2_lead_TATB.json
for f in m2_lead_*.json m2_summary.json; do [ -f "$f" ] && cp "$f" results/"$f" || true; done

# Make daemon executable + start it in background.
chmod +x monitor_daemon.sh || true
bash monitor_daemon.sh > /tmp/monitor.log 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > /tmp/monitor.pid
echo "[setup] monitor_daemon started pid=$MONITOR_PID"

echo "[setup] === RUN ==="
set +e
python3 -u m2_anchors_extension.py --results results 2>&1 | tee -a /tmp/job.log
RC=${PIPESTATUS[0]}
set -e

echo "[ext] main exited rc=$RC"
# signal RC to daemon trap via env file (the trap reads MAIN_RC at exit-time;
# but since the daemon is a separate process we instead let it infer success
# from anchor-count + log markers, plus we kill it which triggers its trap)
kill -TERM "$MONITOR_PID" 2>/dev/null || true
sleep 5
echo "[train] === DONE ==="
exit $RC
