#!/bin/bash
# monitor_daemon.sh
# Live diagnostics daemon for the M2 6-anchor DFT calibration job.
# Pushes heartbeat.json, job.log.live, progress.json, and per-anchor
# m2_anchor_*.json results to RunPod S3 every 60s.
#
# Reads env vars exported by the runpod_runner.py bootstrap:
#   RUNPOD_STORAGE_ENDPOINT
#   RUNPOD_STORAGE_ACCESS_KEY
#   RUNPOD_STORAGE_SECRET_KEY
#   RUNPOD_STORAGE_VOLUME_ID
#   RUNPOD_STORAGE_JOB_PREFIX  (the job_id)
#
# On EXIT (trap), writes done.json or error.json with final status.

set -u

LOG=/tmp/job.log
LIVE=/tmp/job.log.live
HB=/tmp/heartbeat.json
PROG=/tmp/progress.json
DONE=/tmp/done.json
ERRJ=/tmp/error.json
RES_DIR=results
PUSHED_LIST=/tmp/pushed_anchors.list
LAST_PHASE_FILE=/tmp/last_phase.txt
START_TS=$(date +%s)

mkdir -p "$RES_DIR"
touch "$PUSHED_LIST"
echo "image_install_started" > "$LAST_PHASE_FILE"

# --- helper: push a single local file to RunPod S3 via boto3 ---
push_to_s3() {
    local LOCAL="$1"
    local KEY="$2"
    python3 - <<PYEOF 2>>/tmp/monitor.err
import os, boto3, sys
from botocore.config import Config
ep  = os.environ['RUNPOD_STORAGE_ENDPOINT']
ak  = os.environ['RUNPOD_STORAGE_ACCESS_KEY']
sk  = os.environ['RUNPOD_STORAGE_SECRET_KEY']
vol = os.environ['RUNPOD_STORAGE_VOLUME_ID']
import re
m = re.search(r's3api-([a-z0-9-]+)\.runpod\.io', ep)
region = m.group(1) if m else 'us-ks-2'
s3 = boto3.client('s3', endpoint_url=ep,
    aws_access_key_id=ak, aws_secret_access_key=sk,
    region_name=region,
    config=Config(retries={'max_attempts':3}, s3={'addressing_style':'path'}))
try:
    s3.upload_file("$LOCAL", vol, "$KEY")
except Exception as e:
    print(f"push_to_s3 ERR {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# --- helper: build heartbeat.json ---
build_heartbeat() {
    local NVS NDONE CUR_ANCHOR PIDS PHASE
    NVS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "<no nvidia-smi>")
    NDONE=$(ls "$RES_DIR"/m2_anchor_*.json 2>/dev/null | wc -l)
    CUR_ANCHOR=$(grep -oE '\[ext\] === anchor [A-Za-z0-9_-]+ ===' "$LOG" 2>/dev/null | tail -1 | sed -E 's/.*anchor ([A-Za-z0-9_-]+).*/\1/' || echo "")
    PIDS=$(pgrep -af 'm2_anchors|m2_dft_pipeline' 2>/dev/null | tr '\n' '|' | sed 's/|$//')
    PHASE=$(cat "$LAST_PHASE_FILE" 2>/dev/null || echo "unknown")
    cat > "$HB" <<JEOF
{
  "ts_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "elapsed_sec": $(($(date +%s) - START_TS)),
  "current_phase": "$PHASE",
  "current_anchor": "$CUR_ANCHOR",
  "n_anchors_done": $NDONE,
  "nvidia_smi": "$NVS",
  "main_pids": "$PIDS",
  "job_id": "${RUNPOD_STORAGE_JOB_PREFIX:-}"
}
JEOF
}

# --- helper: detect phase advance from log and update progress.json ---
update_phase_from_log() {
    local NEW_PHASE=""
    if   grep -q '\[setup\] gpu4pyscf imported OK' "$LOG" 2>/dev/null; then NEW_PHASE="image_install_done"; fi
    if   grep -q 'atom_refs.*loaded\|m2_atom_refs.json' "$LOG" 2>/dev/null; then NEW_PHASE="atom_refs_done"; fi
    for A in HMX PETN FOX7 NTO RDX TATB; do
        if grep -q "\[ext\] === anchor ${A} ===" "$LOG" 2>/dev/null; then NEW_PHASE="${A}_started"; fi
        if [ -f "$RES_DIR/m2_anchor_${A}.json" ]; then NEW_PHASE="${A}_done"; fi
    done
    if grep -q '\[ext\] === DONE ===\|=== DONE ===' "$LOG" 2>/dev/null; then NEW_PHASE="all_done"; fi
    if [ -n "$NEW_PHASE" ]; then
        local PREV=$(cat "$LAST_PHASE_FILE" 2>/dev/null || echo "")
        if [ "$NEW_PHASE" != "$PREV" ]; then
            echo "$NEW_PHASE" > "$LAST_PHASE_FILE"
            cat > "$PROG" <<JEOF
{"phase": "$NEW_PHASE", "ts_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "elapsed_sec": $(($(date +%s) - START_TS))}
JEOF
            push_to_s3 "$PROG" "${RUNPOD_STORAGE_JOB_PREFIX}/progress.json"
            echo "[monitor] phase advance: $PREV -> $NEW_PHASE"
        fi
    fi
}

# --- helper: push any new per-anchor JSON results ---
push_new_results() {
    for f in "$RES_DIR"/m2_anchor_*.json; do
        [ -f "$f" ] || continue
        local BASE=$(basename "$f")
        if ! grep -qxF "$BASE" "$PUSHED_LIST"; then
            push_to_s3 "$f" "${RUNPOD_STORAGE_JOB_PREFIX}/results/${BASE}"
            echo "$BASE" >> "$PUSHED_LIST"
            echo "[monitor] pushed result: $BASE"
        fi
    done
    # Also push refit summaries when they appear
    for f in m2_calibration_6anchor.json m2_summary_6anchor.json; do
        local local_path="$RES_DIR/$f"
        [ -f "$local_path" ] || continue
        if ! grep -qxF "$f" "$PUSHED_LIST"; then
            push_to_s3 "$local_path" "${RUNPOD_STORAGE_JOB_PREFIX}/results/$f"
            echo "$f" >> "$PUSHED_LIST"
        fi
    done
}

# --- EXIT trap: push final state ---
on_exit() {
    local RC=${MAIN_RC:-0}
    local NDONE=$(ls "$RES_DIR"/m2_anchor_*.json 2>/dev/null | wc -l)
    if [ "$RC" = "0" ] && [ "$NDONE" -gt 0 ]; then
        cat > "$DONE" <<JEOF
{
  "job_id": "${RUNPOD_STORAGE_JOB_PREFIX:-}",
  "total_seconds": $(($(date +%s) - START_TS)),
  "anchors_completed": $NDONE,
  "ts_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JEOF
        push_to_s3 "$DONE" "${RUNPOD_STORAGE_JOB_PREFIX}/done.json"
    else
        local TB=$(tail -100 "$LOG" 2>/dev/null | tr '"' "'" | tr '\n' ' ' | head -c 4000)
        cat > "$ERRJ" <<JEOF
{
  "job_id": "${RUNPOD_STORAGE_JOB_PREFIX:-}",
  "rc": $RC,
  "anchors_completed": $NDONE,
  "tail_log": "$TB",
  "ts_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JEOF
        push_to_s3 "$ERRJ" "${RUNPOD_STORAGE_JOB_PREFIX}/error.json"
    fi
    # one last push of heartbeat + tail
    build_heartbeat
    push_to_s3 "$HB" "${RUNPOD_STORAGE_JOB_PREFIX}/heartbeat.json"
    [ -f "$LOG" ] && tail -500 "$LOG" > "$LIVE" && push_to_s3 "$LIVE" "${RUNPOD_STORAGE_JOB_PREFIX}/job.log.live"
}
trap on_exit EXIT

# --- main loop ---
while true; do
    build_heartbeat
    push_to_s3 "$HB" "${RUNPOD_STORAGE_JOB_PREFIX}/heartbeat.json"
    if [ -f "$LOG" ]; then
        tail -200 "$LOG" > "$LIVE"
        push_to_s3 "$LIVE" "${RUNPOD_STORAGE_JOB_PREFIX}/job.log.live"
    fi
    update_phase_from_log
    push_new_results
    sleep 60
done
