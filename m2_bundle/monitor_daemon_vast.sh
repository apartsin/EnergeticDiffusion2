#!/bin/bash
# monitor_daemon_vast.sh
# Live diagnostics daemon for the M2 6-anchor DFT calibration job on vast.ai.
# Pushes heartbeat.json, job.log.live, progress.json, and per-anchor
# m2_anchor_*.json results to Cloudflare R2 every 60s.
#
# Reads env vars exported by the gpu2vast onstart.sh:
#   R2_ACCOUNT_ID
#   R2_ACCESS_KEY
#   R2_SECRET_KEY
#   R2_BUCKET
#   JOB_ID  (used as prefix label inside heartbeat)
#
# All keys are pushed at the bucket root (no job_id prefix, since the bucket
# itself is the job-specific namespace).
#
# On EXIT (trap), writes done.json or error.json with final status.

set -u

LOG=/tmp/job.log
LIVE=/tmp/job.log.live
HB=/tmp/heartbeat.json
PROG=/tmp/progress.json
DONE=/tmp/done.json
ERRJ=/tmp/error_daemon.json
RES_DIR=results
PUSHED_LIST=/tmp/pushed_anchors.list
LAST_PHASE_FILE=/tmp/last_phase.txt
START_TS=$(date +%s)

mkdir -p "$RES_DIR"
touch "$PUSHED_LIST"
echo "image_install_started" > "$LAST_PHASE_FILE"

# --- helper: push a single local file to R2 via boto3 ---
push_to_r2() {
    local LOCAL="$1"
    local KEY="$2"
    python3 - <<PYEOF 2>>/tmp/monitor.err
import os, boto3, sys
from botocore.config import Config
acc = os.environ['R2_ACCOUNT_ID']
ak  = os.environ['R2_ACCESS_KEY']
sk  = os.environ['R2_SECRET_KEY']
bk  = os.environ['R2_BUCKET']
endpoint = f'https://{acc}.r2.cloudflarestorage.com'
s3 = boto3.client('s3', endpoint_url=endpoint,
    aws_access_key_id=ak, aws_secret_access_key=sk,
    region_name='auto',
    config=Config(retries={'max_attempts':3}))
try:
    s3.upload_file("$LOCAL", bk, "$KEY")
except Exception as e:
    print(f"push_to_r2 ERR {e}", file=sys.stderr)
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
  "job_id": "${JOB_ID:-}"
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
            push_to_r2 "$PROG" "progress.json"
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
            push_to_r2 "$f" "results/${BASE}"
            echo "$BASE" >> "$PUSHED_LIST"
            echo "[monitor] pushed result: $BASE"
        fi
    done
    for f in m2_calibration_6anchor.json m2_summary_6anchor.json; do
        local local_path="$RES_DIR/$f"
        [ -f "$local_path" ] || continue
        if ! grep -qxF "$f" "$PUSHED_LIST"; then
            push_to_r2 "$local_path" "results/$f"
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
  "job_id": "${JOB_ID:-}",
  "status": "success",
  "exit_code": 0,
  "total_seconds": $(($(date +%s) - START_TS)),
  "anchors_completed": $NDONE,
  "ts_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JEOF
        push_to_r2 "$DONE" "done_daemon.json"
    else
        local TB=$(tail -100 "$LOG" 2>/dev/null | tr '"' "'" | tr '\n' ' ' | head -c 4000)
        cat > "$ERRJ" <<JEOF
{
  "job_id": "${JOB_ID:-}",
  "rc": $RC,
  "anchors_completed": $NDONE,
  "tail_log": "$TB",
  "ts_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JEOF
        push_to_r2 "$ERRJ" "error_daemon.json"
    fi
    build_heartbeat
    push_to_r2 "$HB" "heartbeat.json"
    [ -f "$LOG" ] && tail -500 "$LOG" > "$LIVE" && push_to_r2 "$LIVE" "job.log.live"
}
trap on_exit EXIT

# --- main loop ---
while true; do
    build_heartbeat
    push_to_r2 "$HB" "heartbeat.json"
    if [ -f "$LOG" ]; then
        tail -200 "$LOG" > "$LIVE"
        push_to_r2 "$LIVE" "job.log.live"
    fi
    update_phase_from_log
    push_new_results
    sleep 60
done
