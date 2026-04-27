#!/usr/bin/env bash
# Retry wrapper around gpu_runner.py — detects silent crashes (empty log,
# stuck in "provisioning" for > UPLOAD_TIMEOUT seconds) and re-launches.
#
# Usage: bash vast_retry.sh <log_file> <max_retries> <upload_timeout_s> -- <gpu_runner args...>

set -u
LOG="$1"; MAX_RETRIES="${2:-3}"; UPLOAD_TIMEOUT="${3:-600}"; shift 3
[[ "$1" == "--" ]] && shift

RUNNER="/c/Users/apart/Projects/claude-skills/gpu2vast/gpu_runner.py"
attempt=0
while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt + 1))
    echo "[retry] attempt $attempt/$MAX_RETRIES — $(date '+%H:%M:%S')"
    > "$LOG"   # zero log

    # Launch in background within this script so we can monitor
    /c/Python314/python "$RUNNER" run "$@" > "$LOG" 2>&1 &
    PID=$!

    # Watch progress: log size growing or instance creation message
    elapsed=0
    instance_seen=0
    while (( elapsed < UPLOAD_TIMEOUT )); do
        sleep 30
        elapsed=$((elapsed + 30))
        if grep -q "Instance created" "$LOG" 2>/dev/null; then
            echo "[retry] attempt $attempt: instance created OK (after ${elapsed}s)"
            instance_seen=1
            break
        fi
        if ! kill -0 "$PID" 2>/dev/null; then
            # Process exited
            if grep -q "COMPLETE\|Job finished: success" "$LOG" 2>/dev/null; then
                echo "[retry] attempt $attempt: completed successfully"
                exit 0
            else
                echo "[retry] attempt $attempt: process exited early — retrying"
                break
            fi
        fi
    done

    if (( instance_seen == 1 )); then
        # Wait for the run to complete (no timeout from us)
        wait "$PID"
        rc=$?
        if (( rc == 0 )); then
            echo "[retry] attempt $attempt: success"
            exit 0
        fi
        echo "[retry] attempt $attempt: instance run failed (rc=$rc) — retrying"
    else
        echo "[retry] attempt $attempt: stuck > ${UPLOAD_TIMEOUT}s without instance creation — killing"
        kill "$PID" 2>/dev/null
        sleep 5
    fi
done

echo "[retry] all $MAX_RETRIES attempts failed"
exit 1
