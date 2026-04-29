#!/bin/bash
# GPU2Vast On-Start Script for vast.ai instances

# STAGE marker for trap-based error sentinel: each section updates STAGE.
# If trap fires before STAGE=complete, an error.json is uploaded to R2 with
# the stage, exit code, and last 200 lines of the job log.
STAGE="init"

_emit_error_sentinel() {
    local exit_code=$?
    if [ "$STAGE" = "complete" ]; then
        return
    fi
    echo "[GPU2Vast] TRAP: stage=$STAGE exit_code=$exit_code, uploading error.json" >&2
    local log_tail=""
    if [ -f /tmp/job.log ]; then
        log_tail=$(tail -200 /tmp/job.log 2>/dev/null || echo "")
    elif [ -f /workspace/stdout.log ]; then
        log_tail=$(tail -200 /workspace/stdout.log 2>/dev/null || echo "")
    fi
    python3 - <<PYEOF 2>/dev/null || true
import boto3, json, os, time
acc = os.environ.get('R2_ACCOUNT_ID', '')
ak = os.environ.get('R2_ACCESS_KEY', '')
sk = os.environ.get('R2_SECRET_KEY', '')
bucket = os.environ.get('R2_BUCKET', '')
if not (acc and ak and sk and bucket):
    raise SystemExit(0)
s3 = boto3.client('s3',
    endpoint_url=f'https://{acc}.r2.cloudflarestorage.com',
    aws_access_key_id=ak, aws_secret_access_key=sk, region_name='auto')
log_tail = """${log_tail//\"/\\\"}"""
s3.put_object(Bucket=bucket, Key='error.json', Body=json.dumps({
    'stage': '${STAGE}',
    'exit_code': ${exit_code},
    'log_tail': log_tail[-20000:],
    'timestamp': time.time(),
}))
try:
    for src in ('/tmp/job.log', '/workspace/stdout.log'):
        if os.path.exists(src):
            s3.upload_file(src, bucket, 'logs/job.log')
            break
except Exception:
    pass
PYEOF
}
trap _emit_error_sentinel EXIT

set -eo pipefail

SECONDS=0
ts() { echo "[GPU2Vast] [${SECONDS}s] $1" | tee -a /tmp/job.log; }

ts "Starting job: $JOB_ID"
ts "Image: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 || echo unknown)"
ts "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'detecting...')"

# Pre-flight CUDA driver check (fail fast if PyTorch can't see CUDA)
if command -v python3 >/dev/null 2>&1 && python3 -c "import torch" 2>/dev/null; then
    if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        ts "FATAL: torch present but torch.cuda.is_available() == False"
        ts "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
        ts "Torch CUDA build: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
        ts "Aborting before training to avoid silent billing"
        exit 42
    fi
    ts "CUDA preflight OK ($(python3 -c 'import torch; print(torch.version.cuda)'))"
fi

# 1. Install packages (skip torch if already present to avoid re-downloading CUDA libs)
STAGE="install"
ts "Installing packages..."
T0=$SECONDS
EXTRA_PKGS="boto3 transformers accelerate peft trl bitsandbytes sentence-transformers datasets requests tensorboard sentencepiece protobuf"
if python3 -c "import torch" 2>/dev/null; then
    ts "torch found in base image, preserving ($(python3 -c 'import torch; print(torch.__version__)'))"
    CONSTRAINT="--constraint /tmp/torch_constraint.txt"
    python3 -c "import torch; print(f'torch=={torch.__version__}')" > /tmp/torch_constraint.txt
else
    EXTRA_PKGS="torch $EXTRA_PKGS"
    CONSTRAINT=""
    ts "torch not found in base image, will install"
fi
if command -v uv &> /dev/null; then
    uv pip install --system $EXTRA_PKGS
else
    pip install $CONSTRAINT $EXTRA_PKGS
fi
ts "Packages installed ($((SECONDS - T0))s)"

# 2. HuggingFace token
if [ -n "$HF_TOKEN" ]; then
    ts "Configuring HuggingFace token..."
    python3 -c "
from huggingface_hub import login
import os
token = os.environ.get('HF_TOKEN', '')
if token:
    login(token=token, add_to_git_credential=False)
    print('  HuggingFace token configured')
" 2>/dev/null || echo "  HF login skipped"
fi

# 3. Download data from R2
STAGE="download"
ts "Downloading data from R2..."
T0=$SECONDS
python3 -c "
import boto3, os, time
from pathlib import Path

ws = Path('/workspace/data')
ws.mkdir(parents=True, exist_ok=True)

s3 = boto3.client('s3',
    endpoint_url=f'https://{os.environ[\"R2_ACCOUNT_ID\"]}.r2.cloudflarestorage.com',
    aws_access_key_id=os.environ['R2_ACCESS_KEY'],
    aws_secret_access_key=os.environ['R2_SECRET_KEY'],
    region_name='auto')

bucket = os.environ['R2_BUCKET']
count = 0
total_bytes = 0
t0 = time.time()
for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix='data/'):
    for obj in page.get('Contents', []):
        key = obj['Key']
        local = ws / key.replace('data/', '', 1)
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local))
        count += 1
        total_bytes += obj['Size']
        print(f'  {key} ({obj[\"Size\"]:,} bytes)')
elapsed = time.time() - t0
speed = total_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
print(f'[GPU2Vast] Downloaded {count} files ({total_bytes:,} bytes) in {elapsed:.1f}s ({speed:.1f} MB/s)')
"
ts "Data download complete ($((SECONDS - T0))s)"

# 4. Extra requirements
if [ -f /workspace/data/requirements.txt ]; then
    ts "Installing extra requirements..."
    pip install -q -r /workspace/data/requirements.txt
    ts "Extra requirements installed"
fi

# 5. Progress reporter
if [ -f /workspace/data/progress_reporter.py ]; then
    cp /workspace/data/progress_reporter.py /workspace/progress_reporter.py
fi
if [ -f /workspace/progress_reporter.py ]; then
    ts "Starting progress reporter"
    python3 /workspace/progress_reporter.py &
    REPORTER_PID=$!
else
    REPORTER_PID=""
fi

# 6. TensorBoard
ts "Starting TensorBoard on port 6006"
mkdir -p /workspace/data/runs
nohup tensorboard --logdir=/workspace/data/runs --host=0.0.0.0 --port=6006 > /dev/null 2>&1 &

# 7. Run experiment
STAGE="training"
ts "Running: $EXPERIMENT_CMD"
T0=$SECONDS
cd /workspace/data
eval "$EXPERIMENT_CMD" > >(tee /workspace/stdout.log) 2>&1
EXIT_CODE=$?
ts "Training finished (exit=$EXIT_CODE, ${SECONDS}s total, $((SECONDS - T0))s training)"

# 8. Stop reporter
[ -n "$REPORTER_PID" ] && kill $REPORTER_PID 2>/dev/null || true

# 9. Upload results
STAGE="upload"
ts "Uploading results to R2..."
T0=$SECONDS
python3 -c "
import boto3, json, os, glob, time
from pathlib import Path

s3 = boto3.client('s3',
    endpoint_url=f'https://{os.environ[\"R2_ACCOUNT_ID\"]}.r2.cloudflarestorage.com',
    aws_access_key_id=os.environ['R2_ACCESS_KEY'],
    aws_secret_access_key=os.environ['R2_SECRET_KEY'],
    region_name='auto')
bucket = os.environ['R2_BUCKET']
exit_code = $EXIT_CODE

uploaded = {}
total_bytes = 0
t0 = time.time()
for fp in glob.glob('results/**/*', recursive=True):
    path = Path(fp)
    if path.is_file():
        # Preserve subdirectory structure: results/model/config.json -> results/model/config.json
        rel = str(path).replace(os.sep, '/')
        key = rel  # keep full path from results/ onward
        s3.upload_file(str(path), bucket, key)
        size = path.stat().st_size
        uploaded[key] = {'size': size}
        total_bytes += size
        print(f'  {rel} ({size:,} bytes)')

if Path('/workspace/stdout.log').exists():
    s3.upload_file('/workspace/stdout.log', bucket, 'logs/stdout.log')

elapsed = time.time() - t0
speed = total_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
s3.put_object(Bucket=bucket, Key='done.json', Body=json.dumps({
    'status': 'success' if exit_code == 0 else 'failed',
    'exit_code': exit_code, 'files': uploaded,
    'total_bytes': total_bytes, 'timestamp': time.time()
}))
print(f'[GPU2Vast] Uploaded {len(uploaded)} files ({total_bytes:,} bytes) in {elapsed:.1f}s ({speed:.1f} MB/s)')
"
STAGE="complete"
ts "ALL DONE (total ${SECONDS}s)"
