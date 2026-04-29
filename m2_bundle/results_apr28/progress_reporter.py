"""Background progress reporter. Reads stdout.log, extracts progress, uploads to R2."""
import boto3, json, os, re, sys, time, subprocess
from pathlib import Path

s3 = boto3.client("s3",
    endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY"],
    aws_secret_access_key=os.environ["R2_SECRET_KEY"],
    region_name="auto",
)
bucket = os.environ["R2_BUCKET"]
interval = int(os.environ.get("PROGRESS_INTERVAL", "15"))

# Training metrics patterns
METRIC_PATTERNS = {
    "step": r"(\d+)/(\d+)",
    "loss": r"loss[=: ]+([0-9.]+)",
    "epoch": r"epoch[=: ]+([0-9.]+)",
    "accuracy": r"(?:accuracy|acc|hit@1|exact_match|f1)[=: ]+([0-9.]+%?)",
    "lr": r"(?:learning.rate|lr)[=: ]+([0-9.e-]+)",
    "val_loss": r"val(?:idation)?[_ ]loss[=: ]+([0-9.]+)",
    "val_acc": r"val(?:idation)?[_ ](?:accuracy|acc)[=: ]+([0-9.]+%?)",
    "eval": r"eval[_ ](?:loss|accuracy|f1)[=: ]+([0-9.]+)",
}

# Phase detection patterns (order matters: first match wins)
PHASE_PATTERNS = [
    (r"loading.*model|from_pretrained|downloading.*model|fetching.*model", "model_loading"),
    (r"loading.*data|reading.*data|loading.*dataset|loading.*train", "data_loading"),
    (r"tokeniz", "tokenizing"),
    (r"downloading.*weight|downloading.*checkpoint", "downloading_weights"),
    (r"(?:start|begin).*train|training.*(?:start|begin)|phase.*train", "training"),
    (r"(?:start|begin).*eval|evaluat|validat", "evaluating"),
    (r"saving.*model|save_pretrained|saving.*weight|saving.*checkpoint", "saving_model"),
    (r"uploading.*result|uploading.*model", "uploading_results"),
    (r"(?:all )?done|finished|complete", "done"),
]


def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode != 0:
            return {}
        parts = result.stdout.strip().split(", ")
        if len(parts) < 4:
            return {}
        return {"gpu_util": int(parts[0]), "mem_used": int(parts[1]),
                "mem_total": int(parts[2]), "temp": int(parts[3])}
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return {}
    except Exception:
        return {}


def parse_progress():
    log = Path("/workspace/stdout.log")
    if not log.exists():
        log = Path("/var/log/onstart.log")
    if not log.exists():
        return {}

    content = log.read_text(errors="replace")
    lines = content.split("\n")
    tail = lines[-150:]

    # Extract metrics from recent lines (scan backwards)
    metrics = {}
    for line in reversed(tail):
        for key, pattern in METRIC_PATTERNS.items():
            if key not in metrics:
                m = re.search(pattern, line, re.I)
                if m:
                    metrics[key] = m.group(1)
                    if key == "step" and m.lastindex and m.lastindex >= 2:
                        metrics["total"] = m.group(2)
        if len(metrics) >= 4:
            break

    # Detect current phase (scan backwards for most recent phase marker)
    current_phase = "unknown"
    for line in reversed(tail):
        for pattern, phase in PHASE_PATTERNS:
            if re.search(pattern, line, re.I):
                current_phase = phase
                break
        if current_phase != "unknown":
            break

    metrics["phase"] = current_phase

    # Count total lines as rough progress indicator
    metrics["log_lines"] = len(lines)

    # Extract last N meaningful log lines for display
    recent = []
    for line in reversed(tail):
        line = line.strip()
        if line and not line.startswith("  ") and len(line) > 5:
            recent.append(line)
            if len(recent) >= 5:
                break
    metrics["recent_lines"] = list(reversed(recent))

    return metrics


while True:
    try:
        progress = parse_progress()
        progress["gpu"] = get_gpu_info()
        progress["timestamp"] = time.time()
        progress["job_id"] = os.environ.get("JOB_ID", "")

        s3.put_object(
            Bucket=bucket, Key="progress.json",
            Body=json.dumps(progress),
        )

        # Upload log tail every cycle (last 100KB)
        for log_path in [Path("/workspace/stdout.log"), Path("/var/log/onstart.log")]:
            if log_path.exists():
                content = log_path.read_bytes()
                if len(content) > 100000:
                    content = b"[...truncated...]\n" + content[-100000:]
                s3.put_object(Bucket=bucket, Key=f"logs/{log_path.name}", Body=content)
                break

    except Exception as e:
        print(f"[progress] error: {e}", file=sys.stderr)

    time.sleep(interval)
