"""
GPU2Vast Observer: Drop-in observability for training scripts.

Import this at the top of your training script to get automatic
progress reporting to R2 (visible in real-time from your local machine).

Usage in your training script:
    from gpu2vast_observer import observer

    # Reports phase changes
    observer.phase("loading_model")
    model = load_model()

    observer.phase("training")
    for step in range(total_steps):
        loss = train_step()
        # Reports step/loss/epoch/custom metrics
        observer.step(step + 1, total_steps, loss=loss, epoch=epoch, lr=lr)

    observer.phase("saving")
    save_model()
    observer.done(final_loss=loss)

All methods are safe to call even if R2 env vars are not set (no-ops).
Progress is uploaded to R2 every 10 seconds (batched, not per-call).
"""

import json
import os
import sys
import time
import threading
import subprocess
from pathlib import Path


class GPU2VastObserver:
    def __init__(self):
        self._metrics = {}
        self._phase = "initializing"
        self._log_lines = []
        self._lock = threading.Lock()
        self._s3 = None
        self._bucket = None
        self._job_id = os.environ.get("JOB_ID", "")
        self._interval = int(os.environ.get("PROGRESS_INTERVAL", "10"))
        self._start_time = time.time()
        self._step_count = 0

        # Try to connect to R2
        try:
            import boto3
            acct = os.environ.get("R2_ACCOUNT_ID", "")
            if acct:
                self._s3 = boto3.client("s3",
                    endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
                    aws_access_key_id=os.environ["R2_ACCESS_KEY"],
                    aws_secret_access_key=os.environ["R2_SECRET_KEY"],
                    region_name="auto",
                )
                self._bucket = os.environ.get("R2_BUCKET", "")
        except Exception:
            pass

        # TensorBoard writer (auto-creates runs/ directory)
        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir="runs")
        except ImportError:
            pass

        # Start background reporter thread
        if self._s3 and self._bucket:
            self._reporter = threading.Thread(target=self._report_loop, daemon=True)
            self._reporter.start()

    def phase(self, name: str, detail: str = ""):
        """Report a phase change (e.g., 'loading_model', 'training', 'evaluating')."""
        with self._lock:
            self._phase = name
            self._metrics["phase"] = name
        msg = f"[gpu2vast] Phase: {name}"
        if detail:
            msg += f" ({detail})"
        print(msg, flush=True)

    def step(self, current: int, total: int, **metrics):
        """Report training step progress with optional metrics. Also logs to TensorBoard."""
        with self._lock:
            self._metrics["step"] = str(current)
            self._metrics["total"] = str(total)
            self._step_count = current
            for k, v in metrics.items():
                if v is not None:
                    self._metrics[k] = f"{v:.4f}" if isinstance(v, float) else str(v)

        # Log to TensorBoard
        if self._tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, current)
            self._tb_writer.flush()

        # Print compact progress line
        parts = [f"{current}/{total}"]
        for k, v in metrics.items():
            if v is not None:
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        print(f"  {'  '.join(parts)}", flush=True)

    def log(self, message: str):
        """Log a message (shown in streaming output and saved to R2)."""
        with self._lock:
            self._log_lines.append(message)
            if len(self._log_lines) > 20:
                self._log_lines = self._log_lines[-20:]
        print(f"[gpu2vast] {message}", flush=True)

    def metric(self, **kwargs):
        """Report arbitrary metrics without a step."""
        with self._lock:
            for k, v in kwargs.items():
                if v is not None:
                    self._metrics[k] = f"{v:.4f}" if isinstance(v, float) else str(v)

    def done(self, status: str = "success", **final_metrics):
        """Signal completion. Uploads final progress and done.json."""
        self.phase("done")
        with self._lock:
            for k, v in final_metrics.items():
                self._metrics[k] = f"{v:.4f}" if isinstance(v, float) else str(v)
        self._upload_progress()
        elapsed = time.time() - self._start_time
        print(f"[gpu2vast] Done: {status} ({elapsed:.0f}s, {self._step_count} steps)", flush=True)

    def _get_gpu_info(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            parts = result.stdout.strip().split(", ")
            return {"gpu_util": int(parts[0]), "mem_used": int(parts[1]),
                    "mem_total": int(parts[2]), "temp": int(parts[3])}
        except Exception:
            return {}

    def _upload_progress(self):
        if not self._s3 or not self._bucket:
            return
        try:
            with self._lock:
                data = dict(self._metrics)
                data["recent_lines"] = list(self._log_lines[-5:])
            data["gpu"] = self._get_gpu_info()
            data["timestamp"] = time.time()
            data["elapsed_s"] = round(time.time() - self._start_time, 1)
            data["job_id"] = self._job_id
            self._s3.put_object(
                Bucket=self._bucket, Key="progress.json",
                Body=json.dumps(data),
            )
        except Exception as e:
            print(f"[gpu2vast] progress upload error: {e}", file=sys.stderr)

    def _report_loop(self):
        while True:
            time.sleep(self._interval)
            self._upload_progress()


# Singleton instance
observer = GPU2VastObserver()
