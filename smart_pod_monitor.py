"""
Smart monitor for the three running 6-anchor calibration pods:
  1. RunPod A6000 SECURE (pod 19g52uvrxup26l)
  2. vast.ai RTX 4090 spot (instance 35744391)
  3. vast.ai A100 PCIE spot (instance 35744397)

Probes each pod through three independent channels:
1. Vendor API: pod state, uptime, GPU utilisation, container exit
2. SSH (best-effort, BatchMode): nvidia-smi, /tmp/job.log tail, anchor JSONs on disk
3. Storage listing: anchor JSONs already uploaded back to RunPod-S3 (RunPod) or
   Cloudflare R2 (vast)

Outputs a single human-readable status table; safe to run repeatedly.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RUNNER_DIR = Path(r"C:\Users\apart\Projects\claude-skills\gpu2runpod")
VAST_DIR   = Path(r"C:\Users\apart\Projects\claude-skills\gpu2vast")
sys.path.insert(0, str(RUNNER_DIR))
sys.path.insert(0, str(VAST_DIR))

import runpod_manager as rpm   # type: ignore
from runpod_storage import RunPodStorage   # type: ignore

SSH_KEY_RP   = RUNNER_DIR / "keys" / "ssh" / "gpu2runpod_ed25519"
SSH_KEY_VAST = VAST_DIR / "keys" / "ssh" / "gpu2vast_ed25519"
_STORAGE_CFG = json.loads((RUNNER_DIR / "keys" / "runpod_storage.key").read_text())
_STORE = RunPodStorage(_STORAGE_CFG)
_R2_CFG = json.loads((VAST_DIR / "keys" / "r2.key").read_text())

PODS = [
    {"kind": "runpod",
     "name": "m2-anchors-6cal-v2 (RTX_A6000 SECURE)", "id": "19g52uvrxup26l",
     "job": "m2-anchors-6cal-v2-20260428-145330", "results_prefix": "results"},
    {"kind": "vast",
     "name": "m2-anchors-vastA (RTX_4090 spot, retried)", "id": "35744531",
     "job": "m2-anchors-vasta-20260428-145836",
     "bucket": "gpu2vast-m2-anchors-vasta-20260428-145836"},
    {"kind": "vast",
     "name": "m2-anchors-vastB (A100_PCIE spot)", "id": "35744397",
     "job": "m2-anchors-vastb-20260428-145840",
     "bucket": "gpu2vast-m2-anchors-vastb-20260428-145840"},
]


def ssh_exec(host: str, port: int, cmd: str, key: Path,
             timeout: int = 12) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=4", "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5",
             "-i", str(key), "-p", str(port), f"root@{host}", cmd],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        return (r.returncode == 0, (r.stdout or "") + (r.stderr or ""))
    except Exception as e:
        return (False, f"<ssh exec error: {e}>")


def probe_pod_runpod_api(pod_id: str) -> dict[str, Any]:
    try:
        pod = rpm.get_pod(pod_id) or {}
    except Exception as e:
        return {"error": f"get_pod failed: {e}"}
    rt = pod.get("runtime") or {}
    gpus = rt.get("gpus") or []
    gpu0 = gpus[0] if gpus else {}
    ports = rt.get("ports") or []
    ssh_host = ssh_port = None
    for p in ports:
        if p.get("privatePort") == 22:
            ssh_host = p.get("ip")
            ssh_port = p.get("publicPort")
    return {
        "desiredStatus": pod.get("desiredStatus"),
        "lastStatusChange": pod.get("lastStatusChange"),
        "uptimeSec": rt.get("uptimeInSeconds"),
        "gpuUtil": gpu0.get("gpuUtilPercent"),
        "memUsed": gpu0.get("memoryInUse"),
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
        "n_ports": len(ports),
    }


def probe_pod_vast_api(instance_id: str) -> dict[str, Any]:
    """Fetch instance runtime info from vast.ai API."""
    try:
        import vastai_manager as vast  # type: ignore
        info = vast.get_instance(int(instance_id)) or {}
    except Exception as e:
        return {"error": f"vast.get_instance failed: {e}"}
    if not isinstance(info, dict):
        return {"error": f"unexpected vast info type {type(info)}"}
    return {
        "actual_status": info.get("actual_status"),
        "intended_status": info.get("intended_status"),
        "cur_state": info.get("cur_state"),
        "gpu_name": info.get("gpu_name"),
        "dph_total": info.get("dph_total"),
        "ssh_host": info.get("ssh_host", ""),
        "ssh_port": info.get("ssh_port", ""),
        "start_date": info.get("start_date"),
        "image": info.get("image_uuid"),
    }


def probe_pod_ssh(host: str, port: int, key: Path) -> dict[str, str]:
    if not host or not port:
        return {"reachable": "no-ip"}
    reachable, _ = ssh_exec(host, int(port), "echo SSH_OK", key, timeout=8)
    if not reachable:
        return {"reachable": "no"}
    out: dict[str, str] = {"reachable": "yes"}
    probes = [
        ("nvidia_smi",
         "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
         "--format=csv,noheader,nounits"),
        ("py_proc",
         "ps -o pid,etime,cmd -C python3 --no-headers 2>/dev/null | head -3; "
         "ps -o pid,etime,cmd -C python --no-headers 2>/dev/null | head -3"),
        ("job_log_tail", "tail -25 /tmp/job.log 2>/dev/null"),
        ("monitor_log_tail", "tail -10 /tmp/monitor.log 2>/dev/null"),
        ("anchor_json_local",
         "ls -la /workspace/data/results/m2_anchor_*.json results/m2_anchor_*.json "
         "2>/dev/null"),
        ("disk_usage", "df -h / | tail -1"),
    ]
    for label, cmd in probes:
        _, payload = ssh_exec(host, int(port), cmd, key, timeout=15)
        out[label] = payload.strip()
    return out


def probe_runpod_s3(job: str) -> list[str]:
    try:
        prefix = f"{job}/"
        keys = _STORE._list_all(prefix=prefix)
        names = [(o.get("Key") if isinstance(o, dict) else o) for o in keys]
        interesting = [k for k in names
                       if "m2_anchor_" in k or "done.json" in k
                       or "progress.json" in k or "error.json" in k
                       or "heartbeat.json" in k]
        return interesting or [f"<no result objects under {prefix} ({len(names)} keys total)>"]
    except Exception as e:
        return [f"<storage list error: {e}>"]


def probe_r2_cloudflare(bucket: str) -> list[str]:
    """List interesting objects in a Cloudflare R2 bucket."""
    try:
        import boto3  # type: ignore
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{_R2_CFG['account_id']}.r2.cloudflarestorage.com",
            aws_access_key_id=_R2_CFG["access_key"],
            aws_secret_access_key=_R2_CFG["secret_key"],
            region_name="auto",
        )
        keys: list[str] = []
        token = None
        while True:
            kw = {"Bucket": bucket}
            if token:
                kw["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kw)
            for o in resp.get("Contents", []):
                keys.append(o["Key"])
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        interesting = [k for k in keys
                       if "m2_anchor_" in k or k == "done.json"
                       or k == "progress.json" or k == "error.json"
                       or k == "heartbeat.json" or k == "error_daemon.json"
                       or k == "done_daemon.json" or k == "job.log.live"]
        return interesting or [f"<no interesting objects in {bucket} ({len(keys)} keys total)>"]
    except Exception as e:
        return [f"<r2 list error: {e}>"]


def fetch_r2_text(bucket: str, key: str, max_bytes: int = 4096) -> str:
    try:
        import boto3  # type: ignore
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{_R2_CFG['account_id']}.r2.cloudflarestorage.com",
            aws_access_key_id=_R2_CFG["access_key"],
            aws_secret_access_key=_R2_CFG["secret_key"],
            region_name="auto",
        )
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read(max_bytes)
        return body.decode("utf-8", errors="replace")
    except Exception as e:
        return f"<fetch err: {e}>"


def fmt_section(title: str) -> str:
    return f"\n{'=' * 70}\n  {title}\n{'=' * 70}"


def render_pod(pod_meta: dict) -> str:
    out = []
    out.append(fmt_section(f"{pod_meta['name']}  [{pod_meta['id']}]"))

    if pod_meta["kind"] == "runpod":
        api = probe_pod_runpod_api(pod_meta["id"])
        out.append("[runpod API]")
        for k, v in api.items():
            out.append(f"  {k}: {v}")
        if "error" in api:
            return "\n".join(out)
        ssh_info = probe_pod_ssh(api.get("ssh_host") or "",
                                 api.get("ssh_port") or 0, SSH_KEY_RP)
        out.append("[ssh probes]")
        for k, v in ssh_info.items():
            if "\n" in str(v):
                out.append(f"  {k}:")
                for line in str(v).splitlines():
                    out.append(f"    {line}")
            else:
                out.append(f"  {k}: {v}")
        out.append("[runpod-s3 listing]")
        for k in probe_runpod_s3(pod_meta["job"]):
            out.append(f"  {k}")

    elif pod_meta["kind"] == "vast":
        api = probe_pod_vast_api(pod_meta["id"])
        out.append("[vast API]")
        for k, v in api.items():
            out.append(f"  {k}: {v}")
        if "error" in api:
            return "\n".join(out)
        ssh_info = probe_pod_ssh(api.get("ssh_host") or "",
                                 api.get("ssh_port") or 0, SSH_KEY_VAST)
        out.append("[ssh probes]")
        for k, v in ssh_info.items():
            if "\n" in str(v):
                out.append(f"  {k}:")
                for line in str(v).splitlines():
                    out.append(f"    {line}")
            else:
                out.append(f"  {k}: {v}")
        out.append(f"[r2 listing  bucket={pod_meta['bucket']}]")
        listing = probe_r2_cloudflare(pod_meta["bucket"])
        for k in listing:
            out.append(f"  {k}")
        # If heartbeat exists, fetch a snippet for diagnostic richness
        if any("heartbeat.json" == k for k in listing):
            hb = fetch_r2_text(pod_meta["bucket"], "heartbeat.json")
            out.append("  [heartbeat.json]")
            for line in hb.splitlines()[:20]:
                out.append(f"    {line}")
    return "\n".join(out)


def main() -> None:
    t0 = time.time()
    print(fmt_section(f"smart_pod_monitor  {time.strftime('%H:%M:%S')}"))
    for p in PODS:
        print(render_pod(p))
    print(f"\n[done in {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
