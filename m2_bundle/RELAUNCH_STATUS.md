# M2 6-anchor DFT calibration relaunch status

**Date:** 2026-04-28 14:48 UTC
**Status:** BLOCKED — RunPod SECURE cloud out of capacity

## Summary

Authored the live-diagnostics bundle and attempted three SECURE-cloud launches.
All three failed at the GPU search step (step 2/6) with "GPU not found.
Available: []" — RunPod SECURE has zero capacity right now for any 48GB+ SKU.

## Bundle assets created

- `m2_bundle/monitor_daemon.sh` — 60s heartbeat / job.log.live / per-anchor result push / progress.json / done.json|error.json (Bourne-Again, LF, +x)
- `m2_bundle/m2_anchors_run_v2.sh` — entry point: CUDA gate, deps, stage cached refs, launch daemon, run `m2_anchors_extension.py`, signal daemon to flush on exit (Bourne-Again, LF, +x)
- Both scripts use the runner's actual env-var names (`RUNPOD_STORAGE_*`), not the spec's `RUNPOD_S3_*`.

## Pre-flight cleanup

- pod-1 `p3i9y7jjd9i669` (m2-anchors-6cal-20260428-141506) — pod gone, S3 prefix deleted.
- pod-2 `s3jrxo7cfqqj1d` (m2-anchors-6cal-pod2-20260428-142312) — pod gone, S3 prefix deleted.

## Capacity probes (all failed at step 2/6)

| Attempt | Job ID | GPU | Cloud | Max $/hr | Result |
|---|---|---|---|---|---|
| 1 | job-20260428-144636 | NVIDIA_A40 | SECURE | 0.80 | "not available in Secure Cloud" |
| 2 | job-20260428-144659 | RTX_A6000 | SECURE | 0.80 | "not available in Secure Cloud" |
| 3 | job-20260428-144721 | A100_PCIE | SECURE | 2.00 | "not available in Secure Cloud" |
| 4 | job-20260428-144744 | RTX_A6000 | SECURE | 5.00 | "not available in Secure Cloud" |

All four S3 prefixes uploaded successfully (volume accessible) but the pod was
never created — capacity check rejected them and the runner cleaned up.

## Verification checkpoint reached

**0 of 5.** Pod was never created on SECURE cloud, so we never got to the
"minute 2 / runtime.uptimeSec > 0" check, let alone heartbeat or progress.
No heartbeat.json snippet exists yet because no daemon ever ran.

## Decision required from parent

Per spec constraint "Do NOT use community cloud (the failure mode we just hit)",
I am NOT proceeding to a COMMUNITY cloud relaunch without explicit approval.

Three options for the parent thread:

1. **Wait + retry SECURE.** Poll RunPod every 15-30 min for SECURE A40 /
   RTX_A6000 / A100 capacity to come back. Bundle is staged and ready.
2. **Allow COMMUNITY with stricter watchdog.** The original community-cloud
   failures were 21+ / 13+ minutes stuck at image-pull. With the new
   heartbeat-every-60s daemon we'd actually see that earlier and could
   auto-kill at, say, 8-10 min of empty `runtime` and retry. Risk: same
   capacity issue may recur on community.
3. **Switch to a different provider.** vast.ai (gpu2vast skill) typically has
   A40 / A6000 inventory; the same bundle scripts work with minor env-var
   renames.

Bundle is ready to relaunch the moment a path is chosen — `m2_anchors_run_v2.sh`
+ `monitor_daemon.sh` + cached refs. Estimated job runtime ~3-4h on A40.
