# DFT Validation Guide

How to run the M2 DFT validation pipeline (`m2_bundle/m2_dft_pipeline.py`) on
demand on vast.ai for energetic CHNO molecules. Recovered from 4+ cycles of
silent CPU fallback, image-name confusion, stale-timeout kills, and version
mismatch.

## TL;DR — the working recipe (current best)

```bash
# 1. DO NOT use the vastai/pytorch image — it is misleadingly named
#    and ships Ubuntu 22.04 with NO torch pre-installed. Use a real
#    pytorch image where torch + matching CUDA are baked in.
IMAGE="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime"

# 2. Pin torch BEFORE pip-installing gpu4pyscf — otherwise pip resolves
#    the latest torch 2.11 with mismatched CUDA, and gpu4pyscf-cuda12x's
#    CUDA kernels fail to load (silently, in the existing pipeline).
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install gpu4pyscf-cuda12x pyscf rdkit-pypi geometric

# 3. Probe the GPU stack BEFORE running the pipeline:
python3 -c "from gpu4pyscf import dft; import torch; \
  assert torch.cuda.is_available(); print('[smoke] gpu4pyscf+cuda OK')"

# 4. Now launch the pipeline
python3 -u m2_dft_pipeline.py --smiles m2_smiles.json --results results --anchors

# 5. Always set --stale-timeout 7200 on the gpu_runner.py 'run' command;
#    the default 600s kills the pod during slow installs.
# 6. NEVER spawn a 'gpu_runner.py recover' alongside a DFT job — recover's
#    default stale-timeout is also 600s and will force-destroy the pod
#    during long-running SCF iterations.
```

## What goes wrong, in cycle order

### Cycle 1 — `vastai/pytorch` image, default 600s stale-timeout
**Symptom**: pod hits stale-timeout at 600s while still pip-installing torch
(530 MB download). Job destroyed before any DFT step starts. R2 bucket has
~9 metadata files and an `error.json` with `"stage": "stale-timeout"`.
**Cause**: the `vastai/pytorch` image, despite the name, ships Ubuntu 22.04
without torch. The pip install of `gpu4pyscf-cuda12x` triggers a fresh
torch download, which the default 600s stale-timeout cannot accommodate.
**Fix**: pass `--stale-timeout 7200` to `gpu_runner.py run`.

### Cycle 2 — recover-monitor footgun
**Symptom**: a second pod (`--stale-timeout 7200`) running fine, then I
spawn a `gpu_runner.py recover --job-id ...` to peek at progress — the
recover's polling uses a default 600s stale-timeout and force-destroys
the pod mid-pip-install.
**Cause**: `gpu_runner.py recover` does not inherit the parent run's
stale-timeout; it polls with its own 600s default.
**Fix**: **never** spawn a separate recover monitor for a DFT job. The
original `gpu_runner.py run` process self-monitors safely with whatever
stale-timeout you passed to it. Use `vastai show instances --raw` if
you need a real-time peek without spawning another monitor.

### Cycle 3 — silent CPU fallback in `_get_mf()`
**Symptom**: pod alive, status `running`, GPU util **0%**, CPU util **12%**
(≈ 3 cores busy on a 24-core node), 65 minutes elapsed, no progress visible.
Killed and inspected: in `m2_dft_pipeline.py` lines 105-122, `_get_mf()`
catches `ImportError` from `from gpu4pyscf import dft` and silently falls
through to CPU PySCF. CPU-mode B3LYP/6-31G(d) on a 21-atom RDX takes
6–30+ hours per molecule, so the pod was actually grinding through one
SCF iteration on CPU the entire time.
**Cause**: chained from cycle 4 below — gpu4pyscf-cuda12x failed to import
because of a torch+CUDA version mismatch, and the existing pipeline hides
the failure.
**Fix**: patch `_get_mf()` to log the ImportError loudly instead of
silently falling back; add an early-fail probe in `main()` that exits
non-zero if `from gpu4pyscf import dft` fails AND `--cpu` was not passed.

### Cycle 4 — torch + gpu4pyscf-cuda12x version mismatch
**Symptom**: cycle 3's silent-CPU-fallback root cause. `pip install
gpu4pyscf-cuda12x` without pinning torch first causes pip to resolve
the latest torch (currently 2.11 with CUDA 12.x runtime), but
`gpu4pyscf-cuda12x`'s CUDA kernels are built against a specific torch
ABI (commonly 2.4.x) and silently fail to load at runtime.
**Cause**: pip dependency resolution does not pin torch when installing
gpu4pyscf-cuda12x.
**Fix**: install torch **first** with an explicit version + CUDA index,
then install gpu4pyscf:
```
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install gpu4pyscf-cuda12x pyscf rdkit-pypi geometric
```

## Diagnostic checklist when DFT seems stuck

**Prevention first**: pass `--host-probe probes/dft_probe.sh --destroy-on-probe-fail` to `gpu_runner.py run`. The probe SCPs `dft_probe.sh` to the pod after SSH is up but before the main onstart job runs, and validates `nvidia-smi`, `import torch`, and `torch.cuda.is_available()` with a 300 s timeout. Non-zero exit aborts the run before any DFT-side cost is spent. Probe lives at `C:\Users\apart\Projects\claude-skills\gpu2vast\probes\dft_probe.sh`. Caveat: the probe runs on the same pod that will be billed, so a failed probe still costs ~30 s of pod time, still far cheaper than discovering the failure 5+ min into SCF.

The three classic failure-mode signatures, in increasing severity:

| Signature | Meaning | Action |
|---|---|---|
| `actual_status=loading` AND `duration>600` AND `inet_down` flat | Image pull never completed; broken host | Kill, relaunch on a different offer (filter by `reliability>0.99`) |
| `actual_status=running` AND `gpu_util=0%` AND `cpu_util` 10–20% AND `inet_*` flat | **Silent CPU fallback**: gpu4pyscf failed to load CUDA kernels; PySCF is grinding on CPU at hours-per-molecule | Kill immediately; check torch+gpu4pyscf ABI |
| `actual_status=running` AND `gpu_util=0%` AND `cpu_util<5%` AND duration>10 min | Pod alive but pipeline crashed silently (or smoke script never started) | SSH-inspect; if recovery isn't fast, kill |

### Diagnose

In order, fastest to slowest:

1. **Vast.ai metadata** (free, instant):
   ```bash
   vastai show instances --raw | jq '.[] | select(.id==N) | {actual_status, gpu_util, cpu_util, gpu_temp, duration, inet_up, inet_down, mem_usage}'
   ```
   GPU util > 50% during SCF means the stack works. GPU util 0% + CPU util ~10–20% on a busy node = silent CPU fallback. GPU 0% + CPU <5% = pipeline crashed or stalled.

2. **Local launcher buffer**: `tail -50 ~/.claude/.../tasks/<jobid>.output`. **Note**: gpu_runner.py buffers SSH-polled stdout and only flushes on completion, so this lags reality.

3. **R2 bucket incremental sync** (free): the bucket is named `gpu2vast-<job-name>-<timestamp>`. Partial outputs (e.g. `dft_smoke.log`, per-lead JSONs from a resume-aware pipeline) end up there. Pull via boto3 even before the job finishes.

4. **Direct SSH** (free, ~5 s per command):
   ```bash
   vastai ssh-url N    # prints ssh://root@host:port
   ssh -i C:/Users/apart/.claude/skills/gpu2vast/keys/ssh/gpu2vast_ed25519 -p PORT root@HOST
   ```
   Then `nvidia-smi`, `ps -ef`, `tail /var/log/onstart.log`, `tail dft_smoke.log`. **Caveat**: vast.ai's bastion IP (3.81.22.154 region-dependent) sometimes blocks unsolicited SSH connections from new client IPs; the gpu_runner.py polling channel works because the key was registered at instance-creation time. If direct SSH closes immediately, fall back to `vastai ssh-host N` which uses a pre-authorised tunnel.

5. **Heartbeat thread** (already in `m2_dft_pipeline.py`): "elapsed=Xs" every 60 s. Visible in the local stdout buffer once the pipeline starts.

### Alleviate (recover gracefully)

- `--stale-timeout 7200` on every `gpu_runner.py run`. Default 600 s is too aggressive for slow installs.
- **Early-fail probes in `run.sh`** before launching the pipeline. The `dft_smoke_run.sh` template runs `from gpu4pyscf import dft; assert torch.cuda.is_available()` first; aborts in 5 s if the GPU stack is broken.
- **Per-molecule wall-clock alarm** via `signal.alarm(7200)` (already in `m2_dft_pipeline.py`). One hung optimisation no longer kills the whole bundle.
- **Resume-from-cache** at top of `run_lead()` (already in `m2_dft_pipeline.py`). A failed pod restart skips completed leads.
- `--max-hours` cap on every launch. Quietly stuck pods can't bill > $0.43 × max-hours.
- **Never spawn `gpu_runner.py recover`** alongside `run`. The recover loop's default 600 s stale-timeout will destroy a pod doing legitimate slow work; the `run` process self-monitors safely.

### Prevent (kill failure modes upstream)

- **Use `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`** (real torch image, ~5 GB). NOT `vastai/pytorch` (Ubuntu 22.04 with no torch despite the name — cycles 1–2 root cause).
- **Pin torch BEFORE installing gpu4pyscf-cuda12x**:
  ```bash
  pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
  pip install gpu4pyscf-cuda12x pyscf rdkit-pypi geometric
  ```
  Without this, pip resolves the latest torch with mismatched CUDA, and gpu4pyscf's CUDA kernels silently fail (cycle 4).
- **Hard-fail on silent CPU fallback** (already added): `_get_mf()` no longer swallows `ImportError`; logs a loud warning. `main()` exits non-zero unless `--cpu` is explicit.
- **Filter the offer pool**: `vastai search offers 'gpu_name=A100_PCIE rentable=true reliability>0.99 verification=verified inet_down>500'`. Most stuck-loading hours come from low-reliability hosts.
- **Pre-launch host-probe** (now wired into `gpu_runner.py` — see CLAUDE.md): runs a 30-s SSH-side probe (`from gpu4pyscf import dft; assert torch.cuda.is_available()`) BEFORE uploading the data + main script. If the probe fails, exit non-zero so the caller can retry on a different offer. Saves the 60-min stuck-loading hours.
- **Smoke before pipeline**: water → RDX in `dft_smoke.py` is < 5 min on A100. If it passes, full pipeline launches on the same pod via `gpu_runner.py rerun --instance-id N` (no second pod boot).
- **`--keep-alive` for iteration**: don't pay for fresh pod boots between cycles.

## Per-molecule cost on the working stack (when it works)

Estimates from the Apr-27 successful run (12 leads + RDX + TATB anchors):

| GPU | Per-molecule wall-clock (CHNO 12–25 atoms) | Pod cost |
|---|---|---|
| A100_PCIE 40GB | 3–8 min | $0.40–0.60/hr |
| RTX 4090 24GB | 5–15 min | $0.30–0.50/hr |
| RTX 2060 6GB (local) | not feasible — out of memory on Hessian | n/a |

Total wall for 12 leads + 2 anchors on A100_PCIE: **~60–80 min**, ~$0.50.

## File locations on `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`

| What | Path |
|---|---|
| Python | `/opt/conda/bin/python3` (3.11) |
| pip-installed gpu4pyscf | `/opt/conda/lib/python3.11/site-packages/gpu4pyscf/` |
| pip-installed pyscf | `/opt/conda/lib/python3.11/site-packages/pyscf/` |
| pip-installed geometric | `/opt/conda/lib/python3.11/site-packages/geometric/` |
| CUDA runtime | bundled with torch (`torch.version.cuda` reports CUDA 12.4) |

## Smoke vs full pipeline

- **`dft_smoke.py`** (1 molecule, water → RDX, ~5 min on A100): use as a
  go/no-go canary on every cold pod start. Exits non-zero with a probe
  number on first failure. Add it as the **first** thing called by run.sh
  after the pip install completes; if the smoke fails, do not start the
  full pipeline.
- **`m2_dft_pipeline.py`** (N molecules, 5–15 min each): only run after
  the smoke passes. With the resume-from-cache patch (#4 in the audit),
  any restart will skip already-completed leads.

## Do-not-do checklist

- [x] Don't use `vastai/pytorch`. Use `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`.
- [x] Don't run `pip install gpu4pyscf-cuda12x` before pinning torch.
- [x] Don't rely on `_get_mf()`'s silent CPU fallback. Patch it to fail loudly.
- [x] Don't spawn `gpu_runner.py recover` for a DFT job. Period.
- [x] Don't omit `--stale-timeout 7200` on `gpu_runner.py run`.
- [x] Don't omit `--keep-alive` if you want to iterate via `rerun` on the same pod.
- [x] Don't run `cleanup` until you have what you need. Use `--keep-alive`
      and `rerun` for iteration, then explicit `cleanup --job-id X` once
      the data is downloaded and verified.

## When to invoke this guide

- User asks to "rerun DFT", "validate a new lead with DFT", "extend the
  DFT bundle", "calibrate against a new anchor".
- Reviewer requests additional first-principles validation.
- A new chem-pass lead emerges from a generation run and needs a `kj_dft_cal`
  K-J recompute on calibrated DFT inputs.

## Pipeline patches assumed to be present

(All applied in commits e9d957e and follow-ups; see `m2_bundle/m2_dft_pipeline.py`.)

- **#4 resume-from-cache** at top of `run_lead()`: skips compounds whose
  per-lead JSON already has `HOF_kJmol_wb97xd`.
- **#5 per-molecule wall-clock guard** via `signal.alarm(timeout_s)`
  (POSIX-only); finally-block clears the alarm on exit. Default 7200s.
- **#6 anchor-calibrated rho/HOF** post-loop: fits `rho_cal = a*rho_DFT + b`
  and a constant `HOF_cal = HOF_DFT + c` on RDX/TATB literature values,
  recomputes K-J on calibrated inputs, writes `m2_calibration.json`.
- **#7 min_real_freq_cm1** corrected to select from real-mode subset.
- **H5 freqs_cm1** now stores wavenumbers (was hartree).

## Companion files

- `m2_bundle/m2_dft_pipeline.py` — main pipeline.
- `m2_bundle/dft_smoke.py` — single-molecule smoke test (water → RDX),
  written by the debug-cycle agent currently in flight.
- `m2_bundle/dft_smoke_run.sh` — vast.ai run.sh that does the pip pinning
  and runs the smoke before the full pipeline.
- `DFT_DEBUG_LOG.md` — append-only narrative of what was tried and what
  worked across debugging cycles.
- `m2_bundle/results/m2_summary.json` — per-lead aggregates with raw and
  calibrated rho/HOF/K-J fields.
- `m2_bundle/results/m2_calibration.json` — fit coefficients for the
  current 2-anchor calibration.
