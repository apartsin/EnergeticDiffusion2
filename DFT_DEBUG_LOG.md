# DFT Pipeline vast.ai Debug Log

Running narrative of every cycle attempted to get `m2_bundle/m2_dft_pipeline.py`
working reliably on a vast.ai A100 pod. Companion file: `DFTGuide.md` (the
distilled how-to once the recipe stabilises).

Reference target: water (H2O, 3 atoms; canary) and RDX (21 atoms; full pipeline).

## Cycle 0 (offline preparation)

- Wrote `m2_bundle/dft_smoke.py`: standalone instrumented smoke test, 8 probes,
  exits with the failing probe number. No silent CPU fallback.
- Wrote `m2_bundle/dft_smoke_run.sh`: pins `torch==2.4.1+cu124` BEFORE
  installing `gpu4pyscf-cuda12x pyscf rdkit-pypi geometric`. Early-fail probe
  on `from gpu4pyscf import dft`.
- Patched `m2_bundle/m2_dft_pipeline.py`:
  - `main()` now hard-fails (sys.exit(1)) if gpu4pyscf import or CUDA is
    unavailable when `--cpu` was not passed.
  - `_get_mf()` no longer silently swallows ImportError; logs a loud "THIS
    WILL BE CPU AND VERY SLOW" warning if it has to fall back.

## Cycle 1 (planned)

- Image: `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`
- Pip order: torch is pre-baked; only need `pip install gpu4pyscf-cuda12x pyscf rdkit-pypi geometric`.
- gpu_runner.py invocation:
  ```
  python gpu_runner.py run \
      --script m2_bundle/dft_smoke_run.sh \
      --data m2_bundle \
      --gpu A100_PCIE --max-price 1.20 --max-hours 1 \
      --stale-timeout 7200 \
      --image pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime \
      --skip-smoke --keep-alive \
      --results-pattern 'results/*' \
      --local-results m2_bundle/results_smoke_cycle1
  ```
- DO NOT spawn `gpu_runner.py recover` for the run; it self-monitors.
- DO NOT `cleanup` until the user confirms.
- Expected probe sequence to pass: 1, 2, 3, 4, 5, 6, 7, 8.
- Status: not yet executed (pending user authorisation to spend on a pod).

## Notes

- Failure history (already known) prior to cycle 1:
  - `vastai/pytorch` image is misleading; despite the name, base is
    Ubuntu 22.04 with NO torch pre-baked. Every job pip-installed torch
    (~530 MB), which got killed by the default 600 s stale-timeout.
  - Spawning `gpu_runner.py recover` after `run` polled with 600 s
    stale-timeout and destroyed the pod mid-pip. Do not do this.
  - Silent CPU fallback in `_get_mf()` lines 115-121 (now patched).
  - Without pinning torch first, `gpu4pyscf-cuda12x` pulls latest torch
    (currently 2.11) with mismatched CUDA -> import fails or silent kernel
    degradation.

- The user has explicitly asked NOT to spend on a new pod cycle until they
  see and approve the deliverables. Cycles 1+ will be appended once the user
  authorises.
