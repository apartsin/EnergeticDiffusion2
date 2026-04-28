"""Pod-2 driver: hedge against pod-1's 6-anchor DFT calibration job.

Computes ONLY the four new anchors (HMX, PETN, FOX-7, NTO) under wB97X-D3BJ
single-points on B3LYP/6-31G** geometries, with HMX FIRST so the rate-limiter
is attacked early on a separate hardware instance.

Writes per-anchor JSONs to results_pod2/ (separate namespace from pod-1's
results/), then exits BEFORE any refit or lead-recompute. The client-side
aggregator will compare pod-1 vs pod-2 outputs per anchor as a sanity check
and run the joint refit once both pods finish.

Imports compute_anchor + ANCHOR_LIT from m2_anchors_extension.py to keep the
two pods running identical SCF code paths.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import m2_dft_pipeline as dft
from m2_anchors_extension import ANCHOR_LIT, compute_anchor

# Pod-2 ordering: HMX first (28 atoms, the rate-limiter), then the rest.
POD2_ORDER = ["HMX", "PETN", "FOX7", "NTO"]


def main():
    dft._start_heartbeat()
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_pod2")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    use_gpu = not args.cpu

    if use_gpu:
        try:
            from gpu4pyscf import dft as _gpu_dft  # noqa
            import torch
            assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
            print(f"[pod2] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()
        except (ImportError, AssertionError) as e:
            print(f"[pod2] FATAL: gpu4pyscf or CUDA unavailable ({e})", flush=True)
            sys.exit(1)

    results_dir = Path(args.results); results_dir.mkdir(parents=True, exist_ok=True)

    atom_refs_path = results_dir / "m2_atom_refs.json"
    if not atom_refs_path.exists():
        print(f"[pod2] FATAL: missing {atom_refs_path}; upload the cached atom refs",
              flush=True)
        sys.exit(1)
    atom_refs = json.loads(atom_refs_path.read_text())
    print(f"[pod2] loaded cached atom refs from {atom_refs_path}"); sys.stdout.flush()

    print(f"[pod2] anchor order: {POD2_ORDER} (HMX first as rate-limiter hedge)",
          flush=True)

    completed, failed = [], []
    for aid in POD2_ORDER:
        t0 = time.time()
        print(f"\n[pod2] === anchor {aid} ===", flush=True)
        try:
            compute_anchor(aid, atom_refs, results_dir, use_gpu=use_gpu)
            print(f"[pod2] {aid} done in {time.time()-t0:.0f}s", flush=True)
            completed.append(aid)
        except Exception as e:
            print(f"[pod2] {aid} FAILED: {e}", flush=True)
            failed.append(aid)

    print(f"\n[pod2] === SUMMARY ===")
    print(f"[pod2] completed: {completed}")
    print(f"[pod2] failed:    {failed}")
    print(f"[pod2] outputs in {results_dir}/m2_anchor_<ID>.json")
    print(f"[pod2] (no refit / no lead-recompute; client aggregates both pods)")
    print(f"[pod2] === DONE ===")


if __name__ == "__main__":
    main()
