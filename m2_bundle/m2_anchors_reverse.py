"""Reverse-order driver: sibling #4 to the live RunPod A6000 (default order)
and to the two vast.ai pods. Iterates [NTO, FOX7, PETN, HMX] (smallest molecules
first) so that NTO (12 atoms) and FOX-7 (14 atoms) are produced quickly,
hedging against the larger anchors stalling on a sibling.

Writes per-anchor JSONs to results_reverse/m2_anchor_<ID>.json. No refit, no
lead-recompute; the client-side aggregator collects winners across siblings.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import m2_dft_pipeline as dft
from m2_anchors_extension import ANCHOR_LIT, compute_anchor

# Reverse ordering: smallest first -> early wins on NTO/FOX7.
REVERSE_ORDER = ["NTO", "FOX7", "PETN", "HMX"]


def main():
    dft._start_heartbeat()
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_reverse")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    use_gpu = not args.cpu

    if use_gpu:
        try:
            from gpu4pyscf import dft as _gpu_dft  # noqa
            import torch
            assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
            print(f"[ext] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()
        except (ImportError, AssertionError) as e:
            print(f"[ext] FATAL: gpu4pyscf or CUDA unavailable ({e})", flush=True)
            sys.exit(1)

    results_dir = Path(args.results); results_dir.mkdir(parents=True, exist_ok=True)

    atom_refs_path = results_dir / "m2_atom_refs.json"
    if not atom_refs_path.exists():
        # Fall back to top-level upload location.
        top = Path("m2_atom_refs.json")
        if top.exists():
            results_dir.joinpath("m2_atom_refs.json").write_text(top.read_text())
        else:
            print(f"[ext] FATAL: missing {atom_refs_path}; upload the cached atom refs",
                  flush=True)
            sys.exit(1)
    atom_refs = json.loads((results_dir / "m2_atom_refs.json").read_text())
    print(f"[ext] loaded cached atom refs from {atom_refs_path}"); sys.stdout.flush()

    print(f"[ext] anchor order: {REVERSE_ORDER} (smallest first; reverse hedge)",
          flush=True)

    completed, failed = [], []
    for aid in REVERSE_ORDER:
        t0 = time.time()
        print(f"\n[ext] === anchor {aid} ===", flush=True)
        try:
            compute_anchor(aid, atom_refs, results_dir, use_gpu=use_gpu)
            print(f"[ext] {aid} done in {time.time()-t0:.0f}s", flush=True)
            completed.append(aid)
        except Exception as e:
            print(f"[ext] {aid} FAILED: {e}", flush=True)
            failed.append(aid)

    print(f"\n[ext] === SUMMARY ===")
    print(f"[ext] completed: {completed}")
    print(f"[ext] failed:    {failed}")
    print(f"[ext] outputs in {results_dir}/m2_anchor_<ID>.json")
    print(f"[ext] === DONE ===")


if __name__ == "__main__":
    main()
