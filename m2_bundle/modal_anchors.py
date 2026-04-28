"""
Modal app for the 6-anchor DFT calibration extension.

Runs the four added anchors (HMX, PETN, FOX-7, NTO) as four parallel A100
containers, then refits (a, b, c) on the union with the cached RDX/TATB.

Usage (after `modal setup`):
    python -m modal run modal_anchors.py

Outputs land in:
    E:/Projects/EnergeticDiffusion2/m2_bundle/results_modal/m2_anchor_<ID>.json
    E:/Projects/EnergeticDiffusion2/m2_bundle/results_modal/m2_calibration_6anchor.json

Design notes:
- One container per anchor, gpu="A100", parallel via map().
- Image is built once on first run (~5 min) and cached forever after, so
  subsequent launches are ~30 s cold start.
- The pipeline source files (m2_dft_pipeline.py, m2_anchors_extension.py)
  are mounted as a Modal Mount, not COPY'd into the image, so iterating on
  the driver does not invalidate the image cache.
- Cached atomic-reference SCF results (m2_atom_refs.json, m2_lead_RDX.json,
  m2_lead_TATB.json) are passed into each container as base64 strings via
  function args - simpler than mounting and re-reading from disk.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
RESULTS_LOCAL = HERE / "results_modal"
RESULTS_LOCAL.mkdir(exist_ok=True)

# -------- image --------------------------------------------------------------
# nvidia/cuda 12.4 devel image is the published gpu4pyscf-cuda12x match.
# Add Python 3.11 (gpu4pyscf wheels exist for 3.11; not yet for 3.14).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install(
        # Pin torch FIRST so gpu4pyscf-cuda12x links against the correct ABI.
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "pyscf==2.7.0",
        "gpu4pyscf-cuda12x==1.4.0",
        "rdkit-pypi",
        "geometric",
        "numpy<2",
    )
    # Modal 1.x: add local Python source files to the image at runtime
    # (no rebuild on edit).
    .add_local_dir(
        str(HERE),
        remote_path="/m2_bundle",
        ignore=lambda p: not str(p).endswith(".py"),
    )
)

app = modal.App("dgld-anchor-calibration", image=image)


# -------- per-anchor function -----------------------------------------------
@app.function(
    gpu="A100",
    timeout=4 * 60 * 60,  # 4 h hard cap per anchor
)
def compute_anchor_remote(
    anchor_id: str,
    atom_refs_b64: str,
) -> dict:
    """Run the full B3LYP opt + Hessian + wB97X-D SP for one anchor.

    Returns the per-anchor JSON dict (same schema as m2_anchor_<ID>.json).
    """
    sys.path.insert(0, "/m2_bundle")
    import m2_dft_pipeline as dft  # type: ignore
    from m2_anchors_extension import (   # type: ignore
        ANCHOR_LIT, compute_anchor as _compute_anchor,
    )

    # Sanity check: CUDA must be alive in the container.
    import torch  # type: ignore
    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    print(f"[modal:{anchor_id}] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Decode and write atomic refs to a tmp dir the driver expects.
    workdir = Path("/tmp/m2_work")
    workdir.mkdir(exist_ok=True, parents=True)
    atom_refs = json.loads(base64.b64decode(atom_refs_b64).decode())
    (workdir / "m2_atom_refs.json").write_text(json.dumps(atom_refs, indent=2))

    # Compute (idempotent - the function checks for cached output first).
    _compute_anchor(anchor_id, atom_refs, workdir, use_gpu=True)

    out_path = workdir / f"m2_anchor_{anchor_id}.json"
    if not out_path.exists():
        # The cached path uses m2_lead_<id>.json fallback; check both.
        alt = workdir / f"m2_lead_{anchor_id}.json"
        if alt.exists():
            out_path = alt
    return json.loads(out_path.read_text())


# -------- local entrypoint --------------------------------------------------
@app.local_entrypoint()
def main():
    """Drive the 4 added anchors in parallel and refit locally."""
    print("[local] reading cached atomic refs and prior 2-anchor results")
    atom_refs_path = HERE / "results" / "m2_atom_refs.json"
    if not atom_refs_path.exists():
        raise SystemExit(
            f"Cannot find {atom_refs_path}. Run the 2-anchor pipeline first "
            "or copy the cached file in."
        )
    atom_refs = json.loads(atom_refs_path.read_text())
    atom_refs_b64 = base64.b64encode(
        json.dumps(atom_refs).encode()
    ).decode()

    # Spawn one container per anchor, all in parallel.
    anchors = ["HMX", "PETN", "FOX7", "NTO"]
    print(f"[local] dispatching {len(anchors)} parallel A100 containers: {anchors}")
    args = [(aid, atom_refs_b64) for aid in anchors]
    results: dict[str, dict] = {}
    for aid, payload in zip(anchors, compute_anchor_remote.starmap(args)):
        if isinstance(payload, dict):
            results[aid] = payload
            (RESULTS_LOCAL / f"m2_anchor_{aid}.json").write_text(
                json.dumps(payload, indent=2)
            )
            print(f"[local] {aid}: rho_dft={payload.get('rho_dft')}, "
                  f"HOF={payload.get('HOF_kJmol_wb97xd')}")
        else:
            print(f"[local] {aid}: FAILED, payload={payload}")

    # Refit (a, b, c) on the union {RDX, TATB, HMX, PETN, FOX7, NTO}.
    print("[local] running 6-anchor refit")
    sys.path.insert(0, str(HERE))
    from m2_anchors_extension import (   # type: ignore
        ANCHOR_LIT, fit_calibration, loo_calibration,
    )

    # Pull cached RDX/TATB from prior results.
    cached_dir = HERE / "results"
    anchor_dft = {}
    for aid in ["RDX", "TATB"]:
        path = cached_dir / f"m2_lead_{aid}.json"
        if path.exists():
            d = json.loads(path.read_text())
            anchor_dft[aid] = {
                "rho_dft": d.get("rho_dft"),
                "HOF_kJmol_wb97xd": d.get("HOF_kJmol_wb97xd"),
            }
    for aid, d in results.items():
        anchor_dft[aid] = {
            "rho_dft": d.get("rho_dft"),
            "HOF_kJmol_wb97xd": d.get("HOF_kJmol_wb97xd"),
        }

    if len(anchor_dft) < 3:
        raise SystemExit(f"only {len(anchor_dft)} anchors converged; need >= 3")

    a, b, c, fit_info = fit_calibration(anchor_dft, ANCHOR_LIT)
    loo = loo_calibration(anchor_dft, ANCHOR_LIT)
    cal = {
        "n_anchors": len(anchor_dft),
        "anchors_used": list(anchor_dft.keys()),
        "a_rho": a, "b_rho": b, "c_hof_kJmol": c,
        "fit_residual": fit_info, "loo_residual": loo,
        "note": "6-anchor calibration on Modal (RDX+TATB cached, HMX/PETN/FOX7/NTO computed).",
    }
    out = RESULTS_LOCAL / "m2_calibration_6anchor.json"
    out.write_text(json.dumps(cal, indent=2))

    print(f"\n[local] === 6-anchor fit ===")
    print(f"  rho_cal = {a:.4f} * rho_DFT + {b:+.4f}  "
          f"(LOO RMS {loo['loo_rho_rms_g_cm3']:.3f} g/cm3)")
    print(f"  HOF_cal = HOF_DFT + ({c:+.1f}) kJ/mol  "
          f"(LOO RMS {loo['loo_hof_rms_kJmol']:.1f} kJ/mol)")
    print(f"  -> {out}")
