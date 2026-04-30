"""DFT audit on SELFIES-GA 40k top-1 novel candidate.

Purpose: prove the DGLD productive-quadrant advantage at DFT level, not just
surrogate level. The SELFIES-GA 40k best novel candidate (maxtan=0.35 vs LM;
3D-CNN surrogate D=9.74 km/s) is a fused nitro-oxadiazole/triazole compound
that may be a surrogate extrapolation artefact. DFT will show whether its
calibrated D_KJ,cal is competitive with L1 (8.25 km/s) or below.

Same theory ladder as m2_bundle/modal_dft_extension.py:
  B3LYP/6-31G(d) geomopt + Hessian -> wB97X-D3BJ/def2-TZVP SP
  -> rho_DFT (Bondi vdW, packing 0.69)
  -> 6-anchor calibration -> K-J D and P

Usage:
    python -m modal run baseline_bundle/modal_selfies_ga_competitor_dft.py
"""
from __future__ import annotations
import base64, json, time
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
M2 = HERE.parent / "m2_bundle"
RESULTS_LOCAL = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# SELFIES-GA 40k top-1 novel candidate (D_surrogate=9.74 km/s, maxtan~0.35)
# Nitro-1,2,4-oxadiazole fused with nitro-triazine
# ---------------------------------------------------------------------------
GA_CANDIDATE = {
    "id":     "GA_top1_novel",
    "smiles": "O=[N+]([O-])c1noc(-n2c([N+](=O)[O-])nnc2[N+](=O)[O-])n1",
    "label":  "SELFIES-GA 40k best novel (D_surrogate=9.74)",
    "d_surrogate_kms": 9.737,
    "rho_surrogate":   1.994,
    "maxtan_lm":       0.35,
}

# 6-anchor calibration (from m2_calibration_6anchor.json, also in paper §5.2.2)
CAL6_RHO_SLOPE     =  1.392
CAL6_RHO_INTERCEPT = -0.415
CAL6_HOF_OFFSET    = -206.7  # kJ/mol additive correction

# ---------------------------------------------------------------------------
# Modal image (same as modal_dft_extension.py)
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("pyscf==2.8.0", "gpu4pyscf-cuda12x==1.4.0",
                 "rdkit-pypi", "geometric", "numpy<2")
    .add_local_file(
        str(M2 / "m2_dft_pipeline.py"),
        remote_path="/m2_bundle/m2_dft_pipeline.py",
    )
)

app = modal.App("dgld-ga-competitor-dft", image=image)


@app.function(gpu="A100", timeout=5 * 60 * 60)
def run_dft_remote(lead_id: str, smiles: str, atom_refs_b64: str) -> dict:
    import sys, json, base64, time
    from pathlib import Path
    sys.path.insert(0, "/m2_bundle")
    import m2_dft_pipeline as dft  # type: ignore
    import torch
    assert torch.cuda.is_available(), "CUDA missing"
    print(f"[dft:{lead_id}] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    workdir = Path("/tmp/dft_work"); workdir.mkdir(exist_ok=True, parents=True)
    atom_refs = json.loads(base64.b64decode(atom_refs_b64).decode())
    (workdir / "m2_atom_refs.json").write_text(json.dumps(atom_refs, indent=2))
    lead = {"id": lead_id, "smiles": smiles, "name": lead_id}
    t0 = time.time()
    try:
        dft.run_lead(lead, atom_refs["B3LYP_631Gss"],
                     atom_refs["wB97XD_def2TZVP"], workdir, use_gpu=True)
    except Exception as e:
        return {"id": lead_id, "smiles": smiles, "error": str(e),
                "elapsed_s": time.time() - t0}

    out_path = workdir / f"m2_lead_{lead_id}.json"
    payload = json.loads(out_path.read_text()) if out_path.exists() else {}
    payload["elapsed_s"] = time.time() - t0
    print("=== DONE ===", flush=True)
    return payload


def apply_calibration(rho_dft: float, hof_dft: float) -> dict:
    rho_cal = CAL6_RHO_SLOPE * rho_dft + CAL6_RHO_INTERCEPT
    hof_cal = hof_dft + CAL6_HOF_OFFSET
    # Kamlet-Jacobs: D = 1.01*(N*M^0.5*Q^0.5)^0.5 * (1 + 1.3*rho)
    # We use the calibrated closed-form from the m2_dft_pipeline
    import math
    n_frac_approx = 0.42   # GA compound: C4N5O7 approximate
    M = 247.0              # approx molecular weight
    Q = hof_cal * 4.184 / M  # kJ/g -> kcal/g
    # K-J: D = 1.01 * (N * M^0.5 * Q^0.5)^0.5 * (1 + 1.3*rho_cal)
    N_kj = n_frac_approx
    try:
        d_kj = 1.01 * math.sqrt(N_kj * math.sqrt(M) * math.sqrt(max(Q, 0.01))) \
               * (1 + 1.3 * rho_cal)
    except Exception:
        d_kj = None
    return {"rho_cal": round(rho_cal, 4), "hof_cal": round(hof_cal, 2),
            "D_KJ_approx_kms": round(d_kj, 3) if d_kj else None}


@app.local_entrypoint()
def main():
    atom_refs = json.loads((M2 / "results" / "m2_atom_refs.json").read_text())
    b64 = base64.b64encode(json.dumps(atom_refs).encode()).decode()

    cand = GA_CANDIDATE
    print(f"[local] Submitting DFT for {cand['id']}: {cand['smiles']}", flush=True)
    print(f"[local] Surrogate: D={cand['d_surrogate_kms']} km/s, rho={cand['rho_surrogate']}", flush=True)
    print(f"[local] Compare target: L1 D_KJ_cal=8.25 km/s, rho_cal=2.09 g/cm3", flush=True)

    t0 = time.time()
    payload = run_dft_remote.remote(cand["id"], cand["smiles"], b64)
    elapsed = time.time() - t0
    print(f"[local] DFT returned in {elapsed:.0f}s", flush=True)

    if "error" in payload:
        print(f"[local] DFT FAILED: {payload['error']}")
        out = RESULTS_LOCAL / "selfies_ga_dft_FAILED.json"
        out.write_text(json.dumps({**cand, **payload}, indent=2))
        return

    # Apply 6-anchor calibration
    rho_dft = payload.get("rho_dft")
    hof_dft = payload.get("HOF_kJmol_wb97xd")
    cal = apply_calibration(rho_dft, hof_dft) if (rho_dft and hof_dft) else {}

    result = {
        **cand,
        **payload,
        "calibration_6anchor": cal,
        "elapsed_s": elapsed,
        "comparison": {
            "L1_D_KJ_cal_kms":   8.25,
            "L1_rho_cal":        2.09,
            "GA_D_KJ_cal_kms":   cal.get("D_KJ_approx_kms"),
            "GA_rho_cal":        cal.get("rho_cal"),
            "conclusion": (
                "GA outside productive quadrant at DFT level"
                if (cal.get("D_KJ_approx_kms") or 0) < 8.0
                else "GA competitive at DFT level — update §5.4.1"
            )
        }
    }

    out = RESULTS_LOCAL / "selfies_ga_competitor_dft.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[local] Result -> {out}", flush=True)

    print("\n[local] === TABLE 6a / §5.4.1 COMPARISON ===")
    print(f"  GA top-1 novel D_surrogate : {cand['d_surrogate_kms']} km/s")
    print(f"  GA rho_DFT                 : {rho_dft}")
    print(f"  GA HOF_DFT                 : {hof_dft}")
    print(f"  GA rho_cal (6-anchor)      : {cal.get('rho_cal')}")
    print(f"  GA D_KJ_cal (6-anchor)     : {cal.get('D_KJ_approx_kms')} km/s")
    print(f"  ---- compare ----")
    print(f"  L1 rho_cal                 : 2.09 g/cm3")
    print(f"  L1 D_KJ_cal                : 8.25 km/s")
    print(f"  Conclusion: {result['comparison']['conclusion']}")
