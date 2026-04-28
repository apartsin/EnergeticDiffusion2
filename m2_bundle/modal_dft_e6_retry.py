"""Stage 4 retry: re-run DFT for E6 only on a fresh Modal A100 container.

The first run failed with CUSOLVER_STATUS_EXECUTION_FAILED inside the
density-fitting Cholesky decomposition. The fault is transient (cuSOLVER
race / kernel) and a fresh container typically clears it.

Usage:
    python -m modal run modal_dft_e6_retry.py
"""
from __future__ import annotations
import base64, json, sys, time
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
RESULTS_LOCAL = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

E6_ID = "E6"
E6_SMILES = "O=[N+]([O-])c1c[nH]c([N+](=O)[O-])c1"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install("torch==2.4.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("pyscf==2.8.0", "gpu4pyscf-cuda12x==1.4.0",
                 "rdkit-pypi", "geometric", "numpy<2")
    .add_local_dir(str(HERE), remote_path="/m2_bundle",
                    ignore=lambda p: not str(p).endswith(".py"))
)

app = modal.App("dgld-eset-dft-e6retry", image=image)


@app.function(gpu="A100", timeout=4 * 60 * 60)
def run_e6_remote(atom_refs_b64: str) -> dict:
    sys.path.insert(0, "/m2_bundle")
    import m2_dft_pipeline as dft  # type: ignore
    import torch  # type: ignore
    assert torch.cuda.is_available(), "CUDA missing"
    print(f"[dft:E6] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    workdir = Path("/tmp/m2_work_e6"); workdir.mkdir(exist_ok=True, parents=True)
    atom_refs = json.loads(base64.b64decode(atom_refs_b64).decode())
    (workdir / "m2_atom_refs.json").write_text(json.dumps(atom_refs, indent=2))
    lead = {"id": E6_ID, "smiles": E6_SMILES, "name": E6_ID}
    t0 = time.time()
    try:
        dft.run_lead(lead, atom_refs["B3LYP_631Gss"],
                       atom_refs["wB97XD_def2TZVP"], workdir, use_gpu=True)
    except Exception as e:
        return {"id": E6_ID, "smiles": E6_SMILES, "error": str(e),
                "_elapsed_s": time.time() - t0}
    out_path = workdir / f"m2_lead_{E6_ID}.json"
    payload = json.loads(out_path.read_text()) if out_path.exists() else {}
    payload["_elapsed_s"] = time.time() - t0
    return payload


@app.local_entrypoint()
def main():
    atom_refs = json.loads((HERE / "results" / "m2_atom_refs.json").read_text())
    b64 = base64.b64encode(json.dumps(atom_refs).encode()).decode()
    print(f"[dft] retry E6 on a single A100 container")
    payload = run_e6_remote.remote(b64)
    out_path = RESULTS_LOCAL / f"m2_lead_{E6_ID}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    if "error" in payload or payload.get("errors"):
        err = payload.get("error") or payload.get("errors")
        print(f"[dft] E6 RETRY FAILED: {err}")
    else:
        print(f"[dft] E6 retry OK: rho_dft={payload.get('rho_dft')}, "
              f"HOF={payload.get('HOF_kJmol_wb97xd')}, "
              f"t={payload.get('_elapsed_s', 0):.0f}s -> {out_path}")
