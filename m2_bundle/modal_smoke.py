"""Modal smoke test — run NTO (12 atoms) on one A100 to verify image, mounts,
CUDA, and the compute_anchor pipeline before spending Stage-4 budget.

Usage:
    python -m modal run modal_smoke.py
"""
from __future__ import annotations
import base64, json, sys, time
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
RESULTS_LOCAL = HERE / "results_modal"
RESULTS_LOCAL.mkdir(exist_ok=True)

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

app = modal.App("dgld-anchor-smoke", image=image)


@app.function(gpu="A100", timeout=4 * 60 * 60)
def smoke_remote(smiles: str, atom_refs_b64: str) -> dict:
    sys.path.insert(0, "/m2_bundle")
    import m2_dft_pipeline as dft  # type: ignore
    import torch  # type: ignore
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"[smoke] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    workdir = Path("/tmp/m2_work"); workdir.mkdir(exist_ok=True, parents=True)
    atom_refs = json.loads(base64.b64decode(atom_refs_b64).decode())
    (workdir / "m2_atom_refs.json").write_text(json.dumps(atom_refs, indent=2))
    lead = {"id": "NTO_SMOKE", "smiles": smiles, "name": "NTO_SMOKE"}
    t0 = time.time()
    out = dft.run_lead(lead, atom_refs["B3LYP_631Gss"],
                        atom_refs["wB97XD_def2TZVP"], workdir, use_gpu=True)
    elapsed = time.time() - t0
    out_path = workdir / f"m2_lead_NTO_SMOKE.json"
    payload = json.loads(out_path.read_text()) if out_path.exists() else {}
    payload["_elapsed_s"] = elapsed
    return payload


@app.local_entrypoint()
def main():
    smi = "O=C1NN=C(N1)[N+](=O)[O-]"
    atom_refs = json.loads((HERE / "results" / "m2_atom_refs.json").read_text())
    b64 = base64.b64encode(json.dumps(atom_refs).encode()).decode()
    print(f"[smoke] launching A100 for SMILES={smi}")
    res = smoke_remote.remote(smi, b64)
    out = RESULTS_LOCAL / "m2_smoke_NTO.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[smoke] -> {out}")
    print(f"[smoke] rho_dft={res.get('rho_dft')}, "
          f"HOF={res.get('HOF_kJmol_wb97xd')}, "
          f"elapsed={res.get('_elapsed_s'):.0f}s")
    if res.get("rho_dft") is None or res.get("HOF_kJmol_wb97xd") is None:
        raise SystemExit("[smoke] FAILED: missing rho or HOF in returned payload")
    print("[smoke] PASSED")
