"""Stage 4: parallel A100 DFT (B3LYP opt + Hessian + wB97X-D SP) on the 10
extension-set leads E1..E10.

Mirrors modal_anchors.py — one A100 container per molecule, .starmap fan-out.

Usage:
    python -m modal run modal_dft_extension.py
"""
from __future__ import annotations
import base64, json, sys, time
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
PICKED = ROOT / "results" / "extension_set" / "e_set_picked_10.json"
RESULTS_LOCAL = HERE / "results"
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

app = modal.App("dgld-eset-dft", image=image)


@app.function(gpu="A100", timeout=4 * 60 * 60)
def run_lead_remote(lead_id: str, smiles: str, atom_refs_b64: str) -> dict:
    sys.path.insert(0, "/m2_bundle")
    import m2_dft_pipeline as dft  # type: ignore
    import torch  # type: ignore
    assert torch.cuda.is_available(), f"CUDA missing for {lead_id}"
    print(f"[dft:{lead_id}] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    workdir = Path("/tmp/m2_work"); workdir.mkdir(exist_ok=True, parents=True)
    atom_refs = json.loads(base64.b64decode(atom_refs_b64).decode())
    (workdir / "m2_atom_refs.json").write_text(json.dumps(atom_refs, indent=2))
    lead = {"id": lead_id, "smiles": smiles, "name": lead_id}
    t0 = time.time()
    try:
        dft.run_lead(lead, atom_refs["B3LYP_631Gss"],
                       atom_refs["wB97XD_def2TZVP"], workdir, use_gpu=True)
    except Exception as e:
        return {"id": lead_id, "smiles": smiles, "error": str(e),
                "_elapsed_s": time.time() - t0}
    out_path = workdir / f"m2_lead_{lead_id}.json"
    payload = json.loads(out_path.read_text()) if out_path.exists() else {}
    payload["_elapsed_s"] = time.time() - t0
    return payload


@app.local_entrypoint()
def main():
    picked = json.loads(PICKED.read_text())["picked"]
    atom_refs = json.loads((HERE / "results" / "m2_atom_refs.json").read_text())
    b64 = base64.b64encode(json.dumps(atom_refs).encode()).decode()
    args = [(r["id"], r["smiles"], b64) for r in picked]
    print(f"[dft] dispatching {len(args)} parallel A100 containers")
    for (lid, smi, _), payload in zip(args, run_lead_remote.starmap(args)):
        out_path = RESULTS_LOCAL / f"m2_lead_{lid}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        if "error" in payload:
            print(f"[dft] {lid} FAILED: {payload['error']}")
        else:
            print(f"[dft] {lid}: rho_dft={payload.get('rho_dft')}, "
                  f"HOF={payload.get('HOF_kJmol_wb97xd')}, "
                  f"t={payload.get('_elapsed_s', 0):.0f}s -> {out_path}")
