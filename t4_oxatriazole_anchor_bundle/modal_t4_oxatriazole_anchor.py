"""T4: true 1,2,3,5-oxatriazole-class DFT anchor for E1.

Template (DISABLED at the local entrypoint level): copy-modified from
m8_bundle/modal_m8_oxatriazole_anchor.py.  See PRE_FLIGHT.md in this
directory for why this is currently disabled.

To activate:
    1. Set CANDIDATE_SMILES + CANDIDATE_RHO_EXP + CANDIDATE_HOF_EXP
       + CANDIDATE_D_EXP from a literature reference for a confirmed
       1,2,3,5-oxatriazole-class compound.
    2. Remove the `raise RuntimeError(...)` line in main().
    3. modal run modal_t4_oxatriazole_anchor.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
RESULTS_LOCAL = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# CANDIDATE STUB -- requires literature pre-flight before activation.
# ---------------------------------------------------------------------------
CANDIDATE_ID     = "OXAT-TBD"
CANDIDATE_SMILES = ""        # e.g. "Nc1nnon1" (4-amino-1,2,3,5-oxatriazole)
CANDIDATE_RHO_EXP    = 0.0   # g/cm3, experimental crystal density
CANDIDATE_HOF_EXP    = 0.0   # kJ/mol, condensed-phase 298 K
CANDIDATE_D_EXP      = 0.0   # km/s, experimental detonation velocity

# Existing 6-anchor calibration coefficients (paper Table 5 / §5.2.2)
CAL6_RHO_SLOPE     =  1.392
CAL6_RHO_INTERCEPT = -0.415
CAL6_HOF_OFFSET    = -206.7

ANCHOR_6 = {
    "RDX":   {"rho_exp": 1.806, "HOF_exp":  +66.0},
    "TATB":  {"rho_exp": 1.938, "HOF_exp": -154.0},
    "HMX":   {"rho_exp": 1.891, "HOF_exp":  +74.8},
    "PETN":  {"rho_exp": 1.778, "HOF_exp": -538.5},
    "FOX-7": {"rho_exp": 1.885, "HOF_exp": -133.9},
    "NTO":   {"rho_exp": 1.919, "HOF_exp": -110.4},
}
for _name, _d in ANCHOR_6.items():
    _d["rho_dft"] = (_d["rho_exp"] - CAL6_RHO_INTERCEPT) / CAL6_RHO_SLOPE
    _d["HOF_dft"] = _d["HOF_exp"] - CAL6_HOF_OFFSET


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install("torch==2.4.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("pyscf==2.8.0", "gpu4pyscf-cuda12x==1.4.0",
                 "rdkit-pypi", "geometric", "numpy<2")
    .add_local_dir(str(HERE), remote_path="/t4_bundle",
                   ignore=lambda p: not str(p).endswith(".py"))
)

app = modal.App("dgld-t4-oxatriazole-anchor", image=image)


@app.function(gpu="A100", timeout=6 * 60 * 60)
def run_oxatriazole_remote(smiles: str, mol_id: str) -> dict:
    """Same DFT pipeline as m8: ETKDGv3 -> B3LYP/6-31G(d) opt + Hessian
    -> wB97X-D3BJ/def2-TZVP single point -> Bondi-vdW density -> HOF."""
    # Implementation mirrors run_dntf_remote() in
    # m8_bundle/modal_m8_oxatriazole_anchor.py.  Kept intentionally as
    # an import-free stub to make the failure mode visible if someone
    # forgets to activate.
    raise RuntimeError(
        "T4 DFT pipeline body has been intentionally elided. "
        "Copy the run_dntf_remote() function from "
        "m8_bundle/modal_m8_oxatriazole_anchor.py once a viable "
        "1,2,3,5-oxatriazole-class candidate has been identified "
        "(see PRE_FLIGHT.md)."
    )


@app.local_entrypoint()
def main():
    raise RuntimeError(
        "T4 PRE-FLIGHT FAILED: no 1,2,3,5-oxatriazole-class compound "
        "with BOTH experimental rho_exp and experimental D_exp was "
        "located in the open literature. See PRE_FLIGHT.md for the "
        "literature search and §7 paper-language replacement. To "
        "activate this bundle once a candidate exists, see the docstring."
    )
