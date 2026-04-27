"""Light local smoke for M2 pipeline (Windows + Python 3.14, no pyscf).

Tests only the non-pyscf bits:
    1. RDKit imports + SMILES parse
    2. m2_dft_pipeline non-DFT helpers (formula, density, K-J)
    3. SMILES list loads
The pyscf-dependent path is verified once the runpod pod boots.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def step(s):
    print(f"[smoke] {s} ...", end=" ", flush=True)


def ok():
    print("OK")


def fail(m):
    print(f"FAIL: {m}")
    sys.exit(1)


# 1. SMILES list
step("load m2_smiles.json")
data = json.loads((HERE / "m2_smiles.json").read_text())
assert "leads" in data and len(data["leads"]) >= 3, "need >=3 leads"
assert "anchors" in data and len(data["anchors"]) >= 1, "need anchor refs"
ok()

# 2. RDKit on each SMILES
step("RDKit parse all leads + anchors")
from rdkit import Chem
for entry in data["leads"] + data["anchors"]:
    smi = entry["smiles"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        fail(f"RDKit fails on {entry['id']}: {smi}")
ok()

# 3. Pipeline non-DFT helpers
step("import m2_dft_pipeline non-DFT helpers")
from m2_dft_pipeline import (smiles_to_xyz, molecular_formula,
                              density_from_volume, kamlet_jacobs,
                              hof_from_atomization, ATOMIC_HOF_298)
ok()

step("smiles_to_xyz on all leads + anchors")
for entry in data["leads"] + data["anchors"]:
    atoms, charge, n = smiles_to_xyz(entry["smiles"])
    if n < 5: fail(f"{entry['id']} has only {n} atoms, suspicious")
ok()

step("molecular_formula round-trip")
atoms, _, _ = smiles_to_xyz(data["leads"][0]["smiles"])
formula = molecular_formula(atoms)
print(f"\n[smoke]  L1 formula: {formula}", end=" ")
assert "N" in formula and "O" in formula
ok()

step("density_from_volume on RDX (expt 1.806 g/cm3)")
rdx_smi = next(a["smiles"] for a in data["anchors"] if a["id"] == "RDX")
atoms_rdx, _, _ = smiles_to_xyz(rdx_smi)
formula_rdx = molecular_formula(atoms_rdx)
rho_rdx = density_from_volume(atoms_rdx, formula_rdx)
print(f"\n[smoke]  RDX rho_estimated = {rho_rdx:.3f}", end=" ")
if not (1.0 < rho_rdx < 2.5):
    fail(f"RDX density {rho_rdx} far off expected 1.8")
ok()

step("Kamlet-Jacobs on RDX with literature values")
kj = kamlet_jacobs(rho_g_cm3=1.806, hof_kJmol=70.5, formula="C3H6N6O6")
print(f"\n[smoke]  RDX K-J: D={kj['D_kms']} P={kj['P_GPa']}", end=" ")
if kj.get("D_kms") is None:
    fail("K-J returned None D for RDX")
if not (5.0 < kj["D_kms"] < 11.0):
    fail(f"RDX K-J D={kj['D_kms']} far off (expect ~8.75; some K-J variants give ~7-9)")
if not (15 < kj["P_GPa"] < 60):
    fail(f"RDX K-J P={kj['P_GPa']} far off (expect ~33.7 GPa)")
ok()

step("hof_from_atomization shape")
fake_refs = {"H": -0.5, "C": -37.8, "N": -54.5, "O": -75.0}    # rough B3LYP/6-31G* atom Es
fake_e = -300.0
hof = hof_from_atomization(atoms_rdx, fake_e, fake_refs)
print(f"\n[smoke]  HOF(RDX, fake refs) = {hof:.0f} kJ/mol", end=" ")
ok()

print(f"\n[smoke] === LOCAL SMOKE PASS ===")
print(f"[smoke] DFT-dependent steps must be tested in the runpod environment "
      f"(Python 3.14 on Windows has no pyscf wheel).")
