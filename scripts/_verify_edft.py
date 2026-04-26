"""Quick verification of the edft conda env tools."""
import sys
print(f"Python: {sys.version.split()[0]}")
for name in ("psi4", "rdkit", "ase", "tblite", "pyxtal", "numpy", "pandas"):
    try:
        m = __import__(name)
        v = getattr(m, "__version__", "?")
        print(f"  {name:12s}: {v}")
    except Exception as e:
        print(f"  {name:12s}: FAILED - {type(e).__name__}: {e}")

print("\nDoes CP2K binary exist?")
import shutil
for tool in ("cp2k.psmp", "cp2k.ssmp", "cp2k.popt", "cp2k"):
    p = shutil.which(tool)
    if p:
        print(f"  {tool}: {p}")
        break
else:
    print("  NOT FOUND - may need `micromamba install -n edft cp2k -c conda-forge`")

print("\nPsi4 quick sanity check (HF/STO-3G on H2)...")
try:
    import psi4
    import os
    null_path = "NUL" if os.name == "nt" else "/dev/null"
    psi4.set_output_file(null_path, True)
    psi4.set_options({"basis": "sto-3g"})
    mol = psi4.geometry("H 0 0 0\nH 0 0 0.74")
    E = psi4.energy("scf", molecule=mol)
    print(f"  H2 HF/STO-3G energy: {E:.6f} Hartree (expected ~-1.117)")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
