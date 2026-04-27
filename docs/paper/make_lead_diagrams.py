"""Render the top leads as 2D structure diagrams for the paper appendix."""
from __future__ import annotations
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

ROOT = Path("E:/Projects/EnergeticDiffusion2")
OUT  = ROOT / "docs/paper/figs/leads"
OUT.mkdir(parents=True, exist_ok=True)

LEADS = [
    ("L1_trinitro_isoxazole",
     "O=[N+]([O-])c1noc([N+](=O)[O-])c1[N+](=O)[O-]",
     "L1 trinitro-isoxazole (top, gap 2.6 eV)"),
    ("L2_oxime_nitrate",
     "O=[N+]([O-])N=C1NC(O)ON=C1[N+](=O)[O-]",
     "L2 oxime-nitrate-imidazoline"),
    ("L3_polyazene_chain",
     "O=[N+]([O-])N=NNC=NN[N+](=O)[O-]",
     "L3 polyazene-NN-NO2 chain"),
    ("L4_triazole_nitrate",
     "O=[N+]([O-])C1=NCN=NN1[N+](=O)[O-]",
     "L4 triazoline-N-nitramine"),
    ("L5_oxime_nitrate_acyl",
     "O=C(C=NO[N+](=O)[O-])[N+](=O)[O-]",
     "L5 acyl-oxime-nitrate"),
    ("L6_propenediene_NN",
     "C=C(N=CN=N[N+](=O)[O-])[N+](=O)[O-]",
     "L6 propenediene-amidine (gap 1.21, marginal)"),
    ("L7_imidazole_N_oxide",
     "O=[N+]([O-])Cc1n[n+]([O-])cn1[N+](=O)[O-]",
     "L7 imidazole-N-oxide (cycle 10 top)"),
    ("L8_oxadiazinone",
     "O=C1N=C([N+](=O)[O-])C(N[N+](=O)[O-])=NON1",
     "L8 oxadiazinone-N-nitramine (cycle 12 top)"),
    ("L9_pyrazole_trinitro",
     "O=[N+]([O-])c1cnn([N+](=O)[O-])c1[N+](=O)[O-]",
     "L9 1,3,4-trinitropyrazole (pool=80k)"),
    ("L10_RDX_anchor",
     "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]",
     "L10 RDX (anchor)"),
    ("L11_HMX_anchor",
     "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]",
     "L11 HMX (anchor)"),
    ("L12_TATB_anchor",
     "Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]",
     "L12 TATB (anchor, 1,3,5-triamino-2,4,6-trinitrobenzene)"),
]


def main():
    print(f"Rendering {len(LEADS)} 2D diagrams")
    mols, captions = [], []
    for name, smi, desc in LEADS:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print(f"  fail: {name}"); continue
        AllChem.Compute2DCoords(m)
        mols.append(m); captions.append(name)
        # also save individual
        img = Draw.MolToImage(m, size=(380, 280), kekulize=True)
        img.save(OUT / f"{name}.png")
    # Also build a 5x2 grid
    grid = Draw.MolsToGridImage(mols, molsPerRow=5,
                                  subImgSize=(280, 220),
                                  legends=captions)
    grid.save(OUT / "all_leads_grid.png")
    print(f"  -> {OUT}")
    print(f"  grid: all_leads_grid.png")


if __name__ == "__main__":
    main()
