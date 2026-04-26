"""
Resolve Kroonblawd abbreviations → SMILES via four-stage pipeline:

  Stage 1: cm4c01978_si_001.xls Compound Name + CHIRAL SMILES lookup
  Stage 2: PubChem PUG REST name → CanonicalSMILES
  Stage 3: NCI/CADD CIR name → SMILES (synonym fallback)
  Stage 4: Manual override table (hand-curated for stubborn abbrevs)

Output: data/raw/energetic_external/resolver/energetic_name_to_smiles.csv
        with columns [name, smiles, source]
"""
import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

BASE   = Path("E:/Projects/EnergeticDiffusion2")
OUT_DIR = BASE / "data/raw/energetic_external/resolver"
OUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_CSV = OUT_DIR / "energetic_name_to_smiles.csv"
FAIL_CSV = OUT_DIR / "unresolved_names.csv"

# Hand-curated overrides for stubborn energetic-materials abbreviations.
# Expand as needed.
MANUAL = {
    "CL-20":   "O=[N+]([O-])N1CC2N(C1N1[N+](=O)[O-])C1N([N+](=O)[O-])C2N([N+](=O)[O-])C1[N+](=O)[O-]",
    "HMX":     "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]",
    "RDX":     "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "TATB":    "O=[N+]([O-])c1c(N)c([N+](=O)[O-])c(N)c([N+](=O)[O-])c1N",
    "TNT":     "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
    "PETN":    "O=[N+]([O-])OCC(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]",
    "NTO":     "O=C1NN=C(N1)[N+](=O)[O-]",
    "FOX-7":   "NC(=C([N+](=O)[O-])[N+](=O)[O-])N",
    "TNAZ":    "O=[N+]([O-])C1(CN(C1)[N+](=O)[O-])[N+](=O)[O-]",
    "DATB":    "Nc1c(cc([N+](=O)[O-])cc1N)[N+](=O)[O-]",
    "HNS":     "O=[N+]([O-])c1cc(cc(c1)[N+](=O)[O-])/C=C/c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1",
    "5-NT":    "O=[N+]([O-])c1[nH]nnn1",
    "CL-14":   "Nc1c([N+](=O)[O-])c2onc(c2c1[N+](=O)[O-])N",
    "NQ":      "NC(=N[N+](=O)[O-])N",
    "DAAF":    "NC1=C(N=NC1=Nc1c(N)onn1)N=O",
    "DAAzF":   "Nc1onc(n1)N=Nc1onc(N)n1",
    "TATP":    "O1OC(C)(C)OOC(C)(C)OOC1(C)C",
    "ANFO":    "[O-][N+](=O)[N-]",
    "TNP":     "O=[N+]([O-])c1c(O)c([N+](=O)[O-])cc([N+](=O)[O-])c1",
    "NG":      "O=[N+]([O-])OCC(O[N+](=O)[O-])CO[N+](=O)[O-]",
    "MNDPy":   "Cc1cc(nc(n1)[N+](=O)[O-])C",
    "2-MNDPy": "Cc1nc(cc(n1)C)[N+](=O)[O-]",
    "TNB":     "O=[N+]([O-])c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1",
    "HNB":     "O=[N+]([O-])c1c([N+](=O)[O-])c([N+](=O)[O-])c([N+](=O)[O-])c([N+](=O)[O-])c1[N+](=O)[O-]",
    "Tetryl":  "CN([N+](=O)[O-])c1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
    "PA":      "O=[N+]([O-])c1cc([N+](=O)[O-])c(c(c1)[N+](=O)[O-])O",
    "DNBF":    "O=[N+]([O-])c1cc2onc(c2c1[N+](=O)[O-])N",
    "ETN":     "O=[N+]([O-])OCC(O[N+](=O)[O-])C(O[N+](=O)[O-])CO[N+](=O)[O-]",
    "HNBP":    "Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c([N+](=O)[O-])c1[N+](=O)[O-]",
    "PNB":     "O=[N+]([O-])c1c([N+](=O)[O-])c([N+](=O)[O-])c([N+](=O)[O-])cc1[N+](=O)[O-]",
    "DNTF":    "O=[N+]([O-])c1onc(c1-c1c([N+](=O)[O-])on1)[N+](=O)[O-]",
    "BTATz":   "Nc1nn(nn1)c1nn(nn1)N",
    "HMTD":    "C1N2COON1COON2",
    "DINA":    "O=[N+]([O-])OCCN([N+](=O)[O-])CCO[N+](=O)[O-]",
    "FOX-12":  "NC(=N[N+](=O)[O-])NC(=N)N",
    "DNFP":    "O=[N+]([O-])c1cc2[nH]nnc2c([N+](=O)[O-])c1",
    "LLM-105": "Nc1nc(=O)nc(c1[N+](=O)[O-])N",
    "MEDINA":  "CN([N+](=O)[O-])CN([N+](=O)[O-])C",
    "NG":      "O=[N+]([O-])OCC(O[N+](=O)[O-])CO[N+](=O)[O-]",
    "TEX":     "O=[N+]([O-])N1C2OC3OC(OC(C2N1[N+](=O)[O-])N1[N+](=O)[O-]C3[N+](=O)[O-])N=1",
    "TNAD":    "O=[N+]([O-])N1CCN([N+](=O)[O-])CC1",
}

# ── Stage 1: cm4c01978 lookup ────────────────────────────────────────────────
def build_cm4c_lookup():
    cm4 = pd.read_excel(
        BASE / "data/raw/energetic_external/high_explosive_crystal_properties/"
               "PMC11603605_extracted/PMC11603605/cm4c01978_si_001.xls",
        usecols=["NAME", "Compound Name", "CHIRAL SMILES"])
    cm4 = cm4.dropna(subset=["CHIRAL SMILES"])

    def canon(s):
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None

    cm4["smi"] = cm4["CHIRAL SMILES"].apply(canon)
    cm4 = cm4.dropna(subset=["smi"])

    lut = {}
    for _, r in cm4.iterrows():
        for key in ("NAME", "Compound Name"):
            k = str(r.get(key, "")).strip().upper()
            if k and k != "NAN":
                lut.setdefault(k, r["smi"])
    return lut

# ── Stage 2: PubChem PUG REST ────────────────────────────────────────────────
def pubchem_name(name, timeout=8):
    try:
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               f"{urllib.parse.quote(name)}/property/CanonicalSMILES/TXT")
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip().split("\n")[0]
    except requests.RequestException:
        pass
    return None

# ── Stage 3: NCI CADD CIR ────────────────────────────────────────────────────
def nci_cir(name, timeout=8):
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(name)}/smiles"
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.text.strip() and "Page not found" not in r.text:
            return r.text.strip().split("\n")[0]
    except requests.RequestException:
        pass
    return None

def canonicalize(s):
    if not s:
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None

# ── pipeline ─────────────────────────────────────────────────────────────────
def resolve(names):
    print("  Building cm4c lookup …")
    cm4_lut = build_cm4c_lookup()
    print(f"  cm4c names indexed: {len(cm4_lut):,}")

    # normalize manual keys
    manual_upper = {k.upper(): canonicalize(v) for k, v in MANUAL.items()}

    results = []
    unresolved = []
    for i, name in enumerate(names):
        key = str(name).strip()
        if not key:
            continue
        src = smi = None

        # stage 1: cm4c
        smi = cm4_lut.get(key.upper())
        if smi:
            src = "cm4c01978"
        # stage 4: manual (try early since curated)
        if smi is None and key.upper() in manual_upper:
            smi = manual_upper[key.upper()]
            src = "manual"
        # stage 2: pubchem
        if smi is None:
            s = pubchem_name(key)
            if s:
                smi = canonicalize(s)
                src = "pubchem" if smi else None
            time.sleep(0.2)   # rate limit courtesy
        # stage 3: NCI CADD
        if smi is None:
            s = nci_cir(key)
            if s:
                smi = canonicalize(s)
                src = "nci_cir" if smi else None
            time.sleep(0.3)

        if smi:
            results.append({"name": key, "smiles": smi, "source": src})
        else:
            unresolved.append(key)
        if (i+1) % 20 == 0:
            print(f"    progress: {i+1}/{len(names)}  resolved: {len(results)}  unresolved: {len(unresolved)}")

    return results, unresolved

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # Collect names from Kroonblawd and Deng
    kroon_root = BASE / "data/raw/energetic_external/kroonblawd_2025_jctc/extracted"
    vod  = pd.read_csv(kroon_root / "VOD_data.csv")
    hof  = pd.read_csv(kroon_root / "paper_HOF_dataset.csv")
    names = set(vod["Molecule"].astype(str).str.strip())
    names.update(hof["Molecule"].astype(str).str.strip())

    # Deng — names at positional col 0 after header row 4 (messy Excel)
    try:
        deng = pd.read_excel(
            BASE / "data/raw/energetic_external/deng_2021_iscience/PMC7957118/mmc3.xlsx",
            sheet_name="tableS2", header=4)
        for row in deng.iloc[:, 0].dropna().astype(str).str.strip():
            if row and row not in ("Name", "Unnamed: 0"):
                names.add(row)
    except Exception as e:
        print(f"  Deng parse skipped: {e}")

    names = sorted({n for n in names if n and n.lower() != "nan"})
    print(f"Total unique names to resolve: {len(names):,}")

    results, unresolved = resolve(names)

    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nResolved {len(results):,} → {OUT_CSV}")
    print(f"  by source: {out['source'].value_counts().to_dict()}")

    un = pd.DataFrame({"name": unresolved})
    un.to_csv(FAIL_CSV, index=False)
    print(f"Unresolved {len(unresolved):,} → {FAIL_CSV}")

if __name__ == "__main__":
    main()
