"""
Pass 2 of name resolver: parse IUPAC-with-abbreviation strings like
    "1,3,5-Triamino-2,4,6-trinitrobenzene (TATB)"

Splits on " (" and " /" to extract independently-queryable fragments,
retries each against cm4c lookup + PubChem + NCI CIR. Also normalises
whitespace, unicode, and polymorph prefixes (α-, β-, γ-).
"""
import re
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

BASE    = Path("E:/Projects/EnergeticDiffusion2")
OUT_DIR = BASE / "data/raw/energetic_external/resolver"
IN_CSV  = OUT_DIR / "unresolved_names.csv"
OUT_CSV = OUT_DIR / "energetic_name_to_smiles_v2.csv"
STILL_FAIL = OUT_DIR / "unresolved_names_v2.csv"

# polymorph prefixes that obscure name matching
POLY_RE = re.compile(r"^(α|β|γ|δ|α'|β')-?", re.UNICODE)

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

def pubchem(name, timeout=8):
    try:
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               f"{urllib.parse.quote(name)}/property/CanonicalSMILES/TXT")
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip().split("\n")[0]
    except requests.RequestException:
        pass
    return None

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

def candidates_from(name):
    """Generate candidate query strings from a messy raw name."""
    name = str(name).strip()
    seen = []
    def add(x):
        x = x.strip().strip(".,;:")
        if x and x not in seen and len(x) >= 2:
            seen.append(x)

    add(name)

    # Split off parenthesised abbreviation
    m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", name)
    if m:
        iupac, abbrev = m.group(1).strip(), m.group(2).strip()
        add(iupac)
        add(abbrev)
        # strip polymorph prefix from either
        a = POLY_RE.sub("", abbrev)
        if a != abbrev:
            add(a)
        i = POLY_RE.sub("", iupac)
        if i != iupac:
            add(i)

    # split cocrystal / solvate
    if "/" in name:
        for piece in name.split("/"):
            add(piece.strip())

    # strip polymorph from full name too
    pm = POLY_RE.sub("", name)
    if pm != name:
        add(pm)

    return seen

def resolve_one(name, cm4_lut):
    cands = candidates_from(name)
    for c in cands:
        # Stage 1: cm4c
        s = cm4_lut.get(c.upper())
        if s:
            return s, f"cm4c01978({c})"
        # Stage 2: PubChem
        s = pubchem(c)
        if s:
            canon = canonicalize(s)
            if canon:
                time.sleep(0.15)
                return canon, f"pubchem({c})"
        time.sleep(0.15)
        # Stage 3: NCI CADD
        s = nci_cir(c)
        if s:
            canon = canonicalize(s)
            if canon:
                time.sleep(0.25)
                return canon, f"nci_cir({c})"
        time.sleep(0.25)
    return None, None

def main():
    print("  Building cm4c lookup …")
    cm4_lut = build_cm4c_lookup()
    print(f"  cm4c indexed: {len(cm4_lut):,}")

    unresolved = pd.read_csv(IN_CSV)["name"].tolist()
    print(f"  Resolving {len(unresolved):,} previously-failed names …")

    ok = []
    still_fail = []
    for i, name in enumerate(unresolved):
        smi, src = resolve_one(name, cm4_lut)
        if smi:
            ok.append({"name": name, "smiles": smi, "source": src})
        else:
            still_fail.append(name)
        if (i+1) % 20 == 0:
            print(f"    {i+1}/{len(unresolved)}  ok:{len(ok)}  fail:{len(still_fail)}")

    pd.DataFrame(ok).to_csv(OUT_CSV, index=False)
    pd.DataFrame({"name": still_fail}).to_csv(STILL_FAIL, index=False)
    print(f"\nResolved {len(ok):,} → {OUT_CSV}")
    print(f"Still unresolved {len(still_fail):,} → {STILL_FAIL}")

if __name__ == "__main__":
    main()
