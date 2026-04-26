"""
Merge Kroonblawd 2025 JCTC experimental VOD + HOF into labeled_master.csv.

Source: data/raw/energetic_external/kroonblawd_2025_jctc/extracted/
        VOD_data.csv (column VODexp, km/s)
        paper_HOF_dataset.csv (column HOF_exp, likely kJ/mol or kcal/mol TBD)
Name→SMILES mapping: resolver/energetic_name_to_smiles.csv (+ v2.csv if present)

Policy:
  - For SMILES already in master: fill missing DV / HOF at Tier A,
    or UPGRADE current non-A tier to A (experimental wins over anything).
    When the existing cell is already Tier A, leave the value unchanged
    (two different experimental references — keep the incumbent).
  - For SMILES NOT in master: add NEW rows (minimal schema: smiles, selfies
    placeholder, source_dataset="kroonblawd_2025_jctc", label_source_type=
    "compiled_observed", tier="A", VOD/HOF populated with Tier A tagging).

HOF unit check before merge: spot-check known values (RDX lit +62 kJ/mol).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

BASE    = Path("E:/Projects/EnergeticDiffusion2")
MASTER  = BASE / "data/training/master/labeled_master.csv"
BACKUP  = BASE / "data/training/master/labeled_master_pre_kroonblawd.csv.bak"
RESOLVE = BASE / "data/raw/energetic_external/resolver/energetic_name_to_smiles.csv"
RESOLVE_V2 = BASE / "data/raw/energetic_external/resolver/energetic_name_to_smiles_v2.csv"
VOD_CSV = BASE / "data/raw/energetic_external/kroonblawd_2025_jctc/extracted/VOD_data.csv"
HOF_CSV = BASE / "data/raw/energetic_external/kroonblawd_2025_jctc/extracted/paper_HOF_dataset.csv"

def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None

# ── lookup ───────────────────────────────────────────────────────────────────
resolved = pd.read_csv(RESOLVE)
if RESOLVE_V2.exists():
    resolved = pd.concat([resolved, pd.read_csv(RESOLVE_V2)], ignore_index=True)
resolved["smi_canon"] = resolved["smiles"].apply(canon)
resolved = resolved.dropna(subset=["smi_canon"]).drop_duplicates("name", keep="first")
name2smi = dict(zip(resolved["name"].astype(str).str.upper(), resolved["smi_canon"]))
print(f"Name-to-SMILES table size: {len(name2smi):,}")

vod = pd.read_csv(VOD_CSV)
hof = pd.read_csv(HOF_CSV)
vod["smi"] = vod["Molecule"].astype(str).str.upper().map(name2smi)
hof["smi"] = hof["Molecule"].astype(str).str.upper().map(name2smi)
vod_ok = vod.dropna(subset=["smi","VODexp"]).drop_duplicates("smi", keep="first")
hof_ok = hof.dropna(subset=["smi","HOF_exp"]).drop_duplicates("smi", keep="first")
print(f"VOD with resolved SMILES: {len(vod_ok):,}")
print(f"HOF with resolved SMILES: {len(hof_ok):,}")

# HOF unit sanity-check: RDX lit = +62 kJ/mol
if len(hof_ok):
    rdx_smi = canon("O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1")
    r = hof_ok[hof_ok["smi"] == rdx_smi]
    if len(r):
        v = r["HOF_exp"].iloc[0]
        print(f"HOF unit check — RDX: {v}  (lit +62 kJ/mol, +15 kcal/mol)")
        if 50 < v < 75:
            print("  HOF_exp is in kJ/mol ✓")
        elif 10 < v < 25:
            print("  HOF_exp is in kcal/mol → will convert x4.184")
            hof_ok = hof_ok.copy()
            hof_ok["HOF_exp"] = hof_ok["HOF_exp"] * 4.184
        else:
            print(f"  HOF units ambiguous (RDX val {v}); treating as kJ/mol")

# ── load master ──────────────────────────────────────────────────────────────
print(f"\nBackup → {BACKUP}")
shutil.copy2(MASTER, BACKUP)
lm = pd.read_csv(MASTER, low_memory=False)
print(f"Master rows: {len(lm):,}")

smi2idx = {}
for i, s in enumerate(lm["smiles"]):
    if isinstance(s, str) and s not in smi2idx:
        smi2idx[s] = i

n_dv_upgrade = n_dv_fill_new_col = n_hof_upgrade = n_hof_fill = 0
new_rows = []

def apply_DV(idx, value):
    global n_dv_upgrade, n_dv_fill_new_col
    cur_tier = lm.at[idx, "detonation_velocity_tier"]
    if cur_tier == "A":
        # keep incumbent experimental value
        return
    lm.at[idx, "detonation_velocity"] = value
    lm.at[idx, "detonation_velocity_source_dataset"] = "kroonblawd_2025_jctc"
    lm.at[idx, "detonation_velocity_source_type"]    = "compiled_observed"
    lm.at[idx, "detonation_velocity_tier"]           = "A"
    if pd.isna(cur_tier):
        n_dv_fill_new_col += 1
    else:
        n_dv_upgrade += 1

def apply_HOF(idx, value):
    global n_hof_upgrade, n_hof_fill
    cur_tier = lm.at[idx, "heat_of_formation_tier"]
    if cur_tier == "A":
        return
    lm.at[idx, "heat_of_formation"] = value
    lm.at[idx, "heat_of_formation_source_dataset"] = "kroonblawd_2025_jctc"
    lm.at[idx, "heat_of_formation_source_type"]    = "compiled_observed"
    lm.at[idx, "heat_of_formation_tier"]           = "A"
    if pd.isna(cur_tier):
        n_hof_fill += 1
    else:
        n_hof_upgrade += 1

# apply VOD
for _, r in vod_ok.iterrows():
    s = r["smi"]
    if s in smi2idx:
        apply_DV(smi2idx[s], float(r["VODexp"]))
    else:
        # collect for new-rows batch
        new_rows.append({"smiles": s, "VOD": float(r["VODexp"]), "HOF": None, "name": r["Molecule"]})

# apply HOF: overlap with VOD-new rows + apply to master rows
for _, r in hof_ok.iterrows():
    s = r["smi"]
    hof_v = float(r["HOF_exp"])
    if s in smi2idx:
        apply_HOF(smi2idx[s], hof_v)
    else:
        # find in new_rows or add
        found = False
        for nr in new_rows:
            if nr["smiles"] == s:
                nr["HOF"] = hof_v
                found = True
                break
        if not found:
            new_rows.append({"smiles": s, "VOD": None, "HOF": hof_v, "name": r["Molecule"]})

print(f"\nIn-place updates:")
print(f"  DV upgrades (lower tier → A): {n_dv_upgrade}")
print(f"  DV cells newly filled:        {n_dv_fill_new_col}")
print(f"  HOF upgrades (lower tier→A):  {n_hof_upgrade}")
print(f"  HOF cells newly filled:       {n_hof_fill}")
print(f"  New rows to append:           {len(new_rows)}")

# append new rows
if new_rows:
    import selfies as sf
    base_cols = lm.columns.tolist()
    template = {c: pd.NA for c in base_cols}
    to_append = []
    import uuid
    for i, nr in enumerate(new_rows):
        row = dict(template)
        smi = nr["smiles"]
        row["molecule_id"]        = f"kroonblawd_{uuid.uuid4().hex[:12]}"
        row["smiles"]             = smi
        try:
            row["selfies"]        = sf.encoder(smi)
        except Exception:
            row["selfies"]        = ""
        row["source_dataset"]     = "kroonblawd_2025_jctc"
        row["source_path"]        = "kroonblawd_2025_jctc/VOD_data.csv"
        row["label_source_type"]  = "compiled_observed"
        row["tier"]               = "A"
        # atom counts
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mh = Chem.AddHs(m)
            row["n_count"] = int(sum(1 for a in mh.GetAtoms() if a.GetSymbol()=="N"))
            row["o_count"] = int(sum(1 for a in mh.GetAtoms() if a.GetSymbol()=="O"))
            row["has_nitro"] = bool(m.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")))
            row["has_azide"] = bool(m.HasSubstructMatch(Chem.MolFromSmarts("N=[N+]=[N-]")))
            row["energetic_proxy_score"] = row["n_count"] + row["o_count"]
        if nr["VOD"] is not None:
            row["detonation_velocity"]                = nr["VOD"]
            row["detonation_velocity_source_dataset"] = "kroonblawd_2025_jctc"
            row["detonation_velocity_source_type"]    = "compiled_observed"
            row["detonation_velocity_tier"]           = "A"
        if nr["HOF"] is not None:
            row["heat_of_formation"]                = nr["HOF"]
            row["heat_of_formation_source_dataset"] = "kroonblawd_2025_jctc"
            row["heat_of_formation_source_type"]    = "compiled_observed"
            row["heat_of_formation_tier"]           = "A"
        to_append.append(row)
    if to_append:
        lm = pd.concat([lm, pd.DataFrame(to_append)], ignore_index=True)
        print(f"  appended {len(to_append):,} new rows")

# Recompute row-level tier
PROPS = ["density","heat_of_formation","detonation_velocity","detonation_pressure","explosion_heat"]
rank = {"A":0,"B":1,"C":2,"D":3,None:99}
def row_tier(r):
    out = []
    for p in PROPS:
        c = r.get(f"{p}_tier", None)
        if isinstance(c, str) and c in ("A","B","C","D"):
            out.append(c)
    if not out: return None
    return min(out, key=lambda c: rank[c])
lm["tier"] = lm.apply(row_tier, axis=1)

print(f"\nFinal master rows: {len(lm):,}")
print("Post-merge per-property Tier-A counts:")
for p in PROPS:
    tc = f"{p}_tier"
    if tc in lm.columns:
        print(f"  {p}: A={(lm[tc]=='A').sum():,}")

print(f"\nSaving → {MASTER}")
lm.to_csv(MASTER, index=False)
print(f"Backup at {BACKUP}")
