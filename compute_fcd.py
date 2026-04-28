"""Compute Frechet ChemNet Distance for each M6/M1-sweep condition vs labelled-master.

For each pool of generated SMILES (one .txt per condition×seed), compute FCD
between the pool's RDKit-valid canonical SMILES and a 5000-row sample of
labelled_master. Report mean ± std across seeds per condition.
"""
from __future__ import annotations
import json, sys, glob, random
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from fcd_torch import FCD

OUT = Path("results"); OUT.mkdir(exist_ok=True, parents=True)
random.seed(0)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[fcd] init FCD on {device}")
fcd = FCD(device=device, n_jobs=1)

# Reference: 5000-row sample of labelled-master
print("[fcd] loading reference (labelled-master)")
lm = pd.read_csv("m6_postprocess_bundle/labelled_master.csv", usecols=["smiles"], low_memory=False)
ref = lm["smiles"].dropna().sample(n=5000, random_state=0).tolist()
ref = [s for s in ref if Chem.MolFromSmiles(s) is not None]
print(f"[fcd] reference n={len(ref)}")

def canon_filter(path):
    smis = Path(path).read_text().splitlines()
    out = []
    for s in smis:
        s = s.strip()
        if not s: continue
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        out.append(Chem.MolToSmiles(m))
    return out

results = {}
files = sorted(glob.glob("m6_postprocess_bundle/m1_3seed_*.txt") +
                glob.glob("results/m1_sweep_*.txt") +
                glob.glob("baseline_bundle/results/smiles_lstm_samples.txt"))
print(f"[fcd] {len(files)} pool files")

for f in files:
    name = Path(f).stem
    smis = canon_filter(f)
    if len(smis) < 100:
        print(f"  {name}: only {len(smis)} valid, skipping")
        continue
    # Cap at 5000 for speed
    if len(smis) > 5000:
        smis = random.sample(smis, 5000)
    print(f"  {name}: n={len(smis)}", flush=True)
    try:
        d = fcd(smis, ref)
        results[name] = float(d)
        print(f"    FCD = {d:.3f}", flush=True)
        # incremental persistence — a kill mid-run still leaves usable data
        (OUT / "fcd_results.json").write_text(json.dumps(
            {"per_pool": results, "per_condition": {}}, indent=2))
    except Exception as e:
        print(f"    error: {e}")
        continue

# Aggregate by condition
def cond_of(name):
    for prefix in ("m1_3seed_", "m1_sweep_"):
        if name.startswith(prefix):
            stem = name.replace(prefix, "")
            if "_seed" in stem:
                return stem.rsplit("_seed", 1)[0]
            return stem
    if name == "smiles_lstm_samples":
        return "smiles_lstm"
    return name

agg = {}
for name, d in results.items():
    c = cond_of(name)
    agg.setdefault(c, []).append(d)

print()
print(f"{'condition':<28} {'n':>4} {'FCD mean':>10} {'std':>8}")
for c, vs in sorted(agg.items()):
    print(f"{c:<28} {len(vs):>4} {np.mean(vs):>10.3f} {np.std(vs):>8.3f}")

(OUT / "fcd_results.json").write_text(json.dumps(
    {"per_pool": results,
     "per_condition": {c: {"n": len(vs), "mean": float(np.mean(vs)),
                            "std": float(np.std(vs))} for c, vs in agg.items()}},
    indent=2))
print(f"[fcd] -> {OUT/'fcd_results.json'}")
print("[fcd] === DONE ===")
