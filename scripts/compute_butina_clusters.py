"""
Pre-compute ECFP4 fingerprints and Butina clusters for EDA.
Cached output consumed by generate_eda.py.

Strategy: stratified sample across label_source_type (labeled) + proxy-ranked
slice of unlabeled, total ~12k. Butina clustering at Tanimoto cutoff 0.4.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

BASE  = Path("E:/Projects/EnergeticDiffusion2")
CACHE = BASE / "data/training/metadata/butina_clusters.pkl"
CACHE.parent.mkdir(parents=True, exist_ok=True)

CUTOFF       = 0.4     # Tanimoto distance
SAMPLE_SIZE  = 12000   # total
LABELED_FRAC = 0.65    # ~7800 labeled, ~4200 unlabeled

print("Loading labeled/unlabeled masters …")
lm = pd.read_csv(BASE / "data/training/master/labeled_master.csv", low_memory=False,
                 usecols=["smiles", "label_source_type", "source_dataset",
                          "energetic_proxy_score"])
um = pd.read_csv(BASE / "data/training/master/unlabeled_master.csv", low_memory=False,
                 usecols=["smiles", "source_dataset", "energetic_proxy_score"])
lm["bucket"] = lm["label_source_type"]
um["bucket"] = "unlabeled"

# stratified sample: equal-ish per label bucket (cap), then fill with unlabeled
BUCKET_CAPS = {
    "compiled_observed": 2500,
    "kj_calculated":     2500,
    "model_predicted":   2000,
    "qsar_predicted":    1500,
    "unlabeled":         4200,   # mix of high-proxy + random
}

rng = np.random.default_rng(42)

picks = []
for b, cap in BUCKET_CAPS.items():
    if b == "unlabeled":
        # take top proxy-score half, plus a random half for coverage
        top = um.nlargest(cap // 2, "energetic_proxy_score")
        rest = um.drop(top.index).sample(cap // 2, random_state=42)
        picks.append(pd.concat([top, rest]).assign(bucket=b))
    else:
        sub = lm[lm["bucket"] == b]
        n = min(cap, len(sub))
        picks.append(sub.sample(n, random_state=42))

sample = pd.concat(picks, ignore_index=True)[
    ["smiles", "bucket", "source_dataset"]].drop_duplicates("smiles").reset_index(drop=True)
print(f"Sample size: {len(sample):,}  (unique SMILES)")
print(sample["bucket"].value_counts())

# ECFP4 fingerprints
print("\nComputing ECFP4 fingerprints …")
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = []
valid_idx = []
for i, smi in enumerate(sample["smiles"]):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    fps.append(mfpgen.GetFingerprint(m))
    valid_idx.append(i)
    if (len(fps) % 1000) == 0:
        print(f"  {len(fps):,} fingerprints done")
print(f"Valid fingerprints: {len(fps):,}")

sample = sample.iloc[valid_idx].reset_index(drop=True)

# Pairwise Tanimoto distances (condensed, 1 - similarity)
print("\nComputing pairwise Tanimoto distances …")
n = len(fps)
dists = []
for i in range(1, n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend(1.0 - s for s in sims)
    if (i % 500) == 0:
        print(f"  {i}/{n} rows processed")
print(f"Distance pairs: {len(dists):,}")

print(f"\nButina clustering (cutoff={CUTOFF}) …")
clusters = Butina.ClusterData(dists, n, CUTOFF, isDistData=True)
clusters = sorted(clusters, key=len, reverse=True)
print(f"Clusters: {len(clusters):,}  (largest: {len(clusters[0]):,})")

# membership
mem = np.full(n, -1, dtype=int)
for cid, members in enumerate(clusters):
    for m in members:
        mem[m] = cid
sample["cluster"] = mem

# nearest-neighbor Tanimoto (approximate — use cluster center)
nn_sim = np.zeros(n)
centers = [c[0] for c in clusters]
for cid, members in enumerate(clusters):
    center = centers[cid]
    if len(members) == 1:
        nn_sim[members[0]] = 0.0
        continue
    for m in members:
        if m == center:
            nn_sim[m] = DataStructs.TanimotoSimilarity(fps[m],
                         fps[members[1] if len(members) > 1 else m])
        else:
            nn_sim[m] = DataStructs.TanimotoSimilarity(fps[m], fps[center])
sample["nn_tanimoto"] = nn_sim

print("\nSaving cache …")
with open(CACHE, "wb") as f:
    pickle.dump({
        "sample":    sample,
        "clusters":  [list(c) for c in clusters],
        "cutoff":    CUTOFF,
    }, f)
print(f"Cache → {CACHE}")
print(f"Total clusters: {len(clusters):,}")
print(f"Singletons: {sum(1 for c in clusters if len(c)==1):,}")
