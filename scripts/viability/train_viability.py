"""Train a binary viability classifier: P(EM-like and chemically plausible).

Positives:  the 66k labelled_master energetic compounds
Negatives:  random sample from ZINC-250k (general drug-like, not energetic)
            + hard negatives from candidates rejected by chem_redflags.screen
            (added at predict time, optional)

Features: Morgan fingerprint (2048 bits, radius 2) + RDKit physicochemical
descriptors (~30). RandomForestClassifier from scikit-learn — fast, no GPU
needed, robust on small datasets, ships uncertainty via .predict_proba.

Usage:
    python scripts/viability/train_viability.py \
        --positives data/training/master/labeled_master.csv \
        --negatives external/LIMO/zinc250k.smi \
        --out experiments/viability_rf_v1
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


DESCRIPTOR_FNS = [
    ("mw",        Descriptors.MolWt),
    ("logp",      Descriptors.MolLogP),
    ("tpsa",      Descriptors.TPSA),
    ("nrings",    Descriptors.RingCount),
    ("naromatic", Descriptors.NumAromaticRings),
    ("nrotat",    Descriptors.NumRotatableBonds),
    ("nhdon",     Descriptors.NumHDonors),
    ("nhacc",     Descriptors.NumHAcceptors),
    ("nheter",    Descriptors.NumHeteroatoms),
    ("frac_sp3",  Descriptors.FractionCSP3),
]


def featurize(smi: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_arr = np.zeros(2048, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, fp_arr)

    desc = []
    for name, fn in DESCRIPTOR_FNS:
        try: desc.append(float(fn(mol)))
        except Exception: desc.append(0.0)

    nC = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    nN = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    nO = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    nH = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    nNO2 = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N+](=O)[O-]")))
    mw = Descriptors.MolWt(mol)
    ob = -1600 * (2*nC + nH/2.0 - nO) / max(mw, 1e-3)
    nHv = mol.GetNumHeavyAtoms()
    n_ring_N = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.IsInRing())
    extra = [nC, nN, nO, nH, nNO2, ob,
             nO/max(nC, 1), nN/max(nC, 1), nNO2/max(nHv, 1),
             n_ring_N/max(nN, 1), nHv]
    return np.concatenate([fp_arr.astype(np.float32),
                             np.array(desc + extra, dtype=np.float32)])


def feature_names() -> List[str]:
    fp = [f"fp{i}" for i in range(2048)]
    desc = [n for n, _ in DESCRIPTOR_FNS]
    extra = ["nC", "nN", "nO", "nH", "nNO2", "OB",
             "ratio_O_C", "ratio_N_C", "nitro_per_heavy",
             "frac_n_in_ring", "n_heavy"]
    return fp + desc + extra


def load_smiles_csv(path: Path, col: str = "smiles") -> List[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            s = r.get(col) or r.get("canonical_smiles") or r.get("SMILES")
            if s: out.append(s)
    return out


def load_zinc(path: Path, n: int) -> List[str]:
    """zinc250k.smi has SMILES in the first whitespace-separated token."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else None
            if s: out.append(s)
            if len(out) >= n: break
    return out


def build_dataset(pos_smis, neg_smis):
    Xs, ys = [], []
    n_skipped = 0
    for s in pos_smis:
        f = featurize(s)
        if f is None: n_skipped += 1; continue
        Xs.append(f); ys.append(1)
    for s in neg_smis:
        f = featurize(s)
        if f is None: n_skipped += 1; continue
        Xs.append(f); ys.append(0)
    print(f"  featurized: pos={ys.count(1)}  neg={ys.count(0)}  skipped={n_skipped}")
    return np.stack(Xs), np.array(ys, dtype=np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", required=True)
    ap.add_argument("--negatives", required=True)
    ap.add_argument("--neg_samples", type=int, default=80000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=20)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("Loading positives ...")
    pos = load_smiles_csv(Path(args.positives))
    print(f"  {len(pos)} positives")
    print("Loading negatives (ZINC) ...")
    neg = load_zinc(Path(args.negatives), n=args.neg_samples)
    print(f"  {len(neg)} negatives")

    print("Featurizing ...")
    X, y = build_dataset(pos, neg)
    print(f"  X.shape={X.shape}, prevalence positive = {y.mean():.3f}")

    print(f"Training RandomForest ({args.n_estimators} trees, max_depth={args.max_depth}) ...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        n_jobs=args.n_jobs, random_state=42, class_weight="balanced",
        min_samples_leaf=4)
    rf.fit(Xtr, ytr)

    print("\nValidation:")
    yhat = rf.predict_proba(Xte)[:, 1]
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    auc = roc_auc_score(yte, yhat)
    ap_ = average_precision_score(yte, yhat)
    acc = accuracy_score(yte, (yhat > 0.5).astype(int))
    print(f"  AUC = {auc:.4f}  AP = {ap_:.4f}  ACC@0.5 = {acc:.4f}")

    # Save
    import joblib
    model_path = out_dir / "model.joblib"
    joblib.dump(rf, model_path)
    print(f"  -> {model_path}")
    summary = {"n_pos": int((y==1).sum()), "n_neg": int((y==0).sum()),
               "auc": float(auc), "ap": float(ap_), "acc": float(acc),
               "n_estimators": args.n_estimators, "max_depth": args.max_depth,
               "elapsed_s": time.time() - t0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  -> {out_dir/'summary.json'}")
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
