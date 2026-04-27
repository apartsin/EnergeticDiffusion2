"""Retrain the SMILES-space viability classifier with hard negatives mined
from the diffusion candidate pools. The original v1 model used ZINC drug-like
negatives only, which makes the boundary too easy. Hard negatives = generated
SMILES that the chem_redflags filter REJECTS (model cheats: tiny polynitro,
gem-dinitro 4-ring, open-chain N-chain length >= 4, etc.).

Inputs:
    positives:  data/training/master/labeled_master.csv
    zinc_negs:  external/LIMO/zinc250k.smi
    pool_mds:   list of *.md files containing generated candidates
    (we read the SMILES from the rerank tables and pass each through screen())

Output:
    experiments/viability_rf_v2_hardneg/{model.joblib, summary.json}
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts/viability")
sys.path.insert(0, "scripts/diffusion")
from train_viability import featurize, load_smiles_csv, load_zinc, build_dataset
from chem_redflags import screen


def smiles_from_md(path):
    md = Path(path).read_text(encoding="utf-8")
    out = []
    for line in md.split("\n"):
        if not line.startswith("| ") or "rank" in line or "---" in line: continue
        cells = [c.strip(" `") for c in line.split("|")]
        if len(cells) >= 12:
            out.append(cells[-2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", required=True)
    ap.add_argument("--zinc",      required=True)
    ap.add_argument("--pool_mds",  nargs="+", required=True,
                    help="rerank markdowns to mine hard negatives from")
    ap.add_argument("--zinc_n",    type=int, default=80000)
    ap.add_argument("--out",       required=True)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading positives ...")
    pos = load_smiles_csv(Path(args.positives))
    print(f"  {len(pos)} positives")

    print("Loading ZINC negatives ...")
    zinc_neg = load_zinc(Path(args.zinc), n=args.zinc_n)

    print("Mining hard negatives from pool MDs ...")
    cand_smis = set()
    for md in args.pool_mds:
        if not Path(md).exists():
            print(f"  skip {md} (missing)"); continue
        for s in smiles_from_md(md): cand_smis.add(s)
    print(f"  {len(cand_smis)} unique candidate SMILES")

    hard_neg = []
    for s in cand_smis:
        scr = screen(s)
        if scr["status"] == "ok": continue
        hard_neg.append(s)
    print(f"  {len(hard_neg)} hard negatives (rejected by chem_redflags)")

    print(f"\nFeaturizing: {len(pos)} pos / {len(zinc_neg)} ZINC neg / {len(hard_neg)} hard neg")
    X, y = build_dataset(pos, zinc_neg + hard_neg)
    # Hard negatives weight
    print(f"  total: {len(y)}  prevalence_pos={y.mean():.3f}")

    print("\nTraining RandomForest with class_weight=balanced ...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=22, n_jobs=-1, random_state=42,
        class_weight="balanced", min_samples_leaf=4)
    rf.fit(Xtr, ytr)

    yhat = rf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, yhat)
    ap_ = average_precision_score(yte, yhat)
    acc = accuracy_score(yte, (yhat > 0.5).astype(int))
    print(f"\nValidation:  AUC={auc:.4f}  AP={ap_:.4f}  ACC={acc:.4f}")

    # Specifically score the hard negatives in the test set
    test_hard_mask = np.zeros(len(yte), dtype=bool)
    # we lost original index but can re-featurize hard_neg subset
    h_X = []
    for s in hard_neg[:min(2000, len(hard_neg))]:
        f = featurize(s)
        if f is not None: h_X.append(f)
    if h_X:
        h_X = np.stack(h_X)
        h_yhat = rf.predict_proba(h_X)[:, 1]
        print(f"\nHard negatives: P(viable) mean={h_yhat.mean():.3f} median={np.median(h_yhat):.3f} max={h_yhat.max():.3f}")

    import joblib
    joblib.dump(rf, out_dir / "model.joblib")
    summary = {
        "n_pos": int((y == 1).sum()),
        "n_neg_total": int((y == 0).sum()),
        "n_zinc_neg": len(zinc_neg),
        "n_hard_neg": len(hard_neg),
        "auc": float(auc), "ap": float(ap_), "acc": float(acc),
        "elapsed_s": time.time() - t0,
    }
    if h_X is not None and len(h_X):
        summary["hard_neg_mean_pviable"] = float(h_yhat.mean())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out_dir/'model.joblib'}")
    print(f"-> {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
