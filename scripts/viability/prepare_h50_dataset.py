"""2.5 prep: build a literature h50 (impact-sensitivity) latent dataset.

Sources used:
    Huang & Massa "Machine Learning Energetic Molecules" combined_data.xlsx
        (downloaded from Mathieu / Macken et al. compilation)
    307 valid (SMILES, h50_obs) pairs after filtering.

Pipeline:
    1. canonicalise SMILES with RDKit, drop duplicates, drop SELFIES > 72 tokens
    2. encode through the frozen fine-tuned LIMO VAE -> z_mu (N, 1024)
    3. derive a sensitivity proxy in [0, 1]: high = sensitive (low h50),
       low = insensitive (high h50). Mapping is
           sens = sigmoid( (log10(40) - log10(h50)) * 1.5 )
       so h50=40 cm -> 0.5; h50=5 cm -> 0.93; h50=200 cm -> 0.10.
    4. save as experiments/sens_h50_dataset.pt with z_mu + h50_obs +
       sens_target + smiles list.

Output is consumed by retrain_sens_head_h50.py (separate script).

Run:
    /c/Python314/python scripts/viability/prepare_h50_dataset.py \
        --ckpt experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt \
        --out experiments/sens_h50_dataset.pt
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

H50_XLSX = Path("data/raw/energetic_external/Machine-Learning-Energetic-Molecules-Notebooks/datasets/combined_data.xlsx")
H50_REF_PIVOT = 40.0   # cm; below = sensitive, above = insensitive
H50_LOG_SLOPE = 1.5    # steepness of the sigmoid in log-h50 units


def canon(smi: str):
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def h50_to_sens(h50: float) -> float:
    """Map observed h50 (cm) to a sensitivity proxy in [0, 1]."""
    if h50 <= 0 or not np.isfinite(h50):
        return 0.5
    x = (np.log10(H50_REF_PIVOT) - np.log10(h50)) * H50_LOG_SLOPE
    return float(1.0 / (1.0 + np.exp(-x)))


def load_h50_table(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx)
    df = df[df["SMILES"].notna() & df["h50 (obs)"].notna()].copy()
    df["smiles_canon"] = df["SMILES"].apply(canon)
    df = df[df["smiles_canon"].notna()].drop_duplicates("smiles_canon")
    df["h50"] = df["h50 (obs)"].astype(float)
    df["sens_target"] = df["h50"].apply(h50_to_sens)
    return df[["smiles_canon", "h50", "sens_target", "Name"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="experiments/sens_h50_dataset.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    base = Path.cwd()
    sys.path.insert(0, str(base / "scripts" / "vae"))
    from limo_model import (
        LIMOVAE, SELFIESTokenizer, load_vocab, build_limo_vocab, save_vocab,
        LIMO_MAX_LEN, find_limo_repo,
    )

    print(f"Loading h50 table from {H50_XLSX} ...")
    df = load_h50_table(H50_XLSX)
    print(f"  rows after canonicalisation + dedup: {len(df)}")
    print(f"  h50 range: {df['h50'].min():.1f}  -  {df['h50'].max():.1f} cm")
    print(f"  sens_target range: {df['sens_target'].min():.3f} - {df['sens_target'].max():.3f}")

    print("\nLoading LIMO tokenizer ...")
    limo_dir = find_limo_repo(base)
    vocab_cache = limo_dir / "vocab_cache.json"
    if vocab_cache.exists():
        alphabet = load_vocab(vocab_cache)
    else:
        alphabet = build_limo_vocab(limo_dir / "zinc250k.smi")
        save_vocab(alphabet, vocab_cache)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)

    print(f"Loading LIMO checkpoint {args.ckpt} ...")
    blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = LIMOVAE()
    model.load_state_dict(blob["model_state"], strict=True)
    model.to(args.device).eval()

    smiles_list = df["smiles_canon"].tolist()
    tokens = []
    keep = []
    for i, smi in enumerate(smiles_list):
        t = tok.smiles_to_tensor(smi)
        if t is None:
            continue
        tokens.append(t[0])
        keep.append(i)
    df_k = df.iloc[keep].reset_index(drop=True)
    if not tokens:
        raise RuntimeError("No SMILES survived tokenisation.")
    X = torch.stack(tokens)
    print(f"  tokenised: {len(X)} / {len(smiles_list)}")

    print("Encoding ...")
    t0 = time.time()
    z_chunks = []
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            batch = X[i:i + args.batch].to(args.device)
            if len(batch) == 1:
                batch = torch.cat([batch, batch])
                _z, mu, _lv = model.encode(batch)
                z_chunks.append(mu[:1].cpu().float())
            else:
                _z, mu, _lv = model.encode(batch)
                z_chunks.append(mu.cpu().float())
    z_mu = torch.cat(z_chunks, dim=0)
    print(f"  encoded {len(z_mu)} in {time.time() - t0:.1f}s")

    out_blob = {
        "z_mu": z_mu,                                                # (N, 1024)
        "h50_obs": torch.tensor(df_k["h50"].to_numpy(), dtype=torch.float32),
        "sens_target": torch.tensor(df_k["sens_target"].to_numpy(),
                                    dtype=torch.float32),
        "smiles": df_k["smiles_canon"].tolist(),
        "names": df_k["Name"].tolist(),
        "meta": {
            "source": str(H50_XLSX),
            "h50_pivot_cm": H50_REF_PIVOT,
            "h50_log_slope": H50_LOG_SLOPE,
            "n": len(z_mu),
            "ckpt": args.ckpt,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_blob, out_path)
    print(f"\nSaved -> {out_path}  ({len(z_mu)} rows)")


if __name__ == "__main__":
    main()
