"""C2C generation around v2 top leads using the v3.3 noise-aware decoder.

For each seed (top-5 from rerank v2), encode through v1 -> z, then sample
N variants by adding Gaussian noise (sigma in {0.5, 1.0, 1.5}) and decoding
with v3.3 in sample mode. Output: a SMILES list passed through the v2 reranker.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch
from rdkit import Chem

sys.path.insert(0, "scripts/vae"); sys.path.insert(0, "external/LIMO")
from limo_factory import load_limo
from limo_model import SELFIESTokenizer, build_limo_vocab, LIMO_MAX_LEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# v2 top-5 (read from joint_rerank_pool40k_v2.md, manually fixed for stability)
SEEDS = [
    ("isoxazole_top1", "O=[N+]([O-])c1noc([N+](=O)[O-])c1[N+](=O)[O-]"),
    ("oxadiazoline_2", "O=[N+]([O-])CC1=NONN1[N+](=O)[O-]"),
    ("bicyclic_3",    "N=C1NC2=C([N+](=O)[O-])N2C1O[N+](=O)[O-]"),
    ("dinitroguan_5", "N=C(NC=N[N+](=O)[O-])[N+](=O)[O-]"),
    ("imidazole_no",  "O=[N+]([O-])Cc1n[n+]([O-])cn1[N+](=O)[O-]"),
]

def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except Exception: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_seed", type=int, default=64)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--out", default="experiments/c2c_v3_3_v2_seeds/candidates.smi")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    vocab = build_limo_vocab("external/LIMO/zinc250k.smi")
    tok = SELFIESTokenizer(vocab)

    v1_ckpt = "experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt"
    v3_3_ckpt = "experiments/limo_v3_3_diff_aware_AR_20260426T122722Z/checkpoints/best.pt"
    v1, _ = load_limo(".", version="v1", ckpt_override=v1_ckpt, device=DEVICE)
    v3_3, _ = load_limo(".", version="v3_1", ckpt_override=v3_3_ckpt, device=DEVICE)
    v1.eval(); v3_3.eval()

    all_canon = set()
    rows = []
    t0 = time.time()
    for name, smi in SEEDS:
        seq_pair = tok.smiles_to_tensor(smi)
        if seq_pair is None:
            print(f"[{name}] could not tokenize"); continue
        seq, _ = seq_pair
        if seq.shape[0] > LIMO_MAX_LEN:
            print(f"[{name}] too long"); continue
        seq = seq.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, mu, _ = v1._m.encode(seq)[:3]
        z0 = mu

        for sigma in args.sigmas:
            n = args.n_per_seed
            z = z0.expand(n, -1) + sigma * torch.randn(n, z0.shape[1], device=DEVICE)
            with torch.no_grad():
                logp = v3_3._m._generate_autoregressive(z, emb=None, sample=True)
            toks = logp.argmax(-1).cpu().numpy()
            n_valid = 0; n_new = 0
            for i in range(n):
                s = tok.indices_to_smiles(toks[i].tolist())
                c = canon(s) if s else None
                if not c: continue
                n_valid += 1
                if c in all_canon: continue
                all_canon.add(c); n_new += 1
                rows.append({"seed": name, "sigma": sigma, "smiles": c})
            print(f"  [{name}] sigma={sigma}  valid={n_valid}/{n}  new_unique={n_new}")
    elapsed = time.time() - t0
    print(f"\nTotal unique canonical: {len(all_canon)}  ({elapsed:.1f}s)")

    # Save SMILES (one per line) for downstream rerank
    with out.open("w", encoding="utf-8") as f:
        for r in rows: f.write(r["smiles"] + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
