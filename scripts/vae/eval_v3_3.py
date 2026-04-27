"""Evaluate v3.3 on its actual training distribution: in-distribution z's
(encoded from real molecules + noise), not pure N(0,I) prior.

Tests:
  1. Greedy + sampled self-consistency on 7 energetic seeds (encode -> decode)
  2. SDEdit-style C2C: encode a seed, add noise sigma, decode N times, measure
     Tanimoto distribution of variants. Useful for compound-to-compound
     exploration around known leads.
  3. Compare to v1 (production) and v3.1 baselines.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

sys.path.insert(0, "scripts/vae")
sys.path.insert(0, "external/LIMO")
from limo_factory import load_limo
from limo_model import SELFIESTokenizer, build_limo_vocab, LIMO_MAX_LEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = {
    "TNT":   "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
    "RDX":   "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]",
    "HMX":   "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]",
    "PETN":  "C(C(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-])O[N+](=O)[O-]",
    "TATB":  "Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]",
    "FOX-7": "NC(=C([N+](=O)[O-])[N+](=O)[O-])N",
    "CL-20": "C1(N2[N+](=O)[O-])C3N([N+](=O)[O-])C(N4[N+](=O)[O-])C2N([N+](=O)[O-])C1N3[N+](=O)[O-]",
}


def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def tanimoto(a, b):
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None: return 0.0
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, 2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, 2048)
    return DataStructs.TanimotoSimilarity(fa, fb)


def smiles_to_seq(smi, tok):
    out = tok.smiles_to_tensor(smi)
    if out is None: return None
    seq, length = out
    return seq.unsqueeze(0)


def encode_seed(model, seq):
    with torch.no_grad():
        z, mu, log_var = model.encode(seq.to(DEVICE))[:3]
    return mu  # use deterministic mean


def decode_v3(m, z, tok, sample=True, n=8):
    """Decode same z `n` times under multinomial sampling, return canonical SMILES list."""
    z_rep = z.expand(n, -1).contiguous()
    out = []
    with torch.no_grad():
        logp = m._generate_autoregressive(z_rep, emb=None, sample=sample)
    toks = logp.argmax(-1).cpu().numpy()
    for i in range(n):
        smi = tok.indices_to_smiles(toks[i].tolist())
        c = canon(smi)
        if c: out.append(c)
    return out


def decode_v1(model, z, tok, n=8):
    """v1 parallel decoder (deterministic)."""
    z_rep = z.expand(n, -1).contiguous()
    with torch.no_grad():
        logp = model.decode(z_rep)
    if torch.is_tensor(logp):
        toks = logp.argmax(-1).cpu().numpy()
    else:
        return []
    out = []
    for i in range(n):
        smi = tok.indices_to_smiles(toks[i].tolist())
        c = canon(smi)
        if c: out.append(c)
    return out


def main():
    vocab = build_limo_vocab("external/LIMO/zinc250k.smi")
    tok = SELFIESTokenizer(vocab)

    # Load all three: v1 (production), v3.1, v3.3
    v1, _ = load_limo(".", version="v1",
                      ckpt_override="experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt",
                      device=DEVICE)
    v1.eval()
    v3_3, _ = load_limo(".", version="v3_1",
                        ckpt_override="experiments/limo_v3_3_diff_aware_AR_20260426T122722Z/checkpoints/best.pt",
                        device=DEVICE)
    v3_3.eval()

    print("=" * 78)
    print(f"{'seed':<8} {'v1 self':>10} {'v3.3 greedy':>14} {'v3.3 sample×8 best':>22}")
    print("-" * 78)

    rows = []
    for name, smi in SEEDS.items():
        seq = smiles_to_seq(smi, tok)
        if seq is None or seq.shape[1] > LIMO_MAX_LEN:
            print(f"{name:<8} skipped (too long or untokenisable)"); continue

        # encode through both (same encoder, but verify)
        z = encode_seed(v1._m, seq).to(DEVICE)

        # v1 deterministic decode
        v1_out = decode_v1(v1._m, z, tok, n=1)
        v1_self = tanimoto(smi, v1_out[0]) if v1_out else 0.0

        # v3.3 greedy
        v3_3_g = decode_v3(v3_3._m, z, tok, sample=False, n=1)
        v3_3_g_t = tanimoto(smi, v3_3_g[0]) if v3_3_g else 0.0

        # v3.3 sample best of 8
        v3_3_s = decode_v3(v3_3._m, z, tok, sample=True, n=8)
        v3_3_s_best = max((tanimoto(smi, s) for s in v3_3_s), default=0.0)

        print(f"{name:<8} {v1_self:>10.3f} {v3_3_g_t:>14.3f} {v3_3_s_best:>22.3f}")
        rows.append({"seed": name, "v1": v1_self, "v3_3_greedy": v3_3_g_t,
                     "v3_3_sample_best": v3_3_s_best,
                     "v3_3_sample_all": [tanimoto(smi, s) for s in v3_3_s]})

    print()
    print("=== C2C / SDEdit: encode seed -> add noise -> decode 8x ===")
    for sigma in [0.5, 1.0, 2.0]:
        print(f"\n-- sigma = {sigma} --")
        for name in ["RDX", "HMX", "FOX-7"]:
            smi = SEEDS[name]
            seq = smiles_to_seq(smi, tok)
            z = encode_seed(v1._m, seq).to(DEVICE)
            z_noise = z.expand(8, -1) + sigma * torch.randn(8, z.shape[1], device=DEVICE)
            with torch.no_grad():
                logp = v3_3._m._generate_autoregressive(z_noise, emb=None, sample=True)
            toks = logp.argmax(-1).cpu().numpy()
            cands = []
            for i in range(8):
                s = tok.indices_to_smiles(toks[i].tolist())
                c = canon(s)
                if c: cands.append(c)
            uniq = list({c for c in cands})
            tans = [tanimoto(smi, c) for c in uniq]
            n_nitro = sum(1 for c in uniq if "[N+]" in c and "[O-]" in c)
            print(f"  {name:<6}  uniq={len(uniq)}/8   nitro-like={n_nitro}   "
                  f"max_tan={max(tans, default=0):.2f}   mean_tan={sum(tans)/max(len(tans),1):.2f}")
            for c in sorted(uniq, key=lambda x: -tanimoto(smi, x))[:3]:
                print(f"    [{tanimoto(smi, c):.2f}] {c}")

    out = Path("experiments/limo_v3_3_diff_aware_AR_20260426T122722Z/eval_summary.json")
    out.write_text(json.dumps({"self_consistency": rows}, indent=2))
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
