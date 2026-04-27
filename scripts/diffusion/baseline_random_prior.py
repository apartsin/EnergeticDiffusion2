"""2.3 baseline: sample latents from N(0, I) — no diffusion sampler, no
denoiser, no classifier guidance — and decode through LIMO. Output is in
exactly the same markdown format as joint_rerank.py so it can be fed into
rerank_v2.py for apples-to-apples comparison against the merged top-100.

Usage:
    /c/Python314/python scripts/diffusion/baseline_random_prior.py \
        --exp_v4b experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z \
        --n_pool 3000 --n_keep 200 \
        --target_density 1.95 --target_d 9.5 --target_p 40 --target_hof 220 \
        --hard_sa 5.0 --hard_sc 3.5 --tanimoto_min 0.20 --tanimoto_max 0.55 \
        --require_neutral --with_chem_filter --with_feasibility \
        --out experiments/baseline_random_prior_pool3k.md
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_v4b", required=True,
                    help="Used only to load LIMO ckpt path + training stats.")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--n_pool", type=int, default=3000)
    ap.add_argument("--n_keep", type=int, default=200)
    ap.add_argument("--quantile", type=float, default=1.281)
    ap.add_argument("--target_density", type=float, default=None)
    ap.add_argument("--target_hof", type=float, default=None)
    ap.add_argument("--target_d", type=float, default=None)
    ap.add_argument("--target_p", type=float, default=None)
    ap.add_argument("--require_neutral", action="store_true")
    ap.add_argument("--with_chem_filter", action="store_true")
    ap.add_argument("--with_feasibility", action="store_true")
    ap.add_argument("--w_sa", type=float, default=0.5)
    ap.add_argument("--w_sc", type=float, default=0.25)
    ap.add_argument("--hard_sa", type=float, default=5.0)
    ap.add_argument("--hard_sc", type=float, default=3.5)
    ap.add_argument("--tanimoto_min", type=float, default=None)
    ap.add_argument("--tanimoto_max", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = Path(args.base)
    sys.path.insert(0, "scripts/diffusion")
    sys.path.insert(0, "scripts/vae")
    from joint_rerank import (load_denoiser_pack, canon, is_neutral,
                                composite_feasibility_penalty,
                                real_sa, real_sc, morgan_fp,
                                load_limo, find_limo_repo)
    from limo_model import (SELFIESTokenizer, load_vocab, build_limo_vocab,
                              save_vocab, LIMO_MAX_LEN)
    from unimol_validator import UniMolValidator

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    exp = Path(args.exp_v4b);  exp = exp if exp.is_absolute() else base / exp
    print("Loading v4-B (for stats + LIMO ckpt path) ...")
    d_v4b, sch_v4b, l_v4b, pn, n_props = load_denoiser_pack(
        exp, args.ckpt, args.device, base)
    stats = l_v4b["stats"]

    print("Loading LIMO ...")
    limo_dir = find_limo_repo(base); vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    ckpt_limo = Path(l_v4b["meta"]["checkpoint"])
    if not ckpt_limo.is_absolute(): ckpt_limo = base / ckpt_limo
    limo, _ = load_limo(base, str(ckpt_limo), args.device)

    name_map = {"density": "density", "heat_of_formation": "HOF_S",
                 "detonation_velocity": "DetoD", "detonation_pressure": "DetoP"}
    val = UniMolValidator(model_dir=str(
        base / "data/raw/energetic_external/EMDP/Data/smoke_model"))

    # Targets (unused by sampler, but used by composite + reported)
    target_raw = {p: args.quantile * stats[p]["std"] + stats[p]["mean"] for p in pn}
    if args.target_density is not None: target_raw["density"] = args.target_density
    if args.target_hof is not None: target_raw["heat_of_formation"] = args.target_hof
    if args.target_d is not None: target_raw["detonation_velocity"] = args.target_d
    if args.target_p is not None: target_raw["detonation_pressure"] = args.target_p

    latent_dim = d_v4b.latent_dim
    print(f"\nSampling {args.n_pool} latents from N(0, I_{latent_dim}) ...")
    t0 = time.time()
    z = torch.randn(args.n_pool, latent_dim, device=args.device)
    print(f"  z drawn ({time.time()-t0:.1f}s)")

    print("Decoding ...")
    t0 = time.time()
    chunks = []
    bs = 256
    with torch.no_grad():
        for i in range(0, args.n_pool, bs):
            zb = z[i:i + bs]
            logits = limo.decode(zb)
            chunks.append(logits.argmax(-1).cpu())
    toks = torch.cat(chunks, dim=0).tolist()
    smis_raw = [tok.indices_to_smiles(t) for t in toks]
    print(f"  decoded {len(smis_raw)} ({time.time()-t0:.1f}s)")

    # Canonicalise + filter
    canons = [canon(s) for s in smis_raw]
    canons = [c for c in canons if c]
    print(f"valid canonical: {len(canons)} / {args.n_pool}")
    if args.require_neutral:
        canons = [c for c in canons if is_neutral(c)]
        print(f"  neutral: {len(canons)}")
    seen = {}
    for c in canons:
        if c not in seen: seen[c] = {"random_prior"}
    smis = list(seen.keys())
    print(f"  unique: {len(smis)}")

    print("3DCNN scoring ...")
    pdict = val.predict(smis)
    cols = {p: np.asarray(pdict.get(name_map[p]), dtype=float) for p in pn}
    keep = np.ones(len(smis), dtype=bool)
    for p in pn: keep &= ~np.isnan(cols[p])
    smis = [s for s, k in zip(smis, keep) if k]
    cols = {p: cols[p][keep] for p in pn}
    print(f"  fully validated: {len(smis)}")

    if args.with_chem_filter:
        from chem_filter import chem_filter_batch
        keep_idx, reasons = chem_filter_batch(smis, cols, pn)
        print(f"  chem_filter: kept {len(keep_idx)}/{len(smis)}")
        smis = [smis[i] for i in keep_idx]
        cols = {p: cols[p][np.asarray(keep_idx)] for p in pn}

    sa = np.array([real_sa(s) for s in smis])
    sc = np.array([real_sc(s) for s in smis])
    if args.with_feasibility or args.hard_sa < 99 or args.hard_sc < 99:
        keep = np.ones(len(smis), dtype=bool)
        keep &= ~((~np.isnan(sa)) & (sa > args.hard_sa))
        keep &= ~((~np.isnan(sc)) & (sc > args.hard_sc))
        idx_keep = np.where(keep)[0]
        smis = [smis[i] for i in idx_keep]
        cols = {p: cols[p][idx_keep] for p in pn}
        sa = sa[idx_keep]; sc = sc[idx_keep]
        print(f"  hard SA/SC caps: kept {len(smis)}")

    if args.tanimoto_min is not None or args.tanimoto_max is not None:
        train_smiles = l_v4b.get("smiles", [])[:5000]
        ref_fps = [morgan_fp(s) for s in train_smiles if morgan_fp(s) is not None]
        max_tans = []
        for s in smis:
            fp = morgan_fp(s)
            if fp is None: max_tans.append(0.0); continue
            tans = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            max_tans.append(max(tans))
        max_tans = np.array(max_tans)
        keep = np.ones(len(smis), dtype=bool)
        if args.tanimoto_min is not None: keep &= max_tans >= args.tanimoto_min
        if args.tanimoto_max is not None: keep &= max_tans <= args.tanimoto_max
        idx = np.where(keep)[0]
        smis = [smis[i] for i in idx]
        cols = {p: cols[p][idx] for p in pn}
        sa = sa[idx]; sc = sc[idx]
        max_tans = max_tans[idx]
        print(f"  Tanimoto-window: kept {len(smis)}")
    else:
        max_tans = np.zeros(len(smis))

    composite = np.zeros(len(smis))
    for p in pn:
        composite += np.abs(cols[p] - target_raw[p]) / max(stats[p]["std"], 1e-6)
    if args.with_feasibility:
        pen = np.array([composite_feasibility_penalty(sa[k], sc[k], args.w_sa, args.w_sc)
                          for k in range(len(smis))])
        composite += pen

    order = np.argsort(composite)
    top_idx = order[:min(args.n_keep, len(order))]

    md = ["# Random-Gaussian-prior baseline (2.3) — no diffusion, no guidance",
          "",
          f"v4-B exp (for stats only): `{exp.name}`",
          f"latent_dim={latent_dim}, n_pool={args.n_pool}, seed={args.seed}",
          "",
          "## Targets",
          "| property | target | std |",
          "|---|---|---|"]
    for p in pn:
        md.append(f"| {p} | {target_raw[p]:+.3f} | {stats[p]['std']:.3f} |")
    md.append("")
    md.append(f"Hard caps: SA ≤ {args.hard_sa}, SC ≤ {args.hard_sc}")
    if args.tanimoto_min is not None or args.tanimoto_max is not None:
        md.append(f"Tanimoto window vs training: [{args.tanimoto_min}, {args.tanimoto_max}]")
    md.append("")
    md.append(f"## Top {min(args.n_keep, len(top_idx))} candidates (composite-ranked)")
    md.append("")
    md.append("| rank | composite | ρ | HOF | D | P | SA | SC | maxTan | source(s) | SMILES |")
    md.append("|" + "|".join(["---"] * 11) + "|")
    for i, idx in enumerate(top_idx):
        s = smis[idx]
        md.append(f"| {i+1} | {composite[idx]:.2f} | {cols['density'][idx]:.3f} | "
                  f"{cols['heat_of_formation'][idx]:+.1f} | "
                  f"{cols['detonation_velocity'][idx]:.2f} | "
                  f"{cols['detonation_pressure'][idx]:.2f} | "
                  f"{sa[idx]:.2f} | {sc[idx]:.2f} | "
                  f"{max_tans[idx]:.2f} | random_prior | `{s}` |")
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n-> {out_path}  ({len(top_idx)} candidates)")
    print(f"  unique decoded valid: {len(seen)} / {args.n_pool}")
    print(f"  fraction post-3DCNN keep: {len(smis)} / {len(seen)} = {len(smis)/max(len(seen),1):.1%}")


if __name__ == "__main__":
    main()
