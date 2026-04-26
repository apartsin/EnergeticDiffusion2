"""Joint v3 + v4-B rerank: best of both worlds.

Generate independent pools from two checkpoints, dedupe canonical SMILES,
validate jointly with the 3DCNN ensemble + chem_filter, score with one
composite ranker. Use case: v4-B for ρ/D/P, v3 for HOF tail — ensemble
captures candidates that either model alone would miss.

Usage:
    python scripts/diffusion/joint_rerank.py \
        --exp_v4b experiments/diffusion_subset_cond_expanded_v4b_<ts> \
        --exp_v3  experiments/diffusion_subset_cond_expanded_v3_<ts> \
        --cfg 7 --n_pool_each 1500 --n_keep 80 \
        --target_density 1.95 --target_d 9.5 --target_p 40 --target_hof 220 \
        --hard_sa 5.0 --hard_sc 3.5 --tanimoto_min 0.20 --tanimoto_max 0.55 \
        --require_neutral --with_chem_filter --with_feasibility
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vae"))

from model import ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample
from limo_model import (LIMOVAE, SELFIESTokenizer, load_vocab,
                        build_limo_vocab, save_vocab, LIMO_MAX_LEN, find_limo_repo)
from limo_factory import load_limo, LIMOInferenceWrapper
from unimol_validator import UniMolValidator
from feasibility_utils import (real_sa, real_sc, SA_DROP_ABOVE, SC_DROP_ABOVE,
                                composite_feasibility_penalty)
from chem_filter import chem_filter


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def is_neutral(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return False
    if Chem.GetFormalCharge(m) != 0: return False
    return not any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms())


def morgan_fp(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def load_denoiser_pack(exp_dir, ckpt_name, device, base):
    cb = torch.load(exp_dir / "checkpoints" / ckpt_name, map_location=device,
                     weights_only=False)
    cfg = cb["config"]
    lblob = torch.load(base / cfg["paths"]["latents_pt"], weights_only=False)
    pn = lblob["property_names"]
    n_props = lblob["values_raw"].shape[1]
    d = ConditionalDenoiser(latent_dim=lblob["z_mu"].shape[1],
                              hidden=cfg["model"]["hidden"],
                              n_blocks=cfg["model"]["n_blocks"],
                              time_dim=cfg["model"]["time_dim"],
                              prop_emb_dim=cfg["model"]["prop_emb_dim"],
                              n_props=n_props,
                              dropout=cfg["model"].get("dropout", 0.0)).to(device)
    d.load_state_dict(cb["model_state"])
    if cb.get("ema_state") is not None:
        ema = EMA(d, decay=cfg["training"]["ema_decay"])
        ema.load_state_dict(cb["ema_state"]); ema.apply_to(d)
    d.eval()
    sch = NoiseSchedule(T=cfg["training"]["T"], device=device)
    return d, sch, lblob, pn, n_props


def generate_pool(denoiser, schedule, limo, tok, n_pool, target_z, n_props,
                    cfg_g, n_steps, device):
    mask = torch.ones(n_pool, n_props, device=device)
    vals = torch.full((n_pool, n_props), 0.0, device=device)
    for j in range(target_z.shape[0]):
        vals[:, j] = target_z[j]
    z = ddim_sample(denoiser, schedule, vals, mask,
                     n_steps=n_steps, guidance_scale=cfg_g, device=device)
    with torch.no_grad():
        logits = limo.decode(z)
    toks = logits.argmax(-1).cpu().tolist()
    return [tok.indices_to_smiles(t) for t in toks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_v4b", required=True)
    ap.add_argument("--exp_v3",  required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--cfg",  type=float, default=7.0)
    ap.add_argument("--n_pool_each", type=int, default=1500)
    ap.add_argument("--n_keep", type=int, default=80)
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--quantile", type=float, default=1.281)
    ap.add_argument("--target_density", type=float, default=None)
    ap.add_argument("--target_hof",     type=float, default=None)
    ap.add_argument("--target_d",       type=float, default=None)
    ap.add_argument("--target_p",       type=float, default=None)
    ap.add_argument("--require_neutral", action="store_true")
    ap.add_argument("--with_chem_filter", action="store_true")
    ap.add_argument("--with_feasibility", action="store_true")
    ap.add_argument("--w_sa", type=float, default=0.5)
    ap.add_argument("--w_sc", type=float, default=0.25)
    ap.add_argument("--hard_sa", type=float, default=SA_DROP_ABOVE)
    ap.add_argument("--hard_sc", type=float, default=SC_DROP_ABOVE)
    ap.add_argument("--tanimoto_min", type=float, default=None,
                    help="Drop candidates with Tanimoto > this to any training row "
                         "(too similar to known)")
    ap.add_argument("--tanimoto_max", type=float, default=None,
                    help="Drop candidates with Tanimoto < this to any training row "
                         "(off-distribution / unrealistic)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    exp_v4b = Path(args.exp_v4b);  exp_v4b = exp_v4b if exp_v4b.is_absolute() else base / exp_v4b
    exp_v3  = Path(args.exp_v3);   exp_v3  = exp_v3  if exp_v3.is_absolute()  else base / exp_v3

    print("Loading v4-B …")
    d_v4b, sch_v4b, l_v4b, pn, n_props = load_denoiser_pack(exp_v4b, args.ckpt, args.device, base)
    print("Loading v3 …")
    d_v3,  sch_v3,  l_v3,  _,  _       = load_denoiser_pack(exp_v3,  args.ckpt, args.device, base)
    stats = l_v4b["stats"]   # use v4-B's training stats as reference

    print("Loading LIMO …")
    limo_dir = find_limo_repo(base); vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    ckpt_limo = Path(l_v4b["meta"]["checkpoint"])
    if not ckpt_limo.is_absolute(): ckpt_limo = base / ckpt_limo
    limo, _limo_ver = load_limo(base, str(ckpt_limo), args.device)

    print("Loading 3DCNN …")
    val = UniMolValidator(model_dir=str(
        base / "data/raw/energetic_external/EMDP/Data/smoke_model"))
    name_map = {"density":"density","heat_of_formation":"HOF_S",
                "detonation_velocity":"DetoD","detonation_pressure":"DetoP"}

    # Targets
    target_raw = {p: args.quantile*stats[p]["std"]+stats[p]["mean"] for p in pn}
    if args.target_density is not None: target_raw["density"] = args.target_density
    if args.target_hof     is not None: target_raw["heat_of_formation"] = args.target_hof
    if args.target_d       is not None: target_raw["detonation_velocity"] = args.target_d
    if args.target_p       is not None: target_raw["detonation_pressure"] = args.target_p
    target_z = torch.tensor([(target_raw[p]-stats[p]["mean"])/stats[p]["std"]
                                 for p in pn], device=args.device)
    print("Targets:", target_raw)

    # Pools
    print(f"Generating v4-B pool (n={args.n_pool_each}) …")
    smis_v4b = generate_pool(d_v4b, sch_v4b, limo, tok, args.n_pool_each,
                                target_z, n_props, args.cfg, args.n_steps, args.device)
    print(f"Generating v3 pool (n={args.n_pool_each}) …")
    smis_v3  = generate_pool(d_v3,  sch_v3,  limo, tok, args.n_pool_each,
                                target_z, n_props, args.cfg, args.n_steps, args.device)
    # tag provenance
    pool = [(s, "v4b") for s in smis_v4b] + [(s, "v3") for s in smis_v3]
    canons = [(canon(s), src) for s, src in pool]
    canons = [(c, src) for c, src in canons if c]
    print(f"valid: {len(canons)} / {2*args.n_pool_each}")
    if args.require_neutral:
        canons = [(c, src) for c, src in canons if is_neutral(c)]
        print(f"  neutral: {len(canons)}")
    # dedupe by canonical SMILES (keep first occurrence + record set of sources)
    seen = {}
    for c, src in canons:
        if c in seen: seen[c].add(src)
        else: seen[c] = {src}
    smis = list(seen.keys())
    print(f"  unique: {len(smis)}")

    print("Validating with 3DCNN …")
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

    # SA + SC
    sa = np.array([real_sa(s) for s in smis])
    sc = np.array([real_sc(s) for s in smis])
    if args.with_feasibility or args.hard_sa < SA_DROP_ABOVE or args.hard_sc < SC_DROP_ABOVE:
        keep = np.ones(len(smis), dtype=bool)
        keep &= ~((~np.isnan(sa)) & (sa > args.hard_sa))
        keep &= ~((~np.isnan(sc)) & (sc > args.hard_sc))
        idx_keep = np.where(keep)[0]
        smis = [smis[i] for i in idx_keep]
        cols = {p: cols[p][idx_keep] for p in pn}
        sa = sa[idx_keep]; sc = sc[idx_keep]
        print(f"  hard SA/SC caps: kept {len(smis)}")

    # Tanimoto-to-training novelty filter
    if args.tanimoto_min is not None or args.tanimoto_max is not None:
        # Use training SMILES as reference. Sample to keep fast.
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
        if args.tanimoto_min is not None:
            keep &= max_tans >= args.tanimoto_min
        if args.tanimoto_max is not None:
            keep &= max_tans <= args.tanimoto_max
        idx = np.where(keep)[0]
        smis = [smis[i] for i in idx]
        cols = {p: cols[p][idx] for p in pn}
        sa = sa[idx]; sc = sc[idx]
        max_tans = max_tans[idx]
        print(f"  Tanimoto-window: kept {len(smis)}")
    else:
        max_tans = np.zeros(len(smis))

    # Composite (per-property error / std + optional feasibility penalty)
    composite = np.zeros(len(smis))
    for p in pn:
        composite += np.abs(cols[p] - target_raw[p]) / max(stats[p]["std"], 1e-6)
    if args.with_feasibility:
        pen = np.array([composite_feasibility_penalty(sa[k], sc[k], args.w_sa, args.w_sc)
                         for k in range(len(smis))])
        composite += pen

    order = np.argsort(composite)
    top_idx = order[:min(args.n_keep, len(order))]

    md = ["# Joint v3 + v4-B rerank — breakthrough Path A", "",
          f"v4-B exp: `{exp_v4b.name}`",
          f"v3  exp: `{exp_v3.name}`",
          f"cfg={args.cfg}, n_pool_each={args.n_pool_each}, n_keep={args.n_keep}",
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
    md.append("|" + "|".join(["---"]*11) + "|")
    for i, idx in enumerate(top_idx):
        s = smis[idx]
        srcs = ",".join(sorted(seen.get(s, {"?"})))
        md.append(f"| {i+1} | {composite[idx]:.2f} | {cols['density'][idx]:.3f} | "
                  f"{cols['heat_of_formation'][idx]:+.1f} | "
                  f"{cols['detonation_velocity'][idx]:.2f} | "
                  f"{cols['detonation_pressure'][idx]:.2f} | "
                  f"{sa[idx]:.2f} | {sc[idx]:.2f} | "
                  f"{max_tans[idx]:.2f} | {srcs} | `{s}` |")

    out = Path(args.out) if args.out else exp_v4b / "joint_rerank.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    sys.exit(main())
