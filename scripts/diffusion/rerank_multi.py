"""Multi-property generate-and-rerank.

Conditions the denoiser on ALL targets simultaneously (mask = 1 for every
property) and ranks candidates by a composite score that requires high
performance across the joint criterion. Avoids the common failure mode
where high-HOF candidates have unusable D / P, or vice versa.

Composite score:
    composite = Σ_p w_p · |pred_p − target_p| / std_p

Lower = better. Default weights uniform; can also weight HOF higher etc.

Usage:
    python scripts/diffusion/rerank_multi.py \
        --exp experiments/diffusion_subset_cond_expanded_v4b_2026... \
        --cfg 7 --n_pool 400 --n_keep 40
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vae"))

from model import ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample
from limo_model import (LIMOVAE, SELFIESTokenizer, load_vocab,
                        build_limo_vocab, save_vocab, LIMO_MAX_LEN, find_limo_repo)
from unimol_validator import UniMolValidator


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def is_physically_valid(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    if Chem.GetFormalCharge(m) != 0: return False
    if any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()): return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--cfg",  type=float, default=7.0)
    ap.add_argument("--n_pool",  type=int, default=400)
    ap.add_argument("--n_keep",  type=int, default=40)
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--quantile", type=float, default=1.281)
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[1.0, 1.0, 1.0, 1.0],
                    help="Per-property weights in composite score, "
                         "ordered density HOF D P.")
    ap.add_argument("--require_neutral", action="store_true",
                    help="Drop candidates with non-zero formal charge or unpaired"
                         " electrons (filters unphysical SELFIES decode artefacts)")
    ap.add_argument("--with_feasibility", action="store_true",
                    help="Add SA + SC penalties to composite, drop above hard caps")
    ap.add_argument("--w_sa", type=float, default=1.0)
    ap.add_argument("--w_sc", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threedcnn_dir",
                    default="data/raw/energetic_external/EMDP/Data/smoke_model")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    exp = Path(args.exp)
    if not exp.is_absolute(): exp = base / exp

    cb = torch.load(exp / "checkpoints" / args.ckpt, map_location=args.device,
                     weights_only=False)
    cfg = cb["config"]
    latents_blob = torch.load(base / cfg["paths"]["latents_pt"], weights_only=False)
    pn = latents_blob["property_names"]
    stats = latents_blob["stats"]
    n_props = latents_blob["values_raw"].shape[1]

    denoiser = ConditionalDenoiser(
        latent_dim=latents_blob["z_mu"].shape[1],
        hidden=cfg["model"]["hidden"], n_blocks=cfg["model"]["n_blocks"],
        time_dim=cfg["model"]["time_dim"], prop_emb_dim=cfg["model"]["prop_emb_dim"],
        n_props=n_props, dropout=cfg["model"].get("dropout", 0.0)).to(args.device)
    denoiser.load_state_dict(cb["model_state"])
    if cb.get("ema_state") is not None:
        ema = EMA(denoiser, decay=cfg["training"]["ema_decay"])
        ema.load_state_dict(cb["ema_state"]); ema.apply_to(denoiser)
    denoiser.eval()
    schedule = NoiseSchedule(T=cfg["training"]["T"], device=args.device)

    print("Loading LIMO …")
    limo_dir = find_limo_repo(base); vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    ckpt_limo = Path(latents_blob["meta"]["checkpoint"])
    if not ckpt_limo.is_absolute(): ckpt_limo = base / ckpt_limo
    limo_b = torch.load(ckpt_limo, map_location=args.device, weights_only=False)
    limo = LIMOVAE(); limo.load_state_dict(limo_b["model_state"])
    limo.to(args.device).eval()

    print("Loading validator …")
    val = UniMolValidator(model_dir=str(base / args.threedcnn_dir))
    name_map = {"density":"density","heat_of_formation":"HOF_S",
                "detonation_velocity":"DetoD","detonation_pressure":"DetoP"}

    # Build all-props-at-q90 conditioning
    targets_raw = {}
    for j, p in enumerate(pn):
        targets_raw[p] = args.quantile * stats[p]["std"] + stats[p]["mean"]
    print("\nMulti-property targets (q90):")
    for p, t in targets_raw.items():
        print(f"  {p:25s} = {t:+.3f}")

    weights = dict(zip(pn, args.weights[:len(pn)]))
    print("\nWeights:")
    for p, w in weights.items():
        print(f"  {p:25s} = {w}")

    mask = torch.ones(args.n_pool, n_props, device=args.device)
    vals = torch.full((args.n_pool, n_props), args.quantile, device=args.device)

    t0 = time.time()
    print(f"\nGenerating pool of {args.n_pool} (cfg={args.cfg}) …")
    z = ddim_sample(denoiser, schedule, vals, mask,
                     n_steps=args.n_steps, guidance_scale=args.cfg,
                     device=args.device)
    with torch.no_grad():
        logits = limo.decode(z)
    toks = logits.argmax(-1).cpu().tolist()
    smiles = [tok.indices_to_smiles(t) for t in toks]
    valid_smi = [canon(s) for s in smiles if s and canon(s)]
    print(f"  valid: {len(valid_smi)} / {args.n_pool}")
    if args.require_neutral:
        n_pre = len(valid_smi)
        valid_smi = [s for s in valid_smi if is_physically_valid(s)]
        print(f"  --require_neutral: {len(valid_smi)} / {n_pre} kept (charge=0, no radicals)")
    if not valid_smi:
        print("  no valid molecules"); return 1

    print("Validating with 3DCNN ensemble …")
    pdict = val.predict(valid_smi)
    # collect all 4 properties; rows where all are present become candidates
    cols = {}
    for p in pn:
        col = pdict.get(name_map[p])
        cols[p] = np.asarray(col, dtype=float) if col is not None else None
    keep = np.ones(len(valid_smi), dtype=bool)
    for p, c in cols.items():
        if c is None: keep = np.zeros_like(keep); break
        keep &= ~np.isnan(c)
    valid_smi = [s for s, k in zip(valid_smi, keep) if k]
    cols = {p: c[keep] for p, c in cols.items()}
    print(f"  fully validated: {len(valid_smi)}")

    if not valid_smi:
        print("  no fully validated; aborting"); return 1

    # composite score (per-property errors)
    composite = np.zeros(len(valid_smi))
    for p in pn:
        sd_p = stats[p]["std"]
        composite += weights[p] * np.abs(cols[p] - targets_raw[p]) / max(sd_p, 1e-6)

    # Optional feasibility add-on
    sa_arr = np.full(len(valid_smi), np.nan)
    sc_arr = np.full(len(valid_smi), np.nan)
    if args.with_feasibility:
        try:
            from feasibility_utils import (real_sa, real_sc,
                                            SA_DROP_ABOVE, SC_DROP_ABOVE,
                                            composite_feasibility_penalty)
            for k, smi in enumerate(valid_smi):
                sa_arr[k] = real_sa(smi)
                sc_arr[k] = real_sc(smi)
            keep_mask = np.ones(len(valid_smi), dtype=bool)
            keep_mask &= ~((~np.isnan(sa_arr)) & (sa_arr > SA_DROP_ABOVE))
            keep_mask &= ~((~np.isnan(sc_arr)) & (sc_arr > SC_DROP_ABOVE))
            penalty = np.array([composite_feasibility_penalty(sa_arr[k], sc_arr[k],
                                                                 args.w_sa, args.w_sc)
                                for k in range(len(valid_smi))])
            composite = composite + penalty
            composite[~keep_mask] = np.inf
            print(f"  feasibility: {int(keep_mask.sum())}/{len(valid_smi)} kept under hard caps")
        except Exception as exc:
            print(f"  feasibility disabled: {exc}")

    order = np.argsort(composite)
    top_idx = order[: min(args.n_keep, len(order))]

    md = ["# Multi-property rerank (joint q90)", "",
          f"checkpoint: `{exp.name}/checkpoints/{args.ckpt}`",
          f"cfg={args.cfg}, n_pool={args.n_pool}, n_keep={args.n_keep}",
          f"q90 z-score = {args.quantile}", "",
          f"Pool valid (all 4 props): {len(valid_smi)} / {args.n_pool}",
          ""]
    md.append("## Targets")
    md.append("")
    md.append("| property | target | weight | std |")
    md.append("|---|---|---|---|")
    for p in pn:
        md.append(f"| {p} | {targets_raw[p]:+.3f} | {weights[p]} | "
                  f"{stats[p]['std']:.3f} |")

    md.append("")
    md.append("## Per-property metrics on top-N (composite-ranked) vs all valid")
    md.append("")
    md.append("| property | target | mean (top) | mean (all) | rel_MAE % top | "
               "rel_MAE % all | within_10 % top | within_10 % all |")
    md.append("|---|---|---|---|---|---|---|---|")
    for p in pn:
        all_v = cols[p]
        top_v = cols[p][top_idx]
        t = targets_raw[p]
        rel_top = 100*np.mean(np.abs(top_v - t))/max(abs(t),1e-6)
        rel_all = 100*np.mean(np.abs(all_v - t))/max(abs(t),1e-6)
        w10_top = 100*np.mean(np.abs(top_v - t) <= 0.1*abs(t))
        w10_all = 100*np.mean(np.abs(all_v - t) <= 0.1*abs(t))
        md.append(f"| {p} | {t:+.3f} | {top_v.mean():+.3f} | {all_v.mean():+.3f} | "
                  f"{rel_top:.1f} | {rel_all:.1f} | {w10_top:.0f} | {w10_all:.0f} |")

    md.append("")
    md.append(f"## Top {min(20, args.n_keep)} candidates (lowest composite)")
    md.append("")
    sa_col_hdr = " SA |" if args.with_feasibility else ""
    sc_col_hdr = " SC |" if args.with_feasibility else ""
    md.append(f"| rank | composite | density | HOF | D | P |{sa_col_hdr}{sc_col_hdr} SMILES |")
    sep = ["---"] * (6 + (2 if args.with_feasibility else 0) + 1)
    md.append("|" + "|".join(sep) + "|")
    for i, idx in enumerate(top_idx[:20]):
        sm = valid_smi[idx]
        c  = composite[idx]
        d  = cols["density"][idx]
        h  = cols["heat_of_formation"][idx]
        dv = cols["detonation_velocity"][idx]
        pp = cols["detonation_pressure"][idx]
        sa_cell = f" {sa_arr[idx]:.2f} |" if args.with_feasibility else ""
        sc_cell = f" {sc_arr[idx]:.2f} |" if args.with_feasibility else ""
        md.append(f"| {i+1} | {c:.3f} | {d:.3f} | {h:+.1f} | {dv:.2f} | "
                  f"{pp:.2f} |{sa_cell}{sc_cell} `{sm}` |")

    out = exp / "rerank_multi.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {out}")
    print(f"Total: {(time.time()-t0)/60:.1f} min")

    # Quick stdout summary
    print("\nMetrics (top vs all):")
    for p in pn:
        all_v = cols[p]; top_v = cols[p][top_idx]; t = targets_raw[p]
        rel_top = 100*np.mean(np.abs(top_v - t))/max(abs(t),1e-6)
        rel_all = 100*np.mean(np.abs(all_v - t))/max(abs(t),1e-6)
        w10_top = 100*np.mean(np.abs(top_v - t) <= 0.1*abs(t))
        print(f"  {p:25s}  top mean={top_v.mean():+.3f} (rel={rel_top:.1f}%, "
              f"w10={w10_top:.0f}%)  vs all rel={rel_all:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
