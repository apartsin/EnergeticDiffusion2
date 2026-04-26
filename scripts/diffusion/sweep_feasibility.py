"""Validation sweep for SA + SC integration.

Two complementary sweeps in one run:

A. Rerank-weight sweep — generates ONE pool, then re-ranks at multiple
   (w_sa, w_sc) settings without re-sampling. Cheap, exhaustive.

B. Sampling-λ sweep — re-samples a smaller pool for each (λ_SA, λ_SC) so
   the sampling-time gradient is exercised. Slower (one full DDIM per λ pair).

Outputs a consolidated markdown report comparing per-property metrics +
SA/SC distribution shifts across all settings.
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
from limo_factory import load_limo, LIMOInferenceWrapper
from unimol_validator import UniMolValidator
from feasibility_utils import (real_sa, real_sc, SA_DROP_ABOVE, SC_DROP_ABOVE,
                                composite_feasibility_penalty,
                                DEFAULT_SA_CKPT, DEFAULT_SC_CKPT)
from feasibility_sampler import ddim_sample_feasibility
sys.path.insert(0, str(HERE.parent / "guidance"))


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def is_neutral(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return False
    if Chem.GetFormalCharge(m) != 0: return False
    return not any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms())


def load_stack(exp, ckpt_name, base, device):
    cb = torch.load(exp / "checkpoints" / ckpt_name, map_location=device,
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
        n_props=n_props, dropout=cfg["model"].get("dropout", 0.0)).to(device)
    denoiser.load_state_dict(cb["model_state"])
    if cb.get("ema_state") is not None:
        ema = EMA(denoiser, decay=cfg["training"]["ema_decay"])
        ema.load_state_dict(cb["ema_state"]); ema.apply_to(denoiser)
    denoiser.eval()
    schedule = NoiseSchedule(T=cfg["training"]["T"], device=device)

    limo_dir = find_limo_repo(base); vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    ckpt_limo = Path(latents_blob["meta"]["checkpoint"])
    if not ckpt_limo.is_absolute(): ckpt_limo = base / ckpt_limo
    limo, _limo_ver = load_limo(base, str(ckpt_limo), device)

    val = UniMolValidator(model_dir=str(
        base / "data/raw/energetic_external/EMDP/Data/smoke_model"))
    return denoiser, schedule, limo, tok, val, latents_blob, stats, pn, n_props


def evaluate_pool(valid_smi, cols, sa_arr, sc_arr, targets_raw, stats,
                    pn, w_sa, w_sc, n_keep):
    """Apply rerank with given weights, return per-property metrics + SA/SC of top-N."""
    composite = np.zeros(len(valid_smi))
    for p in pn:
        sd_p = stats[p]["std"]
        composite += np.abs(cols[p] - targets_raw[p]) / max(sd_p, 1e-6)
    keep_mask = np.ones(len(valid_smi), dtype=bool)
    keep_mask &= ~((~np.isnan(sa_arr)) & (sa_arr > SA_DROP_ABOVE))
    keep_mask &= ~((~np.isnan(sc_arr)) & (sc_arr > SC_DROP_ABOVE))
    pen = np.array([composite_feasibility_penalty(sa_arr[k], sc_arr[k], w_sa, w_sc)
                     for k in range(len(valid_smi))])
    score = composite + pen
    score[~keep_mask] = np.inf
    order = np.argsort(score)
    n = min(n_keep, int(keep_mask.sum()))
    top = order[:n]
    metrics = {}
    for p in pn:
        v = cols[p][top]; t = targets_raw[p]
        metrics[p] = {
            "rel_mae": float(100*np.mean(np.abs(v - t))/max(abs(t),1e-6)),
            "within_10": float(100*np.mean(np.abs(v - t) <= 0.1*abs(t))),
            "mean": float(v.mean()),
        }
    metrics["_sa_mean"] = float(np.nanmean(sa_arr[top]))
    metrics["_sc_mean"] = float(np.nanmean(sc_arr[top]))
    metrics["_n_kept"]  = int(keep_mask.sum())
    metrics["_n_top"]   = int(n)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--cfg",  type=float, default=7.0)
    ap.add_argument("--n_pool", type=int, default=400)
    ap.add_argument("--n_keep", type=int, default=40)
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--quantile", type=float, default=1.281)
    ap.add_argument("--rerank_weights", type=str,
                    default="0.0,0.0;0.5,0.25;1.0,0.5;2.0,1.0",
                    help="Semicolon-separated (w_sa,w_sc) pairs")
    ap.add_argument("--sample_lambdas", type=str,
                    default="0.0,0.0;0.3,0.2;0.5,0.3",
                    help="Semicolon-separated (lambda_sa,lambda_sc) pairs")
    ap.add_argument("--sample_warmup", type=int, default=25)
    ap.add_argument("--sample_pool", type=int, default=200,
                    help="Pool per λ pair (smaller because re-sampling per pair)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    exp = Path(args.exp)
    if not exp.is_absolute(): exp = base / exp

    print("Loading stack …")
    denoiser, schedule, limo, tok, validator, lblob, stats, pn, n_props = \
        load_stack(exp, args.ckpt, base, args.device)
    name_map = {"density":"density","heat_of_formation":"HOF_S",
                "detonation_velocity":"DetoD","detonation_pressure":"DetoP"}
    targets_raw = {p: args.quantile*stats[p]["std"]+stats[p]["mean"] for p in pn}

    md = ["# Feasibility validation sweep (SA + SC)", "",
          f"checkpoint: `{exp.name}/checkpoints/{args.ckpt}`",
          f"cfg={args.cfg}, q90={args.quantile}",
          f"rerank pool={args.n_pool}, keep={args.n_keep}",
          f"sample_lambda pool={args.sample_pool}", "",
          "## Targets (q90)",
          "| property | target | std |",
          "|---|---|---|"]
    for p in pn:
        md.append(f"| {p} | {targets_raw[p]:+.3f} | {stats[p]['std']:.3f} |")

    # ── A. RERANK-WEIGHT SWEEP ON A SHARED POOL ──────────────────────────
    print(f"\n[A] Generating shared pool of {args.n_pool} (numerically-stable DDIM, all-prop q90)…")
    mask = torch.ones(args.n_pool, n_props, device=args.device)
    vals = torch.full((args.n_pool, n_props), args.quantile, device=args.device)
    t0 = time.time()
    # Use the (fixed) feasibility sampler with no guidance for consistent pool generation
    z = ddim_sample_feasibility(denoiser, schedule, vals, mask,
                                  n_steps=args.n_steps, guidance_scale=args.cfg,
                                  device=args.device, lambda_sa=0.0, lambda_sc=0.0)
    with torch.no_grad():
        logits = limo.decode(z)
    toks = logits.argmax(-1).cpu().tolist()
    smiles = [tok.indices_to_smiles(t) for t in toks]
    smi_canon = [canon(s) for s in smiles if s and canon(s)]
    smi_canon = [s for s in smi_canon if is_neutral(s)]
    print(f"  pool valid + neutral: {len(smi_canon)}/{args.n_pool}")
    pdict = validator.predict(smi_canon)
    cols = {}
    for p in pn:
        c = pdict.get(name_map[p])
        cols[p] = np.asarray(c, dtype=float) if c is not None else None
    keep = np.ones(len(smi_canon), dtype=bool)
    for p, c in cols.items():
        if c is None: keep &= False; break
        keep &= ~np.isnan(c)
    smi_canon = [s for s, k in zip(smi_canon, keep) if k]
    cols = {p: c[keep] for p, c in cols.items()}
    print(f"  fully validated: {len(smi_canon)}")
    print("  computing SA + SC on pool …")
    sa_arr = np.array([real_sa(s) for s in smi_canon], dtype=float)
    sc_arr = np.array([real_sc(s) for s in smi_canon], dtype=float)
    print(f"  SA mean={np.nanmean(sa_arr):.2f}, SC mean={np.nanmean(sc_arr):.2f}")
    print(f"  pool prep: {time.time()-t0:.0f}s")

    md += ["", "## A. Rerank-weight sweep (shared pool)", ""]
    md.append("| w_sa | w_sc | n_kept | n_top | "
              "ρ rel_MAE % | ρ in10 % | "
              "HOF rel_MAE % | HOF in10 % | "
              "D rel_MAE % | D in10 % | "
              "P rel_MAE % | P in10 % | "
              "SA mean (top) | SC mean (top) |")
    md.append("|" + "|".join(["---"]*14) + "|")

    rerank_pairs = [tuple(map(float, p.split(",")))
                     for p in args.rerank_weights.split(";")]
    rerank_results = {}
    for w_sa, w_sc in rerank_pairs:
        m = evaluate_pool(smi_canon, cols, sa_arr, sc_arr, targets_raw,
                            stats, pn, w_sa, w_sc, args.n_keep)
        rerank_results[(w_sa, w_sc)] = m
        md.append(f"| {w_sa} | {w_sc} | {m['_n_kept']} | {m['_n_top']} | "
                  f"{m['density']['rel_mae']:.1f} | {m['density']['within_10']:.0f} | "
                  f"{m['heat_of_formation']['rel_mae']:.1f} | {m['heat_of_formation']['within_10']:.0f} | "
                  f"{m['detonation_velocity']['rel_mae']:.1f} | {m['detonation_velocity']['within_10']:.0f} | "
                  f"{m['detonation_pressure']['rel_mae']:.1f} | {m['detonation_pressure']['within_10']:.0f} | "
                  f"{m['_sa_mean']:.2f} | {m['_sc_mean']:.2f} |")
        print(f"  ({w_sa},{w_sc}): SA mean (top) = {m['_sa_mean']:.2f}, "
              f"D rel_MAE={m['detonation_velocity']['rel_mae']:.1f}%")

    # ── B. SAMPLING-λ SWEEP ──────────────────────────────────────────────
    md += ["", "## B. Sampling-λ sweep (one pool per λ pair)", ""]
    md.append("| λ_SA | λ_SC | pool valid | top SA mean | top SC mean | "
              "ρ rel_MAE % | HOF rel_MAE % | D rel_MAE % | P rel_MAE % |")
    md.append("|" + "|".join(["---"]*9) + "|")

    lambda_pairs = [tuple(map(float, p.split(",")))
                     for p in args.sample_lambdas.split(";")]
    for lam_sa, lam_sc in lambda_pairs:
        print(f"\n[B] sampling pool with λ_SA={lam_sa} λ_SC={lam_sc} …")
        mask = torch.ones(args.sample_pool, n_props, device=args.device)
        vals = torch.full((args.sample_pool, n_props), args.quantile,
                            device=args.device)
        t0 = time.time()
        z = ddim_sample_feasibility(
            denoiser, schedule, vals, mask,
            n_steps=args.n_steps, guidance_scale=args.cfg,
            device=args.device,
            lambda_sa=lam_sa, lambda_sc=lam_sc,
            feasibility_warmup_steps=args.sample_warmup,
            use_t_aware=True,
        )
        with torch.no_grad():
            logits = limo.decode(z)
        toks = logits.argmax(-1).cpu().tolist()
        smis = [tok.indices_to_smiles(t) for t in toks]
        smi_canon = [canon(s) for s in smis if s and canon(s)]
        smi_canon = [s for s in smi_canon if is_neutral(s)]
        if not smi_canon:
            md.append(f"| {lam_sa} | {lam_sc} | 0 | – | – | – | – | – | – |")
            continue
        pdict = validator.predict(smi_canon)
        cols2 = {p: np.asarray(pdict.get(name_map[p]), dtype=float) for p in pn}
        keep = np.ones(len(smi_canon), dtype=bool)
        for p in pn:
            keep &= ~np.isnan(cols2[p])
        smi_canon = [s for s, k in zip(smi_canon, keep) if k]
        cols2 = {p: cols2[p][keep] for p in pn}
        sa2 = np.array([real_sa(s) for s in smi_canon], dtype=float)
        sc2 = np.array([real_sc(s) for s in smi_canon], dtype=float)
        m = evaluate_pool(smi_canon, cols2, sa2, sc2, targets_raw, stats, pn,
                            w_sa=1.0, w_sc=0.5, n_keep=args.n_keep)
        md.append(f"| {lam_sa} | {lam_sc} | {len(smi_canon)} | "
                  f"{m['_sa_mean']:.2f} | {m['_sc_mean']:.2f} | "
                  f"{m['density']['rel_mae']:.1f} | "
                  f"{m['heat_of_formation']['rel_mae']:.1f} | "
                  f"{m['detonation_velocity']['rel_mae']:.1f} | "
                  f"{m['detonation_pressure']['rel_mae']:.1f} |")
        print(f"  ({lam_sa},{lam_sc}): SA top mean={m['_sa_mean']:.2f} "
              f"in {time.time()-t0:.0f}s")

    md += ["",
           "## Reading the table",
           "- Goal: lower SA / SC mean WITHOUT regressing per-property rel_MAE > 5 pp.",
           "- (w_sa, w_sc) = (0,0) is the baseline (composite ranks by property error only).",
           "- (λ_SA, λ_SC) = (0,0) under sampling sweep is also baseline (no gradient).",
           ""]
    out_path = exp / "feasibility_sweep.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    sys.exit(main())
