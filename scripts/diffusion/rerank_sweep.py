"""Generate-and-rerank evaluation.

For each (property, q90), sample N_pool latents, decode, validate with the
3DCNN ensemble, and keep the top N_keep by |pred - target|. Reports both
unranked (random subset of N_keep) and ranked metrics for direct comparison.

The expectation: rerank dramatically improves rel_MAE and within_10 % by
filtering out the model's misses while keeping its hits.
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


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def is_physically_valid(smi: str) -> bool:
    """Net formal charge == 0 AND no radical electrons."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    if Chem.GetFormalCharge(m) != 0: return False
    if any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()): return False
    return True


# Feasibility helpers (lazy import so non-feasibility runs don't pay the cost)
_FEAS = {"sa": None, "sc": None, "loaded": False}
def _ensure_feasibility():
    if _FEAS["loaded"]: return
    try:
        from feasibility_utils import (real_sa, real_sc,
                                        SA_DROP_ABOVE, SC_DROP_ABOVE,
                                        composite_feasibility_penalty)
        _FEAS.update({"sa": real_sa, "sc": real_sc,
                       "sa_drop": SA_DROP_ABOVE, "sc_drop": SC_DROP_ABOVE,
                       "penalty": composite_feasibility_penalty,
                       "loaded": True})
    except Exception as exc:
        print(f"[feasibility] disabled (import failed: {exc})")
        _FEAS["loaded"] = True   # don't keep retrying


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--cfg",  type=float, default=7.0)
    ap.add_argument("--n_pool",  type=int, default=200)
    ap.add_argument("--n_keep",  type=int, default=40)
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--quantile", type=float, default=1.281)
    ap.add_argument("--require_neutral", action="store_true",
                    help="Drop candidates with non-zero formal charge or unpaired"
                         " electrons (filters out unphysical SELFIES decode artefacts)")
    ap.add_argument("--with_feasibility", action="store_true",
                    help="Add SA + SC penalties to the composite ranker, drop"
                         " above hard caps. Requires sascorer + SCScorer.")
    ap.add_argument("--w_sa", type=float, default=1.0,
                    help="Weight on SA penalty term in composite")
    ap.add_argument("--w_sc", type=float, default=0.5,
                    help="Weight on SC penalty term in composite")
    ap.add_argument("--with_chem_filter", action="store_true",
                    help="Drop candidates failing physics/chemistry sanity")
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
    limo_dir = find_limo_repo(base)
    vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    ckpt_limo = Path(latents_blob["meta"]["checkpoint"])
    if not ckpt_limo.is_absolute(): ckpt_limo = base / ckpt_limo
    limo, _limo_ver = load_limo(base, str(ckpt_limo), args.device)

    print("Loading validator …")
    val = UniMolValidator(model_dir=str(base / args.threedcnn_dir))
    name_map = {"density":"density","heat_of_formation":"HOF_S",
                "detonation_velocity":"DetoD","detonation_pressure":"DetoP"}

    out = {"exp": str(exp), "cfg": args.cfg, "n_pool": args.n_pool,
            "n_keep": args.n_keep, "quantile_z": args.quantile,
            "results": {}}

    md = ["# Generate-and-rerank summary", "",
          f"checkpoint: `{exp.name}/checkpoints/{args.ckpt}`",
          f"CFG scale: {args.cfg}, n_pool: {args.n_pool}, n_keep: {args.n_keep}",
          f"q90 target z: {args.quantile}", "",
          "Per property: pool of N_pool generations → validate → keep top "
          "N_keep by |pred − target|. The first row is the *unranked* baseline "
          "from a random N_keep subset (matches earlier sweeps). Second row is "
          "after rerank.", ""]

    for j, prop in enumerate(pn):
        st = stats[prop]
        target_raw = args.quantile * st["std"] + st["mean"]
        print(f"\n========== {prop}  q90 target={target_raw:+.3f} ==========")
        mask = torch.zeros(args.n_pool, n_props, device=args.device)
        mask[:, j] = 1.0
        vals = torch.zeros(args.n_pool, n_props, device=args.device)
        vals[:, j] = args.quantile
        t0 = time.time()
        z = ddim_sample(denoiser, schedule, vals, mask, n_steps=args.n_steps,
                          guidance_scale=args.cfg, device=args.device)
        with torch.no_grad():
            logits = limo.decode(z)
        toks = logits.argmax(-1).cpu().tolist()
        smiles = [tok.indices_to_smiles(t) for t in toks]
        valid_pairs = [(canon(s), i) for i, s in enumerate(smiles) if s and canon(s)]
        valid_smi = [c for c, _ in valid_pairs]
        n_pre_neutral = len(valid_smi)
        if args.require_neutral:
            valid_smi = [s for s in valid_smi if is_physically_valid(s)]
            print(f"  --require_neutral: {len(valid_smi)} / {n_pre_neutral} kept")
        if not valid_smi:
            print("  no valid"); continue
        pdict = val.predict(valid_smi)
        col = pdict.get(name_map[prop])
        if col is None:
            print("  validator missing column"); continue
        pv = np.asarray(col, dtype=float)
        # Drop NaNs
        ok = ~np.isnan(pv)
        valid_smi_ok = [s for s, k in zip(valid_smi, ok) if k]
        pv_ok = pv[ok]
        if len(pv_ok) == 0:
            print("  validator NaN"); continue

        # baseline (random N_keep subset)
        rng = np.random.default_rng(42)
        idx_random = rng.choice(len(pv_ok), min(args.n_keep, len(pv_ok)), replace=False)
        base_pv = pv_ok[idx_random]
        # ranked — combine property error with optional feasibility
        if args.with_feasibility:
            _ensure_feasibility()
            sa_arr = np.full(len(valid_smi_ok), np.nan)
            sc_arr = np.full(len(valid_smi_ok), np.nan)
            keep_arr = np.ones(len(valid_smi_ok), dtype=bool)
            if _FEAS.get("loaded") and _FEAS.get("sa"):
                for k, smi in enumerate(valid_smi_ok):
                    sa_arr[k] = _FEAS["sa"](smi)
                    sc_arr[k] = _FEAS["sc"](smi)
                    if (not np.isnan(sa_arr[k]) and sa_arr[k] > _FEAS["sa_drop"]) \
                            or (not np.isnan(sc_arr[k]) and sc_arr[k] > _FEAS["sc_drop"]):
                        keep_arr[k] = False
                pv_keep = pv_ok[keep_arr]
                smi_keep = [s for s, k in zip(valid_smi_ok, keep_arr) if k]
                sa_keep = sa_arr[keep_arr]; sc_keep = sc_arr[keep_arr]
                err = np.abs(pv_keep - target_raw)
                pen = np.array([_FEAS["penalty"](sa_keep[k], sc_keep[k],
                                                    args.w_sa, args.w_sc)
                                for k in range(len(pv_keep))])
                composite = err + pen * stats[prop]["std"]   # rescale pen to err units
                order = np.argsort(composite)
                idx_top = order[: min(args.n_keep, len(pv_keep))]
                top_pv  = pv_keep[idx_top]
                top_smi = [smi_keep[i] for i in idx_top]
                print(f"  feasibility: {keep_arr.sum()}/{len(valid_smi_ok)} kept under hard caps")
            else:
                # graceful fallback when feasibility module fails
                order = np.argsort(np.abs(pv_ok - target_raw))
                idx_top = order[: min(args.n_keep, len(pv_ok))]
                top_pv = pv_ok[idx_top]
                top_smi = [valid_smi_ok[i] for i in idx_top]
        else:
            order = np.argsort(np.abs(pv_ok - target_raw))
            idx_top = order[: min(args.n_keep, len(pv_ok))]
            top_pv = pv_ok[idx_top]
            top_smi = [valid_smi_ok[i] for i in idx_top]

        def metrics(arr):
            return {
                "n":          int(len(arr)),
                "mean":       float(arr.mean()),
                "max":        float(arr.max()),
                "min":        float(arr.min()),
                "mae":        float(np.mean(np.abs(arr - target_raw))),
                "rel_mae_pct": float(100*np.mean(np.abs(arr - target_raw))/max(abs(target_raw),1e-6)),
                "within_10_pct": float(100*np.mean(np.abs(arr - target_raw) <= 0.1*abs(target_raw))),
            }

        m_base = metrics(base_pv)
        m_top  = metrics(top_pv)
        out["results"][prop] = {
            "target_raw": target_raw, "n_pool_valid": int(len(pv_ok)),
            "baseline": m_base, "ranked": m_top,
            "top5_smiles": top_smi[:5],
        }
        print(f"  pool valid: {len(pv_ok)}/{args.n_pool}")
        print(f"  baseline (n={m_base['n']}): mean={m_base['mean']:+.3f} "
              f"rel_MAE={m_base['rel_mae_pct']:.1f}% w10={m_base['within_10_pct']:.0f}%")
        print(f"  ranked   (n={m_top['n']}):  mean={m_top['mean']:+.3f} "
              f"rel_MAE={m_top['rel_mae_pct']:.1f}% w10={m_top['within_10_pct']:.0f}%")
        md.append(f"## {prop}  (q90 target = {target_raw:+.3f})")
        md.append("")
        md.append(f"pool valid: {len(pv_ok)} / {args.n_pool}")
        md.append("")
        md.append("| set | n | mean | max | rel_MAE % | within_10 % |")
        md.append("|---|---|---|---|---|---|")
        md.append(f"| baseline | {m_base['n']} | {m_base['mean']:+.3f} | "
                   f"{m_base['max']:+.3f} | {m_base['rel_mae_pct']:.1f} | "
                   f"{m_base['within_10_pct']:.0f} |")
        md.append(f"| **ranked top-{args.n_keep}** | {m_top['n']} | "
                   f"**{m_top['mean']:+.3f}** | {m_top['max']:+.3f} | "
                   f"**{m_top['rel_mae_pct']:.1f}** | "
                   f"**{m_top['within_10_pct']:.0f}** |")
        md.append("")
        md.append(f"**Top 5 SMILES** (sorted by closeness to target):")
        for s in top_smi[:5]:
            md.append(f"- `{s}`")
        md.append("")
        print(f"  done in {time.time()-t0:.0f}s")

    out_path = exp / "rerank_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    md_path = exp / "rerank_results.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    sys.exit(main())
