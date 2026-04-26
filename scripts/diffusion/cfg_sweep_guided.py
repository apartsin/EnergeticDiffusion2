"""CFG-sweep with classifier guidance from latent property heads.

Tests whether the broken FiLM signal (D10) can be sidestepped by directly
optimising z toward higher predicted property values during DDIM sampling.

Sweeps over (CFG scale × guidance λ) for q90 of each property. Validates
generated SMILES with the 3DCNN smoke ensemble.

Usage:
    python scripts/diffusion/cfg_sweep_guided.py \
        --exp experiments/diffusion_subset_cond_expanded_v4b_2026... \
        --cfg_scales 5 7 \
        --lambdas 0 1 3 10 \
        --n_per_target 50
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

from model import ConditionalDenoiser, NoiseSchedule, EMA
from limo_model import (LIMOVAE, SELFIESTokenizer, load_vocab,
                        build_limo_vocab, save_vocab, LIMO_MAX_LEN, find_limo_repo)
from limo_factory import load_limo, LIMOInferenceWrapper
from guided_sampler import ddim_sample_guided
from unimol_validator import UniMolValidator


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--heads", default="data/training/guidance/property_heads.pt")
    ap.add_argument("--cfg_scales", type=float, nargs="+", default=[7.0])
    ap.add_argument("--lambdas",    type=float, nargs="+", default=[0.0, 1.0, 3.0, 10.0])
    ap.add_argument("--n_per_target", type=int, default=50)
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--guidance_warmup", type=int, default=0,
                    help="Skip classifier guidance for the first N DDIM steps "
                         "(useful since the property heads were trained on "
                         "clean z_mu, not noisy z_t)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quantile", type=float, default=1.281,
                    help="z target (1.281 = q90)")
    ap.add_argument("--threedcnn_dir",
                    default="data/raw/energetic_external/EMDP/Data/smoke_model")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    exp = Path(args.exp)
    if not exp.is_absolute(): exp = base / exp
    ckpt_path = exp / "checkpoints" / args.ckpt
    cb = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    cfg = cb["config"]
    latents_blob = torch.load(base / cfg["paths"]["latents_pt"], weights_only=False)
    property_names = latents_blob["property_names"]
    stats = latents_blob["stats"]

    denoiser = ConditionalDenoiser(
        latent_dim=latents_blob["z_mu"].shape[1],
        hidden=cfg["model"]["hidden"], n_blocks=cfg["model"]["n_blocks"],
        time_dim=cfg["model"]["time_dim"], prop_emb_dim=cfg["model"]["prop_emb_dim"],
        n_props=latents_blob["values_raw"].shape[1],
        dropout=cfg["model"].get("dropout", 0.0)).to(args.device)
    denoiser.load_state_dict(cb["model_state"])
    if cb.get("ema_state") is not None:
        ema = EMA(denoiser, decay=cfg["training"]["ema_decay"])
        ema.load_state_dict(cb["ema_state"]); ema.apply_to(denoiser)
        print("Using EMA weights")
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

    heads_path = base / args.heads

    out_blob = {"sweep": {}, "exp": str(exp),
                 "n_per_target": args.n_per_target,
                 "n_steps": args.n_steps,
                 "quantile_z": args.quantile,
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    n_props = latents_blob["values_raw"].shape[1]

    for prop in property_names:
        j = property_names.index(prop)
        st = stats[prop]
        target_raw = args.quantile * st["std"] + st["mean"]
        out_blob["sweep"][prop] = {"target_raw": target_raw, "results": {}}
        print(f"\n========== {prop}  q90 target={target_raw:+.3f} ==========")
        for g in args.cfg_scales:
            for lam in args.lambdas:
                key = f"g={g}_l={lam}"
                mask = torch.zeros(args.n_per_target, n_props, device=args.device)
                mask[:, j] = 1.0
                vals = torch.zeros(args.n_per_target, n_props, device=args.device)
                vals[:, j] = args.quantile
                t0 = time.time()
                z = ddim_sample_guided(
                    denoiser, schedule, vals, mask,
                    n_steps=args.n_steps, guidance_scale=g,
                    device=args.device,
                    property_targets={prop: target_raw} if lam > 0 else None,
                    property_heads_path=str(heads_path) if lam > 0 else None,
                    guidance_lambda=lam,
                    guidance_warmup_steps=args.guidance_warmup,
                )
                with torch.no_grad():
                    logits = limo.decode(z)
                toks = logits.argmax(-1).cpu().tolist()
                smiles = [tok.indices_to_smiles(t) for t in toks]
                valid = [canon(s) for s in smiles if s and canon(s)]
                if not valid:
                    print(f"  {key}: no valid"); continue
                pdict = val.predict(valid)
                col = pdict.get(name_map[prop])
                if col is None:
                    print(f"  {key}: no validator output"); continue
                pv = np.asarray(col); pv = pv[~np.isnan(pv)]
                if len(pv) == 0:
                    print(f"  {key}: validator NaN"); continue
                mae = float(np.mean(np.abs(pv - target_raw)))
                rel = 100 * mae / max(abs(target_raw), 1e-6)
                w10 = float(np.mean(np.abs(pv - target_raw) <= 0.1*abs(target_raw)) * 100)
                out_blob["sweep"][prop]["results"][key] = {
                    "g": g, "lambda": lam,
                    "n_valid": len(pv), "n_unique": len(set(valid)),
                    "mean_pred": float(pv.mean()),
                    "max_pred":  float(pv.max()),
                    "mae": mae, "rel_mae_pct": rel, "within_10_pct": w10,
                    "elapsed_s": time.time() - t0,
                }
                print(f"  {key}: pred mean={pv.mean():+.2f} max={pv.max():+.2f} "
                      f"rel_MAE={rel:.1f}% w10={w10:.0f}% (n={len(pv)} {(time.time()-t0):.0f}s)")

    out_path = exp / "cfg_sweep_guided.json"
    with open(out_path, "w") as f:
        json.dump(out_blob, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # markdown summary: per-property best λ
    md = ["# Classifier-guided sweep summary", "",
          f"checkpoint: `{ckpt_path}`",
          f"n_per_target: {args.n_per_target}, q90 target z={args.quantile}",
          f"CFG scales: {args.cfg_scales}",
          f"λ values:   {args.lambdas}", ""]
    for prop, blob_p in out_blob["sweep"].items():
        md.append(f"## {prop}  (q90 target = {blob_p['target_raw']:+.3f})")
        md.append("")
        md.append("| g | λ | mean_pred | max_pred | rel_MAE % | within_10 % | n_unique |")
        md.append("|---|---|---|---|---|---|---|")
        for key, r in blob_p["results"].items():
            md.append(f"| {r['g']} | {r['lambda']} | {r['mean_pred']:+.2f} | "
                       f"{r['max_pred']:+.2f} | {r['rel_mae_pct']:.0f} | "
                       f"{r['within_10_pct']:.0f} | {r['n_unique']} |")
        md.append("")
    md_path = exp / "cfg_sweep_guided.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    sys.exit(main())
