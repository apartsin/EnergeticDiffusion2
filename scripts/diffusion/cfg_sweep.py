"""Quick CFG-scale sweep on an existing checkpoint.

Runs only conditional_fidelity (skips uncond + SDEdit) for several guidance
scales, writes a single comparison JSON + a markdown table.

Usage:
    python scripts/diffusion/cfg_sweep.py \
        --exp experiments/diffusion_subset_cond_expanded_20260425T095335Z \
        --scales 2 3 5 7 \
        --n_per_target 50
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vae"))

from model import ConditionalDenoiser, NoiseSchedule, EMA
from limo_model import (LIMOVAE, SELFIESTokenizer, load_vocab,
                        build_limo_vocab, save_vocab, LIMO_MAX_LEN, find_limo_repo)
from limo_factory import load_limo, LIMOInferenceWrapper
from evaluate import conditional_fidelity
from unimol_validator import UniMolValidator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--scales", type=float, nargs="+", default=[2, 3, 5, 7])
    ap.add_argument("--n_per_target", type=int, default=50)
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threedcnn_dir",
                    default="data/raw/energetic_external/EMDP/Data/smoke_model")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    exp = Path(args.exp)
    if not exp.is_absolute(): exp = base / exp
    ckpt_path = exp / "checkpoints" / args.ckpt
    blob = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    cfg = blob["config"]
    latents_blob = torch.load(base / cfg["paths"]["latents_pt"], weights_only=False)
    property_names = latents_blob["property_names"]
    stats = latents_blob["stats"]

    denoiser = ConditionalDenoiser(
        latent_dim=latents_blob["z_mu"].shape[1],
        hidden=cfg["model"]["hidden"],
        n_blocks=cfg["model"]["n_blocks"],
        time_dim=cfg["model"]["time_dim"],
        prop_emb_dim=cfg["model"]["prop_emb_dim"],
        n_props=latents_blob["values_raw"].shape[1],
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(args.device)
    denoiser.load_state_dict(blob["model_state"])
    if blob.get("ema_state") is not None:
        ema = EMA(denoiser, decay=cfg["training"]["ema_decay"])
        ema.load_state_dict(blob["ema_state"])
        ema.apply_to(denoiser)
        print("Using EMA weights.")
    denoiser.eval()
    schedule = NoiseSchedule(T=cfg["training"]["T"], device=args.device)

    print("Loading LIMO …")
    limo_dir = find_limo_repo(base)
    vocab_cache = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vocab_cache) if vocab_cache.exists() \
               else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vocab_cache.exists(): save_vocab(alphabet, vocab_cache)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    ckpt_limo = Path(latents_blob["meta"]["checkpoint"])
    if not ckpt_limo.is_absolute(): ckpt_limo = base / ckpt_limo
    limo_blob = torch.load(ckpt_limo, map_location=args.device, weights_only=False)
    limo, _limo_ver = load_limo(base, str(ckpt_limo), args.device)

    print("Loading 3DCNN validator …")
    validator = UniMolValidator(model_dir=str(base / args.threedcnn_dir))

    out = {"sweep": {}, "exp": str(exp), "ckpt": str(ckpt_path),
           "n_per_target": args.n_per_target, "n_steps": args.n_steps,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for g in args.scales:
        print(f"\n========== guidance = {g} ==========")
        t0 = time.time()
        cond = conditional_fidelity(denoiser, schedule, limo, tok, args.device,
                                     property_names, stats,
                                     n_per_target=args.n_per_target,
                                     n_steps=args.n_steps,
                                     guidance=g, validator=validator)
        elapsed = time.time() - t0
        out["sweep"][str(g)] = {"results": cond, "elapsed_s": elapsed}
        print(f"  done in {elapsed/60:.1f} min")

    out_path = exp / "cfg_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")

    # markdown table
    md = ["# CFG sweep summary", "",
          f"checkpoint: `{ckpt_path}`",
          f"n_per_target: {args.n_per_target}", "",
          "| Property | q | target | " +
          " | ".join(f"g={g} pred" for g in args.scales) + " | " +
          " | ".join(f"g={g} rel%" for g in args.scales) + " |",
          "|---|---|---|" + "|".join(["---"] * (2*len(args.scales))) + "|"]
    for p in property_names:
        for q in ["q10", "q50", "q90"]:
            target = out["sweep"][str(args.scales[0])]["results"][p][q]["target_raw"]
            preds = []
            rels = []
            for g in args.scales:
                v = out["sweep"][str(g)]["results"][p][q].get("validator_3dcnn", {})
                preds.append(f"{v.get('mean_pred', float('nan')):+.2f}")
                rels.append(f"{v.get('rel_mae_pct', float('nan')):.0f}")
            md.append(f"| {p} | {q} | {target:+.2f} | " +
                      " | ".join(preds) + " | " + " | ".join(rels) + " |")
    md_path = exp / "cfg_sweep.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved → {md_path}")


if __name__ == "__main__":
    sys.exit(main())
