"""
Generate molecules with composite guidance:
  - Classifier-free guidance on property subset (ρ, HOF, D, P, any subset)
  - Classifier guidance from SA + SC score predictors (feasibility)
  - SDEdit seeding from known compound (optional)

Outputs a JSON with the generated SMILES + per-sample scores (property targets
satisfied? SA / SC score after guidance?).

Usage:
    python scripts/diffusion/sample_guided.py \\
        --diffusion_exp experiments/diffusion_subset_cond_<ts> \\
        --guidance_exp  experiments/guidance_sa_sc_<ts> \\
        --n 100 --sa_weight 1.0 --sc_weight 1.0 --grad_scale 0.2

Example: property-constrained + feasibility-guided
    python scripts/diffusion/sample_guided.py \\
        --targets density=1.85,detonation_velocity=9.0 \\
        --guidance_prop 2.0 --grad_scale 0.3
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).parent))
from model import ConditionalDenoiser, NoiseSchedule, EMA

sys.path.insert(0, str(Path(__file__).parent.parent / "vae"))
from limo_model import (
    LIMOVAE, SELFIESTokenizer, load_vocab, build_limo_vocab, save_vocab,
    LIMO_MAX_LEN, find_limo_repo,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "guidance"))
from model import ScorePredictor


PROP_ORDER = ["density", "heat_of_formation", "detonation_velocity", "detonation_pressure"]


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def load_guidance_predictors(guidance_exp: Path, device: str
                              ) -> tuple[ScorePredictor, ScorePredictor, dict]:
    sa_blob = torch.load(guidance_exp / "checkpoints/sa_best.pt",
                          map_location=device, weights_only=False)
    sc_blob = torch.load(guidance_exp / "checkpoints/sc_best.pt",
                          map_location=device, weights_only=False)
    cfg_g = sa_blob["config"]
    sa = ScorePredictor(in_dim=1024,
                         hidden=cfg_g["model"]["hidden"],
                         dropout=cfg_g["model"]["dropout"]).to(device)
    sc = ScorePredictor(in_dim=1024,
                         hidden=cfg_g["model"]["hidden"],
                         dropout=cfg_g["model"]["dropout"]).to(device)
    sa.load_state_dict(sa_blob["model_state"])
    sc.load_state_dict(sc_blob["model_state"])
    sa.eval(); sc.eval()
    stats = {
        "sa": sa_blob["stats"],
        "sc": sc_blob["stats"],
    }
    return sa, sc, stats


def parse_targets(spec: str, prop_order: list[str],
                   prop_stats: dict
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Parse '--targets density=1.85,detonation_velocity=9.0' into
    (values_norm, mask) arrays.
    """
    values = np.zeros(len(prop_order), dtype=np.float32)
    mask   = np.zeros(len(prop_order), dtype=np.float32)
    if not spec:
        return values, mask
    for part in spec.split(","):
        key, val = part.split("=")
        key = key.strip(); val = float(val)
        j = prop_order.index(key)
        mu = prop_stats[key]["mean"]; sd = max(prop_stats[key]["std"], 1e-6)
        values[j] = (val - mu) / sd
        mask[j] = 1.0
    return values, mask


@torch.no_grad()
def _decode_latents(limo, tok, z, batch=64):
    smiles = []
    for i in range(0, len(z), batch):
        zb = z[i:i+batch]
        if len(zb) == 1:
            zb = torch.cat([zb, zb])
            preds = limo.decode(zb).argmax(dim=2)[:1]
        else:
            preds = limo.decode(zb).argmax(dim=2)
        for p in preds.cpu().numpy():
            smiles.append(tok.indices_to_smiles(p))
    return smiles


def sample_with_guidance(
    denoiser:     ConditionalDenoiser,
    schedule:     NoiseSchedule,
    sa_predictor: ScorePredictor,
    sc_predictor: ScorePredictor,
    values_norm:  torch.Tensor,          # (B, 4)
    mask:         torch.Tensor,          # (B, 4)
    n_steps:      int   = 40,
    prop_guidance_scale: float = 2.0,    # classifier-free guidance (CFG)
    sa_weight:    float = 1.0,
    sc_weight:    float = 1.0,
    grad_scale:   float = 0.2,           # classifier-guidance scale
    device:       str = "cuda",
    seed_z:       torch.Tensor | None = None,  # (B, 1024) for SDEdit; None = pure noise
    sdedit_strength: float = 1.0,
):
    """Composite guidance: CFG on property mask + classifier-guidance from SA+SC."""
    denoiser.eval()
    B = values_norm.shape[0]

    # Initial z
    if seed_z is None:
        z = torch.randn(B, denoiser.latent_dim, device=device)
        t_start = schedule.T - 1
    else:
        t_start = int(sdedit_strength * schedule.T)
        t_start = max(1, min(t_start, schedule.T - 1))
        noise = torch.randn_like(seed_z)
        ab = schedule.sqrt_alpha_bar[t_start]
        om = schedule.sqrt_one_minus_ab[t_start]
        z = ab * seed_z + om * noise

    # DDIM timestep list
    ts = torch.linspace(t_start, 0, n_steps + 1, device=device).long()

    uncond_mask = torch.zeros_like(mask)

    for i in range(n_steps):
        t_now  = ts[i]
        t_next = ts[i + 1]
        tb = torch.full((B,), int(t_now), device=device, dtype=torch.long)

        # CFG (property) — standard torch.no_grad context
        with torch.no_grad():
            eps_cond = denoiser(z, tb, values_norm, mask)
            if prop_guidance_scale != 1.0 and mask.sum() > 0:
                eps_null = denoiser(z, tb, values_norm, uncond_mask)
                eps = eps_null + prop_guidance_scale * (eps_cond - eps_null)
            else:
                eps = eps_cond

        # Classifier guidance (feasibility): need gradients
        if grad_scale > 0 and (sa_weight != 0 or sc_weight != 0):
            ab_now = schedule.alpha_bar[t_now]
            # Reconstruct predicted z_0
            z0_pred = (z - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()

            with torch.enable_grad():
                z0_pred = z0_pred.detach().requires_grad_(True)
                sa_val = sa_predictor(z0_pred)
                sc_val = sc_predictor(z0_pred)
                # We want to MINIMISE SA and SC → add gradient (so subtract guidance)
                loss = sa_weight * sa_val + sc_weight * sc_val
                grad = torch.autograd.grad(loss.sum(), z0_pred)[0]

            # Shift eps in direction that decreases the scores
            # d eps / d z_0 = -1 / sqrt(1 - ab)    →   eps_shift = -grad * sqrt(1 - ab)
            eps = eps + grad_scale * (1 - ab_now).sqrt() * grad

        # DDIM step
        ab_now  = schedule.alpha_bar[t_now]
        ab_next = schedule.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
        z0_pred = (z - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()
        z = ab_next.sqrt() * z0_pred + (1 - ab_next).sqrt() * eps

    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion_exp",   required=True)
    ap.add_argument("--guidance_exp",    required=True)
    ap.add_argument("--n",               type=int, default=100)
    ap.add_argument("--targets",         default="",
                    help='Comma-separated k=v, e.g. "density=1.85,detonation_velocity=9.0"')
    ap.add_argument("--guidance_prop",   type=float, default=2.0)
    ap.add_argument("--sa_weight",       type=float, default=1.0)
    ap.add_argument("--sc_weight",       type=float, default=1.0)
    ap.add_argument("--grad_scale",      type=float, default=0.2)
    ap.add_argument("--n_steps",         type=int, default=40)
    ap.add_argument("--seed_smiles",     default=None,
                    help="SDEdit: start from this SMILES")
    ap.add_argument("--sdedit_strength", type=float, default=0.5)
    ap.add_argument("--out",             default=None)
    ap.add_argument("--base",            default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    diff_exp = Path(args.diffusion_exp)
    if not diff_exp.is_absolute(): diff_exp = base / diff_exp
    guid_exp = Path(args.guidance_exp)
    if not guid_exp.is_absolute(): guid_exp = base / guid_exp

    # Load diffusion
    blob = torch.load(diff_exp / "checkpoints/best.pt", map_location=args.device,
                       weights_only=False)
    diff_cfg = blob["config"]
    denoiser = ConditionalDenoiser(
        latent_dim=1024,
        hidden=diff_cfg["model"]["hidden"],
        n_blocks=diff_cfg["model"]["n_blocks"],
        time_dim=diff_cfg["model"]["time_dim"],
        prop_emb_dim=diff_cfg["model"]["prop_emb_dim"],
        n_props=4,
        dropout=diff_cfg["model"].get("dropout", 0.0),
    ).to(args.device)
    denoiser.load_state_dict(blob["model_state"])
    if blob.get("ema_state"):
        ema = EMA(denoiser, decay=diff_cfg["training"]["ema_decay"])
        ema.load_state_dict(blob["ema_state"])
        ema.apply_to(denoiser)
    denoiser.eval()
    schedule = NoiseSchedule(T=diff_cfg["training"]["T"], device=args.device)

    # Property stats from latents blob
    latents_path = base / diff_cfg["paths"]["latents_pt"]
    latents_meta = torch.load(latents_path, weights_only=False)
    prop_stats = latents_meta["stats"]

    # Parse targets
    values_np, mask_np = parse_targets(args.targets, PROP_ORDER, prop_stats)
    values = torch.tensor(values_np, device=args.device).unsqueeze(0).repeat(args.n, 1)
    mask   = torch.tensor(mask_np,   device=args.device).unsqueeze(0).repeat(args.n, 1)
    print(f"Targets: {args.targets or '(none, unconditional)'}")

    # Load guidance predictors
    sa_pred, sc_pred, gstats = load_guidance_predictors(guid_exp, args.device)
    print(f"Guidance: SA w={args.sa_weight}  SC w={args.sc_weight}  grad_scale={args.grad_scale}")

    # Load LIMO decoder
    limo_dir = find_limo_repo(base)
    vocab_cache = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vocab_cache) if vocab_cache.exists() \
                else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vocab_cache.exists(): save_vocab(alphabet, vocab_cache)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)

    ckpt_limo = latents_meta["meta"]["checkpoint"]
    ckpt_limo_path = Path(ckpt_limo)
    if not ckpt_limo_path.is_absolute(): ckpt_limo_path = base / ckpt_limo_path
    limo_blob = torch.load(ckpt_limo_path, map_location=args.device, weights_only=False)
    limo = LIMOVAE()
    limo.load_state_dict(limo_blob["model_state"])
    limo.to(args.device).eval()

    # Optional SDEdit seed
    seed_z = None
    if args.seed_smiles:
        print(f"SDEdit seed: {args.seed_smiles}  strength={args.sdedit_strength}")
        t = tok.smiles_to_tensor(args.seed_smiles)
        if t is None:
            print("  WARN: cannot tokenize seed; falling back to pure-noise sampling")
        else:
            seed_ids = t[0].unsqueeze(0).to(args.device)
            seed_ids = torch.cat([seed_ids, seed_ids])  # BN requires >=2
            with torch.no_grad():
                _z, mu, _ = limo.encode(seed_ids)
            seed_z = mu[:1].repeat(args.n, 1)

    # Sample
    t0 = time.time()
    with torch.inference_mode(False):
        z_gen = sample_with_guidance(
            denoiser, schedule, sa_pred, sc_pred,
            values, mask,
            n_steps=args.n_steps,
            prop_guidance_scale=args.guidance_prop,
            sa_weight=args.sa_weight, sc_weight=args.sc_weight,
            grad_scale=args.grad_scale,
            device=args.device,
            seed_z=seed_z, sdedit_strength=args.sdedit_strength,
        )
    print(f"Sampled {args.n} in {time.time()-t0:.1f}s")

    # Decode
    smiles = _decode_latents(limo, tok, z_gen)
    canons = [canon(s) for s in smiles]
    valid  = [c for c in canons if c is not None]

    # Re-predict SA/SC on generated latents for reporting
    with torch.no_grad():
        sa_pred_val = sa_pred(z_gen).cpu().numpy()
        sc_pred_val = sc_pred(z_gen).cpu().numpy()
    sa_raw = sa_pred_val * gstats["sa"]["std"] + gstats["sa"]["mean"]
    sc_raw = sc_pred_val * gstats["sc"]["std"] + gstats["sc"]["mean"]

    # Summary
    print(f"\nGenerated: {args.n}  valid: {len(valid)}  unique: {len(set(valid))}")
    if valid:
        print(f"SA predicted (on latent): mean={np.nanmean(sa_raw):.3f}  median={np.nanmedian(sa_raw):.3f}")
        print(f"SC predicted (on latent): mean={np.nanmean(sc_raw):.3f}  median={np.nanmedian(sc_raw):.3f}")

    # Dump JSON
    if args.out is None:
        args.out = str(diff_exp / "guided_samples.json")
    out_rows = []
    for i, sm in enumerate(smiles):
        out_rows.append({
            "smiles": sm,
            "canonical": canons[i],
            "sa_pred": float(sa_raw[i]),
            "sc_pred": float(sc_raw[i]),
        })
    with open(args.out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n": args.n,
            "targets": args.targets,
            "guidance_prop": args.guidance_prop,
            "sa_weight": args.sa_weight, "sc_weight": args.sc_weight,
            "grad_scale": args.grad_scale,
            "seed_smiles": args.seed_smiles,
            "sdedit_strength": args.sdedit_strength,
            "samples": out_rows,
        }, f, indent=2)
    print(f"Wrote → {args.out}")


if __name__ == "__main__":
    main()
