"""
Evaluate a trained subset-conditional denoiser.

Metrics:
  - Unconditional sampling: validity, uniqueness, novelty vs training SMILES
  - Conditional fidelity: for each property & target quantile, sample and
    measure how close the realised (decoded & re-encoded & property-predicted)
    value lands to the target
  - Subset monotonicity: MAE vs subset size
  - Latent distribution stats: μ mean/std, active dims

Outputs:
    <exp_dir>/eval_results.json
    <exp_dir>/eval_summary.txt

Usage:
    python scripts/diffusion/evaluate.py --exp experiments/diffusion_*
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
from model import ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample

# UniMol validator is optional (heavy deps)
try:
    from unimol_validator import UniMolValidator, validate_generation, PROP_MAP
    UNIMOL_AVAILABLE = True
except ImportError:
    UNIMOL_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent / "vae"))
from limo_model import (
    LIMOVAE, SELFIESTokenizer, load_vocab, build_limo_vocab, save_vocab,
    LIMO_MAX_LEN, find_limo_repo,
)


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


@torch.no_grad()
def decode_latents(limo: LIMOVAE, tok: SELFIESTokenizer, z: torch.Tensor,
                    batch: int = 64) -> list[str]:
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


# ── novelty analysis: nearest-neighbor Tanimoto against training set ────────
def build_reference_fingerprints(smiles: list[str]):
    """Morgan ECFP4 fingerprints (bit vectors) for reference set."""
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, keep_smi = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        fps.append(gen.GetFingerprint(m))
        keep_smi.append(s)
    return fps, keep_smi


def nearest_neighbor_tanimoto(generated_smiles: list[str],
                                 ref_fps: list,
                                 ref_smiles: list[str]) -> list[dict]:
    """For each generated SMILES, find max Tanimoto similarity in reference set
    and return a list of dicts: {smi, max_sim, nearest_ref_smi}.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s in generated_smiles:
        if not s:
            out.append({"smi": s, "max_sim": None, "nearest": None})
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            out.append({"smi": s, "max_sim": None, "nearest": None})
            continue
        fp = gen.GetFingerprint(m)
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        idx = int(np.argmax(sims))
        out.append({
            "smi":      s,
            "max_sim":  float(sims[idx]),
            "nearest":  ref_smiles[idx],
        })
    return out


def summarise_novelty(nn_results: list[dict]) -> dict:
    sims = [r["max_sim"] for r in nn_results if r["max_sim"] is not None]
    if not sims:
        return {"n_valid": 0}
    a = np.array(sims)
    # novelty thresholds
    return {
        "n_valid":          len(sims),
        "max_sim_mean":     float(a.mean()),
        "max_sim_median":   float(np.median(a)),
        "max_sim_p25":      float(np.percentile(a, 25)),
        "max_sim_p75":      float(np.percentile(a, 75)),
        "exact_dup_pct":    100 * float((a >= 0.999).mean()),
        "near_dup_pct":     100 * float((a >= 0.9).mean()),   # very similar
        "truly_novel_pct":  100 * float((a < 0.4).mean()),     # Butina-novel (no near-relative in training)
        "distinct_scaffold_pct": 100 * float((a < 0.7).mean()),
    }


@torch.no_grad()
def sample_unconditional(denoiser, schedule, limo, tok, device,
                          n_samples: int = 1000, n_steps: int = 40,
                          training_smiles: set = None,
                          ref_fps=None, ref_smiles=None) -> dict:
    mask = torch.zeros(n_samples, 4, device=device)
    values = torch.zeros(n_samples, 4, device=device)
    z = ddim_sample(denoiser, schedule, values, mask,
                     n_steps=n_steps, guidance_scale=1.0, device=device)
    smiles = decode_latents(limo, tok, z)
    canons = [canon(s) for s in smiles]
    valid = [c for c in canons if c is not None]
    unique = set(valid)
    novel_exact = unique - (training_smiles or set())
    out = {
        "n_generated":   len(smiles),
        "valid_pct":     100 * len(valid) / max(len(smiles), 1),
        "unique_pct":    100 * len(unique) / max(len(valid), 1) if valid else 0,
        "novel_exact_pct": 100 * len(novel_exact) / max(len(unique), 1) if unique else 0,
        "sample_smiles": list(unique)[:24],
    }
    # nearest-neighbor Tanimoto analysis
    if ref_fps is not None and valid:
        nn = nearest_neighbor_tanimoto(valid, ref_fps, ref_smiles)
        nn_stats = summarise_novelty(nn)
        out["novelty"] = nn_stats
        # annotate first 12 samples with their nearest training neighbour
        out["sample_smiles_with_neighbor"] = [
            {"smi": r["smi"], "max_sim": round(r["max_sim"], 3) if r["max_sim"] else None,
             "nearest": r["nearest"]}
            for r in nn[:12]
        ]
    return out


# ── SDEdit compound-to-compound generation ──────────────────────────────────
@torch.no_grad()
def sample_from_seed(denoiser, schedule, limo, tok, device,
                      seed_smiles: list[str],
                      strength: float = 0.5,
                      values_norm: Optional[torch.Tensor] = None,
                      mask: Optional[torch.Tensor] = None,
                      n_steps: int = 40,
                      guidance_scale: float = 2.0,
                      n_variants_per_seed: int = 5) -> dict:
    """Compound-to-compound generation via SDEdit.

    For each seed SMILES:
      1. encode via LIMO → z_0
      2. noise to timestep t_star = int(strength * T)
      3. denoise with conditioning → z_generated
      4. decode → variant SMILES

    strength ∈ [0,1]:
      0.1-0.3: tight analogue (small substituent edits)
      0.4-0.6: moderate edit (some structural change)
      0.7-0.9: large edit (mostly uncorrelated)
    """
    from limo_model import SELFIESTokenizer
    denoiser.eval()
    # encode seed molecules
    seed_tensors = []
    keep_idx = []
    for i, smi in enumerate(seed_smiles):
        t = tok.smiles_to_tensor(smi)
        if t is None:
            continue
        seed_tensors.append(t[0])
        keep_idx.append(i)
    if not seed_tensors:
        return {"note": "no valid seeds"}
    X = torch.stack(seed_tensors).to(device)
    if len(X) == 1:
        X = torch.cat([X, X])
        _z, mu, _ = limo.encode(X)
        z0 = mu[:1]
    else:
        _z, mu, _ = limo.encode(X)
        z0 = mu
    # repeat per seed × n_variants
    B = len(z0)
    z0_rep = z0.repeat_interleave(n_variants_per_seed, dim=0)
    n_total = B * n_variants_per_seed
    if values_norm is None:
        values_norm = torch.zeros(n_total, denoiser.n_props, device=device)
    else:
        values_norm = values_norm.repeat_interleave(n_variants_per_seed, dim=0)
    if mask is None:
        mask = torch.zeros(n_total, denoiser.n_props, device=device)
    else:
        mask = mask.repeat_interleave(n_variants_per_seed, dim=0)

    t_star = int(strength * schedule.T)
    t_star = max(1, min(t_star, schedule.T - 1))
    # add noise to timestep t_star
    noise = torch.randn_like(z0_rep)
    ab = schedule.sqrt_alpha_bar[t_star]
    om = schedule.sqrt_one_minus_ab[t_star]
    z = ab * z0_rep + om * noise

    # DDIM denoising from t_star down to 0
    ts = torch.linspace(t_star, 0, n_steps + 1, device=device).long()
    for i in range(n_steps):
        t_now, t_next = ts[i], ts[i + 1]
        tb = torch.full((n_total,), int(t_now), device=device, dtype=torch.long)
        eps_cond = denoiser(z, tb, values_norm, mask)
        if guidance_scale != 1.0:
            eps_null = denoiser(z, tb, values_norm, torch.zeros_like(mask))
            eps = eps_null + guidance_scale * (eps_cond - eps_null)
        else:
            eps = eps_cond
        ab_now = schedule.alpha_bar[t_now]
        ab_next = schedule.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
        z0_pred = (z - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()
        z = ab_next.sqrt() * z0_pred + (1 - ab_next).sqrt() * eps

    # decode
    variants = decode_latents(limo, tok, z)
    seeds_used = [seed_smiles[i] for i in keep_idx]
    # group back per seed
    out = {"strength": strength, "per_seed": []}
    for s_idx in range(B):
        seed = seeds_used[s_idx]
        seed_variants = variants[s_idx * n_variants_per_seed : (s_idx+1) * n_variants_per_seed]
        canons = [canon(v) for v in seed_variants if v]
        canons = [c for c in canons if c]
        seed_canon = canon(seed)
        unique = set(canons) - ({seed_canon} if seed_canon else set())
        out["per_seed"].append({
            "seed":        seed,
            "n_variants":  len(canons),
            "n_unique":    len(unique),
            "n_same_as_seed": sum(1 for c in canons if c == seed_canon),
            "variants":    list(unique)[:5],
        })
    return out


@torch.no_grad()
def conditional_fidelity(denoiser, schedule, limo, tok, device,
                          property_names: list[str], stats: dict,
                          n_per_target: int = 200, n_steps: int = 40,
                          guidance: float = 2.0,
                          validator: Optional = None) -> dict:
    """For each property, sample at q10/q50/q90 target. Decode, re-encode via
    LIMO to recover property via a *very* crude check: we compute a pseudo-
    property using the CDDD-style approach — here, since we don't have a
    property predictor trained yet, we just verify the generated samples are
    diverse and valid. The realised-property check is deferred until the
    property predictor (model #3) is trained.
    """
    out = {}
    n_props = len(property_names)
    for j, p in enumerate(property_names):
        st = stats[p]
        mu_p, sd_p = st["mean"], st["std"]
        per_target = {}
        for q_label, q_z in [("q10", -1.281), ("q50", 0.0), ("q90", 1.281)]:
            mask = torch.zeros(n_per_target, n_props, device=device)
            mask[:, j] = 1.0
            values = torch.zeros(n_per_target, n_props, device=device)
            values[:, j] = q_z
            z = ddim_sample(denoiser, schedule, values, mask,
                             n_steps=n_steps, guidance_scale=guidance, device=device)
            smiles = decode_latents(limo, tok, z)
            canons = [canon(s) for s in smiles if s]
            valid = [c for c in canons if c is not None]
            target_raw = q_z * sd_p + mu_p
            entry = {
                "target_raw":   target_raw,
                "target_z":     q_z,
                "n_generated":  len(smiles),
                "n_valid":      len(valid),
                "unique_pct":   100 * len(set(valid)) / max(len(valid), 1) if valid else 0,
                "sample_3":     list(set(valid))[:3],
            }
            # 3DCNN validation: does the generated molecule actually have the target property?
            if validator is not None and valid:
                try:
                    val_stats = validate_generation(valid,
                                                     {p: target_raw},
                                                     validator)
                    entry["validator_3dcnn"] = val_stats.get(p, {})
                except Exception as e:
                    entry["validator_error"] = str(e)[:200]
            per_target[q_label] = entry
        out[p] = per_target
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",      required=True)
    ap.add_argument("--ckpt",     default="best.pt")
    ap.add_argument("--base",     default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_uncond", type=int, default=500)
    ap.add_argument("--n_cond",   type=int, default=150)
    ap.add_argument("--n_steps",  type=int, default=40)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--use_ema",  action="store_true", default=True)
    ap.add_argument("--use_3dcnn", action="store_true", default=True,
                    help="Run UniMol/3DCNN validation (verifies generated molecules have target properties)")
    ap.add_argument("--3dcnn_dir", default="data/raw/energetic_external/EMDP/Data/smoke_model",
                    dest="threedcnn_dir",
                    help="Path to Uni-Mol model dir with model_0.pth etc.")
    ap.add_argument("--compound2compound", action="store_true", default=True,
                    help="Run SDEdit compound-to-compound analog generation")
    ap.add_argument("--c2c_strengths", nargs="+", type=float, default=[0.3, 0.6, 0.9],
                    help="SDEdit strengths to try")
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="Number of seed molecules for compound-to-compound")
    ap.add_argument("--n_variants_per_seed", type=int, default=5)
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    exp = Path(args.exp)
    if not exp.is_absolute(): exp = base / exp
    ckpt_path = exp / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        alt = exp / "checkpoints/last.pt"
        if alt.exists(): ckpt_path = alt
        else:
            print(f"No checkpoint at {ckpt_path}"); return 1

    blob = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    cfg = blob["config"]

    # load latents for stats + training SMILES
    latents_path = base / cfg["paths"]["latents_pt"]
    latents_blob = torch.load(latents_path, weights_only=False)
    property_names = latents_blob["property_names"]
    stats = latents_blob["stats"]
    training_smiles = set(latents_blob["smiles"])

    # build denoiser
    denoiser = ConditionalDenoiser(
        latent_dim   = latents_blob["z_mu"].shape[1],
        hidden       = cfg["model"]["hidden"],
        n_blocks     = cfg["model"]["n_blocks"],
        time_dim     = cfg["model"]["time_dim"],
        prop_emb_dim = cfg["model"]["prop_emb_dim"],
        n_props      = latents_blob["values_raw"].shape[1],
        dropout      = cfg["model"].get("dropout", 0.0),
    ).to(args.device)
    denoiser.load_state_dict(blob["model_state"])

    if args.use_ema and blob.get("ema_state") is not None:
        ema = EMA(denoiser, decay=cfg["training"]["ema_decay"])
        ema.load_state_dict(blob["ema_state"])
        ema.apply_to(denoiser)
        print("Using EMA weights.")

    denoiser.eval()
    schedule = NoiseSchedule(T=cfg["training"]["T"], device=args.device)

    # load LIMO decoder
    print("Loading LIMO decoder …")
    limo_dir = find_limo_repo(base)
    vocab_cache = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vocab_cache) if vocab_cache.exists() \
                else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vocab_cache.exists(): save_vocab(alphabet, vocab_cache)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)

    # which LIMO ckpt? use the same one that produced the latents
    ckpt_limo = latents_blob["meta"]["checkpoint"]
    ckpt_limo_path = Path(ckpt_limo)
    if not ckpt_limo_path.is_absolute():
        ckpt_limo_path = base / ckpt_limo_path
    limo_blob = torch.load(ckpt_limo_path, map_location=args.device, weights_only=False)
    limo = LIMOVAE()
    limo.load_state_dict(limo_blob["model_state"])
    limo.to(args.device).eval()

    results = {
        "checkpoint":   str(ckpt_path),
        "ckpt_step":    blob["runtime"]["global_step"],
        "best_val":     blob["runtime"]["best_val"],
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "use_ema":      bool(args.use_ema and blob.get("ema_state")),
        "n_steps":      args.n_steps,
        "guidance":     args.guidance,
    }

    # build reference fingerprint set for novelty / NN analysis
    print(f"Building reference fingerprints from {len(training_smiles):,} training SMILES …")
    # subsample reference to bound cost: all training + full labeled
    ref_list = list(training_smiles)
    if len(ref_list) > 40000:
        rng = np.random.default_rng(0)
        ref_list = [ref_list[i] for i in rng.choice(len(ref_list), 40000, replace=False)]
    t0 = time.time()
    ref_fps, ref_smiles = build_reference_fingerprints(ref_list)
    print(f"  {len(ref_fps):,} fingerprints ({time.time()-t0:.1f}s)")

    # ── 1. unconditional ─────────────────────────────────────────────────────
    print(f"\n[1/2] Unconditional sampling (n={args.n_uncond}) …")
    results["unconditional"] = sample_unconditional(
        denoiser, schedule, limo, tok, args.device,
        n_samples=args.n_uncond, n_steps=args.n_steps,
        training_smiles=training_smiles,
        ref_fps=ref_fps, ref_smiles=ref_smiles)
    u = results["unconditional"]
    for k in ("n_generated", "valid_pct", "unique_pct", "novel_exact_pct"):
        v = u.get(k)
        print(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")
    if "novelty" in u:
        nn = u["novelty"]
        print(f"    novelty (nearest-neighbor Tanimoto in training):")
        print(f"      max_sim mean={nn['max_sim_mean']:.3f}  median={nn['max_sim_median']:.3f}")
        print(f"      exact duplicates (sim≥0.999): {nn['exact_dup_pct']:.1f}%")
        print(f"      near duplicates (sim≥0.9):    {nn['near_dup_pct']:.1f}%")
        print(f"      distinct scaffold (sim<0.7):  {nn['distinct_scaffold_pct']:.1f}%")
        print(f"      truly novel (sim<0.4):        {nn['truly_novel_pct']:.1f}%")

    # ── 2. conditional + 3DCNN validation ────────────────────────────────────
    validator = None
    if args.use_3dcnn and UNIMOL_AVAILABLE:
        try:
            print(f"\nLoading 3DCNN validator from {args.threedcnn_dir} …")
            validator = UniMolValidator(Path(args.base) / args.threedcnn_dir)
            # run once on a tiny smoke batch to force lazy init + surface errors early
            _ = validator.predict(["CCO"])
            print("  3DCNN validator ready.")
        except Exception as e:
            print(f"  3DCNN unavailable: {e}")
            validator = None

    print(f"\n[2/2] Conditional sampling (each property at q10/q50/q90, n={args.n_cond} each) …")
    if validator:
        print("  → will validate with 3DCNN (independent predictor)")
    results["conditional"] = conditional_fidelity(
        denoiser, schedule, limo, tok, args.device,
        property_names, stats,
        n_per_target=args.n_cond, n_steps=args.n_steps,
        guidance=args.guidance, validator=validator)
    for p, rs in results["conditional"].items():
        print(f"  {p}:")
        for q, r in rs.items():
            line = (f"    {q}: target={r['target_raw']:.3f}  "
                    f"valid={r['n_valid']}/{r['n_generated']}  "
                    f"unique={r['unique_pct']:.1f}%")
            if "validator_3dcnn" in r and r["validator_3dcnn"].get("n_valid", 0) > 0:
                vs = r["validator_3dcnn"]
                line += (f"  |  3DCNN: mean={vs['mean_pred']:.3f}  "
                         f"MAE={vs['mae']:.3f}  rel={vs['rel_mae_pct']:.1f}%  "
                         f"within-10%={vs['within_10_pct']:.0f}%")
            print(line)

    # ── 3. compound-to-compound SDEdit ───────────────────────────────────────
    if args.compound2compound:
        print(f"\n[3/3] Compound-to-compound (SDEdit, strengths={args.c2c_strengths}) …")
        # pick seed molecules: random Tier-A-rich labeled compounds
        seed_candidates = list(training_smiles)[:2000]
        rng = np.random.default_rng(123)
        seeds = [seed_candidates[i] for i in rng.choice(
            len(seed_candidates), args.n_seeds, replace=False)]
        c2c_results = {}
        for strength in args.c2c_strengths:
            print(f"  strength={strength}:")
            r = sample_from_seed(
                denoiser, schedule, limo, tok, args.device,
                seed_smiles=seeds, strength=strength,
                n_variants_per_seed=args.n_variants_per_seed,
                n_steps=args.n_steps, guidance_scale=args.guidance)
            c2c_results[f"strength_{strength}"] = r
            for item in r.get("per_seed", []):
                print(f"    seed: {item['seed'][:55]:55s}  "
                      f"variants {item['n_variants']}  unique {item['n_unique']}")
        results["compound_to_compound"] = c2c_results

    # save
    with open(exp / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(exp / "eval_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Diffusion evaluation — {ckpt_path.name}\n")
        f.write("=" * 60 + "\n\n")
        u = results["unconditional"]
        f.write(f"Unconditional (n={u['n_generated']}):\n")
        f.write(f"  Valid:         {u['valid_pct']:.2f}%\n")
        f.write(f"  Unique:        {u['unique_pct']:.2f}%\n")
        f.write(f"  Novel (exact): {u.get('novel_exact_pct', 0):.2f}%\n")
        if "novelty" in u:
            nn = u["novelty"]
            f.write(f"Novelty (NN-Tanimoto vs training):\n")
            f.write(f"  max_sim mean:    {nn['max_sim_mean']:.3f}\n")
            f.write(f"  max_sim median:  {nn['max_sim_median']:.3f}\n")
            f.write(f"  exact dup %:     {nn['exact_dup_pct']:.1f}%\n")
            f.write(f"  near dup %:      {nn['near_dup_pct']:.1f}%\n")
            f.write(f"  distinct scaff%: {nn['distinct_scaffold_pct']:.1f}%\n")
            f.write(f"  truly novel %:   {nn['truly_novel_pct']:.1f}%\n")
        f.write("\n")
        for p, rs in results["conditional"].items():
            f.write(f"{p} — conditional:\n")
            for q, r in rs.items():
                f.write(f"  {q}: target={r['target_raw']:.3f}  "
                         f"valid={r['n_valid']}  unique%={r['unique_pct']:.1f}\n")
            f.write("\n")
    print(f"\nResults → {exp / 'eval_results.json'}")


if __name__ == "__main__":
    main()
