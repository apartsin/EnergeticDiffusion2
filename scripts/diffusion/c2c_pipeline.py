"""Compound-to-compound (c2c) generation: SDEdit on latent diffusion.

For each seed SMILES, partially noise its latent, then property-conditional
denoise back to produce structurally-related variants. Validates with the
3DCNN ensemble and computes Tanimoto similarity to seed.

Usage:
    python scripts/diffusion/c2c_pipeline.py \
        --exp <denoiser exp dir> \
        --seeds_csv data/c2c/seeds.csv \
        --strengths 0.3 0.5 0.7 \
        --n_variants 100 \
        --target_density 1.85 --target_d 8.5 --target_p 28.0 --target_hof 200 \
        --require_neutral
"""
from __future__ import annotations
import argparse, json, sys, time, csv
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "guidance"))
sys.path.insert(0, str(HERE.parent / "vae"))
sys.path.insert(0, str(HERE))

from model import ConditionalDenoiser, NoiseSchedule, EMA
from limo_model import (LIMOVAE, SELFIESTokenizer, load_vocab,
                        build_limo_vocab, save_vocab, LIMO_MAX_LEN, find_limo_repo)
from unimol_validator import UniMolValidator
from feasibility_utils import real_sa, real_sc


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def is_neutral(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return False
    if Chem.GetFormalCharge(m) != 0: return False
    return not any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms())


def morgan_fp(smi, radius=2, nbits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nbits)


def tanimoto(a_smi, b_smi):
    fa = morgan_fp(a_smi); fb = morgan_fp(b_smi)
    if fa is None or fb is None: return 0.0
    return DataStructs.TanimotoSimilarity(fa, fb)


@torch.no_grad()
def c2c_sample(denoiser, schedule, limo, tok,
                z_seed: torch.Tensor, n_variants: int,
                strength: float,
                target_z: torch.Tensor, mask: torch.Tensor,
                guidance_scale: float = 7.0,
                device: str = "cuda",
                anchor_alpha: float = 0.0,
                anchor_decay: bool = True):
    """SDEdit-style: noise z_seed to t_edit, then denoise.

    anchor_alpha:  per-step blend toward z_seed (0 = no anchor; 0.3 typical).
                   z0_pred ← (1−α_t) z0_pred + α_t z_seed
    anchor_decay:  if True, α_t decays linearly from anchor_alpha at step 0
                   to 0 at the final step (so the very last steps don't snap).

    Returns (smiles_list, final_z).
    """
    T = schedule.T
    t_edit = max(1, min(T - 2, int(strength * (T - 2))))
    # Replicate seed B times, then noise
    z0 = z_seed.expand(n_variants, -1).clone().to(device)
    t = torch.full((n_variants,), t_edit, device=device, dtype=torch.long)
    z_t, _ = schedule.q_sample(z0, t)

    # Build DDIM trajectory from t_edit down to 0 with the same step schedule
    # used by ddim_sample_feasibility.
    n_steps = max(1, t_edit // 25 * 25 // (T // 40) + 5)
    n_steps = max(20, min(40, n_steps))
    ts = torch.linspace(t_edit, 0, n_steps + 1, device=device).long()
    z = z_t

    cfg_dropout_mask = torch.zeros_like(mask)
    for i in range(n_steps):
        t_now = ts[i]
        t_next = ts[i + 1]
        t_batch = torch.full((n_variants,), int(t_now), device=device, dtype=torch.long)
        eps_cond = denoiser(z, t_batch, target_z, mask)
        if guidance_scale != 1.0:
            eps_null = denoiser(z, t_batch, target_z, cfg_dropout_mask)
            eps = eps_null + guidance_scale * (eps_cond - eps_null)
        else:
            eps = eps_cond
        ab_now = schedule.alpha_bar[t_now].clamp(min=0.01)
        ab_next = (schedule.alpha_bar[t_next] if t_next > 0
                    else torch.tensor(1.0, device=device))
        z0_pred = (z - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()
        # mild clip per Imagen-style
        norm = z0_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (norm.clamp(max=30.0) / norm)
        z0_pred = z0_pred * scale
        # ANCHOR: blend predicted z0 toward seed to preserve scaffold
        if anchor_alpha > 0.0:
            a = anchor_alpha
            if anchor_decay:
                a = anchor_alpha * (1.0 - i / max(n_steps - 1, 1))
            z0_pred = (1 - a) * z0_pred + a * z_seed.expand_as(z0_pred)
        z = ab_next.sqrt() * z0_pred + (1 - ab_next).sqrt() * eps

    logits = limo.decode(z)
    toks = logits.argmax(-1).cpu().tolist()
    smis = [tok.indices_to_smiles(t) for t in toks]
    return smis, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--seeds_csv", required=True,
                    help="CSV with at least 'name,smiles' columns")
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7])
    ap.add_argument("--n_variants", type=int, default=100)
    ap.add_argument("--quantile", type=float, default=1.281)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--target_density", type=float, default=None)
    ap.add_argument("--target_hof",     type=float, default=None)
    ap.add_argument("--target_d",       type=float, default=None)
    ap.add_argument("--target_p",       type=float, default=None)
    ap.add_argument("--require_neutral", action="store_true")
    ap.add_argument("--anchor_alpha", type=float, default=0.0,
                    help="Latent-anchor coefficient toward seed (0 = pure SDEdit, "
                         "0.3 typical for scaffold preservation)")
    ap.add_argument("--no_anchor_decay", action="store_true",
                    help="By default anchor_alpha decays to 0 over the trajectory")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    exp = Path(args.exp)
    if not exp.is_absolute(): exp = base / exp
    out_dir = Path(args.out_dir) if args.out_dir else exp / "c2c_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading stack …")
    cb = torch.load(exp / "checkpoints" / args.ckpt, map_location=args.device,
                     weights_only=False)
    cfg = cb["config"]
    lblob = torch.load(base / cfg["paths"]["latents_pt"], weights_only=False)
    pn = lblob["property_names"]
    stats = lblob["stats"]
    n_props = lblob["values_raw"].shape[1]
    denoiser = ConditionalDenoiser(latent_dim=lblob["z_mu"].shape[1],
                                     hidden=cfg["model"]["hidden"],
                                     n_blocks=cfg["model"]["n_blocks"],
                                     time_dim=cfg["model"]["time_dim"],
                                     prop_emb_dim=cfg["model"]["prop_emb_dim"],
                                     n_props=n_props,
                                     dropout=cfg["model"].get("dropout", 0.0)
                                     ).to(args.device)
    denoiser.load_state_dict(cb["model_state"])
    if cb.get("ema_state") is not None:
        ema = EMA(denoiser, decay=cfg["training"]["ema_decay"])
        ema.load_state_dict(cb["ema_state"]); ema.apply_to(denoiser)
    denoiser.eval()
    schedule = NoiseSchedule(T=cfg["training"]["T"], device=args.device)

    limo_dir = find_limo_repo(base); vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    ckpt_limo = Path(lblob["meta"]["checkpoint"])
    if not ckpt_limo.is_absolute(): ckpt_limo = base / ckpt_limo
    lb = torch.load(ckpt_limo, map_location=args.device, weights_only=False)
    limo = LIMOVAE(); limo.load_state_dict(lb["model_state"])
    limo.to(args.device).eval()

    val = UniMolValidator(model_dir=str(
        base / "data/raw/energetic_external/EMDP/Data/smoke_model"))
    name_map = {"density":"density","heat_of_formation":"HOF_S",
                "detonation_velocity":"DetoD","detonation_pressure":"DetoP"}

    # Build target z (q90 or user-supplied)
    target_raw = {p: args.quantile*stats[p]["std"]+stats[p]["mean"] for p in pn}
    if args.target_density is not None: target_raw["density"] = args.target_density
    if args.target_hof     is not None: target_raw["heat_of_formation"] = args.target_hof
    if args.target_d       is not None: target_raw["detonation_velocity"] = args.target_d
    if args.target_p       is not None: target_raw["detonation_pressure"] = args.target_p
    target_z_per = {p: (target_raw[p]-stats[p]["mean"])/stats[p]["std"] for p in pn}
    print("Targets:", target_raw)

    # Read seeds
    seeds = []
    with open(base / args.seeds_csv, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("smiles"):
                seeds.append((row.get("name", "?"), row["smiles"].strip()))
    print(f"Loaded {len(seeds)} seeds")

    summary_rows = []
    for seed_name, seed_smi in seeds:
        seed_canon = canon(seed_smi)
        if not seed_canon:
            print(f"  [skip] {seed_name}: invalid SMILES")
            continue
        print(f"\n=== seed: {seed_name}  ({seed_canon[:50]}…) ===")
        # Encode seed
        try:
            x, _ = tok.smiles_to_tensor(seed_canon)
            x = x.unsqueeze(0).to(args.device)
            with torch.no_grad():
                _, mu, _ = limo.encode(x)
            z_seed = mu
            print(f"  encoded; z norm = {z_seed.norm(dim=-1).item():.2f}")
        except Exception as e:
            print(f"  [skip] encode failed: {e}")
            continue

        # Self-consistency: argmax decode of clean z_seed
        with torch.no_grad():
            sc_logits = limo.decode(z_seed)
        sc_smi = canon(tok.indices_to_smiles(sc_logits.argmax(-1)[0].cpu().tolist()))
        sc_match = (sc_smi == seed_canon)
        sc_tan = tanimoto(seed_canon, sc_smi or "C")
        print(f"  self-consistency: exact={sc_match}  Tanimoto={sc_tan:.2f}")

        # SA + SC of seed
        seed_sa = real_sa(seed_canon)
        seed_sc = real_sc(seed_canon)

        for strength in args.strengths:
            print(f"  strength={strength} …")
            target_z_t = torch.zeros(args.n_variants, n_props, device=args.device)
            mask = torch.ones(args.n_variants, n_props, device=args.device)
            for j, p in enumerate(pn):
                target_z_t[:, j] = target_z_per[p]
            t0 = time.time()
            smis, _ = c2c_sample(denoiser, schedule, limo, tok,
                                   z_seed.squeeze(0), args.n_variants,
                                   strength, target_z_t, mask,
                                   guidance_scale=args.cfg, device=args.device,
                                   anchor_alpha=args.anchor_alpha,
                                   anchor_decay=not args.no_anchor_decay)
            canons = [canon(s) for s in smis]
            valid = [c for c in canons if c]
            if args.require_neutral:
                valid = [c for c in valid if is_neutral(c)]
            unique = list(dict.fromkeys(valid))
            non_trivial = [c for c in unique if c != seed_canon]
            if not non_trivial:
                print("    no non-trivial variants"); continue

            # 3DCNN validation
            try:
                pdict = val.predict(non_trivial)
            except Exception as e:
                print(f"    validator failed: {e}"); continue
            cols = {p: np.asarray(pdict.get(name_map[p]), dtype=float) for p in pn}
            keep = np.ones(len(non_trivial), dtype=bool)
            for p in pn:
                keep &= ~np.isnan(cols[p])
            non_trivial = [s for s, k in zip(non_trivial, keep) if k]
            cols = {p: cols[p][keep] for p in pn}

            # Per-row Tanimoto + composite vs target + SA + SC
            tans = np.array([tanimoto(seed_canon, s) for s in non_trivial])
            sa_arr = np.array([real_sa(s) for s in non_trivial])
            sc_arr = np.array([real_sc(s) for s in non_trivial])
            composite = np.zeros(len(non_trivial))
            for p in pn:
                composite += np.abs(cols[p] - target_raw[p]) / max(stats[p]["std"], 1e-6)
            order = np.argsort(composite)
            top = order[:min(20, len(order))]

            md = [f"# c2c seed={seed_name}  strength={strength}", "",
                  f"seed SMILES: `{seed_canon}`",
                  f"seed properties (real SA={seed_sa:.2f}, SC={seed_sc:.2f})",
                  f"targets: ρ={target_raw['density']:.2f}  HOF={target_raw['heat_of_formation']:.0f}  "
                  f"D={target_raw['detonation_velocity']:.2f}  P={target_raw['detonation_pressure']:.2f}",
                  f"self-consistency: exact={sc_match}  Tanimoto={sc_tan:.2f}",
                  "",
                  f"## Aggregate (n={len(non_trivial)} validated, non-trivial)",
                  "",
                  f"- valid: {len(valid)}/{args.n_variants}  unique: {len(unique)}  non-trivial: {len(non_trivial)}",
                  f"- Tanimoto to seed:  mean={tans.mean():.3f}  p25={np.percentile(tans,25):.3f}  p75={np.percentile(tans,75):.3f}",
                  f"- SA mean: {np.nanmean(sa_arr):.2f}  (seed = {seed_sa:.2f})",
                  f"- SC mean: {np.nanmean(sc_arr):.2f}  (seed = {seed_sc:.2f})",
                  ""]
            for prop in pn:
                v = cols[prop]; t = target_raw[prop]
                rel = 100*np.mean(np.abs(v - t))/max(abs(t), 1e-6)
                md.append(f"- {prop}: mean={v.mean():+.3f}  rel_MAE vs target = {rel:.1f}%")
            md.append("")
            md.append("## Top 20 variants (composite-ranked)")
            md.append("")
            md.append("| rank | composite | Tan(seed) | ρ | HOF | D | P | SA | SC | SMILES |")
            md.append("|" + "|".join(["---"]*10) + "|")
            for i, idx in enumerate(top):
                md.append(f"| {i+1} | {composite[idx]:.2f} | {tans[idx]:.2f} | "
                          f"{cols['density'][idx]:.3f} | "
                          f"{cols['heat_of_formation'][idx]:+.1f} | "
                          f"{cols['detonation_velocity'][idx]:.2f} | "
                          f"{cols['detonation_pressure'][idx]:.2f} | "
                          f"{sa_arr[idx]:.2f} | {sc_arr[idx]:.2f} | "
                          f"`{non_trivial[idx]}` |")
            seed_safe = seed_name.replace("/", "_").replace(" ", "_")
            out_md = out_dir / f"{seed_safe}_str{strength:.1f}.md"
            out_md.write_text("\n".join(md), encoding="utf-8")
            print(f"    saved {out_md.name}  (n_top={len(top)} elapsed={time.time()-t0:.0f}s)")
            summary_rows.append({
                "seed": seed_name, "strength": strength,
                "n_variants": len(non_trivial),
                "tanimoto_mean": float(tans.mean()),
                "sa_mean": float(np.nanmean(sa_arr)),
                "sa_seed": seed_sa,
                "best_composite": float(composite[top[0]]),
            })

    # index
    idx = ["# c2c index", "",
            "| seed | strength | n_variants | Tanimoto mean | SA mean | SA seed | best composite |",
            "|---|---|---|---|---|---|---|"]
    for r in summary_rows:
        idx.append(f"| {r['seed']} | {r['strength']} | {r['n_variants']} | "
                   f"{r['tanimoto_mean']:.3f} | {r['sa_mean']:.2f} | "
                   f"{r['sa_seed']:.2f} | {r['best_composite']:.2f} |")
    (out_dir / "c2c_index.md").write_text("\n".join(idx), encoding="utf-8")
    print(f"\nIndex saved {out_dir / 'c2c_index.md'}")


if __name__ == "__main__":
    sys.exit(main())
