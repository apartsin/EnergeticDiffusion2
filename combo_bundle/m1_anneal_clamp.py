"""E1: anneal-vs-clamp ablation. 4 cells = (anneal=0|2) x (clamp=5|50).
All with viab=1.0, sens=0.3, hazard=1.0 (default-guided). Pool=10k each.

Tests whether the bug fix is attributable to:
  - disabling alpha-anneal (sigma_max_for_anneal=0 vs 2),
  - loosening max_grad_norm (5 vs 50),
  - or both.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

    ap = argparse.ArgumentParser()
    ap.add_argument("--v4b_ckpt", required=True)
    ap.add_argument("--v3_ckpt", required=True)
    ap.add_argument("--limo_ckpt", required=True)
    ap.add_argument("--score_model", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--vocab_json", required=True)
    ap.add_argument("--pool_per_run", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.results_dir); out_dir.mkdir(exist_ok=True, parents=True)
    meta = json.loads(Path(args.meta_json).read_text())
    pn = meta["property_names"]; n_props = meta["n_props"]; latent_dim = meta["latent_dim"]
    stats = meta["stats"]

    from limo_model import LIMOVAE, SELFIESTokenizer, load_vocab, LIMO_MAX_LEN
    from model import ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample
    import guided_v2_sampler as gv2

    print(f"[train] Loading models"); sys.stdout.flush()
    alphabet = load_vocab(Path(args.vocab_json))
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    limo_blob = torch.load(args.limo_ckpt, map_location=device, weights_only=False)
    limo = LIMOVAE().to(device); limo.load_state_dict(limo_blob["model_state"]); limo.eval()
    for p in limo.parameters(): p.requires_grad_(False)

    def load_denoiser(ckpt_path, cfg):
        cb = torch.load(ckpt_path, map_location=device, weights_only=False)
        d = ConditionalDenoiser(latent_dim=latent_dim, hidden=cfg["hidden"],
                                  n_blocks=cfg["n_blocks"], time_dim=cfg["time_dim"],
                                  prop_emb_dim=cfg["prop_emb_dim"], n_props=n_props,
                                  dropout=cfg.get("dropout", 0.0)).to(device)
        d.load_state_dict(cb["model_state"])
        if cb.get("ema_state") is not None:
            ema = EMA(d, decay=cfg["ema_decay"])
            ema.load_state_dict(cb["ema_state"]); ema.apply_to(d)
        d.eval()
        for p in d.parameters(): p.requires_grad_(False)
        return d, NoiseSchedule(T=cfg["T"], device=device)

    d_v4b, sch_v4b = load_denoiser(args.v4b_ckpt, meta["v4b_cfg"])
    d_v3, sch_v3 = load_denoiser(args.v3_ckpt, meta["v3_cfg"])
    score_model, _ = gv2.load_score_model(args.score_model, device=str(device))

    target_raw = {"density": 1.95, "heat_of_formation": 220, "detonation_velocity": 9.5,
                   "detonation_pressure": 40}
    target_z = torch.tensor([(target_raw[p] - stats[p]["mean"]) / stats[p]["std"]
                              for p in pn], device=device)

    cells = []
    for anneal in [0.0, 2.0]:
        for clamp in [5.0, 50.0]:
            name = f"anneal{anneal:g}_clamp{clamp:g}"
            cells.append((name, anneal, clamp))
    print(f"[train] {len(cells)} cells = {{anneal=0|2}} x {{clamp=5|50}}")
    sys.stdout.flush()

    summary = {"cells": [], "args": vars(args)}
    gscales = {"viab": 1.0, "sens": 0.3, "hazard": 1.0, "sa": 0.0, "sc": 0.0}

    # We need to monkey-patch the _guidance_grad defaults
    orig_guidance_grad = gv2._guidance_grad

    def make_patched(anneal_val, clamp_val):
        def patched(score_model, z_t, sigma_t, scales, sigma_max_for_anneal=anneal_val, max_grad_norm=clamp_val):
            return orig_guidance_grad(score_model, z_t, sigma_t, scales,
                                        sigma_max_for_anneal=sigma_max_for_anneal,
                                        max_grad_norm=max_grad_norm)
        return patched

    def sample_one(denoiser, sch, n_pool, scales, seed, anneal, clamp):
        torch.manual_seed(seed); np.random.seed(seed)
        mask = torch.ones(n_pool, n_props, device=device)
        vals = torch.full((n_pool, n_props), 0.0, device=device)
        for j in range(target_z.shape[0]):
            vals[:, j] = target_z[j]
        # Patch the guidance_grad function for this cell
        gv2._guidance_grad = make_patched(anneal, clamp)
        z = gv2.ddim_sample_guided_v2(denoiser, sch, vals, mask, score_model,
                                        n_steps=40, cfg_scale=7.0,
                                        guidance_scales=scales, device=device)
        gv2._guidance_grad = orig_guidance_grad
        with torch.no_grad():
            logits = limo.decode(z)
        toks = logits.argmax(-1).cpu().tolist()
        return [tok.indices_to_smiles(t) for t in toks]

    for i, (name, anneal, clamp) in enumerate(cells, 1):
        t0 = time.time()
        print(f"  {i}/{len(cells)} loss=0.0000 cell={name} anneal={anneal} clamp={clamp}")
        sys.stdout.flush()
        smis = []
        smis += sample_one(d_v4b, sch_v4b, args.pool_per_run, gscales, args.seed, anneal, clamp)
        smis += sample_one(d_v3, sch_v3, args.pool_per_run, gscales, args.seed + 10000, anneal, clamp)
        elapsed = time.time() - t0
        out_path = out_dir / f"m1_anneal_clamp_{name}.txt"
        out_path.write_text("\n".join(smis), encoding="utf-8")
        meta_c = {"cell": name, "anneal": anneal, "clamp": clamp, "n": len(smis),
                  "elapsed_s": elapsed, "out_path": str(out_path)}
        summary["cells"].append(meta_c)
        print(f"[train] {name}: n={len(smis)} elapsed={elapsed:.1f}s")
        sys.stdout.flush()

    (out_dir / "m1_anneal_clamp_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] === DONE === ({len(cells)} cells)"); sys.stdout.flush()


if __name__ == "__main__":
    main()
