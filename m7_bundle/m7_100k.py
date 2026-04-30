"""M7: 100k-pool diverse-lane sampling.

Runs 5 new guidance lanes (seeds 3-4, new configs) to double the
discovered-compound pool. Each lane: pool_per_run molecules through
both DGLD-H (v4b) and DGLD-P (v3) denoisers, fused post-decode.

Lane definitions:
  L0  C0_seed3      unguided CFG w=7,  seed=3
  L1  C0_seed4      unguided CFG w=7,  seed=4
  L2  C4_highw      unguided CFG w=12, seed=3  (stronger property push)
  L3  C5_viab4      viab=4.0, sens=0.3,         seed=3
  L4  C6_hazviab    viab=0.5, hazard=3.0,        seed=3

Total raw: 5 lanes × 2 denoisers × pool_per_run SMILES.
At pool_per_run=10000: 100k raw SMILES.

Designed for vast.ai RTX 4090 (~30 min total at pool=10k).
Self-contained import surface: run from a flat directory.
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
    print(f"[m7] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

    ap = argparse.ArgumentParser()
    ap.add_argument("--v4b_ckpt", required=True)
    ap.add_argument("--v3_ckpt", required=True)
    ap.add_argument("--limo_ckpt", required=True)
    ap.add_argument("--score_model", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--vocab_json", required=True)
    ap.add_argument("--pool_per_run", type=int, default=10000)
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.results_dir); out_dir.mkdir(exist_ok=True, parents=True)
    meta = json.loads(Path(args.meta_json).read_text())
    pn = meta["property_names"]; n_props = meta["n_props"]; latent_dim = meta["latent_dim"]
    stats = meta["stats"]

    from limo_model import LIMOVAE, SELFIESTokenizer, load_vocab, LIMO_MAX_LEN
    from model import ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample
    from guided_v2_sampler import ddim_sample_guided_v2, load_score_model

    print("[m7] Loading models"); sys.stdout.flush()
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
    score_model, _ = load_score_model(args.score_model, device=str(device))

    target_raw = {"density": 1.95, "heat_of_formation": 220,
                  "detonation_velocity": 9.5, "detonation_pressure": 40}
    target_z = torch.tensor([(target_raw[p] - stats[p]["mean"]) / stats[p]["std"]
                              for p in pn], device=device)

    lanes = [
        ("L0_C0_seed3", None,   7.0,  3),
        ("L1_C0_seed4", None,   7.0,  4),
        ("L2_C4_highw", None,  12.0,  3),
        ("L3_C5_viab4",
            {"viab": 4.0, "sens": 0.3, "hazard": 0.0, "sa": 0.0, "sc": 0.0}, 7.0, 3),
        ("L4_C6_hazviab",
            {"viab": 0.5, "sens": 0.0, "hazard": 3.0, "sa": 0.0, "sc": 0.0}, 7.0, 3),
    ]
    summary = {"lanes": [], "args": vars(args)}

    def sample_one(denoiser, sch, n_pool, gscales, cfg_scale, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        mask = torch.ones(n_pool, n_props, device=device)
        vals = torch.full((n_pool, n_props), 0.0, device=device)
        for j in range(target_z.shape[0]):
            vals[:, j] = target_z[j]
        if gscales is None:
            z = ddim_sample(denoiser, sch, vals, mask, n_steps=40,
                             guidance_scale=cfg_scale, device=device)
        else:
            z = ddim_sample_guided_v2(denoiser, sch, vals, mask, score_model,
                                       n_steps=40, cfg_scale=cfg_scale,
                                       guidance_scales=gscales, device=device)
        with torch.no_grad():
            logits = limo.decode(z)
        toks = logits.argmax(-1).cpu().tolist()
        return [tok.indices_to_smiles(t) for t in toks]

    for lane_idx, (name, gscales, cfg_w, seed) in enumerate(lanes):
        t0 = time.time()
        print(f"[m7] Lane {lane_idx+1}/{len(lanes)}: {name} cfg_w={cfg_w} seed={seed}")
        sys.stdout.flush()
        smis = []
        smis += sample_one(d_v4b, sch_v4b, args.pool_per_run, gscales, cfg_w, seed)
        smis += sample_one(d_v3, sch_v3, args.pool_per_run, gscales, cfg_w, seed + 10000)
        elapsed = time.time() - t0
        out_path = out_dir / f"m7_{name}.txt"
        out_path.write_text("\n".join(smis), encoding="utf-8")
        summary["lanes"].append({"lane": name, "seed": seed, "cfg_w": cfg_w,
                                   "gscales": gscales, "n": len(smis),
                                   "elapsed_s": round(elapsed, 1),
                                   "out_path": str(out_path)})
        print(f"[m7]   -> n={len(smis)} in {elapsed:.1f}s"); sys.stdout.flush()

    (out_dir / "m7_summary.json").write_text(json.dumps(summary, indent=2))
    total_smis = sum(r["n"] for r in summary["lanes"])
    print(f"[m7] Done. {len(lanes)} lanes, {total_smis} total SMILES.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
