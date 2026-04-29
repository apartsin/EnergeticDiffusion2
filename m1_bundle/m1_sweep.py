"""M1 (reviewer concern): guidance-vs-unguided head-to-head, matched compute,
multiple seeds. Sampling-only — outputs SMILES per (condition, seed); the
3DCNN/Phase-A/xTB post-processing runs locally afterwards.

Conditions:
    C0: unguided                          (no score_model)
    C1: viab-only        s_v=1.0, s_s=0,   s_SA=0
    C2: viab+sens        s_v=1.0, s_s=0.3, s_SA=0     [paper recommended]
    C3: viab+sens+SA     s_v=1.0, s_s=0.3, s_SA=0.15  [diversity]

3 seeds each -> 12 sampling runs.

Output: results/m1_sweep_<condition>_<seed>.txt (one SMILES per line) +
        results/m1_summary.json (per-run metadata).

Designed to run on vast.ai RTX_4090 in ~5 min total at pool=10000 per run,
or ~25 min at pool=40000 per run. Bundle requirements: see m1_bundle.py.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

# Self-contained import surface — runs from a flat directory on vast.ai
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main():
    assert torch.cuda.is_available(), "M1 requires GPU."
    device = torch.device("cuda")
    print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

    ap = argparse.ArgumentParser()
    ap.add_argument("--v4b_ckpt", required=True)
    ap.add_argument("--v3_ckpt", required=True)
    ap.add_argument("--limo_ckpt", required=True)
    ap.add_argument("--score_model", required=True)
    ap.add_argument("--meta_json", required=True,
                    help="JSON with property_names, n_props, latent_dim, stats, "
                         "v3_cfg, v4b_cfg")
    ap.add_argument("--vocab_json", required=True)
    ap.add_argument("--pool_per_run", type=int, default=10000)
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--cfg_scale", type=float, default=7.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--target_density", type=float, default=1.95)
    ap.add_argument("--target_hof", type=float, default=220.0)
    ap.add_argument("--target_d", type=float, default=9.5)
    ap.add_argument("--target_p", type=float, default=40.0)
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.results_dir); out_dir.mkdir(exist_ok=True, parents=True)

    # ── Load metadata ──────────────────────────────────────────────────────
    print(f"[train] Loading meta {args.meta_json}"); sys.stdout.flush()
    meta = json.loads(Path(args.meta_json).read_text())
    pn = meta["property_names"]
    n_props = meta["n_props"]
    latent_dim = meta["latent_dim"]
    stats = meta["stats"]

    # ── Load LIMO ──────────────────────────────────────────────────────────
    print(f"[train] Loading LIMO from {args.limo_ckpt}"); sys.stdout.flush()
    from limo_model import LIMOVAE, SELFIESTokenizer, load_vocab, LIMO_MAX_LEN
    alphabet = load_vocab(Path(args.vocab_json))
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    limo_blob = torch.load(args.limo_ckpt, map_location=device, weights_only=False)
    limo = LIMOVAE().to(device)
    limo.load_state_dict(limo_blob["model_state"])
    limo.eval()
    for p in limo.parameters(): p.requires_grad_(False)

    # ── Load denoisers ─────────────────────────────────────────────────────
    from model import ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample
    from guided_v2_sampler import ddim_sample_guided_v2, load_score_model

    def load_denoiser(ckpt_path, cfg):
        cb = torch.load(ckpt_path, map_location=device, weights_only=False)
        d = ConditionalDenoiser(latent_dim=latent_dim,
                                hidden=cfg["hidden"], n_blocks=cfg["n_blocks"],
                                time_dim=cfg["time_dim"],
                                prop_emb_dim=cfg["prop_emb_dim"],
                                n_props=n_props,
                                dropout=cfg.get("dropout", 0.0)).to(device)
        d.load_state_dict(cb["model_state"])
        if cb.get("ema_state") is not None:
            ema = EMA(d, decay=cfg["ema_decay"])
            ema.load_state_dict(cb["ema_state"]); ema.apply_to(d)
        d.eval()
        for p in d.parameters(): p.requires_grad_(False)
        sch = NoiseSchedule(T=cfg["T"], device=device)
        return d, sch

    print(f"[train] Loading v4-B denoiser"); sys.stdout.flush()
    d_v4b, sch_v4b = load_denoiser(args.v4b_ckpt, meta["v4b_cfg"])
    print(f"[train] Loading v3 denoiser"); sys.stdout.flush()
    d_v3, sch_v3 = load_denoiser(args.v3_ckpt, meta["v3_cfg"])

    print(f"[train] Loading score model {args.score_model}"); sys.stdout.flush()
    score_model, _ = load_score_model(args.score_model, device=str(device))

    # ── Targets in z-score space ──────────────────────────────────────────
    target_raw = {
        "density": args.target_density, "heat_of_formation": args.target_hof,
        "detonation_velocity": args.target_d, "detonation_pressure": args.target_p,
    }
    target_z = torch.tensor([(target_raw[p] - stats[p]["mean"]) / stats[p]["std"]
                              for p in pn], device=device)
    print(f"[train] target_raw={target_raw}"); sys.stdout.flush()

    # ── Conditions ─────────────────────────────────────────────────────────
    conditions = [
        ("C0_unguided",   None),
        ("C1_viab",       {"viab": 1.0, "sens": 0.0, "sa": 0.0, "sc": 0.0}),
        ("C2_viab_sens",  {"viab": 1.0, "sens": 0.3, "sa": 0.0, "sc": 0.0}),
        ("C3_viab_sens_sa", {"viab": 1.0, "sens": 0.3, "sa": 0.15, "sc": 0.0}),
    ]

    summary = {"runs": [], "args": vars(args)}
    total_runs = len(conditions) * len(args.seeds)
    run_idx = 0

    def sample_one(denoiser, sch, n_pool, gscales, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        mask = torch.ones(n_pool, n_props, device=device)
        vals = torch.full((n_pool, n_props), 0.0, device=device)
        for j in range(target_z.shape[0]):
            vals[:, j] = target_z[j]
        if gscales is None:
            z = ddim_sample(denoiser, sch, vals, mask,
                            n_steps=args.n_steps,
                            guidance_scale=args.cfg_scale, device=device)
        else:
            z = ddim_sample_guided_v2(denoiser, sch, vals, mask, score_model,
                                       n_steps=args.n_steps,
                                       cfg_scale=args.cfg_scale,
                                       guidance_scales=gscales,
                                       device=device)
        with torch.no_grad():
            logits = limo.decode(z)
        toks = logits.argmax(-1).cpu().tolist()
        return [tok.indices_to_smiles(t) for t in toks]

    # ── Loop ───────────────────────────────────────────────────────────────
    for cond_name, gscales in conditions:
        for seed in args.seeds:
            run_idx += 1
            t0 = time.time()
            print(f"  {run_idx}/{total_runs} loss=0.0000 cond={cond_name} seed={seed}")
            sys.stdout.flush()
            # Two pools (v4b + v3) per run, matching joint_rerank.py
            smis = []
            smis += sample_one(d_v4b, sch_v4b, args.pool_per_run, gscales, seed)
            smis += sample_one(d_v3, sch_v3, args.pool_per_run, gscales, seed + 10000)
            elapsed = time.time() - t0
            out_path = out_dir / f"m1_sweep_{cond_name}_seed{seed}.txt"
            out_path.write_text("\n".join(smis), encoding="utf-8")
            run_meta = {
                "condition": cond_name,
                "seed": seed,
                "guidance_scales": gscales,
                "n_smiles_raw": len(smis),
                "elapsed_s": elapsed,
                "out_path": str(out_path),
            }
            summary["runs"].append(run_meta)
            print(f"[train] cond={cond_name} seed={seed} n={len(smis)} "
                  f"elapsed={elapsed:.1f}s -> {out_path}")
            sys.stdout.flush()

    summary_path = out_dir / "m1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[train] === DONE === ({run_idx} runs, summary -> {summary_path})")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
