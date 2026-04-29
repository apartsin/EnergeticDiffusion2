"""M3 sample-time hyperparameter grid (no denoiser retraining).

Closes the §5.12 limitation that "hyperparameter sensitivity is not
exhaustively characterised". Sweeps sample-time knobs only (cfg_scale,
n_steps), keeping the trained denoisers, score model, and LIMO encoder
fixed.

Cells (all at pool=10000 per pool x 2 pools = 20k SMILES per cell, single seed):
    G0 reference   cfg=7,  steps=40
    G1 cfg=3       cfg=3,  steps=40
    G2 cfg=5       cfg=5,  steps=40
    G3 cfg=9       cfg=9,  steps=40
    G4 cfg=12      cfg=12, steps=40
    G5 steps=20    cfg=7,  steps=20
    G6 steps=80    cfg=7,  steps=80

Output: results/m3_grid_<cell>.txt per cell + m3_summary.json.

Bundle uses m1_bundle/ infrastructure (same checkpoints).

Run on remote:
    python3 m3_hparam_grid.py --v4b_ckpt v4b_best.pt --v3_ckpt v3_best.pt \
        --limo_ckpt limo_best.pt --score_model score_model_v3e.pt \
        --meta_json meta.json --vocab_json vocab.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main():
    assert torch.cuda.is_available(), "M3 grid requires GPU."
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
    ap.add_argument("--target_density", type=float, default=1.95)
    ap.add_argument("--target_hof", type=float, default=220.0)
    ap.add_argument("--target_d", type=float, default=9.5)
    ap.add_argument("--target_p", type=float, default=40.0)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.results_dir); out_dir.mkdir(exist_ok=True, parents=True)
    meta = json.loads(Path(args.meta_json).read_text())
    pn = meta["property_names"]; n_props = meta["n_props"]; latent_dim = meta["latent_dim"]
    stats = meta["stats"]

    from limo_model import LIMOVAE, SELFIESTokenizer, load_vocab, LIMO_MAX_LEN
    from model import ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample
    from guided_v2_sampler import ddim_sample_guided_v2, load_score_model

    print(f"[train] Loading LIMO"); sys.stdout.flush()
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

    print(f"[train] Loading denoisers"); sys.stdout.flush()
    d_v4b, sch_v4b = load_denoiser(args.v4b_ckpt, meta["v4b_cfg"])
    d_v3, sch_v3 = load_denoiser(args.v3_ckpt, meta["v3_cfg"])

    print(f"[train] Loading score model"); sys.stdout.flush()
    score_model, _ = load_score_model(args.score_model, device=str(device))

    target_raw = {"density": args.target_density, "heat_of_formation": args.target_hof,
                   "detonation_velocity": args.target_d, "detonation_pressure": args.target_p}
    target_z = torch.tensor([(target_raw[p] - stats[p]["mean"]) / stats[p]["std"]
                              for p in pn], device=device)

    # Default guidance: paper's recommended (s_v=1.0, s_s=0.3)
    gscales = {"viab": 1.0, "sens": 0.3, "sa": 0.0, "sc": 0.0}

    cells = [
        ("G0_ref",       7.0,  40),
        ("G1_cfg3",      3.0,  40),
        ("G2_cfg5",      5.0,  40),
        ("G3_cfg9",      9.0,  40),
        ("G4_cfg12",    12.0,  40),
        ("G5_steps20",   7.0,  20),
        ("G6_steps80",   7.0,  80),
    ]
    print(f"[train] {len(cells)} cells, pool_per_run={args.pool_per_run}"); sys.stdout.flush()

    summary = {"cells": [], "args": vars(args)}

    def sample_one(denoiser, sch, n_pool, cfg_scale, n_steps, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        mask = torch.ones(n_pool, n_props, device=device)
        vals = torch.full((n_pool, n_props), 0.0, device=device)
        for j in range(target_z.shape[0]):
            vals[:, j] = target_z[j]
        z = ddim_sample_guided_v2(denoiser, sch, vals, mask, score_model,
                                    n_steps=n_steps, cfg_scale=cfg_scale,
                                    guidance_scales=gscales, device=device)
        with torch.no_grad():
            logits = limo.decode(z)
        toks = logits.argmax(-1).cpu().tolist()
        return [tok.indices_to_smiles(t) for t in toks]

    for i, (name, cfg_s, n_steps) in enumerate(cells, 1):
        t0 = time.time()
        print(f"  {i}/{len(cells)} loss=0.0000 cell={name} cfg={cfg_s} steps={n_steps}")
        sys.stdout.flush()
        smis = []
        smis += sample_one(d_v4b, sch_v4b, args.pool_per_run, cfg_s, n_steps, args.seed)
        smis += sample_one(d_v3, sch_v3, args.pool_per_run, cfg_s, n_steps, args.seed + 10000)
        elapsed = time.time() - t0
        out_path = out_dir / f"m3_grid_{name}.txt"
        out_path.write_text("\n".join(smis), encoding="utf-8")
        meta_cell = {"cell": name, "cfg_scale": cfg_s, "n_steps": n_steps,
                      "n_smiles_raw": len(smis), "elapsed_s": elapsed,
                      "out_path": str(out_path)}
        summary["cells"].append(meta_cell)
        print(f"[train] {name}: n={len(smis)} elapsed={elapsed:.1f}s -> {out_path}")
        sys.stdout.flush()

    (out_dir / "m3_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] === DONE === ({len(cells)} cells)"); sys.stdout.flush()


if __name__ == "__main__":
    main()
