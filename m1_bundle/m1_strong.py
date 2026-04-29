"""M1' stronger-scale guidance rerun. Same conditions as m1_sweep, but with
guidance scales scaled 5x to test whether the gradient becomes visible at
larger scales. Single seed (since this is a probing experiment).

Conditions:
    C0 unguided        (no score_model)
    C1 viab-strong     s_v=5.0, s_s=0,   s_SA=0
    C2 viab+sens       s_v=5.0, s_s=2.0, s_SA=0
    C3 viab+sens+SA    s_v=5.0, s_s=2.0, s_SA=1.0
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
    ap.add_argument("--target_density", type=float, default=1.95)
    ap.add_argument("--target_hof", type=float, default=220.0)
    ap.add_argument("--target_d", type=float, default=9.5)
    ap.add_argument("--target_p", type=float, default=40.0)
    ap.add_argument("--results_dir", default="results")
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

    conditions = [
        ("C0_unguided",       None),
        ("C1_viab_strong",    {"viab": 5.0, "sens": 0.0, "sa": 0.0, "sc": 0.0}),
        ("C2_viab_sens_strong", {"viab": 5.0, "sens": 2.0, "sa": 0.0, "sc": 0.0}),
        ("C3_all_strong",     {"viab": 5.0, "sens": 2.0, "sa": 1.0, "sc": 0.0}),
    ]

    summary = {"conditions": [], "args": vars(args)}

    def sample_one(denoiser, sch, n_pool, gscales, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        mask = torch.ones(n_pool, n_props, device=device)
        vals = torch.full((n_pool, n_props), 0.0, device=device)
        for j in range(target_z.shape[0]):
            vals[:, j] = target_z[j]
        if gscales is None:
            z = ddim_sample(denoiser, sch, vals, mask, n_steps=40,
                             guidance_scale=7.0, device=device)
        else:
            z = ddim_sample_guided_v2(denoiser, sch, vals, mask, score_model,
                                       n_steps=40, cfg_scale=7.0,
                                       guidance_scales=gscales, device=device)
        with torch.no_grad():
            logits = limo.decode(z)
        toks = logits.argmax(-1).cpu().tolist()
        return [tok.indices_to_smiles(t) for t in toks]

    for i, (name, gscales) in enumerate(conditions, 1):
        t0 = time.time()
        print(f"  {i}/{len(conditions)} loss=0.0000 cond={name} gscales={gscales}")
        sys.stdout.flush()
        smis = []
        smis += sample_one(d_v4b, sch_v4b, args.pool_per_run, gscales, args.seed)
        smis += sample_one(d_v3, sch_v3, args.pool_per_run, gscales, args.seed + 10000)
        elapsed = time.time() - t0
        out_path = out_dir / f"m1_strong_{name}.txt"
        out_path.write_text("\n".join(smis), encoding="utf-8")
        meta_c = {"condition": name, "guidance_scales": gscales, "n": len(smis),
                  "elapsed_s": elapsed, "out_path": str(out_path)}
        summary["conditions"].append(meta_c)
        print(f"[train] {name}: n={len(smis)} elapsed={elapsed:.1f}s -> {out_path}")
        sys.stdout.flush()

    (out_dir / "m1_strong_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] === DONE === ({len(conditions)} conditions)"); sys.stdout.flush()


if __name__ == "__main__":
    main()
