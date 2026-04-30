"""M10: 200k-pool 10-lane sampling on Modal A10G.

Extends M7 (5-lane, 100k) to 10 lanes, targeting higher max-D and testing
two novel guidance configurations not in M7:

  L0  C0_seed5      unguided CFG w=7,  seed=5       (diversity)
  L1  C0_seed6      unguided CFG w=7,  seed=6       (diversity)
  L2  C0_seed7      unguided CFG w=7,  seed=7       (diversity)
  L3  C4_highw12    unguided CFG w=12, seed=5       (mode sharpening)
  L4  C5_viab4      viab=4.0, sens=0.3              (M7 repeat for consistency)
  L5  C6_hazviab    viab=0.5, hazard=3.0            (M7 repeat)
  L6  C7_str_haz    viab=1.0, hazard=6.0, sens=0.5  (stronger hazard)
  L7  C8_perf05     viab=1.0, perf=0.5, sens=0.3    (NEW: perf head active)
  L8  C9_perf1      viab=1.0, perf=1.0              (NEW: stronger perf push)
  L9  C10_w10       unguided CFG w=10, seed=5       (CFG sweet-spot test)

Total raw: 10 lanes x 2 denoisers x 10k = 200k SMILES.

Key novelty vs M7:
  L7/L8 activate the performance (D/rho/P) score head at sample time.
  This is the first test of perf-guided sampling; the paper §5.5.4 found
  SA head hurt, hazard head helped — perf head is unexplored.

Usage:
    python -m modal run m10_bundle/modal_m10_200k.py

Results:
    m10_bundle/results/m10_summary.json
    m10_bundle/results/m10_L*.txt   (raw SMILES per lane, not committed to git)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import modal

HERE         = Path(__file__).parent.resolve()
PROJECT_ROOT = HERE.parent
COMBO        = PROJECT_ROOT / "combo_bundle"
M7           = PROJECT_ROOT / "m7_bundle"
M1           = PROJECT_ROOT / "m1_bundle"
RESULTS      = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Lane definitions: (seed, cfg_w, gscales_or_None)
# gscales keys: viab, sens, hazard, sa, sc, perf  (all default 0 if absent)
# ---------------------------------------------------------------------------
LANES = [
    ("L0_C0_seed5",   5, 7.0,  None),
    ("L1_C0_seed6",   6, 7.0,  None),
    ("L2_C0_seed7",   7, 7.0,  None),
    ("L3_C4_highw",   5, 12.0, None),
    ("L4_C5_viab4",   5, 7.0,  {"viab": 4.0, "sens": 0.3, "hazard": 0.0, "sa": 0.0, "sc": 0.0, "perf": 0.0}),
    ("L5_C6_hazviab", 5, 7.0,  {"viab": 0.5, "sens": 0.0, "hazard": 3.0, "sa": 0.0, "sc": 0.0, "perf": 0.0}),
    ("L6_C7_strhaz",  5, 7.0,  {"viab": 1.0, "sens": 0.5, "hazard": 6.0, "sa": 0.0, "sc": 0.0, "perf": 0.0}),
    ("L7_C8_perf05",  5, 7.0,  {"viab": 1.0, "sens": 0.3, "hazard": 0.0, "sa": 0.0, "sc": 0.0, "perf": 0.5}),
    ("L8_C9_perf1",   5, 7.0,  {"viab": 1.0, "sens": 0.0, "hazard": 0.0, "sa": 0.0, "sc": 0.0, "perf": 1.0}),
    ("L9_C10_w10",    5, 10.0, None),
]

# ---------------------------------------------------------------------------
# Modal image: PyTorch + selfies + rdkit + model weights baked in
# Model files are large (.pt) and are not in git; uploaded at image-build time.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python="3.11"
    )
    .apt_install("libxrender1", "libxext6")
    .pip_install("selfies==2.1.1", "rdkit-pypi")
    # Python source files (all live in combo_bundle, not m7_bundle)
    .add_local_file(str(COMBO / "model.py"),               "/app/model.py")
    .add_local_file(str(COMBO / "limo_model.py"),          "/app/limo_model.py")
    .add_local_file(str(COMBO / "guided_v2_sampler.py"),   "/app/guided_v2_sampler.py")
    .add_local_file(str(COMBO / "train_multihead_latent.py"), "/app/train_multihead_latent.py")
    .add_local_file(str(COMBO / "vocab.json"),             "/app/vocab.json")
    .add_local_file(str(COMBO / "meta.json"),              "/app/meta.json")
    # Model weights (large files; not in git but Modal bakes them in)
    .add_local_file(str(COMBO / "limo_best.pt"),        "/models/limo_best.pt")
    .add_local_file(str(COMBO / "v4b_best.pt"),         "/models/v4b_best.pt")
    .add_local_file(str(COMBO / "v3_best.pt"),          "/models/v3_best.pt")
    .add_local_file(str(M1   / "score_model_v3f.pt"),   "/models/score_model_v3f.pt")
)

app = modal.App("dgld-m10-200k", image=image)


# ---------------------------------------------------------------------------
# Remote: run all 10 lanes in a single container (avoids per-lane startup cost)
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=4 * 60 * 60,
    memory=24_576,
)
def run_all_lanes(pool_per_run: int = 10_000) -> dict:
    """Run all M10 lanes and return {lane_name: [smiles, ...]}."""
    import sys, json, time as _time
    from pathlib import Path

    sys.path.insert(0, "/app")
    import torch
    assert torch.cuda.is_available(), "No CUDA"
    device = torch.device("cuda")
    print(f"[m10] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    from limo_model import LIMOVAE, SELFIESTokenizer, load_vocab, LIMO_MAX_LEN
    from model import ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample
    from guided_v2_sampler import ddim_sample_guided_v2, load_score_model

    meta = json.loads(Path("/app/meta.json").read_text())
    pn = meta["property_names"]; n_props = meta["n_props"]
    latent_dim = meta["latent_dim"]

    print("[m10] Loading models...", flush=True)
    alphabet = load_vocab("/app/vocab.json")
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)

    limo_blob = torch.load("/models/limo_best.pt", map_location=device, weights_only=False)
    limo = LIMOVAE().to(device)
    limo.load_state_dict(limo_blob["model_state"])
    limo.eval()
    for p in limo.parameters(): p.requires_grad_(False)

    def load_denoiser(ckpt_path, cfg):
        cb = torch.load(ckpt_path, map_location=device, weights_only=False)
        d = ConditionalDenoiser(
            latent_dim=latent_dim, hidden=cfg["hidden"], n_blocks=cfg["n_blocks"],
            time_dim=cfg["time_dim"], prop_emb_dim=cfg["prop_emb_dim"],
            n_props=n_props, dropout=cfg.get("dropout", 0.0)
        ).to(device)
        d.load_state_dict(cb["model_state"])
        if cb.get("ema_state"):
            ema = EMA(d, decay=cfg["ema_decay"])
            ema.load_state_dict(cb["ema_state"]); ema.apply_to(d)
        d.eval()
        for p in d.parameters(): p.requires_grad_(False)
        return d, NoiseSchedule(T=cfg["T"], device=device)

    d_v4b, sch_v4b = load_denoiser("/models/v4b_best.pt", meta["v4b_cfg"])
    d_v3,  sch_v3  = load_denoiser("/models/v3_best.pt",  meta["v3_cfg"])
    score_model, score_cfg = load_score_model("/models/score_model_v3f.pt", device=str(device))
    print("[m10] Models loaded.", flush=True)

    # Target condition: production target (same as M7)
    import numpy as np
    stats = meta["stats"]
    target_raw = {"density": 1.95, "heat_of_formation": 220.0,
                  "detonation_velocity": 9.2, "detonation_pressure": 38.0}
    cond_raw = np.array([target_raw[k] for k in pn], dtype=np.float32)
    cond_norm = np.array([(cond_raw[i] - stats[pn[i]]["mean"]) / max(stats[pn[i]]["std"], 1e-6)
                          for i in range(n_props)], dtype=np.float32)
    cond_t = torch.tensor(cond_norm, dtype=torch.float32, device=device).unsqueeze(0)

    def sample_lane(lane_name, seed, cfg_w, gscales):
        torch.manual_seed(seed)
        np.random.seed(seed)
        lane_smiles = []

        # Build condition tensors (same for all denoiser/seed combinations)
        vals = cond_t.expand(pool_per_run, -1)          # (B, n_props) normalized
        mask = torch.ones(pool_per_run, n_props, device=device)

        for denoiser, sched, tag in [(d_v4b, sch_v4b, "v4b"), (d_v3, sch_v3, "v3")]:
            if gscales is None:
                z0 = ddim_sample(denoiser, sched, vals, mask,
                                 n_steps=40, guidance_scale=cfg_w, device=str(device))
            else:
                z0 = ddim_sample_guided_v2(
                    denoiser, sched, vals, mask, score_model,
                    n_steps=40, cfg_scale=cfg_w, guidance_scales=gscales,
                    device=str(device)
                )
            with torch.no_grad():
                logits = limo.decode(z0)
            toks = logits.argmax(-1).cpu().tolist()
            smiles = [tok.indices_to_smiles(t) for t in toks]
            lane_smiles.extend(smiles)
            print(f"[m10] {lane_name}/{tag}: {len(smiles)} SMILES", flush=True)

        return lane_smiles

    all_results = {}
    t0 = _time.time()
    for (lane_name, seed, cfg_w, gscales) in [
        ("L0_C0_seed5",   5, 7.0,  None),
        ("L1_C0_seed6",   6, 7.0,  None),
        ("L2_C0_seed7",   7, 7.0,  None),
        ("L3_C4_highw",   5, 12.0, None),
        ("L4_C5_viab4",   5, 7.0,  {"viab": 4.0, "sens": 0.3, "hazard": 0.0, "sa": 0.0, "sc": 0.0, "perf": 0.0}),
        ("L5_C6_hazviab", 5, 7.0,  {"viab": 0.5, "sens": 0.0, "hazard": 3.0, "sa": 0.0, "sc": 0.0, "perf": 0.0}),
        ("L6_C7_strhaz",  5, 7.0,  {"viab": 1.0, "sens": 0.5, "hazard": 6.0, "sa": 0.0, "sc": 0.0, "perf": 0.0}),
        ("L7_C8_perf05",  5, 7.0,  {"viab": 1.0, "sens": 0.3, "hazard": 0.0, "sa": 0.0, "sc": 0.0, "perf": 0.5}),
        ("L8_C9_perf1",   5, 7.0,  {"viab": 1.0, "sens": 0.0, "hazard": 0.0, "sa": 0.0, "sc": 0.0, "perf": 1.0}),
        ("L9_C10_w10",    5, 10.0, None),
    ]:
        lt = _time.time()
        smiles = sample_lane(lane_name, seed, cfg_w, gscales)
        all_results[lane_name] = smiles
        print(f"[m10] {lane_name} done in {_time.time()-lt:.0f}s, {len(smiles)} SMILES", flush=True)

    elapsed = _time.time() - t0
    summary = {
        "n_lanes": len(all_results),
        "lane_counts": {k: len(v) for k, v in all_results.items()},
        "n_raw_total": sum(len(v) for v in all_results.values()),
        "elapsed_s": round(elapsed, 1),
        "pool_per_run": pool_per_run,
    }
    print(f"[m10] All lanes done in {elapsed:.0f}s. Total raw: {summary['n_raw_total']}", flush=True)
    print("=== DONE ===", flush=True)
    return {"summary": summary, "lane_smiles": all_results}


# ---------------------------------------------------------------------------
# Local entrypoint: run + save results
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("[local] Launching M10 200k 10-lane run on Modal A10G...", flush=True)
    t0 = time.time()

    result = run_all_lanes.remote(pool_per_run=10_000)
    elapsed = time.time() - t0
    print(f"[local] Remote returned in {elapsed:.0f}s", flush=True)

    summary = result["summary"]
    lane_smiles = result["lane_smiles"]

    # Save per-lane raw SMILES (gitignored via *.txt pattern)
    for lane, smiles in lane_smiles.items():
        out = RESULTS / f"m10_{lane}.txt"
        out.write_text("\n".join(smiles), encoding="utf-8")
        print(f"[local] Saved {lane}: {len(smiles)} SMILES -> {out}", flush=True)

    summary["elapsed_total_s"] = round(elapsed, 1)
    (RESULTS / "m10_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[local] Summary -> {RESULTS / 'm10_summary.json'}", flush=True)

    print("\n[local] === M10 RESULTS ===")
    print(f"  Total raw SMILES : {summary['n_raw_total']:,}")
    print(f"  Lanes            : {summary['n_lanes']}")
    for k, v in summary["lane_counts"].items():
        print(f"    {k}: {v}")
    print(f"  Elapsed          : {elapsed:.0f}s")
    print(f"\nNext step: run m10_post.py to filter, UniMol-score, and rank top candidates.")
