"""Build the vast.ai bundle for m1_sweep.py.

Outputs to ./m1_bundle/:
    m1_sweep.py
    model.py                   # diffusion ConditionalDenoiser + ddim_sample
    limo_model.py              # LIMO VAE
    guided_v2_sampler.py       # guided DDIM
    train_multihead_latent.py  # MultiHeadScoreModel (loaded by load_score_model)
    vocab.json                 # LIMO alphabet
    meta.json                  # property_names, n_props, latent_dim, stats, cfgs
    v4b_best.pt                # 682 MB
    v3_best.pt                 # 682 MB
    limo_best.pt               # 388 MB
    score_model_v3e.pt         # 30 MB
    requirements.txt
    run.sh                     # entrypoint

Usage:
    /c/Python314/python scripts/diffusion/m1_bundle.py
    cd m1_bundle/
    python m1_sweep.py --... --pool_per_run 200   # smoke test locally
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC_DIFF = ROOT / "scripts" / "diffusion"
SRC_VAE = ROOT / "scripts" / "vae"
SRC_VIAB = ROOT / "scripts" / "viability"
LIMO_FT = ROOT / "experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt"
V3_BEST = ROOT / "experiments/diffusion_subset_cond_expanded_v3_20260425T140941Z/checkpoints/best.pt"
V4B_BEST = ROOT / "experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/checkpoints/best.pt"
SCORE_V3E = ROOT / "experiments/score_model_v3e/model.pt"
LATENTS = ROOT / "data/training/diffusion/latents.pt"
LIMO_REPO_VOCAB = ROOT / "external/limo/vocab_cache.json"

OUT = ROOT / "m1_bundle"


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Building bundle at {OUT}")

    # ── 1. Source files (small) ────────────────────────────────────────────
    files = [
        SRC_DIFF / "m1_sweep.py",
        SRC_DIFF / "model.py",
        SRC_DIFF / "guided_v2_sampler.py",
        SRC_VAE / "limo_model.py",
        SRC_VIAB / "train_multihead_latent.py",
    ]
    for f in files:
        if not f.exists():
            print(f"  MISSING: {f}")
            continue
        shutil.copy2(f, OUT / f.name)
        print(f"  + {f.name} ({f.stat().st_size//1024} KB)")

    # ── 2. Vocab ───────────────────────────────────────────────────────────
    vocab = None
    for cand in [LIMO_REPO_VOCAB,
                 ROOT / "external" / "limo" / "vocab_cache.json"]:
        if cand.exists():
            shutil.copy2(cand, OUT / "vocab.json")
            print(f"  + vocab.json from {cand}")
            vocab = cand
            break
    if vocab is None:
        # fall back to building it
        sys.path.insert(0, str(SRC_VAE))
        from limo_model import build_limo_vocab, save_vocab, find_limo_repo
        limo_dir = find_limo_repo(ROOT)
        alphabet = build_limo_vocab(limo_dir / "zinc250k.smi")
        save_vocab(alphabet, OUT / "vocab.json")
        print(f"  + vocab.json (built fresh)")

    # ── 3. Extract minimal metadata from latents.pt ────────────────────────
    print(f"  Loading latents stats from {LATENTS} ...")
    blob = torch.load(LATENTS, weights_only=False, map_location="cpu")
    pn = blob["property_names"]
    latent_dim = int(blob["z_mu"].shape[1])
    n_props = int(blob["values_raw"].shape[1])
    stats = blob["stats"]

    # ── 4. Extract denoiser configs from ckpts ─────────────────────────────
    def cfg_from(ckpt):
        cb = torch.load(ckpt, weights_only=False, map_location="cpu")
        c = cb["config"]
        return {
            "hidden": c["model"]["hidden"],
            "n_blocks": c["model"]["n_blocks"],
            "time_dim": c["model"]["time_dim"],
            "prop_emb_dim": c["model"]["prop_emb_dim"],
            "dropout": c["model"].get("dropout", 0.0),
            "T": c["training"]["T"],
            "ema_decay": c["training"]["ema_decay"],
        }

    v3_cfg = cfg_from(V3_BEST)
    v4b_cfg = cfg_from(V4B_BEST)

    meta = {
        "property_names": pn,
        "n_props": n_props,
        "latent_dim": latent_dim,
        "stats": stats,
        "v3_cfg": v3_cfg,
        "v4b_cfg": v4b_cfg,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  + meta.json: pn={pn}, latent={latent_dim}, n_props={n_props}")

    # ── 5. Heavy ckpts ────────────────────────────────────────────────────
    for src, dst in [(V4B_BEST, "v4b_best.pt"),
                      (V3_BEST, "v3_best.pt"),
                      (LIMO_FT, "limo_best.pt"),
                      (SCORE_V3E, "score_model_v3e.pt")]:
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"  + {dst} ({size_mb:.0f} MB) ... ", end="", flush=True)
        shutil.copy2(src, OUT / dst)
        print("done")

    # ── 6. requirements.txt ────────────────────────────────────────────────
    (OUT / "requirements.txt").write_text(
        "rdkit-pypi\nselfies\nnumpy\n"
    )

    # ── 7. run.sh ──────────────────────────────────────────────────────────
    (OUT / "run.sh").write_text(
        "#!/bin/bash\n"
        "set -e\n"
        "pip install rdkit-pypi selfies\n"
        "python -u m1_sweep.py \\\n"
        "  --v4b_ckpt v4b_best.pt --v3_ckpt v3_best.pt \\\n"
        "  --limo_ckpt limo_best.pt --score_model score_model_v3e.pt \\\n"
        "  --meta_json meta.json --vocab_json vocab.json \\\n"
        "  --pool_per_run ${POOL:-10000} --n_steps 40 --cfg_scale 7.0 \\\n"
        "  --seeds ${SEEDS:-0 1 2} \\\n"
        "  --target_density 1.95 --target_hof 220 --target_d 9.5 --target_p 40\n"
    )

    total_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / (1024**2)
    print(f"\nBundle complete: {OUT} ({total_mb:.0f} MB total)")


if __name__ == "__main__":
    main()
