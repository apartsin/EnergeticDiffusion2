"""T3 multi-seed denoiser EVALUATION step.

Closes Reviewer 1's "denoiser seed-variance gap" by sampling at the production
Hz-C2 condition with the two RETRAINED v4b denoisers (seeds 1 and 2) and
fusing their pools with the existing v3 denoiser, then running the production
4-stage validation funnel and reporting top-1 (D, rho, P, composite, max-Tani).

The seed=42 row reuses the published m6_post.json/Table 5 production numbers,
since that exact denoiser is already the production v4b checkpoint
(`combo_bundle/v4b_best.pt`) and re-running it would produce the same value.

Usage:
    python -m modal run --detach t3_denoiser_seeds_bundle/modal_t3_eval.py
    python -m modal run t3_denoiser_seeds_bundle/modal_t3_eval.py::main --seeds 1

The local entrypoint computes the aggregate (mean +/- std with ddof=1) and
writes t3_eval_summary.json.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
PROJ = HERE.parent
RESULTS_LOCAL = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Source layout: combo_bundle for sampling, m6_postprocess_bundle for filter
# ---------------------------------------------------------------------------
COMBO   = PROJ / "combo_bundle"
M6      = PROJ / "m6_postprocess_bundle"
EXT     = PROJ / "external"

# Code files to embed in the image (small)
_SRC_FILES = [
    # samplers + denoiser model
    (COMBO / "limo_model.py",        "/app/limo_model.py"),
    (COMBO / "model.py",             "/app/model.py"),
    (COMBO / "guided_v2_sampler.py", "/app/guided_v2_sampler.py"),
    # postprocess pipeline
    (M6   / "chem_filter.py",        "/app/chem_filter.py"),
    (M6   / "chem_redflags.py",      "/app/chem_redflags.py"),
    (M6   / "unimol_validator.py",   "/app/unimol_validator.py"),
    # SA + SC scorer wrappers (sascorer reads fpscores.pkl.gz from its own dir)
    (EXT  / "LIMO" / "sascorer.py",      "/app/external/LIMO/sascorer.py"),
    (EXT  / "LIMO" / "fpscores.pkl.gz",  "/app/external/LIMO/fpscores.pkl.gz"),
    (EXT  / "scscore" / "scscore" / "__init__.py",
                                       "/app/external/scscore/scscore/__init__.py"),
    (EXT  / "scscore" / "scscore" / "standalone_model_numpy.py",
                                       "/app/external/scscore/scscore/standalone_model_numpy.py"),
    # Score model loader needs scripts/viability/train_multihead_latent.py
    (COMBO / "train_multihead_latent.py",
                                       "/app/scripts/viability/train_multihead_latent.py"),
    # vocab + meta (small)
    (COMBO / "vocab.json",           "/app/vocab.json"),
    (COMBO / "meta.json",            "/app/meta.json"),
]

# Heavy assets uploaded to dgld-t3-data volume (one-time)
_VOL_ASSETS = [
    (COMBO / "limo_best.pt",          "limo_best.pt"),
    (COMBO / "v3_best.pt",            "v3_best.pt"),
    (COMBO / "score_model_v3e.pt",    "score_model_v3e.pt"),
    (M6   / "labelled_master.csv",    "labelled_master.csv"),
    (EXT  / "scscore" / "models" / "full_reaxys_model_1024bool"
                       / "model.ckpt-10654.as_numpy.json.gz",
                                       "scscore_model.ckpt-10654.as_numpy.json.gz"),
]

# The 367 MB smoke_model directory is uploaded as individual files so we
# avoid a tarball round-trip.
SMOKE_DIR_LOCAL = M6 / "_smoke_model" / "smoke_model"
SMOKE_FILES = [
    "config.yaml",
    "cv.data",
    "metric.result",
    "model_0.pth",
    "model_1.pth",
    "target_scaler.ss",
    "train_set.sdf",
]


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
    .apt_install("libxrender1", "libxext6", "libsm6", "libgl1")
    .pip_install(
        "numpy<2",
        "pandas",
        "pyyaml>=6.0",
        "selfies",
        "rdkit-pypi",
        "scikit-learn",
        "six",   # SCScorer pure-numpy uses six
        "huggingface_hub",  # unimol-tools needs this to fetch backbone weights
    )
    .pip_install("unimol-tools")
)
# Only attach local files when running locally (PROJ exists with combo_bundle).
# Inside the remote container, this module is imported again but PROJ paths
# don't exist there; skip the file-existence check in that case.
if (PROJ / "combo_bundle").exists():
    for local_p, remote_p in _SRC_FILES:
        if not local_p.exists():
            raise FileNotFoundError(f"Missing src file: {local_p}")
        image = image.add_local_file(str(local_p), remote_p)

app = modal.App("dgld-t3-denoiser-eval", image=image)

# Volumes
data_vol    = modal.Volume.from_name("dgld-t3-data",    create_if_missing=True)
results_vol = modal.Volume.from_name("dgld-t3-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helper: full sample + validate funnel for one seed
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100",
    timeout=2 * 60 * 60,
    volumes={"/data": data_vol, "/results": results_vol},
)
def eval_seed_remote(seed: int, pool_per_run: int = 10000,
                      sample_seed: int = 0) -> dict:
    """Sample 10k from new v4b ckpt + 10k from existing v3 ckpt at Hz-C2 and
    run the §4.10 4-stage validation funnel. Returns top-1 metrics."""
    import os
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/external/LIMO")
    sys.path.insert(0, "/app/external/scscore")
    sys.path.insert(0, "/app/scripts/viability")

    import numpy as np
    import pandas as pd
    import torch

    print(f"[eval seed={seed}] GPU: {torch.cuda.get_device_name(0)}",
          flush=True)

    # -----------------------------------------------------------------
    # Local copies of paths
    # -----------------------------------------------------------------
    v4b_ckpt   = f"/results/denoiser_v4b_seed{seed}.pt"
    v3_ckpt    = "/data/v3_best.pt"
    limo_ckpt  = "/data/limo_best.pt"
    score_pt   = "/data/score_model_v3e.pt"
    meta_json  = "/app/meta.json"
    vocab_json = "/app/vocab.json"

    for p in [v4b_ckpt, v3_ckpt, limo_ckpt, score_pt, meta_json, vocab_json]:
        if not Path(p).exists():
            raise FileNotFoundError(f"missing: {p}")

    smoke_dir = "/data/smoke_model"
    if not Path(smoke_dir).exists():
        raise FileNotFoundError(f"smoke_model dir missing: {smoke_dir}")
    labelled_master = "/data/labelled_master.csv"

    # -----------------------------------------------------------------
    # 1. Load models
    # -----------------------------------------------------------------
    device = torch.device("cuda")
    meta = json.loads(Path(meta_json).read_text())
    pn = meta["property_names"]
    n_props = meta["n_props"]
    latent_dim = meta["latent_dim"]
    stats = meta["stats"]

    from limo_model import LIMOVAE, SELFIESTokenizer, load_vocab, LIMO_MAX_LEN
    from model import (ConditionalDenoiser, NoiseSchedule, EMA,
                       ddim_sample)
    from guided_v2_sampler import ddim_sample_guided_v2, load_score_model

    print(f"[eval seed={seed}] loading LIMO ...", flush=True)
    alphabet = load_vocab(Path(vocab_json))
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    limo_blob = torch.load(limo_ckpt, map_location=device, weights_only=False)
    limo = LIMOVAE().to(device)
    limo.load_state_dict(limo_blob["model_state"])
    limo.eval()
    for p in limo.parameters(): p.requires_grad_(False)

    def load_denoiser(ckpt_path, cfg):
        cb = torch.load(ckpt_path, map_location=device, weights_only=False)
        d = ConditionalDenoiser(latent_dim=latent_dim, hidden=cfg["hidden"],
                                  n_blocks=cfg["n_blocks"],
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
        return d, NoiseSchedule(T=cfg["T"], device=device)

    print(f"[eval seed={seed}] loading v4b seed{seed} + v3 ...", flush=True)
    d_v4b, sch_v4b = load_denoiser(v4b_ckpt, meta["v4b_cfg"])
    d_v3, sch_v3   = load_denoiser(v3_ckpt,  meta["v3_cfg"])
    score_model, _ = load_score_model(score_pt, device=str(device))

    # -----------------------------------------------------------------
    # 2. Sample 10k each at Hz-C2 (viab=1.0, sens=0.3, hazard=1.0)
    # -----------------------------------------------------------------
    target_raw = {"density": 1.95, "heat_of_formation": 220,
                  "detonation_velocity": 9.5, "detonation_pressure": 40}
    target_z = torch.tensor(
        [(target_raw[p] - stats[p]["mean"]) / stats[p]["std"] for p in pn],
        device=device,
    )
    gscales = {"viab": 1.0, "sens": 0.3, "hazard": 1.0, "sa": 0.0, "sc": 0.0}

    def sample_one(denoiser, sch, n_pool, gscales, ss):
        torch.manual_seed(ss); np.random.seed(ss)
        mask = torch.ones(n_pool, n_props, device=device)
        vals = torch.full((n_pool, n_props), 0.0, device=device)
        for j in range(target_z.shape[0]):
            vals[:, j] = target_z[j]
        z = ddim_sample_guided_v2(denoiser, sch, vals, mask, score_model,
                                    n_steps=40, cfg_scale=7.0,
                                    guidance_scales=gscales, device=device)
        with torch.no_grad():
            logits = limo.decode(z)
        toks = logits.argmax(-1).cpu().tolist()
        return [tok.indices_to_smiles(t) for t in toks]

    t0 = time.time()
    smis = []
    print(f"[eval seed={seed}] sampling v4b pool=10k ...", flush=True)
    smis += sample_one(d_v4b, sch_v4b, pool_per_run, gscales, sample_seed)
    print(f"[eval seed={seed}] sampling v3 pool=10k ...", flush=True)
    smis += sample_one(d_v3, sch_v3, pool_per_run, gscales, sample_seed + 10000)
    sample_elapsed = time.time() - t0
    print(f"[eval seed={seed}] sampled {len(smis)} SMILES in "
          f"{sample_elapsed:.0f}s",
          flush=True)

    # -----------------------------------------------------------------
    # 3. Validation funnel (mirrors m6_post.py)
    # -----------------------------------------------------------------
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else None

    def is_neutral(s):
        m = Chem.MolFromSmiles(s)
        if m is None: return False
        if Chem.GetFormalCharge(m) != 0: return False
        return not any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms())

    def morgan_fp(s):
        m = Chem.MolFromSmiles(s)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None

    canons = []
    for s in smis:
        c = canon(s)
        if c and is_neutral(c): canons.append(c)
    seen = {}
    for c in canons:
        if c not in seen: seen[c] = True
    smis = list(seen.keys())
    print(f"[eval seed={seed}] unique-neutral: {len(smis)}", flush=True)

    # 3DCNN
    from unimol_validator import UniMolValidator
    val = UniMolValidator(model_dir=smoke_dir)
    name_map = {"density": "density", "heat_of_formation": "HOF_S",
                "detonation_velocity": "DetoD",
                "detonation_pressure": "DetoP"}
    pdict = val.predict(smis)
    cols = {p: np.asarray(pdict.get(name_map[p]), dtype=float) for p in pn}
    keep = np.ones(len(smis), dtype=bool)
    for p in pn:
        keep &= ~np.isnan(cols[p])
    smis = [s for s, k in zip(smis, keep) if k]
    cols = {p: cols[p][keep] for p in pn}
    print(f"[eval seed={seed}] after 3DCNN: {len(smis)}", flush=True)
    if len(smis) == 0:
        return {"seed": seed, "n_validated": 0,
                "error": "3DCNN returned 0 valid predictions",
                "elapsed_s": round(time.time() - t0, 1)}

    # chem_filter
    from chem_filter import chem_filter_batch
    keep_idx, _ = chem_filter_batch(smis, cols, pn)
    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    smis = [smis[i] for i in keep_idx]
    cols = {p: cols[p][keep_idx_arr] for p in pn}

    # SA / SC: load locally without the heavy feasibility_utils paths
    sys.path.insert(0, "/app/external/LIMO")
    import sascorer

    sys.path.insert(0, "/app/external/scscore")
    from scscore.standalone_model_numpy import SCScorer
    sc_model = SCScorer()
    sc_model.restore("/data/scscore_model.ckpt-10654.as_numpy.json.gz")

    def real_sa(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: return float("nan")
            return float(sascorer.calculateScore(mol))
        except Exception:
            return float("nan")

    def real_sc(smi):
        try:
            _, val = sc_model.get_score_from_smi(smi)
            return float(val)
        except Exception:
            return float("nan")

    sa = np.array([real_sa(s) for s in smis])
    sc = np.array([real_sc(s) for s in smis])
    keep = np.ones(len(smis), dtype=bool)
    keep &= ~((~np.isnan(sa)) & (sa > 5.0))
    keep &= ~((~np.isnan(sc)) & (sc > 3.5))
    idx = np.where(keep)[0]
    smis = [smis[i] for i in idx]
    cols = {p: cols[p][idx] for p in pn}
    sa = sa[idx]; sc = sc[idx]

    # Tanimoto window
    lm = pd.read_csv(labelled_master, usecols=["smiles"], nrows=5000)
    train_smiles = lm["smiles"].tolist()
    ref_fps = [morgan_fp(s) for s in train_smiles if morgan_fp(s) is not None]

    max_tans = []
    for s in smis:
        fp = morgan_fp(s)
        if fp is None: max_tans.append(0.0); continue
        tans = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        max_tans.append(max(tans))
    max_tans = np.array(max_tans)
    keep = (max_tans >= 0.20) & (max_tans <= 0.55)
    idx = np.where(keep)[0]
    smis = [smis[i] for i in idx]
    cols = {p: cols[p][idx] for p in pn}
    sa = sa[idx]; sc = sc[idx]; max_tans = max_tans[idx]
    print(f"[eval seed={seed}] Phase-A: {len(smis)}", flush=True)

    if len(smis) < 1:
        return {"seed": seed, "n_validated": 0,
                "error": "phase-A funnel produced 0 candidates",
                "elapsed_s": round(time.time() - t0, 1)}

    # composite + penalty
    SA_PENALTY_THRESHOLD = 4.0
    SC_PENALTY_THRESHOLD = 3.0

    def feas_pen(s_a, s_c, w_sa=0.5, w_sc=0.25):
        pen = 0.0
        if s_a == s_a:
            pen += w_sa * max(0.0, s_a - SA_PENALTY_THRESHOLD)
        if s_c == s_c:
            pen += w_sc * max(0.0, s_c - SC_PENALTY_THRESHOLD)
        return pen

    comp = np.zeros(len(smis))
    for p in pn:
        comp += np.abs(cols[p] - target_raw[p]) / max(stats[p]["std"], 1e-6)
    pen = np.array([feas_pen(sa[k], sc[k], 0.5, 0.25)
                    for k in range(len(smis))])
    comp += pen
    order = np.argsort(comp)
    top1_idx = int(order[0])

    # Top-1 metrics
    top1 = {
        "seed": int(seed),
        "n_validated": int(len(smis)),
        "top1_smiles": smis[top1_idx],
        "top1_composite": float(comp[top1_idx]),
        "top1_D": float(cols["detonation_velocity"][top1_idx]),
        "top1_P": float(cols["detonation_pressure"][top1_idx]),
        "top1_rho": float(cols["density"][top1_idx]),
        "top1_max_tanimoto": float(max_tans[top1_idx]),
        "elapsed_s": round(time.time() - t0, 1),
        "sample_elapsed_s": round(sample_elapsed, 1),
    }

    # Persist a per-seed json under /results/eval/
    out_dir = Path("/results/eval"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"t3_eval_seed{seed}.json"
    out_path.write_text(json.dumps(top1, indent=2))
    results_vol.commit()
    print(f"[eval seed={seed}] -> {out_path}", flush=True)
    return top1


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    seeds: str = "1,2",
    pool_per_run: int = 10000,
    sample_seed: int = 0,
    skip_upload: bool = False,
):
    """Run Hz-C2 sampling + validation funnel at each comma-separated seed.

    seed=42 is NOT included by default since its number is reused from
    the published m6_post.json / Table 5 production run.
    """
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    print(f"[eval] seeds = {seed_list}  pool_per_run = {pool_per_run}",
          flush=True)

    # ---- 1. Ensure assets are uploaded to data_vol ------------------------
    if not skip_upload:
        existing: set[str] = set()
        try:
            for entry in data_vol.listdir("/"):
                existing.add(Path(entry.path).name)
        except Exception as exc:
            print(f"[eval] could not list data_vol ({exc})", flush=True)

        # Top-level assets
        with data_vol.batch_upload() as batch:
            for local_p, remote_name in _VOL_ASSETS:
                if remote_name in existing:
                    continue
                if not local_p.exists():
                    raise FileNotFoundError(f"Missing asset: {local_p}")
                size_mb = local_p.stat().st_size / 1024 / 1024
                print(f"[eval] uploading {remote_name} ({size_mb:.1f} MB) ...",
                      flush=True)
                batch.put_file(str(local_p), remote_name)

        # smoke_model directory
        smoke_existing: set[str] = set()
        try:
            for entry in data_vol.listdir("/smoke_model"):
                smoke_existing.add(Path(entry.path).name)
        except Exception:
            pass
        with data_vol.batch_upload() as batch:
            for f in SMOKE_FILES:
                if f in smoke_existing:
                    continue
                lp = SMOKE_DIR_LOCAL / f
                if not lp.exists():
                    raise FileNotFoundError(f"Missing smoke file: {lp}")
                size_mb = lp.stat().st_size / 1024 / 1024
                print(f"[eval] uploading smoke_model/{f} ({size_mb:.1f} MB) ...",
                      flush=True)
                batch.put_file(str(lp), f"smoke_model/{f}")

    # ---- 2. Dispatch each seed sequentially -------------------------------
    summaries = []
    for s in seed_list:
        print(f"\n[eval] dispatching seed={s}", flush=True)
        try:
            r = eval_seed_remote.remote(seed=s,
                                          pool_per_run=pool_per_run,
                                          sample_seed=sample_seed)
        except Exception as exc:
            r = {"seed": s, "error": str(exc)}
        summaries.append(r)
        out = RESULTS_LOCAL / f"t3_eval_seed{s}_summary.json"
        out.write_text(json.dumps(r, indent=2))
        print(f"[eval] seed={s} -> {out}", flush=True)

    full = RESULTS_LOCAL / "t3_eval_remote_summary.json"
    full.write_text(json.dumps(summaries, indent=2))
    print(f"\n[eval] remote summaries -> {full}", flush=True)

    # The aggregation into t3_eval_summary.json (with seed=42 row pre-filled)
    # is performed by the bundled aggregator script after this entrypoint
    # exits, so the local Python process can compute mean/std with the seed=42
    # production numbers from the paper.
