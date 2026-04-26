"""Run the full diagnostic suite (D1 D2 D3 D5 D10) in one go.

Saves:
    docs/diag_d1.md  validator self-consistency
    docs/diag_d2.md  LIMO encoder-decoder roundtrip on high-D molecules
    docs/diag_d3.md  property predictability from latents
    docs/diag_d5.md  out-of-range conditioning (model-dependent)
    docs/diag_d10.md cond-signal correlation (model-dependent)
    docs/diag_summary.md  combined verdict + recommended v6 changes

Model-dependent diagnostics use --exp argument (default = v4-B).
"""
from __future__ import annotations
import argparse, sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

BASE = Path("E:/Projects/EnergeticDiffusion2")
sys.path.insert(0, str(BASE / "scripts/diffusion"))
sys.path.insert(0, str(BASE / "scripts/vae"))


# ─── helpers ────────────────────────────────────────────────────────────────
def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def tanimoto(s1, s2):
    m1 = Chem.MolFromSmiles(s1); m2 = Chem.MolFromSmiles(s2)
    if not m1 or not m2: return 0.0
    f1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 2048)
    f2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 2048)
    return DataStructs.TanimotoSimilarity(f1, f2)


# ─── D1: validator self-consistency ─────────────────────────────────────────
def diag_D1(blob):
    print("\n=== D1: validator self-consistency ===")
    from unimol_validator import UniMolValidator
    smiles = blob["smiles"]
    raw = blob["values_raw"].numpy()
    cv = blob["cond_valid"].numpy().astype(bool)
    cw = blob["cond_weight"].numpy()
    prop_names = blob["property_names"]
    # Tier-A/B rows only, max 1500 to keep cost bounded
    md = ["# D1: Validator self-consistency (3DCNN smoke ensemble)", "",
          "Validator runs on Tier-A/B SMILES whose values are *known*. Compare"
          " predicted to ground truth.", "",
          "| Property | n | MAE | rel_MAE % | r | verdict |",
          "|---|---|---|---|---|---|"]
    name_map = {"density": "density", "heat_of_formation": "HOF_S",
                "detonation_velocity": "DetoD", "detonation_pressure": "DetoP"}
    PROP_NAMES_3DCNN = ["density","DetoD","DetoP","DetoQ","DetoT","DetoV",
                         "HOF_S","BDE"]
    print("Loading validator …")
    val = UniMolValidator(model_dir=str(
        BASE / "data/raw/energetic_external/EMDP/Data/smoke_model"))
    for j, p in enumerate(prop_names):
        trusted = cv[:, j] & (cw[:, j] >= 0.99)
        idx = np.where(trusted)[0]
        if len(idx) > 1500:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, 1500, replace=False)
        smi_list = [smiles[i] for i in idx]
        truth = raw[idx, j]
        print(f"  {p}: predicting {len(smi_list):,} SMILES …", flush=True)
        try:
            preds_dict = val.predict(smi_list)
        except Exception as e:
            print(f"    failed: {e}")
            md.append(f"| {p} | err | err | err | err | error |")
            continue
        # preds_dict has keys = PROP_NAMES_3DCNN
        col = preds_dict.get(name_map[p])
        if col is None:
            md.append(f"| {p} | – | – | – | – | n/a |")
            continue
        pv = np.asarray(col)
        mask = ~(np.isnan(pv) | np.isnan(truth))
        if mask.sum() < 10:
            md.append(f"| {p} | {int(mask.sum())} | – | – | – | too few |"); continue
        truth_v = truth[mask]; pv = pv[mask]
        mae = float(np.mean(np.abs(truth_v - pv)))
        rel = 100 * mae / max(abs(truth_v.mean()), 1e-6)
        r = float(np.corrcoef(truth_v, pv)[0, 1])
        sd = float(truth_v.std())
        rel_to_std = 100 * mae / max(sd, 1e-6)
        verdict = ("strong" if rel_to_std < 25 else
                   "ok" if rel_to_std < 50 else "weak")
        md.append(f"| {p} | {len(truth_v)} | {mae:.3f} | {rel:.1f} % | "
                  f"{r:.3f} | **{verdict}** (MAE/std={rel_to_std:.0f}%) |")
    md += ["",
           "verdict: strong = validator < 25 % of std error; ok = 25–50 %; "
           "weak = > 50 % (unreliable as ground truth)"]
    (BASE / "docs/diag_d1.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  saved docs/diag_d1.md")


# ─── D2: encoder-decoder roundtrip on high-property molecules ───────────────
def diag_D2(blob, device):
    print("\n=== D2: LIMO encoder-decoder roundtrip on high-D molecules ===")
    from limo_model import (LIMOVAE, SELFIESTokenizer, load_vocab,
                              build_limo_vocab, save_vocab, LIMO_MAX_LEN,
                              find_limo_repo)
    smiles = blob["smiles"]
    raw = blob["values_raw"].numpy()
    cv = blob["cond_valid"].numpy().astype(bool)
    cw = blob["cond_weight"].numpy()
    prop_names = blob["property_names"]
    # Pick top-50 high-D rows from Tier-A/B
    j_d = prop_names.index("detonation_velocity")
    trusted = cv[:, j_d] & (cw[:, j_d] >= 0.99)
    idx = np.where(trusted)[0]
    order = np.argsort(-raw[idx, j_d])
    pick = idx[order[:50]]
    smi_pick = [smiles[i] for i in pick]
    truth_d = raw[pick, j_d]
    print(f"  picked {len(pick)} high-D Tier-A/B mols, D range {truth_d.min():.2f}-{truth_d.max():.2f}")

    limo_dir = find_limo_repo(BASE)
    vocab_cache = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vocab_cache) if vocab_cache.exists() \
               else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vocab_cache.exists(): save_vocab(alphabet, vocab_cache)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    ckpt_limo = Path(blob["meta"]["checkpoint"])
    if not ckpt_limo.is_absolute(): ckpt_limo = BASE / ckpt_limo
    limo_blob = torch.load(ckpt_limo, map_location=device, weights_only=False)
    limo = LIMOVAE()
    limo.load_state_dict(limo_blob["model_state"])
    limo.to(device).eval()

    exact = 0; tan_sum = 0.0; nrg_keep = 0; tan_n = 0
    bad_examples = []
    for i, smi in enumerate(smi_pick):
        res = tok.smiles_to_tensor(smi)
        if res is None:
            bad_examples.append((smi, "smiles_to_tensor returned None")); continue
        x_one, _ = res
        x = x_one.unsqueeze(0).to(device)
        with torch.no_grad():
            _, mu, _ = limo.encode(x)
            dec_logits = limo.decode(mu)
        toks = dec_logits.argmax(-1)[0].cpu().tolist()
        try:
            dec_smi = tok.indices_to_smiles(toks)
        except Exception as e:
            bad_examples.append((smi, f"decode err: {e}")); continue
        c0 = canon(smi); c1 = canon(dec_smi)
        if c0 and c1 and c0 == c1:
            exact += 1
        if c0 and c1:
            t = tanimoto(c0, c1)
            tan_sum += t; tan_n += 1
            from rdkit.Chem import Mol
            m1 = Chem.MolFromSmiles(c1)
            if m1 and m1.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")):
                nrg_keep += 1
        elif len(bad_examples) < 5:
            bad_examples.append((smi, dec_smi))
    n = len(smi_pick)
    pct_exact = 100 * exact / n
    avg_tan = tan_sum / max(tan_n, 1)
    pct_nrg = 100 * nrg_keep / n
    verdict = ("strong" if pct_exact >= 50 else
               "ok" if pct_exact >= 20 or avg_tan >= 0.7 else "weak")
    md = ["# D2: LIMO encoder–decoder roundtrip on high-D molecules", "",
          f"50 highest-D Tier-A/B SMILES → encode → argmax-decode → canonicalise.",
          "",
          f"- exact recovery: **{exact}/{n}** ({pct_exact:.0f} %)",
          f"- mean Tanimoto when both decode: **{avg_tan:.3f}** (n={tan_n})",
          f"- decoded SMILES still containing NO2: **{nrg_keep}/{n}** ({pct_nrg:.0f} %)",
          f"- verdict: **{verdict}**",
          "", "## Failure examples (truth → decoded)",
          ""]
    for s_orig, s_dec in bad_examples[:5]:
        md.append(f"- `{s_orig}` → `{s_dec}`")
    (BASE / "docs/diag_d2.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  exact={exact}/{n} avg_tan={avg_tan:.3f} nrg_keep={nrg_keep}/{n}  ({verdict})")


# ─── D3: property predictability from latents ───────────────────────────────
class Reg(nn.Module):
    def __init__(self, d=1024, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Linear(h, h), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def diag_D3(blob, device):
    print("\n=== D3: property predictability from latents ===")
    z = blob["z_mu"].float()
    raw = blob["values_raw"].float()
    cv = blob["cond_valid"]; cw = blob["cond_weight"]
    rng = np.random.default_rng(42)
    md = ["# D3: Property predictability from LIMO latents", "",
          "MLP 1024→512→512→1, 80/20 split on Tier-A/B latents.", "",
          "| Property | n_train | n_test | r | MAE | rel_MAE % | verdict |",
          "|---|---|---|---|---|---|---|"]
    for j, p in enumerate(blob["property_names"]):
        trusted = cv[:, j] & (cw[:, j] >= 0.99)
        idx_all = np.where(trusted.numpy())[0]
        if len(idx_all) < 200:
            md.append(f"| {p} | {len(idx_all)} | – | – | – | – | too few |"); continue
        rng.shuffle(idx_all)
        n_tr = int(len(idx_all) * 0.8)
        tr_idx = idx_all[:n_tr]; te_idx = idx_all[n_tr:]
        x_tr = z[tr_idx].to(device); x_te = z[te_idx].to(device)
        y_tr = raw[tr_idx, j].to(device); y_te = raw[te_idx, j].to(device)
        mu = y_tr.mean(); sd = y_tr.std() + 1e-6
        y_trn = (y_tr - mu) / sd
        m = Reg().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
        n_steps = 1500 if len(tr_idx) > 5000 else 700
        bs = 256
        for s in range(n_steps):
            i = torch.randint(0, len(x_tr), (bs,), device=device)
            loss = ((m(x_tr[i]) - y_trn[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pte = m(x_te) * sd + mu
            mae = (pte - y_te).abs().mean().item()
            yv = y_te.cpu().numpy(); pv = pte.cpu().numpy()
            r = float(np.corrcoef(yv, pv)[0, 1])
            rel = 100 * mae / max(abs(yv.mean()), 1e-6)
        thr = 0.85 if p != "heat_of_formation" else 0.7
        verdict = ("strong" if r >= thr else "ok" if r >= 0.5 else "weak")
        print(f"  {p}: r={r:.3f} MAE={mae:.3f} rel={rel:.1f}%  ({verdict})")
        md.append(f"| {p} | {len(tr_idx):,} | {len(te_idx):,} | {r:.3f} | "
                  f"{mae:.3f} | {rel:.1f} % | **{verdict}** |")
    (BASE / "docs/diag_d3.md").write_text("\n".join(md), encoding="utf-8")


# ─── D5: out-of-range conditioning (model-dependent) ────────────────────────
def diag_D5(blob, exp_dir, device):
    print(f"\n=== D5: out-of-range conditioning ({exp_dir.name}) ===")
    from model import ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample
    from limo_model import (LIMOVAE, SELFIESTokenizer, load_vocab,
                              build_limo_vocab, save_vocab, LIMO_MAX_LEN,
                              find_limo_repo)
    from unimol_validator import UniMolValidator
    ckpt_path = exp_dir / "checkpoints/best.pt"
    cb = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cb["config"]
    denoiser = ConditionalDenoiser(
        latent_dim=blob["z_mu"].shape[1],
        hidden=cfg["model"]["hidden"], n_blocks=cfg["model"]["n_blocks"],
        time_dim=cfg["model"]["time_dim"], prop_emb_dim=cfg["model"]["prop_emb_dim"],
        n_props=blob["values_raw"].shape[1],
        dropout=cfg["model"].get("dropout", 0.0)).to(device)
    denoiser.load_state_dict(cb["model_state"])
    if cb.get("ema_state") is not None:
        ema = EMA(denoiser, decay=cfg["training"]["ema_decay"])
        ema.load_state_dict(cb["ema_state"]); ema.apply_to(denoiser)
    denoiser.eval()
    schedule = NoiseSchedule(T=cfg["training"]["T"], device=device)

    limo_dir = find_limo_repo(BASE); vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() \
               else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    limo_b = torch.load(BASE / blob["meta"]["checkpoint"], map_location=device,
                         weights_only=False)
    limo = LIMOVAE(); limo.load_state_dict(limo_b["model_state"])
    limo.to(device).eval()

    val = UniMolValidator(model_dir=str(
        BASE / "data/raw/energetic_external/EMDP/Data/smoke_model"))

    md = ["# D5: out-of-range conditioning",
          f"Model: `{exp_dir.name}`", "",
          "Tests whether the model can extrapolate beyond q90 (z=+1.281).",
          "If predicted property at z=+3 ≈ z=+1.281, the model has saturated.",
          "",
          "| Property | target z | target raw | pred mean | pred max | rel_MAE % |",
          "|---|---|---|---|---|---|"]
    name_map = {"density":"density","heat_of_formation":"HOF_S",
                "detonation_velocity":"DetoD","detonation_pressure":"DetoP"}
    n_per = 30
    n_props = blob["values_raw"].shape[1]
    for j, p in enumerate(blob["property_names"]):
        st = blob["stats"][p]
        for q_label, qz in [("q90", 1.281), ("z=+2", 2.0), ("z=+3", 3.0)]:
            mask = torch.zeros(n_per, n_props, device=device)
            mask[:, j] = 1.0
            vals = torch.zeros(n_per, n_props, device=device)
            vals[:, j] = qz
            z = ddim_sample(denoiser, schedule, vals, mask, n_steps=40,
                              guidance_scale=7.0, device=device)
            with torch.no_grad():
                logits = limo.decode(z)
            toks = logits.argmax(-1).cpu().tolist()
            smiles = [tok.indices_to_smiles(t) for t in toks]
            valid = [canon(s) for s in smiles if s and canon(s)]
            if not valid:
                md.append(f"| {p} | {q_label} | – | (none valid) | – | – |"); continue
            pdict = val.predict(valid)
            col = pdict.get(name_map[p])
            if col is None:
                md.append(f"| {p} | {q_label} | – | n/a | – | – |"); continue
            pv = np.asarray(col); pv = pv[~np.isnan(pv)]
            tgt = qz * st["std"] + st["mean"]
            rel = 100 * float(np.mean(np.abs(pv - tgt))) / max(abs(tgt), 1e-6)
            md.append(f"| {p} | {q_label} | {tgt:+.2f} | {pv.mean():+.2f} | "
                      f"{pv.max():+.2f} | {rel:.1f} % |")
    md += ["", "If 'z=+3' pred mean is similar to q90 pred mean → saturated."]
    (BASE / "docs/diag_d5.md").write_text("\n".join(md), encoding="utf-8")


# ─── D10: cond signal correlation (does model attend to c?) ─────────────────
def diag_D10(blob, exp_dir, device):
    print(f"\n=== D10: cond signal correlation ({exp_dir.name}) ===")
    from model import ConditionalDenoiser, NoiseSchedule, EMA
    ckpt_path = exp_dir / "checkpoints/best.pt"
    cb = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cb["config"]
    denoiser = ConditionalDenoiser(
        latent_dim=blob["z_mu"].shape[1],
        hidden=cfg["model"]["hidden"], n_blocks=cfg["model"]["n_blocks"],
        time_dim=cfg["model"]["time_dim"], prop_emb_dim=cfg["model"]["prop_emb_dim"],
        n_props=blob["values_raw"].shape[1],
        dropout=cfg["model"].get("dropout", 0.0)).to(device)
    denoiser.load_state_dict(cb["model_state"])
    if cb.get("ema_state") is not None:
        ema = EMA(denoiser, decay=cfg["training"]["ema_decay"])
        ema.load_state_dict(cb["ema_state"]); ema.apply_to(denoiser)
    denoiser.eval()
    schedule = NoiseSchedule(T=cfg["training"]["T"], device=device)
    n_props = blob["values_raw"].shape[1]
    md = ["# D10: conditioning signal correlation",
          f"Model: `{exp_dir.name}`", "",
          "Same z_t,t fed three ways: (A) target_z=+1.281 for prop p, (B) target_z=-1.281, (C) all-zero unconditional. Measures cosine(eps_A − eps_C, eps_B − eps_C) — should be **negative** (opposite targets push eps in opposite directions). Cosine ≈ 0 means the model is ignoring conditioning.",
          "",
          "| Property | t=100 | t=500 | t=900 | mean | verdict |",
          "|---|---|---|---|---|---|"]
    n_samples = 100
    z_t = torch.randn(n_samples, blob["z_mu"].shape[1], device=device)
    for j, p in enumerate(blob["property_names"]):
        rows = []
        for t_val in [100, 500, 900]:
            t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
            mask = torch.zeros(n_samples, n_props, device=device); mask[:, j] = 1.0
            vA = torch.zeros(n_samples, n_props, device=device); vA[:, j] = +1.281
            vB = torch.zeros(n_samples, n_props, device=device); vB[:, j] = -1.281
            mZ = torch.zeros(n_samples, n_props, device=device)
            with torch.no_grad():
                eA = denoiser(z_t, t, vA, mask)
                eB = denoiser(z_t, t, vB, mask)
                eC = denoiser(z_t, t, mZ, mZ)
            dA = (eA - eC).flatten(1); dB = (eB - eC).flatten(1)
            cos = F.cosine_similarity(dA, dB, dim=-1).mean().item()
            rows.append(cos)
        mean_cos = float(np.mean(rows))
        verdict = ("strong" if mean_cos < -0.3 else
                   "weak" if mean_cos < 0.0 else "broken")
        md.append(f"| {p} | {rows[0]:+.3f} | {rows[1]:+.3f} | {rows[2]:+.3f} | "
                  f"{mean_cos:+.3f} | **{verdict}** |")
    md += ["", "verdict: **strong** if cosine < -0.3 (good signal); **weak** if -0.3..0; **broken** if positive."]
    (BASE / "docs/diag_d10.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  saved docs/diag_d10.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None,
                    help="Experiment dir for model-dependent diagnostics. "
                         "Default = newest v4b experiment.")
    ap.add_argument("--skip", default="",
                    help="Comma-separated diagnostics to skip, e.g. d2,d10")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    skip = set(s.strip().lower() for s in args.skip.split(",") if s.strip())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    blob = torch.load(BASE / "data/training/diffusion/latents_expanded.pt",
                       weights_only=False)

    if args.exp:
        exp_dir = Path(args.exp)
        if not exp_dir.is_absolute(): exp_dir = BASE / exp_dir
    else:
        exp_dirs = sorted((BASE / "experiments").glob(
            "diffusion_subset_cond_expanded_v4b_*"))
        exp_dir = exp_dirs[-1] if exp_dirs else None

    t0 = time.time()
    if "d1"  not in skip: diag_D1(blob)
    if "d2"  not in skip: diag_D2(blob, device)
    if "d3"  not in skip: diag_D3(blob, device)
    if "d5"  not in skip and exp_dir: diag_D5(blob, exp_dir, device)
    if "d10" not in skip and exp_dir: diag_D10(blob, exp_dir, device)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
