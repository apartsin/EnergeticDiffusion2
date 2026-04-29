"""Comprehensive diagnostic for the M1 null-result root cause.

Runs 7 tests on the local GPU using fp16 ckpts (since fp32 had S3 issues
and fp16 was confirmed working on runpod):

D1: per-step gradient norm trace
D2: score-model gradient sanity (output differs with input)
D3: final z divergence (cos similarity across guidance conditions)
D4: LIMO decoder distribution entropy
D5: sign convention (guidance increases viab_logit)
D6: per-row vs batch gradient
D7: RNG state isolation
D8: argmax basin width

Output: experiments/diag_guidance.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4b_ckpt", default="v4b_best_fp16.pt")
    ap.add_argument("--limo_ckpt", default="limo_best_fp16.pt")
    ap.add_argument("--score_model", default="score_model_v3f.pt")
    ap.add_argument("--meta_json", default="meta.json")
    ap.add_argument("--vocab_json", default="vocab.json")
    ap.add_argument("--n", type=int, default=8, help="batch size for diagnostics")
    ap.add_argument("--out", default="../experiments/diag_guidance.json")
    args = ap.parse_args()
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

    from limo_model import LIMOVAE, SELFIESTokenizer, load_vocab, LIMO_MAX_LEN
    from model import ConditionalDenoiser, NoiseSchedule, EMA, ddim_sample
    from guided_v2_sampler import _guidance_grad, ddim_sample_guided_v2, load_score_model

    meta = json.loads(Path(args.meta_json).read_text())
    pn = meta["property_names"]; n_props = meta["n_props"]; latent_dim = meta["latent_dim"]
    stats = meta["stats"]

    print("Loading score model + LIMO + denoiser ..."); sys.stdout.flush()
    sm, _ = load_score_model(args.score_model, device=device)
    alphabet = load_vocab(Path(args.vocab_json))
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    limo_blob = torch.load(args.limo_ckpt, map_location=device, weights_only=False)
    limo = LIMOVAE().to(device)
    sd = limo_blob["model_state"]
    sd = {k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v
          for k, v in sd.items()}
    limo.load_state_dict(sd); limo.eval()
    cb = torch.load(args.v4b_ckpt, map_location=device, weights_only=False)
    cfg = meta["v4b_cfg"]
    sd = cb["model_state"]
    sd = {k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v
          for k, v in sd.items()}
    d = ConditionalDenoiser(latent_dim=latent_dim, hidden=cfg["hidden"],
                              n_blocks=cfg["n_blocks"], time_dim=cfg["time_dim"],
                              prop_emb_dim=cfg["prop_emb_dim"], n_props=n_props,
                              dropout=0).to(device)
    d.load_state_dict(sd)
    if cb.get("ema_state") is not None:
        ema_sd = {k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v
                  for k, v in cb["ema_state"].items()}
        ema = EMA(d, decay=cfg["ema_decay"])
        try:
            ema.load_state_dict(ema_sd); ema.apply_to(d)
        except Exception as e:
            print(f"  ema apply failed: {e}; continuing with raw weights"); sys.stdout.flush()
    d.eval()
    sch = NoiseSchedule(T=cfg["T"], device=device)

    target_raw = {"density": 1.95, "heat_of_formation": 220, "detonation_velocity": 9.5,
                   "detonation_pressure": 40}
    target_z = torch.tensor([(target_raw[p] - stats[p]["mean"]) / stats[p]["std"]
                              for p in pn], device=device)

    n = args.n
    mask = torch.ones(n, n_props, device=device)
    vals = torch.zeros(n, n_props, device=device)
    for j in range(target_z.shape[0]):
        vals[:, j] = target_z[j]
    cfg_drop = torch.zeros_like(mask)

    diag = {}

    # ── D2: score-model gradient sanity ─────────────────────────────────
    print("\n[D2] Score-model gradient sanity ..."); sys.stdout.flush()
    z_a = torch.randn(n, latent_dim, device=device)
    z_b = torch.randn(n, latent_dim, device=device)
    sigma = torch.full((n,), 0.5, device=device)
    with torch.no_grad():
        out_a = sm(z_a, sigma); out_b = sm(z_b, sigma)
    diag["D2"] = {
        "viab_logit_a_mean": float(out_a["viab_logit"].mean()),
        "viab_logit_b_mean": float(out_b["viab_logit"].mean()),
        "logit_differs": float((out_a["viab_logit"] - out_b["viab_logit"]).abs().mean()),
    }
    if "hazard_logit" in out_a:
        diag["D2"]["hazard_logit_differs"] = float(
            (out_a["hazard_logit"] - out_b["hazard_logit"]).abs().mean())
    print(f"  viab_logit |a-b|.mean = {diag['D2']['logit_differs']:.4f}")
    if diag["D2"]["logit_differs"] < 1e-6:
        print("  WARNING: score model output is approximately constant w.r.t. z!")

    # Check gradient sanity
    z_g = z_a.detach().clone().requires_grad_(True)
    out_g = sm(z_g, sigma)
    loss_v = F.softplus(-out_g["viab_logit"]).sum()
    g_v = torch.autograd.grad(loss_v, z_g, retain_graph=True)[0]
    diag["D2"]["viab_grad_norm_per_row"] = g_v.norm(dim=-1).tolist()
    print(f"  viab grad norm per row: {[f'{x:.3f}' for x in diag['D2']['viab_grad_norm_per_row'][:4]]}...")

    # ── D5: sign convention via finite-difference ───────────────────────
    print("\n[D5] Sign convention check ..."); sys.stdout.flush()
    eps_step = 0.5
    with torch.no_grad():
        z_eta_p = z_g.detach() - eps_step * g_v / g_v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        out_p = sm(z_eta_p, sigma)
    delta_logit = (out_p["viab_logit"] - out_g["viab_logit"]).mean().item()
    diag["D5"] = {
        "delta_logit_after_descending_loss": delta_logit,
        "expected_sign": "positive (descending loss = ascending logit)",
        "actual_sign": "POSITIVE (correct)" if delta_logit > 0 else "NEGATIVE (BUG: sign flipped!)",
    }
    print(f"  delta_viab_logit after step in -grad direction: {delta_logit:+.4f}")
    print(f"  expected: positive (loss descent = logit ascent)")

    # ── D6: per-row vs batch gradient ───────────────────────────────────
    print("\n[D6] Per-row gradient independence ..."); sys.stdout.flush()
    z_p = z_a.detach().clone().requires_grad_(True)
    out_p = sm(z_p, sigma)
    # Take grad of a single row's output
    loss_single = out_p["viab_logit"][0]
    g_single = torch.autograd.grad(loss_single, z_p, retain_graph=True)[0]
    nonzero_rows = (g_single.norm(dim=-1) > 1e-6).sum().item()
    diag["D6"] = {
        "single_row_loss_grad_nonzero_rows": int(nonzero_rows),
        "expected_nonzero_rows": 1,
        "verdict": "OK (per-row independent)" if nonzero_rows == 1 else f"BUG: {nonzero_rows} rows have nonzero grad (should be 1)",
    }
    print(f"  loss=row[0] only -> nonzero grad on {nonzero_rows} rows (expect 1)")

    # ── D4: LIMO decoder entropy ────────────────────────────────────────
    print("\n[D4] LIMO decoder entropy ..."); sys.stdout.flush()
    z_dec = torch.randn(n, latent_dim, device=device)
    with torch.no_grad():
        logits = limo.decode(z_dec)
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs.clamp(min=1e-10))).sum(-1)  # (n, 72)
    diag["D4"] = {
        "entropy_mean_nats": float(ent.mean()),
        "entropy_max_nats": float(ent.max()),
        "vocab_size": logits.shape[-1],
        "uniform_baseline_nats": float(np.log(logits.shape[-1])),
        "max_token_prob_mean": float(probs.max(-1).values.mean()),
    }
    print(f"  per-token entropy: mean={diag['D4']['entropy_mean_nats']:.3f} max={diag['D4']['entropy_max_nats']:.3f} nats")
    print(f"  uniform baseline: {diag['D4']['uniform_baseline_nats']:.3f} nats (vocab={logits.shape[-1]})")
    print(f"  mean argmax-prob: {diag['D4']['max_token_prob_mean']:.3f}")

    # ── D7: RNG state isolation ────────────────────────────────────────
    print("\n[D7] RNG state isolation ..."); sys.stdout.flush()
    torch.manual_seed(0); a1 = torch.randn(5, device=device)
    torch.manual_seed(0); a2 = torch.randn(5, device=device)
    diag["D7"] = {
        "same_seed_same_output": bool((a1 == a2).all().item()),
        "verdict": "RNG is reset on manual_seed (expected behavior; means seed-fixing pins z₀)",
    }

    # ── D1: per-step gradient norm trace + D3: z divergence ──────────────
    print("\n[D1+D3] Per-step grad trace + z divergence ..."); sys.stdout.flush()
    conds = [
        ("C0_unguided", None),
        ("C1_default",  {"viab": 1.0, "sens": 0.3, "hazard": 0.0, "sa": 0.0, "sc": 0.0}),
        ("C2_high",     {"viab": 5.0, "sens": 2.0, "hazard": 1.0, "sa": 0.0, "sc": 0.0}),
        ("C3_extreme",  {"viab": 50.0, "sens": 20.0, "hazard": 20.0, "sa": 0.0, "sc": 0.0}),
    ]

    final_zs = {}
    grad_traces = {}
    for cond_name, gscales in conds:
        torch.manual_seed(0)
        z = torch.randn(n, latent_dim, device=device)
        ts = torch.linspace(sch.T - 1, 0, 41, device=device).long()
        per_step_norms = []
        for i in range(40):
            t_now = ts[i]; t_next = ts[i + 1]
            t_b = torch.full((n,), int(t_now), device=device, dtype=torch.long)
            with torch.no_grad():
                e_cond = d(z, t_b, vals, mask)
                e_null = d(z, t_b, vals, cfg_drop)
                eps = e_null + 7.0 * (e_cond - e_null)
            ab_now = sch.alpha_bar[t_now]
            ab_next = sch.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
            sigma_t = (1 - ab_now).sqrt() * 2.0
            if gscales is not None:
                grad = _guidance_grad(sm, z, sigma_t, gscales)
                per_step_norms.append(float(grad.norm(dim=-1).mean()))
                eps = eps + (1 - ab_now).sqrt() * grad
            else:
                per_step_norms.append(0.0)
            z0_pred = (z - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()
            z = ab_next.sqrt() * z0_pred + (1 - ab_next).sqrt() * eps
        final_zs[cond_name] = z.detach()
        grad_traces[cond_name] = per_step_norms
        print(f"  {cond_name}: ||z||={z.norm(dim=-1).mean():.2f}  "
              f"max grad_norm = {max(per_step_norms) if per_step_norms else 0:.2f}  "
              f"mean grad_norm = {np.mean(per_step_norms):.2f}")

    # D3 cosine similarity
    z0 = final_zs["C0_unguided"]
    diag["D3"] = {}
    for k, z in final_zs.items():
        if k == "C0_unguided": continue
        cos = F.cosine_similarity(z0, z).mean().item()
        l2 = (z - z0).norm(dim=-1).mean().item()
        diag["D3"][k] = {"cos_to_unguided": cos, "l2_to_unguided": l2}
        print(f"  cos(z, z_C0) for {k}: {cos:.5f}  ||z-z0||={l2:.3f}")
    diag["D1_grad_traces"] = grad_traces

    # ── D8: argmax basin width ─────────────────────────────────────────
    print("\n[D8] argmax basin width ..."); sys.stdout.flush()
    z_ref = torch.randn(n, latent_dim, device=device)
    with torch.no_grad():
        logits_ref = limo.decode(z_ref)
        argmax_ref = logits_ref.argmax(-1)
    deltas = []
    for delta_norm in [0.1, 0.3, 1.0, 3.0, 10.0]:
        with torch.no_grad():
            noise = torch.randn(n, latent_dim, device=device)
            noise = noise / noise.norm(dim=-1, keepdim=True) * delta_norm
            z_pert = z_ref + noise
            logits_pert = limo.decode(z_pert)
            argmax_pert = logits_pert.argmax(-1)
            n_changed = (argmax_pert != argmax_ref).float().sum(-1).mean().item()
        deltas.append({"delta_norm": delta_norm, "tokens_changed": n_changed})
        print(f"  ||dz||={delta_norm:>5.1f}  -> tokens changed (mean): {n_changed:.2f} / 72")
    diag["D8_argmax_basin"] = deltas

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(diag, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
