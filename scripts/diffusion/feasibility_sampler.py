"""DDIM sampler with property-target + SA + SC classifier guidance.

Combines two gradient terms with the standard CFG-corrected ε prediction:
  - property head gradient (push z toward predicted-target)
  - SA + SC surrogate gradient (push z toward predicted-low scores)

All three gradients can be enabled / disabled independently.

Design mirrors `guided_sampler.py` so the existing rerank scripts can swap
implementations cleanly. By default no run actually happens — this module
is *implemented but not invoked* until the validation sweep is approved.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from feasibility_utils import (load_surrogate, DEFAULT_SA_CKPT, DEFAULT_SC_CKPT)


def _maybe_property_grad(z, heads_pkg, targets, norms):
    """Returns ∇_z [- Σ_p (pred_p − target_p)²]. None if disabled."""
    if heads_pkg is None or not targets: return None
    z_req = z.detach().requires_grad_(True)
    total = z_req.new_zeros(())
    for prop, target_raw in targets.items():
        if prop not in heads_pkg:
            continue
        head = heads_pkg[prop]["model"]
        mu, sd = heads_pkg[prop]["mu"], heads_pkg[prop]["sd"]
        pred_norm = head(z_req)
        target_norm = (target_raw - mu) / sd
        total = total - (pred_norm - target_norm) ** 2
    g, = torch.autograd.grad(total.sum(), z_req)
    return g


def _maybe_feasibility_grad(z, sa_pack, sc_pack,
                              lam_sa: float, lam_sc: float,
                              t_batch=None):
    """Returns ∇_z [-λ_SA · f_SA(z, t) − λ_SC · f_SC(z, t)].
    Surrogates may be time-conditional; if so they accept (z_t, t).
    """
    if sa_pack is None and sc_pack is None: return None
    if lam_sa <= 0 and lam_sc <= 0: return None
    z_req = z.detach().requires_grad_(True)
    total = z_req.new_zeros(())
    if sa_pack is not None and lam_sa > 0:
        m = sa_pack["model"]
        if sa_pack.get("time_aware") and t_batch is not None:
            sa_pred = m(z_req, t_batch)
        else:
            sa_pred = m(z_req)
        total = total - lam_sa * sa_pred.sum()
    if sc_pack is not None and lam_sc > 0:
        m = sc_pack["model"]
        if sc_pack.get("time_aware") and t_batch is not None:
            sc_pred = m(z_req, t_batch)
        else:
            sc_pred = m(z_req)
        total = total - lam_sc * sc_pred.sum()
    g, = torch.autograd.grad(total, z_req)
    return g


def ddim_sample_feasibility(
    denoiser,
    schedule,
    values_norm: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_steps: int = 40,
    guidance_scale: float = 7.0,
    cfg_dropout_mask: Optional[torch.Tensor] = None,
    device: str = "cuda",
    # property head guidance (existing pattern)
    property_heads: Optional[dict] = None,         # {prop: {model, mu, sd}}
    property_targets: Optional[dict] = None,       # {prop: target_raw}
    lambda_property: float = 0.0,
    # feasibility guidance
    sa_ckpt: Optional[str] = None,
    sc_ckpt: Optional[str] = None,
    lambda_sa: float = 0.0,
    lambda_sc: float = 0.0,
    feasibility_warmup_steps: int = 25,
    grad_clip: float = 1.0,
    # NEW: time-conditional feasibility surrogates (preferred)
    t_aware_ckpt: Optional[str] = None,            # path to property_heads_t.pt
    use_t_aware: bool = False,
) -> torch.Tensor:
    denoiser.eval()
    B = values_norm.shape[0]
    if cfg_dropout_mask is None:
        cfg_dropout_mask = torch.zeros_like(mask)

    # ── load surrogates (lazy) ──────────────────────────────────────────
    sa_pack = None
    sc_pack = None
    if use_t_aware and (lambda_sa > 0 or lambda_sc > 0):
        # Time-conditional path
        from train_t_aware_surrogates import TimeAwareScorePredictor
        path = t_aware_ckpt or "data/training/guidance/property_heads_t.pt"
        bundle = torch.load(path, map_location=device, weights_only=False)
        if lambda_sa > 0 and bundle.get("sa", {}).get("state_dict"):
            m = TimeAwareScorePredictor().to(device)
            m.load_state_dict(bundle["sa"]["state_dict"])
            m.eval()
            for prm in m.parameters(): prm.requires_grad_(False)
            sa_pack = {"model": m,
                        "stats": {"mean": bundle["sa"]["mu"], "std": bundle["sa"]["sd"]},
                        "time_aware": True}
        if lambda_sc > 0 and bundle.get("sc", {}).get("state_dict"):
            m = TimeAwareScorePredictor().to(device)
            m.load_state_dict(bundle["sc"]["state_dict"])
            m.eval()
            for prm in m.parameters(): prm.requires_grad_(False)
            sc_pack = {"model": m,
                        "stats": {"mean": bundle["sc"]["mu"], "std": bundle["sc"]["sd"]},
                        "time_aware": True}
    else:
        if lambda_sa > 0:
            loaded = load_surrogate(sa_ckpt or DEFAULT_SA_CKPT, device=device)
            if loaded is not None:
                sa_pack = {"model": loaded[0], "stats": loaded[1], "time_aware": False}
            else:
                print("[feasibility] SA surrogate disabled (load failed)")
        if lambda_sc > 0:
            loaded = load_surrogate(sc_ckpt or DEFAULT_SC_CKPT, device=device)
            if loaded is not None:
                sc_pack = {"model": loaded[0], "stats": loaded[1], "time_aware": False}
            else:
                print("[feasibility] SC surrogate disabled (load failed)")

    # NOTE: schedule.alpha_bar[T-1] = 0 with cosine schedule (slicing artefact).
    # Starting at T-1 makes z0_pred = z / sqrt(0) → inf. Use T-2 as the first
    # noise level (small but non-zero alpha_bar) so the gradient correction
    # actually has effect on z.
    start_t = max(schedule.T - 2, 1)
    ts = torch.linspace(start_t, 0, n_steps + 1, device=device).long()
    z = torch.randn(B, denoiser.latent_dim, device=device)

    for i in range(n_steps):
        t_now = ts[i]; t_next = ts[i + 1]
        t_batch = torch.full((B,), int(t_now), device=device, dtype=torch.long)

        with torch.no_grad():
            eps_cond = denoiser(z, t_batch, values_norm, mask)
            if guidance_scale != 1.0:
                eps_null = denoiser(z, t_batch, values_norm, cfg_dropout_mask)
                eps = eps_null + guidance_scale * (eps_cond - eps_null)
            else:
                eps = eps_cond

        ab_now = schedule.alpha_bar[t_now]
        sigma_t = (1 - ab_now).sqrt()

        # property gradient
        if lambda_property > 0 and i >= feasibility_warmup_steps:
            g = _maybe_property_grad(z, property_heads, property_targets,
                                       {p: (property_heads[p]["mu"],
                                             property_heads[p]["sd"])
                                        for p in (property_heads or {})})
            if g is not None:
                if grad_clip:
                    norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    g = g * (norm.clamp(max=grad_clip) / norm)
                eps = eps - sigma_t * lambda_property * g

        # feasibility gradient
        if (lambda_sa > 0 or lambda_sc > 0) and i >= feasibility_warmup_steps:
            g = _maybe_feasibility_grad(z, sa_pack, sc_pack, lambda_sa, lambda_sc,
                                          t_batch=t_batch)
            if g is not None:
                if grad_clip:
                    norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    g = g * (norm.clamp(max=grad_clip) / norm)
                # SA / SC objective is `-λ·f`, so to MINIMISE f we ascend on
                # the objective → eps_correction = +sigma·λ·g_objective
                # which is equivalent to subtracting σ·λ·∇f_score.
                eps = eps + sigma_t * g     # already includes the (−λ_SA, −λ_SC) sign

        ab_next = schedule.alpha_bar[t_next] if t_next > 0 \
                  else torch.tensor(1.0, device=device)
        # Stronger ab_now floor (0.01) keeps z0_pred ≤ ~10× z scale.
        ab_now_safe = ab_now.clamp(min=0.01)
        z0_pred = (z - (1 - ab_now_safe).sqrt() * eps) / ab_now_safe.sqrt()
        # Predicted-x0 dynamic threshold (Imagen-style): LIMO training
        # latents have norm ~8, so clip predictions to norm 15 (a couple of
        # σ above typical). Tight enough to keep z on-manifold without
        # masking the gradient signal at on-manifold scale.
        max_z0 = 30.0
        z0_norm = z0_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (z0_norm.clamp(max=max_z0) / z0_norm)
        z0_pred = z0_pred * scale
        z = ab_next.sqrt() * z0_pred + (1 - ab_next).sqrt() * eps

    return z.detach()
