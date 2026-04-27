"""DDIM sampler with multi-head classifier guidance.

Combines:
    - frozen denoiser ε_θ(z_t, t, c, m)  [v3 + v4-B + (v5)]
    - frozen multi-head score model:
          viab_logit, sa, sc, sens, perf
    - per-head guidance scales s_k (with alpha-anneal)

At each diffusion step:
    eps_cfg = eps_null + w * (eps_cond - eps_null)              # standard CFG
    grad    = dL/dz_t  for L = - s_v log P(viab) + s_s sens + ...
    eps     = eps_cfg + sigma_t * (-grad)                       # subtract grad
    eps     = eps_cfg - sigma_t * grad
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Dict
import torch
import torch.nn.functional as F


def load_score_model(path, device="cuda"):
    sys.path.insert(0, "scripts/viability")
    from train_multihead_latent import MultiHeadScoreModel
    blob = torch.load(path, weights_only=False, map_location=device)
    cfg = blob["config"]
    model = MultiHeadScoreModel(latent_dim=cfg["latent_dim"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, cfg


@torch.enable_grad()
def _guidance_grad(score_model, z_t, sigma_t, scales: Dict[str, float],
                   sigma_max_for_anneal: float = 2.0,
                   max_grad_norm: float = 5.0):
    """Compute -∇_z [- s_v log P(viab) + s_s * sens_norm - s_perf*perf_satisfy + s_sa * sa]
    with per-head clamping and alpha-anneal."""
    z_t = z_t.detach().requires_grad_(True)
    sigma = torch.full((z_t.shape[0],), float(sigma_t.item() if torch.is_tensor(sigma_t) else sigma_t),
                        device=z_t.device)
    out = score_model(z_t, sigma)
    losses = {}
    if scales.get("viab", 0) > 0:
        # ascend log P(viab) -> minimize -log sigmoid(logit) = softplus(-logit)
        losses["viab"] = scales["viab"] * F.softplus(-out["viab_logit"]).sum()
    if scales.get("sa", 0) > 0:
        # SA z-score: lower is better -> directly add positive
        losses["sa"] = scales["sa"] * out["sa"].sum()
    if scales.get("sens", 0) > 0:
        # sens z-score: lower is better
        losses["sens"] = scales["sens"] * out["sens"].sum()
    if scales.get("sc", 0) > 0:
        # SC z-score: lower is better
        losses["sc"] = scales["sc"] * out["sc"].sum()
    if not losses:
        return torch.zeros_like(z_t)
    total = sum(losses.values())
    grad = torch.autograd.grad(total, z_t, create_graph=False)[0]
    # per-row clamp norm
    norms = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale_factor = torch.where(norms > max_grad_norm, max_grad_norm / norms,
                                torch.ones_like(norms))
    grad = grad * scale_factor
    # alpha-anneal: weaken at high sigma (early/noisy steps)
    if sigma_max_for_anneal > 0:
        anneal = max(0.0, 1.0 - float(sigma_t.item() if torch.is_tensor(sigma_t) else sigma_t)
                     / sigma_max_for_anneal)
        grad = grad * anneal
    return grad.detach()


def ddim_sample_guided_v2(
    denoiser,                  # ConditionalDenoiser (frozen)
    schedule,                  # NoiseSchedule
    values_norm,               # (B, n_props)
    mask,                      # (B, n_props)
    score_model,               # MultiHeadScoreModel (frozen)
    n_steps: int = 40,
    cfg_scale: float = 7.0,
    guidance_scales: Optional[Dict[str, float]] = None,
    cfg_dropout_mask=None,
    device: str = "cuda",
):
    """DDIM with CFG + multi-head classifier guidance."""
    if guidance_scales is None:
        guidance_scales = {"viab": 1.5, "sens": 0.5, "sa": 0.0}
    denoiser.eval()
    B = values_norm.shape[0]
    if cfg_dropout_mask is None:
        cfg_dropout_mask = torch.zeros_like(mask)
    ts = torch.linspace(schedule.T - 1, 0, n_steps + 1, device=device).long()
    z = torch.randn(B, denoiser.latent_dim, device=device)

    for i in range(n_steps):
        t_now  = ts[i]
        t_next = ts[i + 1]
        t_batch = torch.full((B,), int(t_now), device=device, dtype=torch.long)
        with torch.no_grad():
            eps_cond = denoiser(z, t_batch, values_norm, mask)
            if cfg_scale != 1.0:
                eps_null = denoiser(z, t_batch, values_norm, cfg_dropout_mask)
                eps = eps_null + cfg_scale * (eps_cond - eps_null)
            else:
                eps = eps_cond

        ab_now  = schedule.alpha_bar[t_now]
        ab_next = schedule.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
        # sigma_t equivalent for the score model: 1-ab_now scaled to ~[0, 2]
        sigma_t = (1 - ab_now).sqrt() * 2.0

        # Multi-head classifier guidance
        grad = _guidance_grad(score_model, z, sigma_t, guidance_scales)
        eps = eps + (1 - ab_now).sqrt() * grad

        z0_pred = (z - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()
        z = ab_next.sqrt() * z0_pred + (1 - ab_next).sqrt() * eps

    return z
