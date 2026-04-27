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
    import torch.nn as nn
    blob = torch.load(path, weights_only=False, map_location=device)
    cfg = blob["config"]
    sd = blob["state_dict"]
    model = MultiHeadScoreModel(latent_dim=cfg["latent_dim"]).to(device)
    # Detect optional hazard head (v3f+) and add it dynamically
    has_hazard = any(k.startswith("head_hazard.") for k in sd.keys())
    if has_hazard:
        hidden = model.input_proj.out_features
        model.head_hazard = nn.Sequential(
            nn.Linear(hidden, 256), nn.SiLU(),
            nn.Linear(256, 1),
        ).to(device)
        # Patch forward to also output hazard_logit
        _orig = model.forward
        def forward_with_hazard(z, sigma, _orig=_orig, _model=model):
            out = _orig(z, sigma)
            s = _model.sig_emb(sigma.view(-1, 1).float())
            h = _model.input_proj(z)
            for b in _model.blocks:
                h = b(h, s)
            out["hazard_logit"] = _model.head_hazard(h).squeeze(-1)
            return out
        model.forward = forward_with_hazard
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, cfg


@torch.enable_grad()
def _guidance_grad(score_model, z_t, sigma_t, scales: Dict[str, float],
                   sigma_max_for_anneal: float = 0.0,           # 0 disables anneal (was 2.0; anneal was killing gradient at high sigma)
                   max_grad_norm: float = 50.0):                # raised from 5 (was clamping legitimate signal)
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
    if scales.get("hazard", 0) > 0 and "hazard_logit" in out:
        # Descend hazard probability: minimize log P(hazard) = -softplus(-logit)
        # Equivalently: minimize sigmoid(logit). Use softplus(logit) =
        # -log(1-sigmoid(logit)) so that high-hazard latents get high loss
        # and gradient flows AWAY from hazardous regions.
        losses["hazard"] = scales["hazard"] * F.softplus(out["hazard_logit"]).sum()
    if not losses:
        return torch.zeros_like(z_t)
    # S4 per-head gradient-norm rebalancing: compute each head's gradient
    # independently, normalise to unit per-row norm, then sum.
    use_unit_norm = scales.get("_unit_norm", True)
    if use_unit_norm and len(losses) > 1:
        z_t.grad = None
        per_head_grads = []
        for name, loss in losses.items():
            g = torch.autograd.grad(loss, z_t, create_graph=False, retain_graph=True)[0]
            unit = g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            per_head_grads.append(g / unit)
        grad = torch.stack(per_head_grads, dim=0).sum(dim=0)
    else:
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
