"""DDIM sampler with latent classifier guidance.

Adds gradients from latent property heads to the diffusion score, pushing
samples toward higher predicted property values regardless of whether the
denoiser's FiLM-conditioning is reading the values correctly.

Use:
    z = ddim_sample_guided(
        denoiser, schedule, values_norm, mask,
        property_targets={"detonation_velocity": 7.86, "density": 1.83},
        property_heads_path="data/training/guidance/property_heads.pt",
        guidance_lambda=2.0, ...)

The guidance term per step:
    eps_guided = eps - sqrt(1-ᾱ_t) · λ · ∇_z [Σ_p sign_p · (f_p(z) - target_p)/std_p]

Sign is +1 to push toward target (gradient ascent if pred < target).
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional


class _PropHead(nn.Module):
    def __init__(self, d=1024, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(h, h), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(h, 1))
    def forward(self, z): return self.net(z).squeeze(-1)


def load_property_heads(path: str | Path, device: str = "cuda"):
    """Returns (heads_dict, normalizer_dict, prop_names).
    heads[prop] = nn.Module that predicts standardised property value
    normalizer[prop] = (y_mean, y_std)
    """
    blob = torch.load(path, weights_only=False)
    heads = {}
    norms = {}
    for p, info in blob["heads"].items():
        m = _PropHead().to(device)
        m.load_state_dict(info["state_dict"])
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)
        heads[p] = m
        norms[p] = (info["y_mean"], info["y_std"])
    return heads, norms, blob["property_names"]


def _compute_guidance(z, heads: dict, targets: dict, norms: dict):
    """Returns scalar = -Σ_p (pred_norm - target_norm)². Maximising this
    pushes z toward the targets. Caller computes ∇_z, adds to score.

    z must already have requires_grad=True before calling.
    """
    total = z.new_zeros(())
    for prop, target_raw in targets.items():
        if prop not in heads:
            continue
        mu, sd = norms[prop]
        pred_norm = heads[prop](z)             # (B,) standardised
        target_norm = (target_raw - mu) / sd
        diff = pred_norm - target_norm
        total = total - (diff ** 2).sum()
    return total


def ddim_sample_guided(
    denoiser,
    schedule,
    values_norm: torch.Tensor,     # (B, n_props) standardized target values
    mask:        torch.Tensor,     # (B, n_props) 0/1
    n_steps: int = 40,
    guidance_scale: float = 2.0,
    cfg_dropout_mask: Optional[torch.Tensor] = None,
    device: str = "cuda",
    # NEW
    property_targets: Optional[dict] = None,   # {prop_name: target_raw}
    property_heads_path: Optional[str] = None,
    guidance_lambda: float = 1.0,
    guidance_warmup_steps: int = 0,            # don't apply guidance for first N steps
) -> torch.Tensor:
    denoiser.eval()
    B = values_norm.shape[0]
    if cfg_dropout_mask is None:
        cfg_dropout_mask = torch.zeros_like(mask)

    use_classifier = (property_targets is not None and property_heads_path is not None
                      and guidance_lambda > 0)
    if use_classifier:
        heads, norms, _ = load_property_heads(property_heads_path, device=device)

    ts = torch.linspace(schedule.T - 1, 0, n_steps + 1, device=device).long()
    z = torch.randn(B, denoiser.latent_dim, device=device)

    for i in range(n_steps):
        t_now  = ts[i]
        t_next = ts[i + 1]
        t_batch = torch.full((B,), int(t_now), device=device, dtype=torch.long)

        with torch.no_grad():
            eps_cond = denoiser(z, t_batch, values_norm, mask)
            if guidance_scale != 1.0:
                eps_null = denoiser(z, t_batch, values_norm, cfg_dropout_mask)
                eps = eps_null + guidance_scale * (eps_cond - eps_null)
            else:
                eps = eps_cond

        # Classifier guidance term
        if use_classifier and i >= guidance_warmup_steps:
            z_req = z.detach().requires_grad_(True)
            with torch.enable_grad():
                total = _compute_guidance(z_req, heads, property_targets, norms)
                g, = torch.autograd.grad(total, z_req)
            ab_now = schedule.alpha_bar[t_now]
            sigma_t = (1 - ab_now).sqrt()
            # obj = -Σ(target − pred)² so grad points toward higher pred when below target.
            # Subtract σ·λ·(−g) — but the negation is folded into the obj sign already.
            eps = eps - sigma_t * guidance_lambda * g

        ab_now  = schedule.alpha_bar[t_now]
        ab_next = schedule.alpha_bar[t_next] if t_next > 0 \
                  else torch.tensor(1.0, device=device)

        z0_pred = (z - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()
        z = ab_next.sqrt() * z0_pred + (1 - ab_next).sqrt() * eps

    return z.detach()
