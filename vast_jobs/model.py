"""
Subset-conditional DDPM denoiser on LIMO's 1024-dim latent.

Architecture:
  - Per-property value embedding (scalar -> sinusoidal PE -> MLP -> 64-dim)
  - Per-property learned "mask" embedding (used when property is not conditioned)
  - Concatenated 4 x 64 = 256-dim conditioning vector
  - Timestep sinusoidal embedding -> MLP -> 256-dim
  - FiLM modulation inside 8 residual MLP blocks (1024 -> 2048 -> 1024)
  - EMA-tracked weights for stable sampling
  - Cosine noise schedule (Nichol & Dhariwal 2021)

Used by scripts/diffusion/train.py and scripts/diffusion/evaluate.py.
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── noise schedule ──────────────────────────────────────────────────────────
def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule. Returns alpha_bar_t for t=0..T."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    return alpha_bar.float()


class NoiseSchedule:
    """Pre-computes all betas/alphas/etc for a cosine schedule."""
    def __init__(self, T: int = 1000, device: str = "cpu"):
        self.T = T
        alpha_bar = cosine_alpha_bar(T).to(device)                 # [T+1]
        self.alpha_bar = alpha_bar[1:]                              # α̅_t, t=1..T, shape [T]
        prev = alpha_bar[:-1]                                       # α̅_{t-1}
        self.alphas = (self.alpha_bar / prev).clamp(min=1e-8)
        self.betas  = (1.0 - self.alphas).clamp(max=0.999)
        self.sqrt_alpha_bar     = self.alpha_bar.sqrt()
        self.sqrt_one_minus_ab  = (1.0 - self.alpha_bar).sqrt()
        self.device = device

    def to(self, device: str):
        for name in ("alpha_bar", "alphas", "betas",
                     "sqrt_alpha_bar", "sqrt_one_minus_ab"):
            setattr(self, name, getattr(self, name).to(device))
        self.device = device
        return self

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample z_t from q(z_t | z_0)."""
        if noise is None:
            noise = torch.randn_like(z0)
        ab = self.sqrt_alpha_bar[t].unsqueeze(1)
        om = self.sqrt_one_minus_ab[t].unsqueeze(1)
        return ab * z0 + om * noise, noise


# ── timestep and value embeddings ────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) or (B, 1) float. Returns (B, dim)."""
        if x.dim() == 1:
            x = x.unsqueeze(1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=x.device) / half
        )                                            # (half,)
        args = x * freqs.unsqueeze(0)                # (B, half)
        enc = torch.cat([args.sin(), args.cos()], dim=-1)   # (B, 2*half)
        if self.dim % 2 == 1:
            enc = F.pad(enc, (0, 1))
        return enc


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.pe = SinusoidalPE(dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(dim // 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pe(t.float()))


class PropertyConditioner(nn.Module):
    """Per-property value embedding + learned mask token. Produces concat c."""
    def __init__(self, n_props: int = 4, emb_dim: int = 64, n_freqs: int = 32):
        super().__init__()
        self.n_props = n_props
        self.emb_dim = emb_dim
        self.pe = SinusoidalPE(2 * n_freqs)
        # Per-property MLP to turn sinusoidal PE into 64-dim
        self.value_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_freqs, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            for _ in range(n_props)
        ])
        # Learned mask embeddings, one per property
        self.mask_embeds = nn.Parameter(torch.randn(n_props, emb_dim) * 0.02)

    @property
    def out_dim(self) -> int:
        return self.n_props * self.emb_dim

    def forward(self, values_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        values_norm: (B, n_props)  standardized values; can be 0 where mask==0
        mask:        (B, n_props)  0/1
        Returns: (B, n_props * emb_dim)
        """
        B = values_norm.shape[0]
        outs = []
        for j in range(self.n_props):
            v = values_norm[:, j:j+1]                       # (B, 1)
            pe = self.pe(v)                                 # (B, 2*n_freqs)
            emb = self.value_mlps[j](pe)                    # (B, emb_dim)
            m = mask[:, j:j+1].float()                      # (B, 1)
            mask_emb = self.mask_embeds[j].unsqueeze(0).expand(B, -1)  # (B, emb_dim)
            outs.append(m * emb + (1 - m) * mask_emb)
        return torch.cat(outs, dim=-1)                      # (B, out_dim)


# ── FiLM residual block ─────────────────────────────────────────────────────
class FiLMResBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.lin1  = nn.Linear(dim, hidden)
        self.lin2  = nn.Linear(hidden, dim)
        self.act   = nn.SiLU()
        self.film  = nn.Linear(cond_dim, 2 * hidden)        # γ, β for hidden
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.lin1(h)                                    # (B, hidden)
        g, b = self.film(cond).chunk(2, dim=-1)
        h = (1 + g) * self.act(h) + b
        h = self.drop(h)
        h = self.lin2(h)                                    # (B, dim)
        return x + h


# ── full denoiser ────────────────────────────────────────────────────────────
class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim:   int = 1024,
        hidden:       int = 2048,
        n_blocks:     int = 8,
        time_dim:     int = 256,
        prop_emb_dim: int = 64,
        n_props:      int = 4,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_props    = n_props

        self.prop_cond = PropertyConditioner(n_props=n_props, emb_dim=prop_emb_dim)
        self.time_emb  = TimeEmbedding(dim=time_dim)

        cond_dim_in    = self.prop_cond.out_dim + time_dim   # 256 + 256
        self.cond_comb = nn.Sequential(
            nn.Linear(cond_dim_in, cond_dim_in),
            nn.SiLU(),
            nn.Linear(cond_dim_in, time_dim),                # final cond dim
        )
        cond_dim       = time_dim

        self.input_proj = nn.Linear(latent_dim, latent_dim)
        self.blocks = nn.ModuleList([
            FiLMResBlock(latent_dim, hidden, cond_dim, dropout)
            for _ in range(n_blocks)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_t, t, values_norm, mask) -> torch.Tensor:
        cond = torch.cat([self.prop_cond(values_norm, mask),
                          self.time_emb(t)], dim=-1)
        cond = self.cond_comb(cond)
        h = self.input_proj(z_t)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_proj(self.out_norm(h))


# ── EMA wrapper ──────────────────────────────────────────────────────────────
class EMA:
    """Exponential moving average of model weights for stable sampling."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                  alpha=1 - self.decay)

    def apply_to(self, model: nn.Module) -> dict:
        """Swap model's weights to EMA. Returns original state dict for restore."""
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        new_sd = dict(model.state_dict())
        for k in self.shadow:
            new_sd[k] = self.shadow[k].to(new_sd[k].dtype).to(new_sd[k].device)
        model.load_state_dict(new_sd)
        return original

    def restore(self, model: nn.Module, original: dict):
        model.load_state_dict(original)

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.shadow[k].device)


# ── sampler (DDIM, deterministic, ~40 steps) ─────────────────────────────────
@torch.no_grad()
def ddim_sample(
    model: ConditionalDenoiser,
    schedule: NoiseSchedule,
    values_norm: torch.Tensor,     # (B, n_props) standardized target values
    mask:        torch.Tensor,     # (B, n_props) 0/1
    n_steps: int = 40,
    guidance_scale: float = 2.0,
    cfg_dropout_mask: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """DDIM sampling with classifier-free guidance.

    cfg_dropout_mask: if provided, used for the unconditional pass (i.e., "drop"
    specified properties during guidance). If None, uses all-zeros mask.
    """
    model.eval()
    B = values_norm.shape[0]
    if cfg_dropout_mask is None:
        cfg_dropout_mask = torch.zeros_like(mask)

    # pick a subset of timesteps for DDIM
    ts = torch.linspace(schedule.T - 1, 0, n_steps + 1, device=device).long()
    z = torch.randn(B, model.latent_dim, device=device)

    for i in range(n_steps):
        t_now  = ts[i]
        t_next = ts[i + 1]
        t_batch = torch.full((B,), int(t_now), device=device, dtype=torch.long)

        eps_cond = model(z, t_batch, values_norm, mask)
        if guidance_scale != 1.0:
            eps_null = model(z, t_batch, values_norm, cfg_dropout_mask)
            eps = eps_null + guidance_scale * (eps_cond - eps_null)
        else:
            eps = eps_cond

        ab_now  = schedule.alpha_bar[t_now]
        ab_next = schedule.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

        z0_pred = (z - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()
        z = ab_next.sqrt() * z0_pred + (1 - ab_next).sqrt() * eps

    return z
