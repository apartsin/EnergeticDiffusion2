"""LIMO v3.3: diffusion-aware decoder.

Same architecture as v3.1 (causal AR transformer decoder; configurable
skip-connection and depth/width via constructor args), but during training
the decoder sees a NOISE-AUGMENTED z drawn from a curriculum that
approximates the diffusion-time prior:

    z_train = z_clean + sigma * eps   where eps ~ N(0, I), sigma ~ Uniform(0, sigma_max)

This bridges the train/sample distribution gap that caused v3.1 and v3.2 to
collapse on diffusion-time z (see registry rows for v3.1 and v3.2).

Config flag in arch:
    noise_aug:
        sigma_max:    1.5          # max noise per-element std injected during training
        prob:         1.0          # probability of injecting noise on a given step (1.0 = always)
        z_only:       true         # only noise z; skip embeddings stay clean
        curriculum:   "linear"     # "linear" ramps sigma_max from 0 over warmup; "constant" fixed
        warmup_steps: 1500
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

from limo_v3_1_model import LIMOVAEv3_1


class LIMOVAEv3_3(LIMOVAEv3_1):
    def __init__(
        self,
        *args,
        noise_aug_sigma_max: float = 1.5,
        noise_aug_prob: float = 1.0,
        noise_aug_z_only: bool = True,
        noise_aug_curriculum: str = "linear",
        noise_aug_warmup_steps: int = 1500,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_aug_sigma_max = float(noise_aug_sigma_max)
        self.noise_aug_prob = float(noise_aug_prob)
        self.noise_aug_z_only = bool(noise_aug_z_only)
        self.noise_aug_curriculum = noise_aug_curriculum
        self.noise_aug_warmup_steps = int(noise_aug_warmup_steps)
        self.register_buffer("_train_step", torch.zeros(1, dtype=torch.long), persistent=False)

    def _current_sigma_max(self) -> float:
        if self.noise_aug_curriculum == "constant":
            return self.noise_aug_sigma_max
        step = int(self._train_step.item())
        warm = max(1, self.noise_aug_warmup_steps)
        return self.noise_aug_sigma_max * min(1.0, step / warm)

    def _maybe_noise(self, z, emb):
        if not self.training:
            return z, emb
        if torch.rand(1, device=z.device).item() > self.noise_aug_prob:
            return z, emb
        sigma_max = self._current_sigma_max()
        if sigma_max <= 0:
            return z, emb
        sigma = torch.rand(z.shape[0], 1, device=z.device) * sigma_max
        z_noisy = z + sigma * torch.randn_like(z)
        if self.noise_aug_z_only or emb is None:
            return z_noisy, emb
        emb_sigma = sigma.unsqueeze(-1)
        emb_noisy = emb + emb_sigma * torch.randn_like(emb)
        return z_noisy, emb_noisy

    def forward(self, x: torch.Tensor):
        if self.skip_connection:
            z, mu, log_var, emb = self.encode(x, return_emb=True)
            z_in, emb_in = self._maybe_noise(z, emb)
            log_probs = self.decode(z_in, x_in=x, emb=emb_in)
        else:
            z, mu, log_var = self.encode(x)
            z_in, _ = self._maybe_noise(z, None)
            log_probs = self.decode(z_in, x_in=x)
        if self.training:
            self._train_step += 1
        return log_probs, z, mu, log_var
