"""
Auxiliary score predictors on LIMO latent.

Both SA and SC predictors are simple MLPs that map z ∈ R^1024 → scalar score.
By construction they're differentiable w.r.t. z, which lets us use gradients
during diffusion sampling (classifier guidance — Dhariwal & Nichol 2021).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ScorePredictor(nn.Module):
    """Tiny MLP for differentiable score prediction.

    Architecture:
        Linear(1024 -> 512) -> SiLU -> Dropout
        Linear(512  -> 512) -> SiLU -> Dropout
        Linear(512  -> 256) -> SiLU
        Linear(256  -> 1)     (no activation; raw standardized score)
    """
    def __init__(self, in_dim: int = 1024, hidden: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 1024)  →  (B,) standardized score."""
        return self.net(z).squeeze(-1)


class ScorePredictorWithDenorm(nn.Module):
    """Wrapper that stores standardization stats and exposes raw-score output.

    Useful during diffusion sampling where you want to display or
    threshold-check the raw (non-standardized) score.
    """
    def __init__(self, predictor: ScorePredictor, mean: float, std: float):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std",  torch.tensor(float(std)))

    def forward(self, z: torch.Tensor, standardized: bool = True) -> torch.Tensor:
        out = self.predictor(z)
        if standardized:
            return out
        return out * self.std + self.mean
