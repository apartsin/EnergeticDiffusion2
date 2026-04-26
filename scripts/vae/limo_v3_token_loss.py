"""Per-token weighted loss for LIMO v3.1.

Up-weights cross-entropy on tokens that participate in rings, charges, or
energetic motifs. Direct fix for L8 failure-taxonomy: lost ring + lost
charge dominate; this gradient bias forces the decoder to pay attention.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


# Token-class weights. Names match SELFIES alphabet conventions.
# Tokens not listed get weight 1.0.
TOKEN_WEIGHTS = {
    # Charged atoms / cations / anions
    "[N+]":      3.0,
    "[N+1]":     3.0,
    "[N+H1]":    3.0,
    "[O-]":      3.0,
    "[O-1]":     3.0,
    "[N-]":      3.0,
    "[N-1]":     3.0,
    # Ring opening / closing — must match for valid molecule
    "[Ring1]":   2.5,
    "[Ring2]":   2.5,
    "[Ring3]":   2.5,
    "[=Ring1]":  2.5,
    "[=Ring2]":  2.5,
    "[Branch1]": 2.0,
    "[Branch2]": 2.0,
    "[=Branch1]":2.0,
    # Aromatic ring tokens for N-rich heterocycles
    "[n]":       2.0,
    "[nH]":      2.0,
    "[N=]":      2.0,
    "[=N]":      2.0,
    # Nitro / N=O patterns
    "[=O]":      1.5,
    "[N]":       1.2,
    "[O]":       1.2,
}


def build_token_weight_vector(alphabet, default: float = 1.0) -> torch.Tensor:
    """Returns (vocab_size,) tensor of weights — one per vocab id."""
    w = torch.full((len(alphabet),), default, dtype=torch.float32)
    for i, tok in enumerate(alphabet):
        if tok in TOKEN_WEIGHTS:
            w[i] = TOKEN_WEIGHTS[tok]
    return w


def weighted_nll(log_probs: torch.Tensor, target: torch.Tensor,
                  token_weights: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """log_probs: (B, T, V), target: (B, T) int, token_weights: (V,).
    Returns scalar = mean over (non-pad) positions of [w(target) · −log p(target)].
    Heavier weight on important tokens → bigger gradient when wrong.
    """
    B, T, V = log_probs.shape
    nll = F.nll_loss(log_probs.reshape(-1, V), target.reshape(-1),
                      ignore_index=pad_idx, reduction="none")     # (B*T,)
    w = token_weights.to(target.device)[target.reshape(-1)]
    w = torch.where(target.reshape(-1) == pad_idx, torch.zeros_like(w), w)
    return (nll * w).sum() / w.sum().clamp(min=1.0)
