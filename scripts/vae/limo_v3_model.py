"""LIMO v3: frozen v1 encoder + transformer decoder.

Replaces the original decoder MLP with a non-autoregressive transformer
decoder that cross-attends to z (treated as memory tokens). Encoder is
loaded from v1 best.pt and frozen — only the new decoder trains.

Compatible with the same forward signature as `limo_model.LIMOVAE` so it
drops into the existing trainer / encode_latents pipelines.
"""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from limo_model import LIMOVAE, LIMO_MAX_LEN, LIMO_VOCAB_SIZE, LIMO_LATENT_DIM


class LIMOVAEv3(nn.Module):
    """Frozen v1 encoder; transformer decoder.

    Latent z (1024-d) → reshaped to (n_memory, d_mem) memory tokens →
    transformer decoder cross-attends from learnable position queries to
    the memory → per-position vocab logits.

    Non-autoregressive: all output tokens are predicted in parallel
    (matches v1 behaviour, no need to change the trainer's loss).
    """

    def __init__(
        self,
        v1_state_dict: Optional[dict] = None,
        max_len:       int = LIMO_MAX_LEN,
        vocab_len:     int = LIMO_VOCAB_SIZE,
        latent_dim:    int = LIMO_LATENT_DIM,
        embedding_dim: int = 64,
        # transformer decoder hyperparameters
        d_model:       int = 384,
        n_heads:       int = 6,
        n_layers:      int = 4,
        ff_mult:       int = 4,
        dropout:       float = 0.1,
        n_memory:      int = 16,
        freeze_encoder: bool = True,
        # NEW v3.1 options
        skip_connection: bool = False,
        skip_dim:        int = 64,
    ):
        super().__init__()
        self.max_len    = max_len
        self.vocab_len  = vocab_len
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.d_model    = d_model
        self.n_memory   = n_memory
        assert latent_dim % n_memory == 0, \
            f"latent_dim ({latent_dim}) must be divisible by n_memory ({n_memory})"
        self.d_mem = latent_dim // n_memory   # 1024 / 16 = 64

        # ── reuse v1 encoder layers (frozen by default) ────────────────────
        # Build a v1 LIMOVAE just to get its layers, then borrow .embedding +
        # .encoder. The v1 .decoder is dropped.
        v1 = LIMOVAE(max_len=max_len, vocab_len=vocab_len,
                      latent_dim=latent_dim, embedding_dim=embedding_dim)
        if v1_state_dict is not None:
            v1.load_state_dict(v1_state_dict)
        self.embedding = v1.embedding
        self.encoder   = v1.encoder
        if freeze_encoder:
            for p in self.embedding.parameters(): p.requires_grad_(False)
            for p in self.encoder.parameters():   p.requires_grad_(False)

        # ── v3.1 skip connection from encoder embeddings ─────────────────
        # If on, the encoder's per-token embeddings (B, max_len, embedding_dim)
        # are projected to d_model and concatenated to the memory tokens →
        # decoder cross-attends to BOTH z-memory AND original token embeddings.
        # Lets the decoder access motif info that the bottleneck z lost.
        self.skip_connection = skip_connection
        if skip_connection:
            self.skip_proj = nn.Linear(embedding_dim, d_model)

        # ── new transformer decoder ────────────────────────────────────────
        # Project memory tokens 64 → d_model
        self.mem_proj = nn.Linear(self.d_mem, d_model)

        # Learnable position queries (one per output token)
        self.pos_query = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # Pre-LN transformer decoder layers
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers,
                                                norm=nn.LayerNorm(d_model))

        self.out_proj = nn.Linear(d_model, vocab_len)

        # ── parameter inventory ──────────────────────────────────────────
        n_total    = sum(p.numel() for p in self.parameters())
        n_train    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._n_params_summary = f"total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M"

    # ── encoder = v1 (frozen) ────────────────────────────────────────────
    def encode(self, x: torch.Tensor, return_emb: bool = False):
        """x: (B, max_len) int → (z, mu, log_var) [, emb])."""
        B = x.shape[0]
        emb = self.embedding(x)                                  # (B, L, emb_dim)
        e = emb.view(B, -1)
        h = self.encoder(e).view(B, 2, self.latent_dim)
        mu, log_var = h[:, 0, :], h[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if return_emb:
            return z, mu, log_var, emb
        return z, mu, log_var

    # ── decoder = new transformer ─────────────────────────────────────────
    def decode(self, z: torch.Tensor, emb: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """z: (B, latent_dim).  emb: optional (B, max_len, embedding_dim)
        for skip connection.
        """
        B = z.shape[0]
        mem = z.view(B, self.n_memory, self.d_mem)
        mem = self.mem_proj(mem)                                 # (B, 16, d_model)
        if self.skip_connection and emb is not None:
            skip = self.skip_proj(emb)                           # (B, max_len, d_model)
            mem = torch.cat([mem, skip], dim=1)                  # (B, 16+L, d_model)
        q = self.pos_query.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(q, mem)
        logits = self.out_proj(out)
        return F.log_softmax(logits, dim=2)

    def forward(self, x: torch.Tensor):
        if self.skip_connection:
            z, mu, log_var, emb = self.encode(x, return_emb=True)
            logits = self.decode(z, emb)
        else:
            z, mu, log_var = self.encode(x)
            logits = self.decode(z)
        return logits, z, mu, log_var

    # ── helpers expected by the rest of the codebase ─────────────────────
    def load_limo_weights(self, *args, **kwargs):
        # v3 is constructed with v1 weights via __init__; this is a no-op
        return [], []

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return (f"LIMOVAEv3({self._n_params_summary}, "
                f"d_model={self.d_model}, n_layers={len(self.decoder.layers)}, "
                f"frozen_encoder=True)")
