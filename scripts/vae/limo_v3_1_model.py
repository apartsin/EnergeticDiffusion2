"""LIMO v3.1: causal autoregressive transformer decoder + skip connection.

Replaces v3's parallel non-autoregressive decoder (which collapsed to
predicting the most-frequent token at every position) with a standard
autoregressive transformer decoder using teacher forcing.

Key differences vs v3:
  - DECODER INPUT during training: shifted ground-truth tokens (BOS + tokens[:-1])
  - causal self-attention mask: position i sees only positions ≤ i
  - skip connection (default ON): cross-attend to z-memory ⊕ encoder embeddings
  - output is per-position logits over vocab, trained with NLL on next-token

At inference, generates left-to-right.
"""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from limo_model import LIMOVAE, LIMO_MAX_LEN, LIMO_VOCAB_SIZE, LIMO_LATENT_DIM


class LIMOVAEv3_1(nn.Module):
    """Frozen v1 encoder; CAUSAL autoregressive transformer decoder.

    Forward signature matches v1 LIMOVAE:
        forward(x) → (log_probs[B, T, V], z, mu, log_var)
    For training, log_probs[i] is the predicted distribution over token i+1
    given tokens[:i]. For evaluation in our pipeline (which still measures
    `argmax(log_probs) vs x`), we left-shift logits so position i still
    refers to predicting token i — just with full causal attention.

    Implementation note: we pad the input embedding sequence with a BOS
    token at position 0 (recycled as pad index 0), then take logits at
    positions 0..T-1, which correspond to predicting tokens 0..T-1.
    """

    def __init__(
        self,
        v1_state_dict: Optional[dict] = None,
        max_len:        int = LIMO_MAX_LEN,
        vocab_len:      int = LIMO_VOCAB_SIZE,
        latent_dim:     int = LIMO_LATENT_DIM,
        embedding_dim:  int = 64,
        d_model:        int = 384,
        n_heads:        int = 6,
        n_layers:       int = 4,
        ff_mult:        int = 4,
        dropout:        float = 0.1,
        n_memory:       int = 16,
        freeze_encoder: bool = True,
        skip_connection: bool = True,
        pad_idx:        int = 0,
    ):
        super().__init__()
        self.max_len    = max_len
        self.vocab_len  = vocab_len
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.d_model    = d_model
        self.n_memory   = n_memory
        self.pad_idx    = pad_idx
        assert latent_dim % n_memory == 0
        self.d_mem = latent_dim // n_memory

        # ── reuse v1 encoder (frozen) ────────────────────────────────────
        v1 = LIMOVAE(max_len=max_len, vocab_len=vocab_len,
                      latent_dim=latent_dim, embedding_dim=embedding_dim)
        if v1_state_dict is not None:
            v1.load_state_dict(v1_state_dict)
        self.embedding = v1.embedding   # (V, embedding_dim)
        self.encoder   = v1.encoder
        if freeze_encoder:
            for p in self.embedding.parameters(): p.requires_grad_(False)
            for p in self.encoder.parameters():   p.requires_grad_(False)

        # ── memory projection ──────────────────────────────────────────
        self.mem_proj = nn.Linear(self.d_mem, d_model)
        self.skip_connection = skip_connection
        if skip_connection:
            self.skip_proj = nn.Linear(embedding_dim, d_model)

        # ── decoder input embedding (separate from frozen encoder embed) ──
        # We use a fresh embedding here to give the decoder its own token
        # representation. Same vocab. Initialised from frozen encoder embed.
        self.dec_token_embed = nn.Embedding(vocab_len, d_model, padding_idx=pad_idx)
        with torch.no_grad():
            # Project frozen embedding (V, embedding_dim) → (V, d_model)
            # as a smarter init than random.
            ein = self.embedding.weight     # (V, embedding_dim)
            proj = nn.Linear(embedding_dim, d_model, bias=False)
            self.dec_token_embed.weight.copy_(proj(ein))

        # ── learnable position embedding for decoder input ──
        self.dec_pos_embed = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # ── transformer decoder layers (causal) ──────────────────────────
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers,
                                                norm=nn.LayerNorm(d_model))
        self.out_proj = nn.Linear(d_model, vocab_len)

        # Pre-built causal mask
        self.register_buffer("causal_mask",
                              torch.triu(torch.ones(max_len, max_len) * float("-inf"), diagonal=1),
                              persistent=False)

        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._summary = f"total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M"

    # ── encoder ─────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor, return_emb: bool = False):
        B = x.shape[0]
        emb = self.embedding(x)                                 # (B, L, embedding_dim)
        e = emb.view(B, -1)
        h = self.encoder(e).view(B, 2, self.latent_dim)
        mu, log_var = h[:, 0, :], h[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return (z, mu, log_var, emb) if return_emb else (z, mu, log_var)

    # ── memory builder ──────────────────────────────────────────────────
    def _build_memory(self, z: torch.Tensor, emb: Optional[torch.Tensor]):
        B = z.shape[0]
        mem = z.view(B, self.n_memory, self.d_mem)
        mem = self.mem_proj(mem)
        if self.skip_connection and emb is not None:
            skip = self.skip_proj(emb)
            mem = torch.cat([mem, skip], dim=1)
        return mem

    # ── decode (training): teacher forcing with causal mask ─────────────
    def decode(self, z: torch.Tensor, x_in: Optional[torch.Tensor] = None,
                emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Training mode: x_in = ground-truth tokens (B, max_len).

        Decoder input at position i is the embedding of x_in[i-1] (BOS at i=0).
        Output at position i is the predicted distribution over x_in[i].
        """
        B = z.shape[0]
        device = z.device
        if x_in is None:
            return self._generate_autoregressive(z, emb)
        # shift right: prepend pad as BOS
        bos = torch.full((B, 1), self.pad_idx, dtype=x_in.dtype, device=device)
        dec_inp = torch.cat([bos, x_in[:, :-1]], dim=1)         # (B, max_len)
        tok_emb = self.dec_token_embed(dec_inp)                 # (B, L, d_model)
        pos = self.dec_pos_embed.unsqueeze(0).expand(B, -1, -1)
        q = tok_emb + pos                                        # (B, L, d_model)

        memory = self._build_memory(z, emb)
        out = self.decoder(q, memory, tgt_mask=self.causal_mask[:self.max_len, :self.max_len])
        logits = self.out_proj(out)                              # (B, L, V)
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def _generate_autoregressive(self, z: torch.Tensor,
                                    emb: Optional[torch.Tensor] = None,
                                    sample: bool = False) -> torch.Tensor:
        """Inference: generate tokens left-to-right. Returns log-probs (B, L, V)
        for compatibility with eval pipelines that argmax."""
        B = z.shape[0]
        device = z.device
        memory = self._build_memory(z, emb)
        x = torch.full((B, 1), self.pad_idx, dtype=torch.long, device=device)
        all_logits = []
        for t in range(self.max_len):
            tok_emb = self.dec_token_embed(x)                    # (B, t+1, d_model)
            pos = self.dec_pos_embed[: tok_emb.shape[1]].unsqueeze(0).expand(B, -1, -1)
            q = tok_emb + pos
            mask = self.causal_mask[: tok_emb.shape[1], : tok_emb.shape[1]]
            out = self.decoder(q, memory, tgt_mask=mask)
            logits = self.out_proj(out[:, -1:, :])                # (B, 1, V)
            all_logits.append(logits)
            if sample:
                probs = logits.softmax(-1).squeeze(1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = logits.argmax(-1)                       # (B, 1)
            x = torch.cat([x, next_tok], dim=1)
        full = torch.cat(all_logits, dim=1)                       # (B, max_len, V)
        return F.log_softmax(full, dim=-1)

    def forward(self, x: torch.Tensor):
        if self.skip_connection:
            z, mu, log_var, emb = self.encode(x, return_emb=True)
            log_probs = self.decode(z, x_in=x, emb=emb)
        else:
            z, mu, log_var = self.encode(x)
            log_probs = self.decode(z, x_in=x)
        return log_probs, z, mu, log_var

    def load_limo_weights(self, *args, **kwargs):
        return [], []

    def __repr__(self):
        return (f"LIMOVAEv3_1({self._summary}, d_model={self.d_model}, "
                f"layers={len(self.decoder.layers)}, "
                f"causal_AR=True, skip={self.skip_connection})")
