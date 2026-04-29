"""
LIMO VAE adapter for EnergeticDiffusion2.

Loads the Rose-STL-Lab/LIMO pretrained checkpoint (ZINC-250k SELFIES, MIT
license) and provides a minimal PyTorch-native interface (no PyTorch-Lightning
dependency) with tokenizer, encode/decode, and state-dict compatible weights.

LIMO architecture (verified against external/LIMO/vae.pt):
    embedding:    vocab=108, embed_dim=64, padding_idx=0 ([nop])
    encoder:      Linear(max_len*64 -> 2000) - ReLU
                  Linear(2000 -> 1000) - BN1d(1000) - ReLU
                  Linear(1000 -> 1000) - BN1d(1000) - ReLU
                  Linear(1000 -> 2 * latent_dim)       [mu, log_var packed]
    decoder:      Linear(latent_dim -> 1000) - BN1d(1000) - ReLU
                  Linear(1000 -> 1000) - BN1d(1000) - ReLU
                  Linear(1000 -> 2000) - ReLU
                  Linear(2000 -> max_len * vocab_len)  [log-softmax over vocab]
    max_len:      72 (max ZINC-250k SELFIES token count)
    latent_dim:   1024
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import selfies as sf


# ── constants (must match LIMO pretrained weights) ──────────────────────────
LIMO_VOCAB_SIZE   = 108
LIMO_EMBED_DIM    = 64
LIMO_MAX_LEN      = 72
LIMO_LATENT_DIM   = 1024
LIMO_PAD_TOKEN    = "[nop]"
LIMO_PAD_IDX      = 0   # [nop] is always index 0 in LIMO's vocab

# ── vocabulary (reproducible from zinc250k.smi) ─────────────────────────────
def build_limo_vocab(zinc250k_path: str | Path) -> list[str]:
    """Rebuild the LIMO SELFIES alphabet exactly as utils.py constructs it."""
    selfies_list = []
    with open(zinc250k_path, "r") as f:
        for line in f:
            smi = line.split()[0]
            try:
                selfies_list.append(sf.encoder(smi))
            except Exception:
                continue
    alphabet = set()
    for s in selfies_list:
        alphabet.update(sf.split_selfies(s))
    alphabet = ["[nop]"] + sorted(list(alphabet))
    assert len(alphabet) == LIMO_VOCAB_SIZE, (
        f"Vocab size mismatch: got {len(alphabet)}, expected {LIMO_VOCAB_SIZE}")
    return alphabet


def save_vocab(alphabet: list[str], path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump({"alphabet": alphabet}, f, indent=2)


def load_vocab(path: str | Path) -> list[str]:
    with open(path) as f:
        return json.load(f)["alphabet"]


# ── tokenizer ────────────────────────────────────────────────────────────────
class SELFIESTokenizer:
    """Encode SMILES → SELFIES → integer tensor (padded to max_len)."""
    def __init__(self, alphabet: Sequence[str], max_len: int = LIMO_MAX_LEN):
        self.alphabet   = list(alphabet)
        self.max_len    = max_len
        self.sym_to_idx = {s: i for i, s in enumerate(alphabet)}
        self.idx_to_sym = {i: s for i, s in enumerate(alphabet)}
        self.pad_idx    = self.sym_to_idx[LIMO_PAD_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    def encode_selfies(self, selfies_str: str) -> tuple[list[int], int]:
        """Return (padded_indices, true_length_without_padding).

        Tokens not in vocab → pad. Returns None if empty after encoding.
        """
        toks = list(sf.split_selfies(selfies_str))
        if not toks:
            return [self.pad_idx] * self.max_len, 0
        indices = [self.sym_to_idx.get(t, self.pad_idx) for t in toks]
        true_len = min(len(indices), self.max_len)
        if len(indices) >= self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [self.pad_idx] * (self.max_len - len(indices))
        return indices, true_len

    def smiles_to_tensor(self, smiles: str) -> tuple[torch.Tensor, int] | None:
        """SMILES → tensor. Returns None if encoding/tokenization fails."""
        try:
            selfies_str = sf.encoder(smiles)
        except Exception:
            return None
        if not selfies_str:
            return None
        indices, true_len = self.encode_selfies(selfies_str)
        return torch.tensor(indices, dtype=torch.long), true_len

    def indices_to_selfies(self, indices: Sequence[int]) -> str:
        """Integer tensor → SELFIES string (pad tokens dropped)."""
        toks = [self.idx_to_sym[int(i)] for i in indices]
        # drop trailing and embedded [nop]
        return "".join(t for t in toks if t != LIMO_PAD_TOKEN)

    def indices_to_smiles(self, indices: Sequence[int]) -> str:
        selfies_str = self.indices_to_selfies(indices)
        try:
            return sf.decoder(selfies_str) or ""
        except Exception:
            return ""


# ── model ────────────────────────────────────────────────────────────────────
class LIMOVAE(nn.Module):
    """Plain PyTorch reimplementation of LIMO's VAE (no pytorch-lightning)."""

    def __init__(
        self,
        max_len:       int = LIMO_MAX_LEN,
        vocab_len:     int = LIMO_VOCAB_SIZE,
        latent_dim:    int = LIMO_LATENT_DIM,
        embedding_dim: int = LIMO_EMBED_DIM,
    ):
        super().__init__()
        self.max_len       = max_len
        self.vocab_len     = vocab_len
        self.latent_dim    = latent_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(max_len * embedding_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, max_len * vocab_len),
        )

    def encode(self, x: torch.Tensor):
        """x: (B, max_len) int → (z, mu, log_var) with z reparameterised."""
        B = x.shape[0]
        e = self.embedding(x).view(B, -1)
        h = self.encoder(e).view(B, 2, self.latent_dim)
        mu, log_var = h[:, 0, :], h[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent) → (B, max_len, vocab_len) log-softmax logits."""
        out = self.decoder(z).view(-1, self.max_len, self.vocab_len)
        return F.log_softmax(out, dim=2)

    def forward(self, x: torch.Tensor):
        z, mu, log_var = self.encode(x)
        logits = self.decode(z)
        return logits, z, mu, log_var

    # ── weight loading ───────────────────────────────────────────────────────
    def load_limo_weights(self, path: str | Path, strict: bool = True):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        # LIMO's pytorch-lightning checkpoint has bare state_dict keys
        # (no "model." prefix), so direct load works.
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        return missing, unexpected


# ── loss ─────────────────────────────────────────────────────────────────────
def vae_loss(
    log_probs: torch.Tensor,    # (B, max_len, vocab_len) log-softmax
    targets:   torch.Tensor,    # (B, max_len)
    mu:        torch.Tensor,
    log_var:   torch.Tensor,
    beta:      float = 0.1,
    free_bits: float = 0.0,
    pad_idx:   int   = 0,
    reduction: str   = "mean",
):
    """LIMO-compatible VAE loss with optional free-bits regularisation.

    Returns (loss, nll, kld, stats).
    """
    B, T, V = log_probs.shape
    # reconstruction: standard NLL averaged over (B*T) like LIMO's setup
    nll = F.nll_loss(
        log_probs.reshape(-1, V), targets.reshape(-1),
        ignore_index=pad_idx, reduction=reduction)

    # KL divergence (standard VAE)
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())   # (B, latent)
    # free-bits: impose a floor per latent dimension
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kld = kl_per_dim.sum(dim=1).mean()    # mean over batch, sum over latent dims

    total = nll + beta * kld
    return total, nll, kld


# ── helpers ──────────────────────────────────────────────────────────────────
def find_limo_repo(base: str | Path = "E:/Projects/EnergeticDiffusion2") -> Path:
    base = Path(base)
    candidates = [
        base / "external/LIMO",
        base / "external" / "LIMO",
    ]
    for c in candidates:
        if c.exists() and (c / "vae.pt").exists() and (c / "zinc250k.smi").exists():
            return c
    raise FileNotFoundError(
        f"LIMO repo not found. Expected external/LIMO/{{vae.pt, zinc250k.smi}}")


def load_limo_model_and_tokenizer(limo_dir: str | Path | None = None,
                                    device: str = "cpu"
                                  ) -> tuple[LIMOVAE, SELFIESTokenizer]:
    """One-liner: load LIMO VAE + tokenizer ready to encode/decode."""
    limo_dir = Path(limo_dir) if limo_dir else find_limo_repo()
    vocab_cache = limo_dir / "vocab_cache.json"
    if vocab_cache.exists():
        alphabet = load_vocab(vocab_cache)
    else:
        alphabet = build_limo_vocab(limo_dir / "zinc250k.smi")
        save_vocab(alphabet, vocab_cache)

    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    model = LIMOVAE()
    model.load_limo_weights(limo_dir / "vae.pt", strict=True)
    model.to(device).eval()
    return model, tok
