"""Unified LIMO loader: v1 (default) or v3.1 (causal AR transformer decoder).

Drop-in for sample-time scripts. Set environment variable LIMO_V3_1 to a
ckpt path, or pass `version="v3_1"` + `ckpt=<path>` directly, to use the
v3.1 decoder. Otherwise falls back to v1 LIMOVAE loaded from the meta
field of latents.pt.

Both versions expose `.decode(z)` that returns log-softmax logits over
SELFIES tokens (B, max_len, vocab) — so downstream argmax + tokenizer
code is unchanged.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Optional, Tuple
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from limo_model import LIMOVAE, SELFIESTokenizer

V3_1_DEFAULT = "experiments/limo_v3_1_AR_20260426T070423Z/checkpoints/best.pt"


class LIMOInferenceWrapper:
    """Drop-in for v1 LIMOVAE. `.decode(z)` routes to v3.1 AR generation
    when version == 'v3_1', else parallel decode."""
    def __init__(self, model, version: str):
        self._m = model
        self.version = version

    def decode(self, z):
        with torch.no_grad():
            if self.version == "v3_1":
                return self._m._generate_autoregressive(z, emb=None)
            return self._m.decode(z)

    def encode(self, x):
        return self._m.encode(x)

    def to(self, device):
        self._m.to(device); return self

    def eval(self):
        self._m.eval(); return self

    @property
    def latent_dim(self):
        return getattr(self._m, "latent_dim", 1024)

    def __getattr__(self, k):    # pass-through for any other attribute
        if k.startswith("_"):
            raise AttributeError(k)
        return getattr(object.__getattribute__(self, "_m"), k)


def load_limo(base: Path,
                latents_meta_ckpt: Optional[str] = None,
                device: str = "cuda",
                version: Optional[str] = None,
                ckpt_override: Optional[str] = None
                ) -> Tuple[torch.nn.Module, str]:
    """Returns (model, version_used).

    Resolution order for which version to use:
      1. Explicit `version` argument ("v1" or "v3_1")
      2. Environment variable LIMO_V3_1 (if set, points at a v3.1 ckpt path)
      3. Default = v1 (loads from latents_meta_ckpt)

    Resolution for ckpt path:
      1. Explicit `ckpt_override`
      2. LIMO_V3_1 env var (when version=="v3_1")
      3. V3_1_DEFAULT (when version=="v3_1")
      4. latents_meta_ckpt (when version=="v1")
    """
    base = Path(base)

    # Decide version
    if version is None:
        env_v3 = os.environ.get("LIMO_V3_1", "").strip()
        version = "v3_1" if env_v3 else "v1"
    version = version.lower()

    if version == "v3_1":
        ckpt = ckpt_override or os.environ.get("LIMO_V3_1") or V3_1_DEFAULT
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = base / ckpt_path
        from limo_v3_1_model import LIMOVAEv3_1
        cb = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = LIMOVAEv3_1()
        m.load_state_dict(cb.get("model_state") or cb)
        m.to(device).eval()
        return LIMOInferenceWrapper(m, "v3_1"), "v3_1"

    # Default v1 path
    ckpt = ckpt_override or latents_meta_ckpt
    if ckpt is None:
        raise ValueError("v1 LIMO load requires either ckpt_override or "
                          "latents_meta_ckpt (from latents.pt['meta']['checkpoint'])")
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = base / ckpt_path
    cb = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = LIMOVAE()
    m.load_state_dict(cb["model_state"])
    m.to(device).eval()
    return LIMOInferenceWrapper(m, "v1"), "v1"


def decode_logits(limo, z: torch.Tensor, version: str = "v1") -> torch.Tensor:
    """Wrap decoding so v3.1 routes through autoregressive generation.
    v1: limo.decode(z) is parallel non-autoregressive — direct call.
    v3.1: must use _generate_autoregressive(z, emb) — but emb is from encoder
    which was bypassed (we only have z from diffusion). Skip-connection is
    optional: if v3.1 was trained with skip_connection=True, we'd need emb
    here. For diffusion-sample decode (no source SMILES), we can pass emb=None
    and rely on z alone; the decoder still cross-attends to z-memory tokens.
    """
    if version == "v3_1":
        # call generate without emb (skip is bypassed at inference for novel z's)
        with torch.no_grad():
            return limo._generate_autoregressive(z, emb=None)
    else:
        with torch.no_grad():
            return limo.decode(z)
