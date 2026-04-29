"""Feasibility helpers: load latent SA/SC surrogates, plus real RDKit/SCScorer
for post-hoc reranking.

Two distinct uses:
  - latent surrogate (differentiable, fast)  → classifier guidance during DDIM
  - real scorers (exact, ms per molecule)    → composite reranking, hard filter
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

BASE = Path("E:/Projects/EnergeticDiffusion2")

DEFAULT_SA_CKPT = BASE / "experiments/guidance_sa_sc_20260424T172820Z/checkpoints/sa_best.pt"
DEFAULT_SC_CKPT = BASE / "experiments/guidance_sa_sc_20260424T172820Z/checkpoints/sc_best.pt"


# ── latent surrogate ──────────────────────────────────────────────────────
class _ScorePredictor(nn.Module):
    """Mirror of scripts/guidance/model.py:ScorePredictor (kept local to
    avoid the heavier guidance module's training-time imports)."""
    def __init__(self, in_dim: int = 1024, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z): return self.net(z).squeeze(-1)


def load_surrogate(path: str | Path, device: str = "cuda"
                    ) -> tuple[nn.Module, dict] | None:
    """Loads (model_eval, stats={'mean','std'}). Returns None on failure
    so callers can fall back gracefully."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        ck = torch.load(p, map_location=device, weights_only=False)
        m = _ScorePredictor().to(device)
        m.load_state_dict(ck["model_state"])
        m.eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        stats = ck.get("stats", {"mean": 0.0, "std": 1.0})
        return m, stats
    except Exception as exc:
        print(f"[feasibility] surrogate load failed for {p.name}: {exc}")
        return None


# ── real scorers (RDKit + SCScorer) ───────────────────────────────────────
_REAL_STATE = {"sa": None, "sc": None}


def _init_real_sa():
    if _REAL_STATE["sa"] is not None: return _REAL_STATE["sa"]
    sys.path.insert(0, str(BASE / "external" / "LIMO"))
    import sascorer
    _REAL_STATE["sa"] = sascorer
    return sascorer


def _init_real_sc():
    if _REAL_STATE["sc"] is not None: return _REAL_STATE["sc"]
    sys.path.insert(0, str(BASE / "external" / "scscore"))
    from scscore.standalone_model_numpy import SCScorer
    m = SCScorer()
    m.restore(str(BASE / "external/scscore/models/full_reaxys_model_1024bool"
                          "/model.ckpt-10654.as_numpy.json.gz"))
    _REAL_STATE["sc"] = m
    return m


def real_sa(smi: str) -> float:
    from rdkit import Chem
    try:
        sa = _init_real_sa()
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return float("nan")
        return float(sa.calculateScore(mol))
    except Exception:
        return float("nan")


def real_sc(smi: str) -> float:
    try:
        m = _init_real_sc()
        _, val = m.get_score_from_smi(smi)
        return float(val)
    except Exception:
        return float("nan")


def real_sa_sc(smi: str) -> tuple[float, float]:
    return real_sa(smi), real_sc(smi)


# ── thresholds (literature defaults; can be overridden in CLIs) ───────────
SA_DROP_ABOVE = 6.5     # very hard to synthesise above this
SC_DROP_ABOVE = 4.5
SA_PENALTY_THRESHOLD = 4.0   # start penalising in composite above this
SC_PENALTY_THRESHOLD = 3.0


def composite_feasibility_penalty(sa: float, sc: float,
                                   w_sa: float = 1.0,
                                   w_sc: float = 0.5) -> float:
    """Returns scalar to ADD to the rerank composite (higher = worse).
    Hard caps are enforced by callers (drop entirely when above
    SA_DROP_ABOVE / SC_DROP_ABOVE).
    """
    pen = 0.0
    if not _is_nan(sa): pen += w_sa * max(0.0, sa - SA_PENALTY_THRESHOLD)
    if not _is_nan(sc): pen += w_sc * max(0.0, sc - SC_PENALTY_THRESHOLD)
    return pen


def _is_nan(x: float) -> bool:
    return x != x   # NaN != NaN
