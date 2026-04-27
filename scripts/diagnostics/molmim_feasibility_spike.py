"""Stage-0 feasibility spike for NVIDIA MolMIM.

Goal: verify whether MolMIM weights can be loaded and used outside the
BioNeMo Docker container, ideally via HuggingFace.

Runs CPU-only (so it doesn't conflict with concurrent GPU training).

Output: docs/diag_molmim_feasibility.md with the verdict.
"""
from __future__ import annotations
import sys, traceback, urllib.request
from pathlib import Path
from typing import Any

BASE = Path("E:/Projects/EnergeticDiffusion2")
REPORT = BASE / "docs/diag_molmim_feasibility.md"
DEVICE = "cpu"

KNOWN_TEST_SMILES = [
    "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",        # TNT
    "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1",         # RDX
    "O=[N+]([O-])N1[C@H]2[C@@H]3N([N+](=O)[O-])[C@@H]4[C@H]"
    "(N2[N+](=O)[O-])N([N+](=O)[O-])[C@H]([C@H]1N3[N+](=O)[O-])"
    "N4[N+](=O)[O-]",                                            # CL-20
    "Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]",   # TATB
]

def out(line: str = ""):
    """Write to stdout AND collect for report."""
    print(line)
    out._buf.append(line)
out._buf = []


# ── 1. HuggingFace catalog probe ────────────────────────────────────────
HF_CANDIDATES = [
    "nvidia/MolMIM",
    "nvidia/molmim",
    "nvidia/molmim-base",
    "ibm/MoLFormer-XL-both-10pct",   # known to exist; control case
]


def try_huggingface():
    out("\n## 1. HuggingFace probe")
    out("")
    try:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
    except ImportError:
        out("- huggingface_hub not installed; install with `pip install huggingface_hub`")
        return False, None

    found = []
    for repo in HF_CANDIDATES:
        try:
            info = api.repo_info(repo)
            files = list(api.list_repo_files(repo))[:10]
            out(f"- ✓ `{repo}` exists ({len(files)}+ files)")
            out(f"  - sample files: {files[:5]}")
            found.append(repo)
        except Exception as e:
            err = str(e).split("\n")[0][:80]
            out(f"- ✗ `{repo}`: {err}")

    return bool(found), found


def try_transformers_load(repo: str):
    out(f"\n## 2. Transformers `from_pretrained` on `{repo}`")
    out("")
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
    except ImportError:
        out("- transformers not installed")
        return False
    try:
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        out(f"- config loaded; arch: `{cfg.__class__.__name__}` "
            f"vocab_size={getattr(cfg, 'vocab_size', '?')}")
    except Exception as e:
        out(f"- ✗ AutoConfig failed: {str(e)[:200]}")
        return False
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        out(f"- tokenizer loaded; class: `{tok.__class__.__name__}`")
        sample = tok(KNOWN_TEST_SMILES[0], return_tensors="pt")
        out(f"  - tokenises TNT to {sample['input_ids'].shape[1]} tokens")
    except Exception as e:
        out(f"- ✗ tokenizer failed: {str(e)[:200]}")
        return False
    try:
        m = AutoModel.from_pretrained(repo, trust_remote_code=True).to(DEVICE).eval()
        n_params = sum(p.numel() for p in m.parameters())
        out(f"- model loaded; {n_params/1e6:.1f} M params on {DEVICE}")
    except Exception as e:
        out(f"- ✗ model load failed: {str(e)[:200]}")
        return False
    # smoke encode
    try:
        import torch
        with torch.no_grad():
            out_ = m(**sample)
        out(f"- forward pass OK; output keys: {list(out_.keys()) if hasattr(out_, 'keys') else 'tensor'}")
        # Try to extract a hidden state
        if hasattr(out_, "last_hidden_state"):
            h = out_.last_hidden_state
            out(f"  - last_hidden_state shape: {tuple(h.shape)}")
        elif hasattr(out_, "pooler_output"):
            out(f"  - pooler shape: {tuple(out_.pooler_output.shape)}")
    except Exception as e:
        out(f"- ✗ forward failed: {str(e)[:200]}")
        return False
    return True


def try_ngc_catalog():
    out("\n## 3. NVIDIA NGC catalog probe (anonymous)")
    out("")
    catalog_urls = [
        "https://catalog.ngc.nvidia.com/api/orgs/nvidia/teams/clara/models",
        "https://api.ngc.nvidia.com/v2/orgs/nvidia/teams/clara/models",
    ]
    for u in catalog_urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                txt = r.read(2000).decode("utf-8", errors="replace")
                hit = "molmim" in txt.lower()
                out(f"- {u}: {'HIT' if hit else 'no MolMIM mention'} "
                    f"(first {min(80, len(txt))}b: `{txt[:80]}…`)")
        except Exception as e:
            out(f"- {u}: {str(e)[:80]}")


def try_bionemo_pypi():
    out("\n## 4. BioNeMo as Python package (no Docker)")
    out("")
    try:
        import bionemo  # noqa
        out("- ✓ `bionemo` already importable")
    except ImportError:
        out("- `bionemo` not installed (would require pip install)")
    # Don't actually pip install — too invasive for a spike.


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    out("# MolMIM stage-0 feasibility spike")
    out("")
    out("Cutoff: training-data Jan 2026 — verify against current state.")
    out("Test SMILES: " + ", ".join(KNOWN_TEST_SMILES[:2]) + "…")

    found, repos = try_huggingface()
    if found:
        for r in repos:
            try_transformers_load(r)

    try_ngc_catalog()
    try_bionemo_pypi()

    out("\n## Verdict")
    out("")
    out("- If section 1 found a MolMIM HF repo AND section 2 loaded it cleanly →")
    out("  **Path A — use HuggingFace mirror, ~30 min integration**")
    out("- If section 3 (NGC) returned a downloadable .nemo URL →")
    out("  **Path B — extract .nemo offline, half-day work**")
    out("- If only Docker route works →")
    out("  **Path C — shelve, pursue ChemBERTa hybrid instead**")
    out("- If nothing works → see ChemBERTa hybrid plan in")
    out("  `docs/limo_diagnostics_extended.md`")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(out._buf), encoding="utf-8")
    print(f"\nSaved {REPORT}")


if __name__ == "__main__":
    sys.exit(main())
