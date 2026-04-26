"""
Offline preprocessing for VAE training.

One-shot: read labeled_master + unlabeled_master, canonicalize SMILES, convert
to SELFIES, tokenize, build vocabulary, save pre-padded int tensors to disk.

Run once, then train.py loads the cached tensors directly. Saves ~15% of
training budget versus tokenizing every epoch.

Usage:
    python scripts/vae/prepare_data.py --config configs/vae.yaml
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import selfies as sf

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("prep")


# ── helpers ──────────────────────────────────────────────────────────────────
def canonicalize(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def smiles_to_selfies_safe(smi: str) -> str | None:
    try:
        return sf.encoder(smi)
    except Exception:
        return None


def build_token_list(smi_series: pd.Series, representation: str) -> pd.DataFrame:
    """Canonicalize, convert to SELFIES (or keep SMILES), tokenize.

    Returns DataFrame with columns: smiles, selfies (may be empty for SMILES mode),
    tokens (list), length.
    """
    rows = []
    n_invalid = n_sf_fail = n_ok = 0
    for s in smi_series:
        if not isinstance(s, str) or not s:
            n_invalid += 1
            continue
        c = canonicalize(s)
        if c is None:
            n_invalid += 1
            continue
        if representation == "selfies":
            sf_str = smiles_to_selfies_safe(c)
            if sf_str is None:
                n_sf_fail += 1
                continue
            toks = list(sf.split_selfies(sf_str))
        else:  # smiles tokenization
            sf_str = ""
            toks = list(c)  # naive char-tokenization; could improve
        rows.append({"smiles": c, "selfies": sf_str,
                     "tokens": toks, "length": len(toks)})
        n_ok += 1
    log.info(f"   tokenized: valid {n_ok:,}  invalid {n_invalid:,}  sf-fail {n_sf_fail:,}")
    return pd.DataFrame(rows)


def build_vocab(token_lists: Iterable[list[str]]) -> dict[str, int]:
    """Build vocab with reserved special tokens. 0=<pad> 1=<bos> 2=<eos> 3=<unk>."""
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for toks in token_lists:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def encode_rows(df: pd.DataFrame, vocab: dict[str, int], max_len: int,
                drop_longer: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (int tensor [N, max_len+2], length tensor [N]). +2 for BOS/EOS."""
    bos = vocab["<bos>"]; eos = vocab["<eos>"]; pad = vocab["<pad>"]; unk = vocab["<unk>"]
    seqs = []
    lens = []
    n_trimmed = 0
    n_dropped = 0
    for toks in df["tokens"]:
        if len(toks) > max_len:
            if drop_longer:
                n_dropped += 1
                continue
            toks = toks[:max_len]
            n_trimmed += 1
        ids = [bos] + [vocab.get(t, unk) for t in toks] + [eos]
        # length *with* BOS and EOS included
        lens.append(len(ids))
        ids = ids + [pad] * (max_len + 2 - len(ids))
        seqs.append(ids)
    if n_trimmed:
        log.info(f"   trimmed {n_trimmed:,} rows to max_len")
    if n_dropped:
        log.info(f"   dropped {n_dropped:,} rows exceeding max_len")
    return torch.tensor(seqs, dtype=torch.int32), torch.tensor(lens, dtype=torch.int32)


def subset_energetic_biased(df: pd.DataFrame, um_full: pd.DataFrame,
                             labeled_smi: set[str]) -> pd.DataFrame:
    """Union of labeled + rnnmgm_ds9 + high-proxy + has_nitro/azide unlabeled."""
    um = um_full.copy()
    for col in ("has_nitro", "has_azide"):
        if col in um.columns:
            um[col] = um[col].astype(str).str.lower().isin(["true", "1"])
    # rnnmgm_ds9 unlabeled + energetic filters
    mask = (
        (um["source_dataset"] == "rnnmgm_ds9") |
        (um.get("energetic_proxy_score", 0) >= 6) |
        (um.get("has_nitro", False)) |
        (um.get("has_azide", False))
    )
    energetic_um = um.loc[mask, ["smiles"]]
    all_smi = pd.concat([pd.Series(list(labeled_smi)), energetic_um["smiles"]],
                        ignore_index=True)
    return pd.DataFrame({"smiles": all_smi.dropna().drop_duplicates()})


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out",    default=None,
                    help="Override cache dir (default: from config)")
    ap.add_argument("--quick",  action="store_true",
                    help="Subsample to 20k rows per stage for smoke test")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base = Path(os.environ.get("EDIFF2_BASE", "E:/Projects/EnergeticDiffusion2"))
    cache_dir = Path(args.out or cfg["data"]["cache_dir"])
    cache_dir = base / cache_dir if not cache_dir.is_absolute() else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"cache → {cache_dir}")

    max_len       = cfg["data"]["max_length"]
    drop_longer   = cfg["data"]["drop_longer"]
    representation = cfg["data"]["representation"]
    log.info(f"representation={representation}  max_len={max_len}")

    # ── load masters ─────────────────────────────────────────────────────────
    lm_path = base / cfg["data"]["labeled_master"]
    um_path = base / cfg["data"]["unlabeled_master"]
    log.info(f"loading {lm_path.name} …")
    lm = pd.read_csv(lm_path, low_memory=False, usecols=["smiles"])
    log.info(f"  {len(lm):,} labeled rows")
    log.info(f"loading {um_path.name} …")
    um = pd.read_csv(um_path, low_memory=False,
                     usecols=["smiles","source_dataset","has_nitro","has_azide",
                              "energetic_proxy_score"])
    log.info(f"  {len(um):,} unlabeled rows")

    if args.quick:
        log.info("QUICK MODE — subsampling to 20k per stage")
        lm = lm.head(20000)
        um = um.head(20000)

    # canonicalize + dedup once
    log.info("canonicalizing labeled SMILES …")
    t0 = time.time()
    lm["canon"] = lm["smiles"].apply(canonicalize)
    lm = lm.dropna(subset=["canon"]).drop_duplicates("canon")
    log.info(f"  {len(lm):,} unique  ({time.time()-t0:.1f}s)")
    log.info("canonicalizing unlabeled SMILES …")
    t0 = time.time()
    um["canon"] = um["smiles"].apply(canonicalize)
    um = um.dropna(subset=["canon"]).drop_duplicates("canon")
    log.info(f"  {len(um):,} unique  ({time.time()-t0:.1f}s)")

    # ── build two stage datasets ─────────────────────────────────────────────
    # Stage 1: all_union  (labeled ∪ unlabeled)
    all_union = pd.concat([lm[["canon"]].rename(columns={"canon": "smiles"}),
                           um[["canon"]].rename(columns={"canon": "smiles"})],
                          ignore_index=True).drop_duplicates("smiles")
    log.info(f"STAGE pretrain_broad: {len(all_union):,} unique SMILES")

    # Stage 2: energetic_biased
    labeled_smi = set(lm["canon"])
    um_renamed = um.rename(columns={"canon": "smiles"})
    energetic = subset_energetic_biased(pd.DataFrame(), um_renamed, labeled_smi)
    log.info(f"STAGE finetune_energetic: {len(energetic):,} unique SMILES")

    # ── tokenize union (shared across stages) ────────────────────────────────
    log.info("tokenizing all_union …")
    t0 = time.time()
    union_tok = build_token_list(all_union["smiles"], representation)
    log.info(f"  tokenized {len(union_tok):,} rows  ({time.time()-t0:.1f}s)")

    # vocab is built from union (so stage 2 is a subset of the same vocab)
    vocab = build_vocab(union_tok["tokens"])
    log.info(f"vocab size (incl. specials): {len(vocab)}")

    # Stage 2 subset by SMILES
    energetic_set = set(energetic["smiles"])
    ft_tok = union_tok[union_tok["smiles"].isin(energetic_set)].reset_index(drop=True)
    log.info(f"finetune subset size: {len(ft_tok):,}")

    # ── encode & save ────────────────────────────────────────────────────────
    log.info("encoding pretrain_broad …")
    pt_seq, pt_len = encode_rows(union_tok, vocab, max_len, drop_longer)
    log.info(f"  tensor shape {tuple(pt_seq.shape)}")
    log.info("encoding finetune_energetic …")
    ft_seq, ft_len = encode_rows(ft_tok, vocab, max_len, drop_longer)
    log.info(f"  tensor shape {tuple(ft_seq.shape)}")

    # Save
    log.info(f"saving → {cache_dir}")
    torch.save({"seq": pt_seq, "len": pt_len,
                "smiles": union_tok["smiles"].tolist()},
               cache_dir / "pretrain_broad.pt")
    torch.save({"seq": ft_seq, "len": ft_len,
                "smiles": ft_tok["smiles"].tolist()},
               cache_dir / "finetune_energetic.pt")
    with open(cache_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    with open(cache_dir / "meta.json", "w") as f:
        json.dump({
            "representation": representation,
            "max_length":     max_len,
            "vocab_size":     len(vocab),
            "pretrain_rows":  int(len(pt_seq)),
            "finetune_rows":  int(len(ft_seq)),
            "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
        }, f, indent=2)

    log.info("done.")


if __name__ == "__main__":
    main()
