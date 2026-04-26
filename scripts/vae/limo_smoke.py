"""
Smoke test for LIMO VAE:

  1. Load pretrained checkpoint + rebuild tokenizer
  2. Run encode/decode round-trip on 200 labeled energetic SMILES
  3. Measure reconstruction accuracy, SELFIES validity, and vocab coverage
  4. Report findings (pass/fail) and save a short JSON summary

Exit code:
    0 = PASS  (fine-tune can proceed)
    1 = FAIL  (vocab incompatible or model loads corruptly)

Usage:
    python scripts/vae/limo_smoke.py [--n 200] [--out smoke_report.json]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Counter
from collections import Counter as _Counter

import pandas as pd
import torch
import selfies as sf
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).parent))
from limo_model import (
    LIMOVAE, SELFIESTokenizer, build_limo_vocab, save_vocab, load_vocab,
    LIMO_PAD_TOKEN, LIMO_MAX_LEN, find_limo_repo,
    load_limo_model_and_tokenizer,
)


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",     type=int, default=200)
    ap.add_argument("--out",   default="external/LIMO/smoke_report.json")
    ap.add_argument("--base",  default="E:/Projects/EnergeticDiffusion2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    base = Path(args.base)
    print(f"Device: {args.device}")
    print(f"Base:   {base}")

    # ── Step 1: sample labeled molecules ─────────────────────────────────────
    print("\n[1/4] Loading sample SMILES from labeled_master …")
    lm = pd.read_csv(base / "data/training/master/labeled_master.csv",
                     low_memory=False, usecols=["smiles"])
    # take energetic-biased sample: first N unique
    sample = lm["smiles"].dropna().drop_duplicates().head(args.n).tolist()
    print(f"  {len(sample)} SMILES to test")

    # ── Step 2: load LIMO model + tokenizer ──────────────────────────────────
    print("\n[2/4] Loading LIMO checkpoint and rebuilding tokenizer …")
    t0 = time.time()
    limo_dir = find_limo_repo(base)
    print(f"  LIMO dir: {limo_dir}")
    vocab_cache = limo_dir / "vocab_cache.json"
    if vocab_cache.exists():
        alphabet = load_vocab(vocab_cache)
        print(f"  loaded cached vocab ({len(alphabet)} tokens)")
    else:
        print("  rebuilding vocab from zinc250k.smi (one-time) …")
        alphabet = build_limo_vocab(limo_dir / "zinc250k.smi")
        save_vocab(alphabet, vocab_cache)
        print(f"  vocab built: {len(alphabet)} tokens, cached")

    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    model = LIMOVAE()
    missing, unexpected = model.load_limo_weights(limo_dir / "vae.pt", strict=True)
    model.to(args.device).eval()
    print(f"  model loaded in {time.time()-t0:.1f}s  missing={len(missing)}  unexpected={len(unexpected)}")

    # ── Step 3: vocabulary coverage ──────────────────────────────────────────
    print("\n[3/4] Checking vocabulary coverage on our SMILES …")
    oov_tokens: Counter[str] = _Counter()
    n_oov_molecules = 0
    n_selfies_fail  = 0
    n_too_long      = 0
    per_mol = []
    for smi in sample:
        try:
            sfs = sf.encoder(smi)
        except Exception:
            n_selfies_fail += 1
            per_mol.append({"smi": smi, "sfs_ok": False})
            continue
        toks = list(sf.split_selfies(sfs))
        if len(toks) > LIMO_MAX_LEN:
            n_too_long += 1
        oov = [t for t in toks if t not in tok.sym_to_idx]
        if oov:
            n_oov_molecules += 1
            for t in oov:
                oov_tokens[t] += 1
        per_mol.append({"smi": smi, "sfs_ok": True,
                        "n_tokens": len(toks),
                        "n_oov": len(oov), "too_long": len(toks) > LIMO_MAX_LEN})
    print(f"  SELFIES encode fails : {n_selfies_fail}")
    print(f"  too long (>{LIMO_MAX_LEN} tokens): {n_too_long}")
    print(f"  molecules with ≥1 OOV token: {n_oov_molecules}")
    print(f"  distinct OOV tokens: {len(oov_tokens)}")
    if oov_tokens:
        top = oov_tokens.most_common(10)
        print(f"  top OOV: {top}")

    # ── Step 4: round-trip reconstruction ────────────────────────────────────
    print("\n[4/4] Round-trip encode/decode reconstruction test …")
    n_exact_smiles   = 0
    n_exact_selfies  = 0
    n_valid_decoded  = 0
    total_tokens     = 0
    correct_tokens   = 0

    # build input batch
    valid_indices = []
    valid_sources = []
    for i, smi in enumerate(sample):
        t = tok.smiles_to_tensor(smi)
        if t is None:
            continue
        indices, _ = t
        valid_indices.append(indices)
        valid_sources.append(smi)
    if not valid_indices:
        print("  FAIL: no molecules survived tokenization")
        return 1

    x = torch.stack(valid_indices).to(args.device)
    with torch.no_grad():
        log_probs, z, mu, log_var = model(x)
        # deterministic decode: argmax over vocab
        preds = log_probs.argmax(dim=2)  # (B, max_len)

    preds_cpu = preds.cpu()
    x_cpu     = x.cpu()
    for i, smi in enumerate(valid_sources):
        gt_indices   = x_cpu[i].tolist()
        pred_indices = preds_cpu[i].tolist()
        # token accuracy over non-pad positions
        for g, p in zip(gt_indices, pred_indices):
            if g != tok.pad_idx:
                total_tokens += 1
                if g == p:
                    correct_tokens += 1
        # reconstructed SMILES
        recon_smi = tok.indices_to_smiles(pred_indices)
        recon_sfs = tok.indices_to_selfies(pred_indices)
        gt_sfs    = tok.indices_to_selfies(gt_indices)
        if recon_smi and Chem.MolFromSmiles(recon_smi) is not None:
            n_valid_decoded += 1
        # canonical SMILES equality
        gt_c    = canon(smi)
        recon_c = canon(recon_smi) if recon_smi else None
        if gt_c and recon_c and gt_c == recon_c:
            n_exact_smiles += 1
        if gt_sfs == recon_sfs:
            n_exact_selfies += 1

    n = len(valid_sources)
    pct_exact_smi  = 100 * n_exact_smiles  / n
    pct_exact_sfs  = 100 * n_exact_selfies / n
    pct_valid_dec  = 100 * n_valid_decoded / n
    pct_token_acc  = 100 * correct_tokens  / max(total_tokens, 1)
    print(f"  round-trip on {n} molecules:")
    print(f"    exact SMILES reconstruction: {pct_exact_smi:.1f}%  ({n_exact_smiles}/{n})")
    print(f"    exact SELFIES reconstruction: {pct_exact_sfs:.1f}%")
    print(f"    decoded molecule valid (parses as SMILES): {pct_valid_dec:.1f}%")
    print(f"    token-level accuracy: {pct_token_acc:.1f}%")

    # ── verdict ──────────────────────────────────────────────────────────────
    PASS_TOKEN_ACC = 70.0    # on OOD energetic molecules this is already ambitious
    PASS_VALID_DEC = 50.0
    verdict_pass = (pct_token_acc >= PASS_TOKEN_ACC
                    and pct_valid_dec >= PASS_VALID_DEC
                    and len(missing) == 0)

    print("\n" + "="*72)
    print(f"VERDICT: {'PASS ✓' if verdict_pass else 'FAIL ✗'}")
    print(f"  weights loaded cleanly:   {len(missing) == 0}  (missing={len(missing)}, unexpected={len(unexpected)})")
    print(f"  token accuracy ≥ {PASS_TOKEN_ACC}%:   {pct_token_acc >= PASS_TOKEN_ACC}  ({pct_token_acc:.1f}%)")
    print(f"  valid decode ≥ {PASS_VALID_DEC}%:    {pct_valid_dec >= PASS_VALID_DEC}  ({pct_valid_dec:.1f}%)")
    print("="*72)

    # save report
    report = {
        "timestamp":                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device":                   args.device,
        "n_molecules_tested":       args.n,
        "n_molecules_valid":        n,
        "vocab_size":               tok.vocab_size,
        "max_len":                  tok.max_len,
        "selfies_encode_failures":  n_selfies_fail,
        "too_long_count":           n_too_long,
        "oov_molecules":            n_oov_molecules,
        "distinct_oov_tokens":      len(oov_tokens),
        "top_oov":                  oov_tokens.most_common(20),
        "exact_smiles_pct":         pct_exact_smi,
        "exact_selfies_pct":        pct_exact_sfs,
        "valid_decoded_pct":        pct_valid_dec,
        "token_accuracy_pct":       pct_token_acc,
        "weights_load_missing":     list(missing),
        "weights_load_unexpected":  list(unexpected),
        "verdict_pass":             verdict_pass,
    }
    out_path = base / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {out_path}")

    return 0 if verdict_pass else 1


if __name__ == "__main__":
    sys.exit(main())
