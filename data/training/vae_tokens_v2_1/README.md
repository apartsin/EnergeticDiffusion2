# Tokenization cache — shared

This directory holds the SELFIES token cache for the **motif-augmented**
dataset (`labeled_motif_aug.csv`, 1.22 M rows after rare-motif × 5 and
polynitro × 2 oversampling).

The cache filename embeds a content-hash of (SMILES set, vocab, max_len,
drop_oov, drop_too_long). Any LIMO trainer with the same data + settings
hits the same hash → all share this file.

**Reuse policy**: every motif-augmented LIMO variant (v2.1b, v3, v3.1,
v3.2, …) should point its `paths.cache_dir` at this folder, not a fresh
one. Re-tokenising 1.22 M rows takes ~3 min and 630 MB; no reason to
duplicate.

| Cache file | Hash | Built by | Bytes |
|---|---|---|---|
| `energetic_ft_0586b876335b.pt` | `0586b876335b` | LIMO v2.1b (2026-04-26) | 630 650 933 |

## v1 cache lives separately

`data/training/vae_tokens/energetic_ft_21ee624f3df7.pt` (190 MB) is the
v1 cache, built from the pre-augmentation 326 k unique SMILES. Don't
mix.
