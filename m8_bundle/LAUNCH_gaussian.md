# Gaussian Latent Baseline: Launch Instructions

Closes §5.5.2: "A compute-matched Gaussian baseline (40k samples) is scoped as future work."

## What this does

1. Spins up a Modal A100 worker.
2. Loads the frozen LIMO VAE from `combo_bundle/limo_best.pt`.
3. Samples 40 000 latents from N(0, I_1024) in chunks of 512.
4. Decodes each chunk through the LIMO decoder to get raw SMILES.
5. Downloads results locally and runs the same post-processing pipeline as DGLD:
   - RDKit validity + canonical dedup
   - Tanimoto novelty window [0.20, 0.55] vs labelled master (5 k rows)
   - SA score <= 4.5 cap
   - SC score <= 4.0 cap (if scscore is installed)
   - Ranks by composite = (1 - max_tanimoto) x N-fraction proxy
6. Saves:
   - `m8_bundle/results/gaussian_latent_40k_raw.txt` (40 000 lines, raw SMILES)
   - `m8_bundle/results/gaussian_latent_40k_top100.json` (top-100 + paper metrics)

## Prerequisites

```bash
pip install modal
modal token new          # one-time auth
```

`combo_bundle/limo_best.pt` must exist (it is uploaded automatically by the script).

## Run

From the project root:

```bash
python -m modal run m8_bundle/modal_gaussian_latent_40k.py
```

Expected wall time: 20-40 minutes total (A100 decode ~15 min; postprocess ~5 min locally).

## Key paper metrics printed on completion

```
n_raw              : 40000
n_valid_smiles     : <N>
n_novel            : <N>
keep_rate          : X.XX%   (DGLD reference: 2.41%)
top1_smiles        : ...
top1_maxtan        : ...
top1_composite     : ...
topN_mean_composite: ...
```

The `keep_rate` vs DGLD's 2.41% (966 / 40 000) is the core §5.5.2 result.
A much lower keep-rate for the Gaussian baseline confirms that DGLD's diffusion
process is essential, not just LIMO decoding from random latents.

## Labelled master CSV

Post-processing expects a CSV with a `smiles` (or `SMILES`) column at one of:

```
data/processed/labelled_master.csv   # preferred
data/labelled_master.csv
data/raw/labelled_master.csv
combo_bundle/corpus.csv              # fallback
```

If none exist, the novelty window filter is skipped and keep-rate reflects
validity + SA/SC only. The JSON output notes this in the filter configuration.

## Reproducing with a different seed

```bash
# Edit the main() call or pass via env:
python -c "
import modal
# Modify n_total / seed in the local_entrypoint before running
"
```

Alternatively, edit the `sample_gaussian_latents_remote.remote(seed=42)` call
in the `main()` local entrypoint and re-run.

## Output interpretation for §5.5.2

Insert into the paper table at §5.5.2:

| Method             | Pool   | Valid  | Novel | Keep-rate | Top-1 composite |
|--------------------|--------|--------|-------|-----------|-----------------|
| Gaussian (N(0,I))  | 40 000 | TBD    | TBD   | TBD %     | TBD             |
| DGLD (this work)   | 40 000 | ...    | 966   | 2.41 %    | ...             |

Fill TBD columns from `gaussian_latent_40k_top100.json` after the run completes.
