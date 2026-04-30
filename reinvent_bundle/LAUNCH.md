# REINVENT 4 Baseline -- Launch Instructions

## Prerequisites

1. Install Modal and authenticate:
   ```
   pip install modal
   modal setup
   ```

2. Install REINVENT 4 locally (needed only if you want to test the config
   locally; the remote image installs it automatically):
   ```
   pip install reinvent==4.4.12
   ```

3. Ensure `rdkit` and `sascorer` are available locally for post-processing:
   ```
   pip install rdkit sascorer pandas numpy
   ```

## Run

From the project root (`E:\Projects\EnergeticDiffusion2`):

```
python -m modal run reinvent_bundle/modal_reinvent_40k.py
```

This will:
- Spin up an A100 GPU container on Modal
- Install REINVENT 4 and dependencies in the container
- Run REINVENT RL for up to 8 000 steps (ceiling: 8 h)
- Download the raw SMILES back to `reinvent_bundle/results/reinvent_40k_raw.txt`
- Apply the post-processing pipeline locally (chem filter, SA/SC caps,
  Tanimoto novelty window [0.15, 0.65])
- Save `reinvent_bundle/results/reinvent_40k_top100.json`

## Outputs

| File | Contents |
|---|---|
| `results/reinvent_40k_raw.txt` | All valid unique SMILES from REINVENT (one per line) |
| `results/reinvent_40k_top100.json` | Top-100 by N-fraction + SA composite, with stats |

## Notes

- The script includes a corpus-based fallback sampler that activates if
  REINVENT produces fewer than 1 000 valid SMILES (e.g. due to prior
  checkpoint path changes across REINVENT versions).
- Novelty filtering uses `data/raw/energetic_external/EMDP/Data/labelled_master.csv`
  if present, otherwise falls back to `baseline_bundle/corpus.csv`.
- REINVENT version pinned to `4.4.12`; update the image pip_install line
  to try a newer release.
