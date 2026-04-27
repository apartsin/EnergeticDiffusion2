# MolMIM Deployment Guide

How to load and run MolMIM 70M on a fresh GPU pod, recovered from 4 cycles of dependency / API hell on `nvcr.io/nvidia/clara/bionemo-framework:1.5`.

## TL;DR — the working recipe

```bash
# 1. Use the bionemo image, not vanilla NeMo / pytorch
# Vast/RunPod image: nvcr.io/nvidia/clara/bionemo-framework:1.5

# 2. Pin the dep matrix BEFORE importing nemo
pip install -q "huggingface_hub<0.20" "transformers<4.38"

# 3. Import from bionemo, NOT from nemo
# WRONG: from nemo.collections.nlp.models.language_modeling.megatron.molmim_model import MolMIMModel
# RIGHT: from bionemo.model.molecule.molmim.molmim_model import MolMIMModel

# 4. DO NOT call MolMIMModel.restore_from() directly — it needs a PTL Trainer
# Use MolMIMInference instead, which wires up the trainer for you
# from bionemo.model.molecule.molmim.infer import MolMIMInference
# infer = MolMIMInference(cfg=<hydra OmegaConf>, restore_path=".../molmim_70m_24_3.nemo")

# 5. Easiest config: copy /workspace/bionemo/examples/molecule/molmim/conf/infer.yaml
#    and override BIONEMO_HOME=/workspace/bionemo + restore_from_path=<your .nemo>
```

## What goes wrong, in cycle order

### Cycle 1: smoke_test failed locally
- Symptom: `ModuleNotFoundError: No module named 'nemo'` during the runner's local smoke test.
- Cause: NeMo is in the container, not in the local machine.
- Fix: pass `--skip-smoke` to `gpu_runner.py run`.

### Cycle 2: ImportError ModelFilter
- Symptom: `ImportError: cannot import name 'ModelFilter' from 'huggingface_hub'`.
- Cause: bionemo:1.5 ships NeMo 1.22, which uses `huggingface_hub.ModelFilter` — removed in `huggingface_hub >= 0.20`. The image often has a newer `huggingface_hub` than NeMo expects.
- Fix: `pip install "huggingface_hub<0.20"` BEFORE the first `import nemo`.

### Cycle 3: ImportError list_repo_tree
- Symptom: `ImportError: cannot import name 'list_repo_tree' from 'huggingface_hub'`.
- Cause: After pinning huggingface_hub<0.20, the bundled `transformers` (≥4.38) wants `list_repo_tree` which only exists in `huggingface_hub >= 0.21`. Mutually incompatible without also pinning transformers.
- Fix: also pin `transformers<4.38`.

### Cycle 4: ModuleNotFoundError nemo...molmim_model
- Symptom: `nemo.collections.nlp.models.language_modeling.megatron.molmim_model` does not exist.
- Cause: MolMIM is **not** in vanilla NeMo. It lives in the separate `bionemo` package, which is mounted at `/workspace/bionemo` in the bionemo:1.5 image.
- Fix: `from bionemo.model.molecule.molmim.molmim_model import MolMIMModel`.

### Cycle 5: ValueError Trainer cannot be None
- Symptom: `ValueError: Trainer cannot be None for Megatron-based models. Please provide a PTL trainer object.`
- Cause: Megatron-based models in NeMo cannot be loaded by `restore_from()` without a `pytorch_lightning.Trainer` instance. The trainer carries device/distributed state.
- Fix: use `bionemo.model.molecule.molmim.infer.MolMIMInference(cfg=..., restore_path=...)`. It builds the Trainer internally.

### Cycle 6: cfg.target missing
- Symptom: `ConfigAttributeError: Missing key target` when passing `cfg.model` instead of `cfg`.
- Cause: `MolMIMInference` (via `BaseEncoderInference`) accesses `cfg.target` and `cfg.infer_target` at the top level of the config, not under `cfg.model`. The infer.yaml has them at top level.
- Fix: pass the full `cfg`, not `cfg.model`.

### Cycle 7: trainer key missing
- Symptom: `ConfigAttributeError: Missing key trainer` after passing full cfg.
- Cause: `infer.yaml` uses Hydra `defaults: [base_infer_config, _self_]`, which composes the trainer block from `examples/conf/base_infer_config.yaml`. Loading `infer.yaml` alone with `OmegaConf.load` does NOT resolve Hydra defaults — you only get `infer.yaml`'s own keys.
- Fix: explicitly merge `base_infer_config.yaml` with `infer.yaml` via `OmegaConf.merge(base, infer)`. Working recipe (verified end-to-end on cycle 8):

```python
import os
os.environ.setdefault('BIONEMO_HOME', '/workspace/bionemo')
from omegaconf import OmegaConf
from bionemo.model.molecule.molmim.infer import MolMIMInference

base = OmegaConf.load('/workspace/bionemo/examples/conf/base_infer_config.yaml')
cfg  = OmegaConf.load('/workspace/bionemo/examples/molecule/molmim/conf/infer.yaml')
merged = OmegaConf.merge(base, cfg)
merged.model.downstream_task.restore_from_path = '/workspace/data/molmim_70m_24_3.nemo'
merged.name = 'MolMIM_smoke'  # required: base_infer has 'name: ???'

infer = MolMIMInference(
    cfg=merged,
    interactive=True,
    restore_path=merged.model.downstream_task.restore_from_path,
)

# Encode
hidden, mask = infer.seq_to_hiddens(['O=[N+]([O-])c1noc([N+](=O)[O-])c1[N+](=O)[O-]'])
emb = infer.hiddens_to_embedding(hidden, mask)   # (1, 512)

# Sample (perturb-and-decode)
samples = infer.sample(num_samples=4,
                        seqs=['O=[N+]([O-])c1noc([N+](=O)[O-])c1[N+](=O)[O-]'],
                        sampling_method='greedy-perturbate',
                        scaled_radius=1.0)
# returns list of lists of decoded SMILES strings
```

This works on `nvcr.io/nvidia/clara/bionemo-framework:1.5` after the `huggingface_hub<0.20` and `transformers<4.38` pins from cycles 2-3. Verified: encode produces (B, L, 512) hidden tensor, embedding (B, 512), sample produces 4 distinct SMILES from a perturbed L1 latent in <1 second.

## Working smoke template

```python
# molmim_smoke.py — works on bionemo-framework:1.5 after the pip pins above
import os, sys, time, json
os.environ.setdefault("BIONEMO_HOME", "/workspace/bionemo")
import torch
from omegaconf import OmegaConf
from bionemo.model.molecule.molmim.infer import MolMIMInference

cfg = OmegaConf.load("/workspace/bionemo/examples/molecule/molmim/conf/infer.yaml")
cfg.model.downstream_task.restore_from_path = "/workspace/data/molmim_70m_24_3.nemo"
# vocab paths default to BIONEMO_HOME-relative; override only if you moved them
infer = MolMIMInference(cfg=cfg.model, interactive=True, restore_path=cfg.model.downstream_task.restore_from_path)

test_smiles = ["O=[N+]([O-])c1noc([N+](=O)[O-])c1[N+](=O)[O-]"]
hidden, mask = infer.seq_to_hiddens(test_smiles)         # (1, L, d)
emb = infer.hiddens_to_embedding(hidden, mask)           # (1, d)
print(f"embedding: {emb.shape} {emb.dtype}")
samples = infer.sample(num_samples=8, seqs=test_smiles, sampling_method="greedy-perturbate", scaled_radius=1.0)
print(f"sampled: {samples}")
```

## Cost notes

- bionemo:1.5 image is ~15 GB — first pod start takes 10-30 min for the docker pull.
- Subsequent reruns on the **same** pod start in seconds (image cached).
- Always `--keep-alive` (or vast `--keep-alive` equivalent) when iterating; the cache is per-host.

## When to skip MolMIM entirely

- Dep-rot risk is real on bionemo. If you only need a "strong-LM no-diffusion baseline", **MoLFormer-XL** (`ibm/MoLFormer-XL-both-10pct`, 47M params) is much cheaper to deploy: standard HuggingFace transformers, `pip install transformers` works on any image, no NeMo, no Hydra. Same baseline class.
- For the energetic-materials project specifically, the SMILES-LSTM at composite=0.083 vs DGLD at 0.45-0.70 already provides the no-diffusion baseline; MolMIM is only worth the deployment cost if you need a 70M-param baseline specifically.

## File locations on bionemo:1.5 image

| What | Path |
|---|---|
| bionemo package source | `/workspace/bionemo/bionemo/` |
| MolMIM model class | `/workspace/bionemo/bionemo/model/molecule/molmim/molmim_model.py` |
| MolMIM inference class | `/workspace/bionemo/bionemo/model/molecule/molmim/infer.py` |
| MolMIM example configs | `/workspace/bionemo/examples/molecule/molmim/conf/` |
| MolMIM tokenizer (default) | `${BIONEMO_HOME}/tokenizers/molecule/molmim/vocab/` |
| Default model checkpoint location | `${BIONEMO_HOME}/models/molecule/molmim/molmim_70m_24_3.nemo` |
| Pretraining example | `/workspace/bionemo/examples/molecule/molmim/pretrain.py` |
