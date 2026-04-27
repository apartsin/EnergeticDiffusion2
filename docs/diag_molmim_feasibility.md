# MolMIM stage-0 feasibility spike

Cutoff: training-data Jan 2026 — verify against current state.
Test SMILES: Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-], O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1…

## 1. HuggingFace probe

- ✗ `nvidia/MolMIM`: 404 Client Error. (Request ID: Root=1-69edb64b-725cb06e719c86eb3d031619;23062376
- ✗ `nvidia/molmim`: 404 Client Error. (Request ID: Root=1-69edb64b-5c0d472b4c634dcf03565355;4b35bad9
- ✗ `nvidia/molmim-base`: 404 Client Error. (Request ID: Root=1-69edb64b-1f01657c4cc4e3bd1005db47;843097e8
- ✓ `ibm/MoLFormer-XL-both-10pct` exists (10+ files)
  - sample files: ['.gitattributes', 'README.md', 'config.json', 'configuration_molformer.py', 'convert_molformer_original_checkpoint_to_pytorch.py']

## 2. Transformers `from_pretrained` on `ibm/MoLFormer-XL-both-10pct`

- ✗ AutoConfig failed: No module named 'transformers.onnx'

## 3. NVIDIA NGC catalog probe (anonymous)

- https://catalog.ngc.nvidia.com/api/orgs/nvidia/teams/clara/models: HTTP Error 404: Not Found
- https://api.ngc.nvidia.com/v2/orgs/nvidia/teams/clara/models: HTTP Error 401: Unauthorized

## 4. BioNeMo as Python package (no Docker)

- `bionemo` not installed (would require pip install)

## Verdict

- If section 1 found a MolMIM HF repo AND section 2 loaded it cleanly →
  **Path A — use HuggingFace mirror, ~30 min integration**
- If section 3 (NGC) returned a downloadable .nemo URL →
  **Path B — extract .nemo offline, half-day work**
- If only Docker route works →
  **Path C — shelve, pursue ChemBERTa hybrid instead**
- If nothing works → see ChemBERTa hybrid plan in
  `docs/limo_diagnostics_extended.md`