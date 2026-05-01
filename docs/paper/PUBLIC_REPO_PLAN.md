# Public reproducibility repo — plan

_For the future companion repo to the DGLD paper. Designed for a fresh user to clone, follow READMEs, download checkpoints from Zenodo, and reproduce every figure and table in the paper. **Plan only, not yet executed.**_

---

## 1. Repo name

Suggested: **`dgld-energetic-materials`** (matches paper title).
Alternative: `dgld-paper-reproducibility` (flags role explicitly).

## 2. Top-level layout

```
dgld-energetic-materials/
├── README.md                       Project overview, quick-start, citation
├── LICENSE                         Apache-2.0 (code) + CC-BY-4.0 (data) dual
├── CITATION.cff                    Machine-readable citation
├── requirements.txt                Pinned Python deps for inference path
├── requirements-train.txt          Adds GPU/PyTorch/Modal deps
├── pyproject.toml                  Package metadata (optional)
├── .gitignore                      Excludes large files, caches, secrets
├── paper/                          Paper artefacts only — no code
├── dgld/                           Importable Python package (the DGLD method)
├── scripts/                        Top-level entry points (train, sample, evaluate, plot)
├── data/                           Small data (≤ 100 MB) committed; large data linked
├── models/                         Checkpoint pointer files (sidecars to Zenodo)
├── experiments/                    One subfolder per experiment in the paper
├── figures/                        Final figure outputs (PNG/SVG) committed; generators in scripts/
└── docs/                           Extended docs: design notes, reproducibility guide, troubleshooting
```

## 3. Per-directory contents

### `paper/`
- `short_paper.html` (final submission state; tag `post_5_rewrite` from current dev repo)
- `short_paper.pdf` (rendered for offline reading)
- `figs/` — every PNG + SVG referenced by `short_paper.html` (so HTML opens locally without downloads)
- **README**: source format note; how to render to PDF; figure-script index (which `scripts/plot_*.py` produces which figure); license.

### `dgld/` — the importable package
```
dgld/
├── __init__.py
├── limo/                LIMO VAE wrapper + checkpoint loader
├── denoiser/            FiLM-ResNet 1000-step DDPM
├── score_model/         Multi-task FiLM-MLP with 6 heads
├── sampling/            DDIM loop + classifier-free guidance + multi-head steering
├── filters/             4-stage validation funnel (SMARTS, Pareto reranker, xTB triage, DFT audit)
├── reranker/            Pareto + scaffold-aware composite scorer
├── data/                Tier-gating, motif-augmented expansion, canonicalisation
└── utils/               Common helpers
```
**README**: module-by-module API summary; the 5-line "minimal sampling example"; entry points.

### `scripts/`
- `train_limo.py`              Fine-tune LIMO on energetic SMILES corpus
- `train_denoiser.py`          Train DGLD-H or DGLD-P denoiser
- `train_score_model.py`       Train the 6-head score model + self-distillation
- `sample.py`                  Production pool=40k Hz-C2 sampling pipeline
- `validate.py`                Run the 4-stage filter on a SMILES list (xTB + DFT audit hooks)
- `evaluate.py`                MOSES-style metrics + composite ranking
- `plot_fig1.py` … `plot_fig26.py`  One figure per script (consolidated from current dev repo)
- `download_assets.sh`         Pull all checkpoints + data from Zenodo by DOI
- **README**: which script reproduces which paper claim; expected runtime per GPU class.

### `data/`
- `labelled_master.parquet`            ~66k SMILES with tier-gated property labels (Tier A/B/C/D)
- `unlabelled_corpus.txt`              ~380k canonical SMILES (one per line)
- `motif_augmented.txt.gz`             ~1.08 M SMILES; gzipped
- `hard_negatives_918.parquet`         918 mined hard-negative latents (~5 MB)
- `dft_anchors_6.json`                 RDX/TATB/HMX/PETN/FOX-7/NTO experimental ρ + HOF + D
- `pubchem_novelty_subset.txt`         Reduced PubChem novelty-check set (filtered to CHNOClF)
- `provenance.csv`                     Per-row source citation for the labelled master
- **README**: per-file schema; license terms; public-source citations (Klapötke 2019, Cooper 1996, LLNL Explosives Handbook UCRL-52997 1985, Casey 2020, ZINC-15); preprocessing pipeline.

### `models/` — pointer files only (no binaries)
Each large file is a small text "sidecar" (≤ 1 KB):

```
models/limo_vae.pt.sidecar
  filename:   limo_vae.pt
  size:       412 MB
  sha256:     <hash>
  zenodo_doi: 10.5281/zenodo.19821953
  url:        https://zenodo.org/record/19821953/files/limo_vae.pt
  download:   wget -O models/limo_vae.pt $URL
```

Sidecars for:
- `limo_vae.pt`
- `denoiser_dgld_h.pt`, `denoiser_dgld_p.pt`
- `denoiser_v4b_seed{1,2,42}.pt` (T3 multi-seed)
- `score_model_5head.pt`, `score_model_6head.pt`
- `random_forest_viability.pkl`
- `smoke_ensemble_fold{1,2}.pt` (3D-CNN / Uni-Mol)
- **README**: total checkpoint size (~5 GB); download script; integrity check command.

### `experiments/` — one subfolder per paper experiment

```
experiments/
├── e_t1_bde/                   xTB BDE on L1, E1 (T1)
├── e_t2_density/               Bondi-vdW packing-factor bracket (T2)
├── e_t3_seed_variance/         Multi-seed denoiser train + eval (T3)
├── e_t4_oxatriazole_anchor/    7th-anchor pre-flight (negative finding)
├── e_dft_audit/                Per-lead B3LYP/6-31G(d) + ωB97X-D3BJ DFT (m2)
├── e_baselines_lstm/           SMILES-LSTM 3-seed memorisation
├── e_baselines_molmim/         MolMIM 70M sampling
├── e_baselines_reinvent/       REINVENT 4 (40k pool + UniMol scoring)
├── e_baselines_selfies_ga/     SELFIES-GA + DFT-collapse audit
├── e_gaussian_control/         Gaussian-latent baseline
├── e_tier_gate_ablation/       --no_tier_gate ablation
├── e_pool_fusion/              5-lane M7 100k run
├── e_aizynth_retro/            AiZynthFinder routes for L-set
└── e_self_distillation/        Round-0 / round-1 / round-2 hard-negative mining
```

Each subfolder contains:
- `run.py`                     Entry point that produces all numbers cited in the paper for this experiment
- `LAUNCH.md`                  How to run (Modal? local? GPU class? wall time?)
- `results/*.json`             The actual outputs (kept in repo; small)
- `expected_outputs.md`        Sanity-check numbers to verify successful reproduction
- **README**: what the experiment shows; which paper claim it supports; dependencies.

### `figures/`
- All final PNGs and SVGs from `paper/figs/`
- One markdown table mapping figure number → script → source data
- **README**: regeneration command; expected diff if seed changes.

### `docs/`
- `REPRODUCIBILITY.md`         Step-by-step guide: clone → install → download → reproduce headline figures
- `DESIGN_NOTES.md`            Architectural notes (why LIMO frozen, why FiLM, etc.)
- `TROUBLESHOOTING.md`         Common failure modes (CUDA OOM, missing xtb binary, Modal auth)
- `CONTRIBUTING.md`            How to extend / fork
- `CHANGELOG.md`               Version history
- **README**: doc index.

## 4. Top-level README structure

```markdown
# DGLD: Domain-Gated Latent Diffusion for Energetic Materials

[paper PDF link] · [Zenodo DOI] · [BibTeX]

## TL;DR
3-line summary + headline number (12 DFT-confirmed leads).

## Quick start: reproduce Figure 1 in 10 minutes
1. `git clone …`
2. `pip install -r requirements.txt`
3. `bash scripts/download_assets.sh --figure-only`
4. `python scripts/plot_fig1.py`

## Directory map
[the layout from §2 with one-line descriptions]

## Reproducing every claim in the paper
Table mapping paper claim → script → expected runtime → required hardware.

## Hosting
Large files (5 GB total) live on Zenodo (DOI 10.5281/zenodo.19821953).
Pointer files in `models/` and `data/` document each.

## Citation
[BibTeX]

## License
- Code: Apache-2.0
- Data: CC-BY-4.0
- Paper: CC-BY-4.0
```

## 5. Large-file hosting strategy

**Primary archive: Zenodo** (DOI already reserved at 10.5281/zenodo.19821953 per §8.1 of the paper).

Layout on Zenodo:
- `dgld-models-v1.0.zip` — all checkpoints (~5 GB combined)
- `dgld-data-v1.0.zip` — labelled master + augmented corpus + motif expansion
- `dgld-experiment-bundles-v1.0.zip` — raw experiment outputs that exceed git size limits

**Each repo sidecar file** documents: filename, size, SHA-256, Zenodo record, direct URL.
**`scripts/download_assets.sh`** reads sidecars and downloads.

**Optional secondary mirror**: HuggingFace Hub (for the score model + denoiser). Cross-link from Zenodo.

## 6. Curation rules

### EXCLUDE from the new repo (drop from current dev repo state):
- All review / planning markdown files: `NMI_REVIEWER_REPORT*`, `PASS4_R*`, `FIX_WAVES_PLAN`, `EXPERIMENTATION_PLAN`, `SECTION5_REWRITE_PLAN`, `COPYEDIT_PASS_REPORT`, `UNFINISHED_TASKS`, `REWRITE_PLAN`, `P4_BIBLIO_FINDINGS`, `PUBLIC_REPO_PLAN` (this file). Keep `CHANGELOG.md` only.
- `experiments/diffusion_subset_cond_expanded_*` (16+ training-run dirs; keep only the production v4b checkpoint via Zenodo).
- `m1_bundle/results_*` variants (7 different result dirs).
- `m2_bundle/results_apr28/`, `results_modal/`, `results_vast{A,B,_a,_b}/` (keep canonical `results/` only).
- `__pycache__/`, `.modal-cache/`, token files.
- `keys/`, `accounts.json`, `*.key` (ANY auth material).
- Old version names in code (v2, v3, v3b, v3d, v3e, v3f) — keep one canonical.
- Modal launch logs, scratch markdown, dev TODO files.

### KEEP / RENAME (curation map):
- `combo_bundle/` → `dgld/` (renamed for clarity)
- `m2_bundle/results/m2_*.json` (12 chem-pass + anchor DFT outputs) → `experiments/e_dft_audit/results/`
- `m6_post.json`, `m7_post.json` → `experiments/e_baselines_*/results/`
- `t{1,2,3,4}_bundle/` → `experiments/e_t{1,2,3,4}_*/`
- All figure scripts → `scripts/plot_fig{N}.py` (one figure per file, consistent naming)

## 7. Reproducibility test (acceptance gate before publishing)

Before pushing the repo public, on a fresh machine:
1. Clone, run `pip install -r requirements.txt`, `bash scripts/download_assets.sh`.
2. Run `python scripts/plot_fig1.py` — must produce a PNG within 5% pixel-diff of the published Fig 1.
3. Run `python scripts/sample.py --pool 40000 --condition Hz-C2 --seed 42` — must produce top-1 D within 0.1 km/s of paper's 9.39.
4. Run `python experiments/e_t1_bde/run.py` (Modal-based) — must reproduce L1 weakest BDE 86 ± 5 kcal/mol.
5. Run `python experiments/e_baselines_selfies_ga/run.py` (the SELFIES-GA DFT-collapse audit) — must show D_surrogate > D_DFT by ≥ 3 km/s.

## 8. Migration steps

1. Create empty repo `dgld-energetic-materials` (GitHub).
2. `git init` locally; add `.gitignore` with the exclusions from §6.
3. Create directory skeleton (10 dirs).
4. Copy + rename source files per the curation map.
5. Write all 11 READMEs (top-level + 10 directory).
6. Generate sidecar files for the ~10 large-file checkpoints + 3 large data files.
7. Upload large files to Zenodo; update sidecars with actual URLs/SHA-256.
8. Run the 5-step reproducibility test on a fresh clone.
9. Initial commit; tag `v1.0`; push.
10. Submit to Software Heritage for archival; cross-link on Zenodo.

## 9. Estimated effort

| Step | Hours |
|---|---|
| Skeleton + .gitignore + READMEs | 4–6 |
| Source-code curation + renaming | 6–8 |
| Sidecar + Zenodo upload | 3–4 |
| Reproducibility test on fresh machine | 2–3 |
| Polish + first-issue triage | 2 |
| **Total** | **17–23 h** |

Total disk: ~150 MB in repo (everything fits) + ~5 GB on Zenodo.

## 10. Recommended execution order (when starting)

1. **Skeleton + READMEs first** — get docs reviewed before any binaries move.
2. **Source-code curation second** — copy + rename + clean version names.
3. **Zenodo upload last** — avoids re-uploading if something gets renamed during steps 1–2.

---

_This plan is intentionally conservative: it relies on the current Zenodo DOI (already reserved), avoids git-LFS (sidecars instead), and keeps the dev repo's history private. Re-evaluate when ready to execute._
