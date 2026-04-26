# Data Description (EnergeticDiffusion2)

Generated on 2026-04-24 from direct inspection of `data/` files in this workspace.

## 1) Data at a Glance

- `data/raw`: 1,107 files, 5,147,814,512 bytes (~4.79 GiB)
- `data/training`: 26 files, 1,752,705,899 bytes (~1.63 GiB)
- `data/logs`: 23 files, 1,238,793 bytes (~1.18 MiB)

The repository contains:
- Large raw molecule corpora (PubChem, ChEMBL, QM9, GuacaMol)
- Energetics-focused external datasets and upstream repos
- Curated master tables and train/validation/test splits for diffusion and supervised learning

## 2) Raw Data Inventory

### `data/raw` subdirectories

| Subdirectory | Files | Size (bytes) | Approx size |
|---|---:|---:|---:|
| `benchmarks` | 1 | 61,841,218 | 58.98 MiB |
| `energetic` | 125 | 119,517,377 | 114.00 MiB |
| `energetic_external` | 976 | 523,718,535 | 499.46 MiB |
| `general` | 4 | 4,397,909,915 | 4.10 GiB |
| `quantum` | 1 | 44,827,467 | 42.75 MiB |
| `reactions` | 0 | 0 | 0 |

### Core raw inputs from readiness audit (`data/training/data_readiness_audit.json`)

Required (all present):
- `data/raw/benchmarks/guacamol_v1_train.smiles` (61,841,218 bytes)
- `data/raw/general/CID-SMILES.gz` (1,476,338,195 bytes)
- `data/raw/general/CID-Identifiers.tsv.gz` (95,094,111 bytes)
- `data/raw/quantum/qm9.zip` (44,827,467 bytes)

Optional inputs listed in audit (all present):
- `data/raw/general/chembl_36.sdf.gz` (936,489,001 bytes)
- `data/raw/general/chembl_36_sqlite.tar.gz` (1,889,988,608 bytes)
- `data/raw/energetic_external/EMDP/Data/train_set.csv` (182,236 bytes)
- `data/raw/energetic_external/EMDP/Data/test_set.csv` (20,389 bytes)

## 3) Curated Training Outputs

### Master tables

| File | Rows | Columns | Size |
|---|---:|---:|---:|
| `data/training/master/unlabeled_master.csv` | 699,969 | 11 | 259,873,685 bytes |
| `data/training/master/labeled_master.csv` | 65,960 | 40 | 50,484,050 bytes |

### Diffusion pretraining split

| File | Rows | Columns | Size |
|---|---:|---:|---:|
| `data/training/diffusion_pretrain/train.csv` | 678,555 | 12 | 271,189,804 bytes |
| `data/training/diffusion_pretrain/validation.csv` | 7,621 | 12 | 2,677,457 bytes |
| `data/training/diffusion_pretrain/test.csv` | 8,342 | 12 | 3,193,648 bytes |
| **Total** | **694,518** |  |  |

### Property supervision splits

`property_supervised` (partial labels allowed):
- train: 37,574
- validation: 4,539
- test: 5,756
- total: 47,869

`property_supervised_all` (all labeled molecules):
- train: 51,624
- validation: 6,046
- test: 8,290
- total: 65,960

`property_supervised_complete` (all 5 numeric targets required):
- `all.csv`: 20,409 rows
- kept rate from summary: 42.64%

`property_supervised_complete4` (4 numeric targets required; excludes `explosion_heat` requirement):
- `all.csv`: 33,348 rows
- kept rate from summary: 69.67%

### Fused aggregation tables

| File | Rows | Columns | Size |
|---|---:|---:|---:|
| `data/training/fused/energetic_fused_all.csv` | 2,198,854 | 26 | 974,300,361 bytes |
| `data/training/fused/energetic_fused_labeled.csv` | 54,066 | 26 | 22,944,909 bytes |

## 4) Schema Summary

### Core molecular identity fields
- `molecule_id`
- `smiles`
- `selfies`
- `canonical_smiles` (in fused tables)

### Structural proxy features
- `n_count`, `o_count`
- `has_nitro`, `has_azide`
- `energetic_proxy_score`

### Numeric energetic targets
- `density`
- `heat_of_formation`
- `detonation_velocity`
- `detonation_pressure`
- `explosion_heat`
- `sensitivity_impact` (present in fused tables)

### Provenance / confidence metadata
- `source_dataset`, `source_path`
- `source_dataset_all`, `source_path_all`
- `label_source_type`, `label_source_type_all`
- `label_confidence_weight`
- per-target source fields (for each numeric target, `*_source_dataset`, `*_source_type`, and `_all` variants)

### Split and aggregation fields
- `split_key` (split files)
- `num_numeric_labels`, `has_numeric_labels`, `num_source_records` (fused tables)
- `sensitivity_class_*`, `has_sensitivity_class` (fused tables)

## 5) Label Coverage and Distributions

From `data/training/master/labeled_master.csv` (65,960 rows):

Numeric target non-empty coverage:
- `density`: 56,452 (85.59%)
- `heat_of_formation`: 54,345 (82.39%)
- `detonation_velocity`: 51,066 (77.42%)
- `detonation_pressure`: 44,372 (67.27%)
- `explosion_heat`: 31,431 (47.65%)

`label_source_type` distribution:
- `compiled_observed`: 33,282
- `model_predicted`: 18,091
- `unknown`: 14,587

High-frequency `source_dataset` values (top):
- `3DCNN`: 26,254
- `cm4c01978_si_001`: 12,040
- `generation`: 7,072
- `denovo_sampling_rl.predict.0_filtered`: 6,097
- `denovo_sampling_tl.predict.0_filtered`: 4,922

Structural flags:
- Labeled with `has_nitro = true`: 47,263
- Labeled with `has_azide = true`: 646
- Unlabeled with `has_nitro = true`: 157,167
- Unlabeled with `has_azide = true`: 946

## 6) Data Quality / Consistency Notes

- SELFIES alignment checks report no missing SELFIES rows in current training files; one audit entry reports 13 encode failures for diffusion train.
- `data/training/summary.json` reports diffusion split totals of 699,969 (train 683,901 / val 7,678 / test 8,390), but current CSV files contain 694,518 rows total (train 678,555 / val 7,621 / test 8,342). This summary appears stale relative to current split files.
- Several JSON metadata files contain absolute paths rooted at `E:\\Projects\\EnergeticDiffusion\\...` rather than this workspace (`E:\\Projects\\EnergeticDiffusion2\\...`).
- `data/raw/reactions/` exists but is currently empty.

## 7) Which Dataset to Use

- Generative pretraining: use `data/training/diffusion_pretrain/*.csv`
- Supervised learning with partial label availability: use `data/training/property_supervised/*.csv`
- Supervised learning on fully observed 4-target set: use `data/training/property_supervised_complete4/*.csv`
- Supervised learning on fully observed 5-target set: use `data/training/property_supervised_complete/*.csv`
- Large aggregate exploration / deduplicated merged corpus: use `data/training/fused/energetic_fused_all.csv` and `energetic_fused_labeled.csv`

