# Results Catalogue

All experiment result files produced during DGLD development.
Files are read-only records; never delete without archiving.

Last updated: 2026-04-29.

---

## `results/` — Main experiment outputs

### Generation sweep outputs (raw SMILES pools)

| File | Contents |
|---|---|
| `m1_sweep_C0_unguided_seed{0,1,2}.txt` | 10 000 SMILES per seed, unguided (CFG-only). SA-axis multi-seed sweep. |
| `m1_sweep_C1_viab_seed{0,1,2}.txt` | 10 000 SMILES per seed, viab-only guidance (s_viab=1.0). SA-axis. |
| `m1_sweep_C2_viab_sens_seed{0,1,2}.txt` | 10 000 SMILES per seed, viab+sens (s_viab=1.0, s_sens=0.3). SA-axis. |
| `m1_sweep_C3_viab_sens_sa_seed{0,1,2}.txt` | 10 000 SMILES per seed, viab+sens+SA (s_SA=0.15). SA-axis. |
| `m1_anneal_clamp_anneal{0,2}_clamp{5,50}.txt` | 4 raw SMILES pools from the alpha-anneal x gradient-clamp 2x2 ablation grid. |
| `e1_pool40k_log.txt` | Training log from the pool=40k generation run on runpod RTX 4090. |

### Sweep summaries (aggregated metrics)

| File | Contents |
|---|---|
| `m1_summary.json` | Per-run top-1 composite/D/rho/P metrics for the 4-condition SA sweep (keys: `runs`, `args`). |
| `m1_anneal_clamp_summary.json` | 2x2 anneal/clamp ablation grid summary (keys: `cells`, `args`). Paper §4.9. |
| `m6_post.json` | 7 conditions x 3-6 seeds multi-seed head-to-head results (keys: `per_run`, per-condition). Primary source for Table 5 and Figure 22. |
| `m6_post.md` | Human-readable summary of m6_post.json. |
| `pool_metrics.json` | MOSES-style metrics per pool and per condition (keys: `per_pool`, `per_condition`). Single-seed. Source for Table E.4. |
| `moses_multiseed_per_file.json` | Per-file MOSES metrics for all 12 sweep files (4 conditions x 3 seeds). Source for Table E.4b. |
| `moses_multiseed_summary.json` | Per-condition mean/std MOSES metrics across 3 seeds. Source for Table E.4b (paper). |
| `fcd_results.json` | Frechet ChemNet Distance per pool and condition (keys: `per_pool`, `per_condition`). Source for Table E.5. |
| `scatter_data.json` | (D, P, rho, novelty) scatter points for Fig 2 (top-200 from pool=40k joint rerank). |
| `novelty_top1.json` | Top-1 per-condition Tanimoto novelty against labelled master (keys: `rows`). |
| `novelty_filtered_top1.json` | Top-1 per-condition novelty filtered to energetic-SMARTS-pass candidates (keys: `per_condition`). |
| `molmim_post.json` | MolMIM 70M baseline run results (keys: `per_run`, `summary`). Source for Table 6a. |
| `baseline_examples.json` | Top-10 most-novel SMILES from MolMIM 70M and SMILES-LSTM baselines. Appendix F. |
| `summary.json` | Training summary for the score model (keys: model, device, epochs, total_steps, final_loss). |

### Validation and audit outputs

| File | Contents |
|---|---|
| `h50_predictions.json` | Drop-weight impact sensitivity (h50, cm) predictions for all leads, two methods (route1: Keshavarz correlation; route2: RF from h50 training data). Keys: `method_route1`, `method_route2`, `comparison`. Source for Table D.1c. |
| `aizynth_L4L5_deep.json` | AiZynthFinder deep retrosynthesis results for leads L4 and L5. Source for §5.3.2 1/12 hit-rate result. |
| `aizynth_deep_log.txt` | AiZynthFinder run log for the L4/L5 deep audit. |
| `cj_cantera.json` | Cantera ideal-gas CJ detonation recompute for all 12 chem-pass leads + anchors. Relative-ranking cross-check (absolute values ~3x low due to no covolume EOS). Source for §5.2.2 cross-check paragraph. |
| `d9_kj_nfrac_table.json` | K-J residual vs N-fraction table for 575 Tier-A rows. Source for Table D.4 / Fig 21 right panel. |
| `uncertainty_propagation.json` | Per-lead K-J sensitivity slopes and propagated calibration uncertainty (delta_D, delta_P) for all 12 leads. Source for Table 7c. |
| `smoke_result.json` | Smoke test result for the Modal inference endpoint (keys: phase, ok, error, elapsed_s). |
| `smoke_stdout.log` | Smoke test stdout for Modal endpoint validation. |

### Pool-fusion and top-100 results

| File | Contents |
|---|---|
| `combo_status.json` | Exit codes and timing for the pool-fusion pipeline steps (step1_e1, step2_aizynth). |

### Scalar result files

| File | Contents |
|---|---|
| `m1_anneal_clamp_anneal0_clamp5.txt` | Anneal=0, clamp=5 pool (raw SMILES). |
| `m1_anneal_clamp_anneal0_clamp50.txt` | Anneal=0, clamp=50 pool (raw SMILES). |
| `m1_anneal_clamp_anneal2_clamp5.txt` | Anneal=2, clamp=5 pool (raw SMILES). |
| `m1_anneal_clamp_anneal2_clamp50.txt` | Anneal=2, clamp=50 pool (raw SMILES). |

### `results/extension_set/`

| File | Contents |
|---|---|
| `e_set_500_smiles.json` | 500-SMILES extension set from pool=80k unguided run. |
| `e_set_lead_table.json` | Property table for the extension set candidates. Source for §5.3.3 extension audit. |
| `e_set_picked_10.json` | 10 scaffold-distinct leads picked from the extension set for DFT. |
| `e_set_xtb_screen.json` | GFN2-xTB HOMO-LUMO gap screening results for the extension set. |

### `results/model/` (score model checkpoint)

| File | Contents |
|---|---|
| `model.safetensors` | Production score-model checkpoint (v3d/v3e; 4-block FiLM-MLP, 6 heads). |
| `config.json` | Model architecture config. |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer files (SELFIES vocabulary). |

---

## `m2_bundle/results/` — DFT and calibration outputs

### Per-lead DFT results

| File | Contents |
|---|---|
| `m2_lead_L{1..5,9,11,13,16,18,19,20}.json` | Per-lead DFT result: formula, rho_DFT, HOF_DFT (wB97X-D), frequencies, rho_cal, HOF_cal, K-J D and P under 6-anchor calibration. |
| `m2_lead_E{1..10}.json` | Per-lead DFT result for extension-set leads (E1-E10). Same schema as L-leads. |
| `m2_lead_R{2,3,14}.json` | DFT results for reference-class SMARTS-rejected candidates (R2, R3, R14). Used in §5.2.2 cross-check. |
| `m2_lead_RDX.json`, `m2_lead_TATB.json` | Anchor compound DFT results for RDX and TATB (primary validation anchors). |
| `m2_summary.json` | Aggregated table of all lead DFT results (rho_DFT, HOF, rho_cal from 2-anchor calibration). WARNING: rho_cal here uses old 2-anchor calibration; use 6-anchor from m2_calibration_6anchor.json instead. |

### Calibration files

| File | Contents |
|---|---|
| `m2_calibration.json` | 2-anchor (RDX, TATB) calibration: a_rho=4.275, b_rho=-5.172. Deprecated; superseded by 6-anchor. |
| `m2_calibration_6anchor.json` | 6-anchor calibration (RDX, TATB, HMX, PETN, FOX-7, NTO): a_rho=1.392, b_rho=-0.415, c_hof=-206.654 kJ/mol; LOO rho RMSE=0.078 g/cm3, LOO HOF RMSE=64.6 kJ/mol. Primary calibration used throughout paper. |
| `m2_atom_refs.json` | Atomic reference energies for HOF computation (wB97X-D/def2-TZVP). |
| `kj_6anchor_recompute.json` | K-J D and P recomputed for all 6 anchor compounds under the 6-anchor calibration for residual validation. |

---

## Top-level result-adjacent files

| File | Contents |
|---|---|
| `dft_5mol_targets.json` | Input SMILES and target properties for the 5-molecule initial DFT batch. |
| `compute_cj_detonation.py` | Script: Cantera CJ recompute; outputs `results/cj_cantera.json`. |
| `compute_uncertainty_propagation.py` | Script: K-J analytic sensitivity + LOO error propagation; outputs `results/uncertainty_propagation.json` and HTML for Table 7c. |
| `compute_moses_multiseed.py` | Script: MOSES metrics for 12 sweep files (4 conditions x 3 seeds); outputs `results/moses_multiseed_*.json`. |
| `compute_pool_metrics.py` | Script: MOSES + FCD per pool; outputs `results/pool_metrics.json` and `fcd_results.json`. |
| `d9_kj_nfrac.py` | Script: K-J residual vs N-fraction analysis for 575 Tier-A rows; outputs `results/d9_kj_nfrac_table.json`. |
| `DATA_DESCRIPTION.md` | High-level description of the training data sources and preprocessing. |
| `RUN_STATUS.md` | Chronological log of all major experiment runs with commit hashes and compute provenance. |

---

## Audit / review documents (not result files but referenced in paper)

| File | Purpose |
|---|---|
| `PAPER_REVIEWER_REPORT.md` | NMI peer-review report (P1-P4 issues, Tier 1 must-fix items). |
| `PAPER_AI_TELL_AUDIT.md` | AI-tell language audit (prior to submission polish). |
| `PAPER_LANG_ALIGN_AUDIT.md` | Language-alignment audit between paper text and code. |
| `PAPER_STYLE_AUDIT.md` | Style/consistency audit. |
| `DFT_DEBUG_LOG.md` | DFT pipeline debug log (imaginary frequency fixes, pod consistency checks). |
