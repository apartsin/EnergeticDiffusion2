# Unfinished Tasks Audit
_Generated 2026-04-30, after Wave 1+2+3+T2 prose, Fig 1/22/23 regen, T1 BDE science fix._

---

## Closed in this session

### NMI committee review fixes (first pass)
- ✅ **All P1**: HMX K-J qualifier in abstract; productive-quadrant defined; max-Tanimoto rephrased; CHNO expansion on first use; GeoLDM in §2.3; npj CompMat 2025 in §2.5; Choi 2023 in §2.4; MolMIM venue.
- ✅ **All P2**: duplicate outline at §1 line 110 deleted; stakes sentence added; CFG jargon decompressed; productive-quadrant in §1; retrosynthesis 1/12 moved to §7; E1 thermal qualifier (line 93).
- ✅ **All P3**: denoiser seed-variance T3 sentence in §7; 89/100 unguided-pool surfaced earlier in §5.2; self-distillation rounds terminology consistent; L1 synthesis recommendation in §8; CSP CGD 2023 follow-up in §7; domain-gate generalisability in §1.
- ✅ **Wave 3 tables**: Table G.2 n/a cells fixed; Table 7c K-J formula bias column added.

### Experiments
- ✅ **T1 BDE**: infrastructure fixed (xtb 6.6.1 install path); science fix applied (coordinate-preserving graph split, no SMILES round-trip, UHF=1 per fragment); positive BDEs across all bonds; L1 weakest = 86.0 kcal/mol C-NO2, E1 weakest = 92.7 kcal/mol N-C; result paragraph added to §7.
- ✅ **T2 density bracket**: Bondi-vdW packing-factor bracket on L1 and E1; pk ∈ {0.65, 0.69, 0.72} yields ρ ∈ [1.69, 1.87] for L1, [1.65, 1.83] for E1 (~14% below 6-anchor-calibrated); paragraph added to §7.
- ✅ **T4 oxatriazole anchor**: pre-flight literature search found no viable 1,2,3,5-oxatriazole compound with both experimental ρ AND experimental D; documented as negative finding in `t4_oxatriazole_anchor_bundle/PRE_FLIGHT.md`.

### Figures
- ✅ **Figure 1** (`novelty_vs_D.png`): added REINVENT 4 marker + SELFIES-GA surrogate→DFT collapse arrow on D panel; caption updated.
- ✅ **Figure 22** (`fig_baseline_forest.png`): added REINVENT 4 + SELFIES-GA rows; MolMIM bar truncated with off-scale arrow + label so x-axis stays in [0, 1.5] for the meaningful comparison region.
- ✅ **Figure 23** (`fig_quadrant_scatter.png`): legend moved below x-axis (full plot width for points); SELFIES-GA collapse arrow + REINVENT 4 + 12-lead coverage; three audit-fix cycles converged.

### Paper structure
- ✅ **§5.5 → Appendix G migration**: §5.5 collapsed to ablation summary (Table 8 + 3 paragraphs); detailed prose, sub-tables, and figures moved to new Appendix G.
- ✅ **§5.1 / Table 11 → Appendix C.5**: production recipe table relocated.

### T7 dropped
- ✅ **T7 future-work pointer in §5.4.1 removed** per user request. EXPERIMENTATION_PLAN.md retains the historical entry but is no longer referenced from the body.

---

## Open tasks (priority-ordered)

### P0 — currently in flight
- ⏳ **T3 multi-seed denoiser training** (Modal `ap-JUyk3qA0ZkI61RjisNK2PU`): expected ~10 hours remaining. Will produce `denoiser_v4b_seed1.pt` and `denoiser_v4b_seed2.pt`. On completion, the next planned step is to re-run the §G.3 pool=10k single-seed-per-condition sampling on each new checkpoint and report top-1 D variance, closing Reviewer 1's seed-variance gap fully (currently flagged as in-flight in §7).
- ⏳ **Figure 22b productive-quadrant snapshot** (background agent `aa7bd...`): single-panel scatter + small inset showing SELFIES-GA collapse, as a Table 6a visual companion. To be inserted in §5.4.1 immediately after Table 6a.
- ⏳ **Second-pass NMI committee review** (background agent `a71a6be...`): writes `NMI_REVIEWER_REPORT_PASS2.md`; will identify any new concerns surfaced by the first-pass revisions plus residual P0/P1 issues.

### P1 — should land this revision cycle
- ❌ **Verify Kamlet-Jacobs 1968 page range**: paper currently cites `J. Chem. Phys. 48:23-35`; some secondary sources list `23-55`. Quick web check against the AIP original required.
- ❌ **EMDB citation upgrade**: currently grey-literature ("Energetic Materials DataBase (EMDB), public extract, accessed 2024"); needs URL, access date, institutional provenance.
- ❌ **Push current commits to remote**: `cb848c2` and any subsequent commits with T1 BDE integration + T7 drop are local-only.
- ❌ **Bibliography placeholder authors**: the four newly-added refs (npj CompMat 2025, Choi 2023, CSP CGD 2023, Chem. Mater. 2024) currently have placeholder author lists ("Anonymous", "et al."). Web-search to fill exact authors.

### P2 — useful polish
- ❌ **Figure 19 colorblind palette**: the lead-cards grid uses red/green border colors (xTB pass/fail); inaccessible to ~8% of readers with red-green colorblindness. Swap to blue/orange or add a shape indicator.
- ❌ **Figure 21 anchor labels + L1 line**: the DFT dumbbell could add a horizontal line at D=8.25 km/s (L1 K-J calibrated) and label RDX/HMX/FOX-7 as named anchor points on the N-fraction scatter for reader orientation.
- ❌ **T3 result integration**: once T3 completes, integrate the 3-seed denoiser variance into §7 (replace the "in-flight" sentence with the actual variance band). This will close Reviewer 1's seed-variance gap fully.
- ❌ **Figure 22b caption + insertion**: the in-flight agent `aa7bd...` should land the figure block in §5.4.1 after Table 6a; verify on completion.

### P3 — defer to next revision
- ❌ **Split Table C.1 hyperparameter dump** into C.1a (LIMO VAE), C.1b (denoiser + score model), C.1c (pipeline configuration). Currently one long table; reading is heavy but not blocking acceptance.
- ❌ **T5 covolume CJ recompute** (open-source BKW EOS): ~1 week development to remove the K-J relative-ranking caveat from the abstract. Future paper.
- ❌ **CSP run on L1**: scoped as future work in §7. Out of scope for this paper.

### Dropped per user direction
- ❌ ~~**T7 REINVENT 4 with D-direct RL**~~ — dropped 2026-04-30.

---

## Files written/touched this session (summary)

### Created
- `docs/paper/EXPERIMENTATION_PLAN.md`
- `docs/paper/FIX_WAVES_PLAN.md`
- `docs/paper/NMI_REVIEWER_REPORT.md`
- `docs/paper/REWRITE_PLAN.md`
- `docs/paper/short_paper.html` (the streamlined paper itself)
- `docs/paper/UNFINISHED_TASKS.md` (this file)
- `docs/paper/figs/fig_quadrant_scatter.png` + `.svg` (Figure 23, regenerated)
- `docs/paper/figs/fig_baseline_forest.png` + `.svg` (Figure 22, regenerated)
- `docs/paper/figs/novelty_vs_D.png` + `.svg` (Figure 1, regenerated)
- `t1_bde_bundle/` (xTB BDE Modal launcher + results)
- `t2_density_bundle/` (Bondi-vdW density Modal launcher + results)
- `t3_denoiser_seeds_bundle/` (multi-seed denoiser Modal launcher; in flight)
- `t4_oxatriazole_anchor_bundle/` (pre-flight: skipped per literature search)
- `plot_fig_quadrant_scatter.py` (Figure 23 generator)

### Modified
- `docs/paper/index.html` (long version): 13 audit fixes
- `plot_baseline_forest.py`: REINVENT 4 + SELFIES-GA rows + MolMIM truncation
- `plot_novelty_scatter.py`: REINVENT 4 marker + SELFIES-GA collapse arrow

### Pending verification on next user touch
- `t1_bde_bundle/results/t1_bde.json`: positive BDEs verified in this session.
- `t3_denoiser_seeds_bundle/results/`: empty until T3 completes.
- `docs/paper/figs/fig22b_baseline_quadrant.png`: pending, in-flight agent.
- `docs/paper/NMI_REVIEWER_REPORT_PASS2.md`: pending, in-flight agent.

---

## Suggested next actions for the user (in priority order)

1. Wait for the two remaining background agents (Fig 22b, committee review pass 2). Their completion will close P0.
2. Push commits to GitHub (`git push`).
3. Have me apply the four small P1 fixes (KJ page, EMDB URL, bibliography author lookup, T3 result integration on T3 completion) in one batch.
4. Defer P3 items to a follow-up revision.
