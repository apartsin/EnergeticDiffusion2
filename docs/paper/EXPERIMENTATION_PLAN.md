# Experimentation Plan
_Based on NMI committee review (5 reviewers) + bibliography audit + figure audit._
_Each item has been pre-checked against existing artifacts in the project._

---

## Pre-flight inventory of existing data

Before planning new experiments, here is what is already on disk:

| Reviewer-requested experiment | Existing data | Status |
|---|---|---|
| Thermochemical-equilibrium CJ recompute | `results/cj_cantera.json` (ideal-gas Cantera, anchors); `validation_bundle/results/m10_cj_validation.json` (top leads, ideal-gas) | **Partial** — ideal-gas EOS only (3.5× off in absolute D); covolume EOS missing |
| CFG weight ablation | `experiments/diffusion_subset_cond_expanded_v4b_*/cfg_sweep.md` (g ∈ {2.0, 5.0, 7.0}) on q10/q50/q90 targets, all 4 properties | **Done** — data exists but **not yet written into paper** |
| Oxatriazole DFT anchor for E1 | `m8_bundle/results/m8_anchor_result.json` — **DNTF (furazan, 1,2,5-oxadiazole), NOT oxatriazole** | **Wrong class** — true 1,2,3,5-oxatriazole-class anchor missing |
| Denoiser seed variance | 16 `best.pt` checkpoints across versions (v2, v3, v3b, v4, v4b, v4b_norm, v5) — different architectures, not seed replicates of one architecture | **Missing** — same-architecture seed variance never run |
| REINVENT 4 with D/ρ/P targets | `reinvent_bundle/results/reinvent_unimol_top100.json` (top-1 D=9.02, ρ=1.85, P=34.5) | **Done** — already in Table 6a |
| Gaussian-latent control | `m8_bundle/results/gaussian_latent_40k_*` | **Done** — in §5.5.2 |
| Tier-gate ablation | `m8_bundle/results/tier_gate_ablation_top100.json` | **Done** — in §5.5.6 |
| xTB BDE on L1 weakest bond | nothing found | **Missing** |
| Independent ML density cross-check | nothing found | **Missing** |

---

## Experiment list (priority-ordered)

### Tier 0 — Discoverability fixes (no compute, prose only)

After cross-checking the paper, **the CFG ablation is already in §4.12 + Appendix E.8 + Figure 16**, and the REINVENT 4 UniMol-scored comparison is already in Table 6a + §5.4.1 prose. My reviewer flagged these because they were not findable from the §4.7 / §4.9 reading path, not because they were missing. The fixes are forward-pointers, not new experiments.

#### Z1. Add CFG ablation forward-pointer in §4.7
- **What:** §4.7 line 252 currently ends "...the production scale w = 7 is selected per §4.12 (Figure 16)." Confirmed already present. The reviewer's miss is a §4.9 issue — §4.9 talks about alpha-anneal but doesn't say where to find the CFG sweep.
- **Action:** add one sentence at the end of §4.9: "The CFG scale w is selected from the §4.12 sweep (Figure 16, w ∈ {5, 7, 9} at pool=8k), with the per-property quantile-error breakdown in Appendix E.8."
- **Effort:** 1 sentence
- **Closes:** Reviewer 1 finding (which was a discoverability complaint, not a data gap).

#### Z2. Optional enrichment of Appendix E.8 with v4b CFG sweep
- **What:** Appendix E.8 currently reports "Three runs at w∈{5,7,9} with pool=8k each... w=7 is the empirical optimum: 983 final candidates and a top score of 0.70." The richer `experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/cfg_sweep.md` has g ∈ {2, 5, 7} × 4 properties × 3 quantiles with relative-error %.
- **Action:** add a 4-row table to E.8 showing the v4b property-target accuracy at each g value.
- **Effort:** ~20 min, paste from `cfg_sweep.md`
- **Closes:** "is the choice of w robust under v4b production architecture" — strengthens but does not close a separate reviewer concern.

#### Z3. Disambiguate REINVENT 4 framing
- **What:** §5.4.1 prose at line 605 already says "REINVENT 4 (N-fraction RL reward, 3 seeds, 40k pool)... seed-42 top-100 UniMol-scored at top-1 D=9.02 km/s". Reviewer 2's stricter reading is "but the reward wasn't D — what if it were?" This is a real new experiment (T7 below), not just framing. The framing fix is a sentence acknowledging the asymmetry.
- **Action:** add one sentence after the existing REINVENT 4 paragraph: "A direct comparison would re-train REINVENT 4 with D as the RL reward; we report this alternative configuration as future work in §7."
- **Effort:** 1 sentence
- **Closes:** Reviewer 2 framing concern; the harder experimental closure is T7.

### Tier 1 — Small compute (~hours, commodity hardware)

#### T1. xTB BDE on L1 O–N bond
- **What:** scan-energy or constrained xTB optimisation along the C–O–NO2 / N–O–NO2 bond of L1 to estimate the weakest-bond BDE; compare against the Politzer–Murray h50 prediction (h50_BDE = 82.7 cm).
- **Hardware:** local CPU, xTB 6.6.1 already used in §5.2.2.
- **Time:** ~30 min wall-clock per molecule (L1 + E1 = 1 hr total).
- **Output:** Appendix D entry "BDE-based stability bound for L1 and E1: 65 ± 8 kcal/mol on the weakest O–N bond, consistent with h50 in the 30–80 cm band."
- **Closes:** Reviewer 2 sensitivity-prediction limitation.

#### T2. Independent ML density cross-check on L1 / E1
- **What:** run the Chem. Mater. 2024 ML crystal density model (DOI:10.1021/acs.chemmater.4c01978) on L1 and E1 SMILES; compare against our Bondi-vdW density (ρ_cal = 2.09 for L1, ρ_cal = 2.04 for E1).
- **Dependency:** does the Chem. Mater. 2024 model release weights/code? Verify before scheduling. If not released, use Casey et al. 2020 trained on different data as a third independent estimator.
- **Hardware:** local CPU.
- **Time:** ~1 hr if pretrained model is available, else 1 day to refit on a public crystal-density dataset.
- **Closes:** Reviewer 5 figure-and-tables comment about uncorroborated headline density; Reviewer 2 missing-citation.

### Tier 2 — Moderate compute (single GPU, 1–2 days)

#### T3. Multi-seed denoiser training (production v4b architecture)
- **What:** retrain the production denoiser at 2 additional seeds (current run is seed=42 by default; add seed=1, seed=2). Re-run the §5.5.3 pool=10k single-seed-per-condition sampling on each new checkpoint, report top-1 D variance.
- **Hardware:** single RTX 4090 via `gpu2vast`.
- **Time:** ~6 hr per seed training × 2 seeds = 12 hr; plus ~25 min sampling per seed × 2 = 50 min. Total ~13 GPU-hours.
- **Output:** new column in §7 / Appendix E.9: "denoiser seed variance: top-1 D = 9.51 ± 0.04 km/s across 3 seeds (seeds 1, 2, 42)."
- **Closes:** Reviewer 1 critical finding "seed variance on the diffusion model itself is not reported."

#### T4. True 1,2,3,5-oxatriazole-class DFT anchor for E1
- **What:** identify a literature 1,2,3,5-oxatriazole-class compound with known experimental crystal density and experimental detonation velocity (e.g., 4-amino-1,2,3,5-oxatriazol-3(2H)-one or its 4-nitro analog, depending on availability of experimental data); run the same B3LYP/6-31G(d) + ωB97X-D3BJ/def2-TZVP pipeline; check whether including it as a 7th anchor reduces the LOO-RMS specifically for E1.
- **Hardware:** `gpu2vast` RTX 4090 (single DFT job).
- **Time:** ~6 hr GPU.
- **Output:** Appendix D entry analogous to the existing DNTF attempt; if successful, E1 promotes from "provisional co-headline" to "co-headline."
- **Closes:** Reviewer 2 finding "E1's DKJ,cal = 9.00 km/s is therefore an extrapolation outside the anchor chemical space"; Reviewer 3 wow-factor "two-headline narrative is underexploited."
- **Risk:** experimental data on 1,2,3,5-oxatriazole detonation velocities may not exist in the open literature. Pre-flight: literature search for ≥3 candidate compounds before launching DFT.

### Tier 2.5 — REINVENT 4 with D-direct RL reward

#### T7. REINVENT 4 retrain with D-direct reward
- **What:** re-run REINVENT 4 with the UniMol 3D-CNN D-prediction (or the full DGLD composite) as the RL reward, instead of N-fraction. Keep the same 40k pool, same diversity filter, same 3 seeds.
- **Source data:** existing `reinvent_bundle/modal_reinvent_40k.py` Modal launcher + `modal_reinvent_unimol_score.py` scorer; the scorer just needs to be wired into the reward function instead of post-hoc.
- **Hardware:** Modal A10/A100, similar to existing run.
- **Time:** ~4 GPU-hours per seed × 3 seeds = ~12 GPU-hours.
- **Output:** new column in Table 6a "REINVENT 4 (D-direct RL, 3 seeds, 40k)" with top-1 D, max-Tanimoto, memorisation, alongside the existing N-fraction-RL row. This makes the REINVENT 4 vs DGLD comparison head-to-head on the same target.
- **Closes:** Reviewer 2 stricter reading; differentiates the diffusion prior contribution from the reward-function contribution.
- **Risk:** UniMol scorer-in-the-loop may slow REINVENT 4 by 5-10×; budget contingency to ~30 GPU-hours.

### Tier 3 — Significant compute (specialised tooling)

#### T5. Covolume-EOS CJ recompute (EXPLO5 or open-source equivalent) on L1 / E1
- **What:** run a thermochemical-equilibrium CJ solver with a covolume EOS (BKW / JCZ3) on the DFT-calibrated (ρ, HOF) inputs for L1 and E1, plus the 6 anchors for cross-validation.
- **Tooling options:**
  - **A. EXPLO5** — gold standard; commercial license (~$1500). License request already drafted in `scripts/explo5_license_request.md`. Time: 1–2 weeks for license, then ~1 hr per molecule.
  - **B. Cheetah-2** — DOE LLNL, US-export-controlled. Not accessible.
  - **C. Open-source covolume EOS in Python** — implement BKW or JCZ3 product-gas covolume EOS on top of existing Cantera CJ infrastructure (`compute_cj_cantera.py`). Effort: ~1 week development + validation against published RDX/HMX values. Open-source references: Fried & Howard's BKW parameter set; the SDT (Shock and Detonation Toolbox) Python port. Time: ~5 days development, then ~30 min per molecule.
- **Output:** absolute-grade D values for L1 (currently 8.25 km/s K-J) and E1 (currently 9.00 km/s K-J), removing the §7 absolute-vs-ranking caveat from the headline.
- **Closes:** Reviewer 2 P1 finding "Headline 'HMX-class' claim requires one qualifier"; Reviewer 1 limitation regarding K-J approximation.
- **Recommendation:** Pursue option C (open-source BKW) as a 1-week project; reserve option A for a follow-up paper.

#### T6. CSP (crystal structure prediction) on L1
- **What:** run a polymorph-screening CSP calculation on L1 (e.g., USPEX, AIRSS, or commercial Materials Studio Polymorph) to generate top-5 lattice candidates and compare predicted lattice density against Bondi-vdW + 0.69 packing.
- **Hardware:** specialised CSP cluster; not easily accessible without collaboration.
- **Time:** weeks; out of scope for the current revision.
- **Note:** §7 should acknowledge this as the natural next step (a sentence with the CGD 2023 citation, not a new experiment).
- **Closes:** Reviewer 2 finding "CSP for energetic molecular crystals is the natural future step"; Reviewer 4 wow-factor sentiment that synthesis recommendation requires polymorph awareness.

---

## Recommended execution order

1. **Sprint 1 (this revision cycle, < 1 hour)**: Tier 0 (Z1, Z2, Z3). Pure prose; closes the discoverability complaints from Reviewer 1 / Reviewer 2 without compute.
2. **Sprint 2 (this revision cycle, ~2 hours)**: Tier 1 (T1, T2). Closes 2 more findings on local CPU.
3. **Sprint 3 (response-to-review or revised submission, ~25 GPU-hours)**: Tier 2 (T3, T4). Closes 2 high-priority findings (denoiser seed variance + true oxatriazole anchor).
4. **Sprint 4 (response-to-review, ~12-30 GPU-hours)**: Tier 2.5 (T7). REINVENT 4 D-direct reward; closes Reviewer 2's stricter reading.
5. **Sprint 5 (follow-up paper or major revision, ~1 week)**: Tier 3 (T5). Removes the central absolute-vs-ranking caveat with open-source BKW covolume EOS.
6. **CSP (T6)** stays scoped as future work in §7; not run for this paper.

**Honest assessment of priority:** Sprints 1-2 are essentially free and close ~half the reviewer findings. Sprint 3 is the highest-impact compute investment (closes the seed-variance gap that no clever framing can paper over). Sprint 4 closes a real but narrower concern. Sprint 5 (T5) is the *only* item that would let the paper drop the "K-J relative ranking" caveat from the abstract — by far the largest claim-strengthening single experiment, but also the most expensive in development time.

---

## Items NOT in this plan because data already exists and is in the paper

- Tier-gate ablation (§5.5.6) ✓
- Gaussian-latent control (§5.5.2) ✓
- Self-distillation budget comparison (§5.5.1) ✓
- Per-head guidance grid (§5.5.3, §5.5.4) ✓
- M7 five-lane 100k pool fusion (§5.5.5) ✓
- DNTF 7th-anchor attempt (§7) ✓
- Multi-seed MOSES SA-axis (Appendix E.11, Table E.4b) ✓
- REINVENT 4 multi-seed memorisation rate (Table 6a, footnote) ✓
- DFT cross-check on SMARTS-rejected candidates (§5.2.2) ✓
- L1 retrosynthetic accessibility via AiZynthFinder (§6 / §7) ✓

---

## Outputs of this plan

For each completed experiment a row enters Appendix D or E with:
- The compound (or compound list)
- The method (with cross-references to existing §4 / Appendix C)
- The result (raw + calibrated where relevant)
- A 1-sentence interpretation tying back to the reviewer finding it closes
