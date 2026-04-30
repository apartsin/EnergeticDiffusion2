# NMI Reviewer Committee Report (Pass 2)
## DGLD: Domain-Gated Latent Diffusion for the Discovery of Novel Energetic Materials

_Prepared for the Editor, Nature Machine Intelligence_
_Date: 2026-04-30_
_Document covers: revised `short_paper.html` after the Pass-1 prose-edit pass (commit `cb848c2` and onward)_

---

## Executive Summary (Editor)

Five reviewers reread the manuscript after the Pass-1 prose pass. The verdict has moved from "Minor Revision" to **borderline Accept / very-light Minor Revision**. All P1 items raised in Pass 1 are closed: the K-J relative-ranking qualifier now appears in the abstract; the "productive quadrant" and "max-Tanimoto" terms are defined inline; GeoLDM, npj CompMat 2025, Choi 2023 PEP, the CSP CGD 2023, and the ML-density Chem. Mater. 2024 references have all been incorporated; MolMIM venue is corrected; Tables G.2 / 7c / 3 are improved; Figs 1, 22, 23 have been redesigned. The §5.5 collapse to ablation summary + Appendix G is a clean structural improvement that materially helps readability. E1 now carries the "(pending thermal stability confirmation / oxatriazole-class anchor)" qualifier consistently in the abstract, §1, §6 item 8, and §7. The retrosynthesis 1/12 result has been promoted to §7. The denoiser seed-variance gap is acknowledged in §7 and flagged as in-flight (T3).

What is most strengthened: (i) abstract framing now lands the K-J relative-ranking qualifier without sacrificing impact; (ii) the §5.5 -> Table 8 + Appendix G split lets a reader see all seven ablations on one page, with the §6 Item-2 tier-gate result now legible at first read; (iii) Table 7c K-J formula bias column makes the per-lead error decomposition genuinely transparent. The two-headline (L1 + E1) story is now consistently surfaced.

What still needs fixing: a small set of residual P1 / P2 items, most of them factual-consistency cross-checks introduced by the prose-edit pass (numbers that no longer agree across §6 and Table 8 / G.5; one composite-direction phrasing inversion; a couple of Pass-1 P3 items that were not picked up). No new experiments are required for acceptance. The in-flight T1/T2/T3 results, when they land, should be folded in via standard editor-handled errata; they are not blockers.

**Recommendation:** Accept with minor textual fixes (~1 day of edits).

---

## Reviewer 1: AI / ML Expert

**What is closed from Pass 1:** GeoLDM is now in §2.3 with the 1D-vs-3D contrast; the 89/100-from-unguided-pool is up front in §5.2; the seed-variance vs guidance-claim issue is honestly addressed in the §5.5 "Seed-variance context for the headline" paragraph, with the production-default justification reframed around the most-novel result rather than a sharp performance lift; the self-distillation language is now consistent (round-0 / round-1 / round-2, with round-2 = production).

**Residual concerns**

1. **Composite "lower = better" vs "higher = better" inconsistency.** Table 6a, Table G.4, and Fig 22 caption all use "top-1 composite (lower = better)" and the values 0.451, 0.485 etc. are penalties. But the §G.3 narrative paragraph above Table G.3 says "multi-head guidance lifts max composite from 0.64 to 0.69" (i.e. higher = better) and §6 item 5 says "from 1.5k to 40k drops the best composite by ~5×" (drop = improvement = penalty interpretation). Figure 23's caption says "y-axis: composite score S (higher = better)" with the L1 marker "at the top of the productive quadrant." This is two different composite scales (the §4.10 ramp `S(x)` is higher = better; the Pareto-reranker "penalty" is lower = better). The paper uses both without flagging the inversion. **P1.** Add one footnote on first use clarifying that the Table-6a/G.4 numbers are penalties and the §4.10 / Fig 23 score is a reward, or rename the penalty as "composite penalty" everywhere. (Minor wording but a reviewer will flag it.)

2. **Pool-fusion lift number drift.** §6 item 5 says "post-filter yield more than doubles (4639 vs 966 candidates)"; Fig 17 caption says "5.1× more than the 40k baseline"; Table 8 ablation row says "+5×"; Table G.5 reports 4639 / 966 = 4.80×. The "more than doubles" phrasing in §6 Item 5 is wrong (should be ~5×). **P2.** One-word fix: "more than quadruples" or "nearly five-fold."

3. **CFG weight ablation status.** Pass-1 R1.4 asked about CFG-weight robustness. The revision has §4.12 and Table 8's CFG-scale row referring readers to Fig 16 (w in {5,7,9}, pool=8k). This closes the structural ask but the actual CFG-w grid is still only three points. Acceptable, but it is worth noting in §7 limitations that the w-sweep is 3-point. **P3.** One-line addendum to §7.

4. **Seed variance on the trained denoiser** is now correctly acknowledged in §7 ("Each denoiser is one ~6 hr training run; seed variance across the diffusion model itself is not reported. The denoiser seed-variance gap noted above is being closed by an in-flight 3-seed retraining of the v4b production architecture (T3); preliminary results will be added in the response-to-review."). This is honest. **Closed.**

5. **§4.8 vs Appendix C.1 round naming.** §4.8 says "production budget-918 checkpoint is round 2 of self-distillation (round 0 = initial Random Forest, round 1 = 137 hard negatives, round 2 = 918 hard negatives + aromatic-heterocycle boost)". Appendix C.1 says "round 0 trains on corpus only and mines 137; round 1 trains on corpus + 137 cheats and mines up to 918; round 2 trains on corpus + 918 cheats and is the production checkpoint." These two descriptions are subtly different on what "round 0" means (initial RF vs corpus-only score-model). Both internally consistent with their own framing but they conflict with each other. **P2.** Reconcile: pick one definition of "round 0" and propagate.

---

## Reviewer 2: Materials Science Expert

**What is closed from Pass 1:** abstract now flags K-J relative ranking explicitly; npj CompMat 2025 and Choi 2023 PEP are cited; CSP CGD 2023 is added to §7; §7 contains the new "Independent Bondi-vdW packing-factor bracket on L1 and E1" paragraph (T2 cross-check) which is exactly the right inclusion; oxatriazole thermal-stability caveat is now consistent in abstract / §1 / §6 / §7; retrosynthesis 1/12 is promoted into §7 as a known weakness paragraph.

**Residual concerns**

1. **L1 D_KJ,cal = 8.25 km/s "HMX-class" still requires careful reading.** Abstract reads correctly now ("by Kamlet-Jacobs relative ranking; absolute D requires a thermochemical equilibrium solver, see §7"). §6 item 1 also reads correctly: "The DFT-calibrated velocity D = 8.25 km/s falls below the nominal D >= 9.0 km/s threshold; it is within the HMX-class band by K-J relative ranking." Good. **Closed.**

2. **L1 P_KJ,cal = 32.9 GPa < 35 GPa target threshold.** §6 item 1 now correctly states "Its P_KJ,cal = 32.9 GPa also falls below the nominal 35 GPa target threshold (3D-CNN surrogate: 40.5 GPa)." This is honest scoping that was missing in Pass 1. **Strengthened.**

3. **Sensitivity-prediction quantification.** Pass-1 R2.4 asked for a one-sentence rough estimate of how Politzer-Murray BDE under/over-estimates h50 vs experiment for the lead chemotypes. The Table D.1c caption now contains this directly: "RDX h50,model = 26.3 cm vs literature 25-30; TATB 89.2 vs literature 140-490 (both routes underpredict TATB; H-bond cushioning is not in the BDE-only class scheme)." And the L1-specific h50 = 30 vs 83 cm discrepancy is explicitly flagged. **Closed.**

4. **CSP follow-up.** §7 now reads: "Crystal structure prediction is the natural follow-up validation; recent benchmarks of CSP on energetic molecular crystals (Crystal Growth & Design 2023, DOI:10.1021/acs.cgd.3c00706) demonstrate that polymorph screening is feasible at this candidate-list scale." **Closed.**

5. **DNTF 7th-anchor honest-failure paragraph** is preserved verbatim in §7. **Strengthened.**

6. **New (revision-introduced) concern — D.7 N-fraction stratification table sign flip.** Table D.4 now has a footnote correctly explaining the +0.54 km/s residual in the [0.55-1.00] N-fraction bin as an artefact of the open-form Q approximation. This is good, but the §5.2.2 prose still says "Pearson r(f_N, residual) = +0.43 (p = 4×10^-27)" without flagging that the population-level open-form K-J residuals (-2.59 to -3.59 km/s in the low/mid-N regime) are ~2 km/s more negative than the per-lead closed-form residuals in Table D.2 (-1.18 to -2.27 km/s). A reader cross-referencing the two tables will be confused. **P2.** One sentence in §5.2.2 explaining that the population K-J residual table uses open-form Q while the per-lead K-J residual uses closed-form Q, and that only the trend (not absolute values) is the diagnostic.

7. **L1's chemotype extrapolation note** is now in §7 ("L1's ρ_cal = 2.09 g/cm³ involves chemotype extrapolation: no nitroisoxazole anchor is present in the 6-anchor set; a packing factor of 0.65 (lower end for aromatic compounds, vs 0.69 used here) would give ρ ≈ 1.97 g/cm³, shifting D_KJ by roughly ±0.3 km/s"). This is exactly the right level of disclosure. **Strengthened.**

---

## Reviewer 3: Wow Factor / Impact

**What is closed from Pass 1:** Abstract qualifier lands without diluting impact; SELFIES-GA 3.5 km/s collapse is in Fig 1 (down-arrow; very effective); E1 is now a co-headline lead in abstract / §1 / §6 / §8; the "candidate-for-synthesis" recommendation paragraph is now in §8 ("L1 satisfies the candidate-for-synthesis criteria... We recommend it for synthesis-and-characterisation"); domain-gate generalisability appears in §1 para 3 ("The tier-gating recipe is domain-agnostic; only the validation funnel changes per application").

**Residual concerns**

1. **Figure 1 caption.** Now mentions both the SELFIES-GA collapse arrow (purple triangle-up to triangle-down) and the MolMIM marker; this makes the central argument self-contained as Pass 1 asked. **Closed.**

2. **L1 chemotype-rediscovery framing in §6 item 1.** The §6 item 1 prose now explicitly notes "L1 is a chemotype rediscovery within the polynitroisoxazole family known to the energetic-materials literature ... the 3,4,5-trinitro-1,2-isoxazole isomer DGLD proposes is absent from the 65 980-row labelled master ... and from PubChem." This is the right framing: not "DGLD invented an unknown ring" but "DGLD proposes a novel positional isomer in a known productive family." This is actually a stronger story scientifically than "novel ring" because it puts L1 inside Hervé / Sabatini / Tang prior art. **Strengthened.**

3. **The "two HMX-class leads" message** is now consistent across abstract ("E1 reaches D = 9.00 km/s... pending thermal stability confirmation"), §1 ("A second lead, E1..."), §6 item 1 / item 8, and §8. The story arc reads cleanly. **Strengthened.**

4. **Residual concern on §6 item 5 number.** As Reviewer 1 flagged, "more than doubles (4639 vs 966)" should read "nearly five-fold" or "~5×" to match Fig 17 caption and Table 8. **P2.**

---

## Reviewer 4: Abstract Reader

**What is closed from Pass 1:** CHNO is expanded on first use ("carbon-hydrogen-nitrogen-oxygen (CHNO)"); CFG jargon is decompressed ("standard guidance methods fail silently when the sampling trajectory is as short as molecular generation requires"); productive quadrant is defined inline ("simultaneously novel and on-target for performance"); max-Tanimoto is defined inline ("structurally dissimilar from all 65 980 training molecules (nearest-neighbour Tanimoto 0.27)"); HMX-class qualifier is now present ("by Kamlet-Jacobs relative ranking; absolute D requires a thermochemical equilibrium solver, see §7"). Pass-1 abstract concerns are essentially fully addressed.

**Residual concern**

1. **Abstract paragraph 1 still leans technical.** "Sparse-label, multi-objective generation problem" survives in para 1; the rest of the para is now decompressed but this opening clause remains jargon-forward. Pass-1 R4 suggested "generation under sparse, heterogeneous labels with multiple competing design targets." This is a stylistic call; the current text is acceptable but not maximally accessible. **P3.** Optional softening.

2. **One Pass-1 nitpick uncaught:** the Zenodo DOI text now reads "reserved DOI 10.5281/zenodo.19821953". Fine for submission. The §8.1 disclosure ("the DOI mints on publication of the deposition") is the correct phrasing for production. **Closed.**

3. **Abstract paragraph 3 closing baseline summary** is well-balanced. Three strong baselines (LSTM memorisation 18.3%; SELFIES-GA 3.5 km/s collapse; REINVENT 4 D=9.02 km/s) each get one clear sentence. **Strengthened.**

---

## Reviewer 5: Figures and Tables

**What is closed from Pass 1:** Fig 1 now shows the SELFIES-GA up/down-triangle collapse and the MolMIM diamond annotation; Fig 22 forest plot has been regenerated to include REINVENT 4 and SELFIES-GA on the same axis; Fig 23 is redesigned with a legend below the x-axis; Table 7c has the K-J formula bias column; Table G.2 n/a cells are explained ("not ranked by D; N-fraction proxy"); Table 6a's † / ‡ comparability footnote is now adequately verbose.

**Residual concerns**

1. **Fig 23 caption complexity.** The new Fig 23 caption is dense (~200 words) and combines "S vs viability" plus four baseline markers plus L1-L20 plus h_50 marker-area encoding. Visually it works; the caption is a reading load. **P3.** Consider splitting the caption into a 2-line plot description + a "Source / encoding" note below.

2. **Table 6a memo rate column** for REINVENT 4 reads "0.04% exact match, seeds 1-2; <1% novelty-window, seed 42" — this footnote is consistent with the prose but the percentages are different categories of "memorisation" and a reader may misread the cell. **P3.** Add a note "(exact match: full-SMILES rediscovery; novelty-window failure: max-Tanimoto > 0.55)."

3. **Fig 19 lead-card border colour scheme** (green / red) is unchanged. Pass-1 flagged red-green colour blindness as a P4 item. Not addressed. **P4.** Optional accessibility fix.

4. **Table C.1 split.** Pass-1 P4 suggested splitting C.1 into C.1a/b/c. Not done. The single Table C.1 is still long but hierarchically grouped by component (rowspan headers). The current form is acceptable for an appendix table; the split is a polish item. **Closed at P4 level.**

5. **Table 13 row for E2 D and P** has a † (cross-reference dagger) but the table-level note is below the table heading rather than at the bottom of the table; layout is fine but a strict typesetter would want the dagger note reformatted. **P4.** Cosmetic.

---

## Editor Summary

### Overall verdict

**Accept with minor textual fixes.** The Pass-1 priority items are essentially all closed. The Pass-2 issues are factual-consistency cross-checks (a couple of numbers that drifted between sections in the edit pass) plus a few stylistic suggestions. None require new experiments and none rise to the bar of "Major Revision."

### What blocks acceptance now

Three small textual items, listed below as P1. All can be fixed in <2 hours of editing.

### What is most strengthened by the revisions

1. **Abstract framing.** The K-J relative-ranking qualifier lands without diluting impact; the productive-quadrant / max-Tanimoto terms are defined inline; CHNO expansion is clean. Pass-1 R4's primary concerns are fully addressed.
2. **The §5.5 → Table 8 + Appendix G split.** This is a structural improvement: a reader can now see all seven ablations on one page with the tier-gate result legible at first read, while the detailed prose lives in Appendix G where it belongs. The Pass-1 concern that the tier-gate result was "buried" is fully addressed.
3. **The L1 + E1 two-headline story.** The "(pending thermal stability confirmation / oxatriazole-class anchor)" qualifier is now consistent across abstract / §1 / §6 / §7 / §8. The two-lead story strengthens rather than dilutes the headline. Pass-1 R3's primary impact concern is closed.

### Focused fix list (priority-ordered)

| # | Priority | Issue | Location | Effort |
|---|---|---|---|---|
| 1 | P1 | Composite "lower = better" vs "higher = better" inconsistency: Table 6a / G.4 / Fig 22 use penalty (lower = better); §G.3 narrative ("lifts max composite from 0.64 to 0.69") and Fig 23 caption use reward (higher = better). Add a one-sentence footnote on first use clarifying the two scales, or rename the Table-6a column "composite penalty." | Table 6a caption + §G.3 first sentence | 1 sentence |
| 2 | P1 | Pool-fusion lift number drift: §6 item 5 says "more than doubles (4639 vs 966)" but Fig 17 caption and Table 8 say "5.1× / +5×" and Table G.5 confirms 4.8×. Replace "more than doubles" with "nearly five-fold" or "~5×". | §6 item 5 | 1 word |
| 3 | P1 | Self-distillation round-0 definition: §4.8 says "round 0 = initial Random Forest" but Appendix C.1 says "round 0 trains on corpus only" (i.e. score-model round, not RF round). Reconcile. | §4.8 vs Appendix C.1 first paragraph | 1 sentence |
| 4 | P2 | Open-form vs closed-form K-J in Table D.4 vs Table D.2: the population K-J residual table uses open-form Q (residuals -2.59 to -3.59 km/s); per-lead Table D.2 uses closed-form Q (residuals -1.18 to -2.27 km/s). The §5.2.2 prose should flag that only the trend is the diagnostic, not the absolute residual values. The Table D.4 footnote already says this; pull it up to §5.2.2 main prose. | §5.2.2 | 1 sentence |
| 5 | P2 | CFG-w sweep is only 3-point ({5, 7, 9}); §7 should add a one-line acknowledgement of this as a small grid. | §7 | 1 sentence |
| 6 | P3 | Abstract para 1 "sparse-label, multi-objective generation problem" is still jargon-forward; optional softening to "generation under sparse, heterogeneous labels with multiple competing design targets" (Pass-1 carry-over). | Abstract | 1 phrase |
| 7 | P3 | Fig 23 caption is ~200 words; split into a plot description + a "Source / encoding" note below. | Fig 23 caption | Cosmetic |
| 8 | P3 | Table 6a memo rate cell for REINVENT 4 mixes "exact match" and "novelty-window" categories; add a one-line definitional note. | Table 6a | Cosmetic |
| 9 | P4 | Fig 19 lead-card colour scheme remains red/green; consider blue/orange or shape indicator for accessibility. | Fig 19 | Optional |
| 10 | P4 | Table C.1 split into C.1a/b/c (Pass-1 carry-over). | Appendix C | Optional |

### In-flight experiments (not blockers for acceptance)

- T1 BDE Modal job (xtb infrastructure fixed, science fix in progress): would tighten the L1 thermal-stability bound. Land as response-to-review or post-acceptance erratum.
- T2 density bracket: already integrated in §7. Closed.
- T3 multi-seed denoiser (3 seeds, ~12 hr): would close the denoiser seed-variance gap acknowledged in §7. Land in response-to-review or as a supplementary table at proof stage.

None of T1/T2/T3 affects the headline numbers. The paper's evidence base is sufficient at the current state.

---

**Final recommendation: Accept after the three P1 fixes (~2 hours of editing). The paper is ready for the Production editor.**
