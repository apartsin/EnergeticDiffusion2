# Referee Report (v1.1+ revision pass): DGLD: Domain-Gated Latent Diffusion for the Discovery of Novel Energetic Materials

**Journal:** Nature Machine Intelligence
**Manuscript:** docs/paper/index.html (Aperstein, Berman, Apartsin), revision since prereview-v1.0
**Reviewer role:** Diffusion-models / energetic-materials ML
**Status of original review:** Major revision (PAPER_REVIEWER_REPORT.md, v1.0)

---

## 1. Summary in reviewer's words

The revised manuscript is the same architecture and the same headline candidate (trinitro-1,2-isoxazole, L1) but the thermochemistry it rests on is no longer the same number. A unit-conversion bug in the ZPE pathway of the DFT pipeline (m2_dft_pipeline.py reading PySCF's freq_au as energy rather than angular frequency) was found and patched (commit 4abe43d); every cached HOF was inflated by roughly 15,600 kJ/mol, the 2-anchor calibration intercept was absorbing that bias, and the post-fix intercept is a defensible -205.7 kJ/mol with a 2-anchor leave-one-out residual band of plus or minus 62.5 kJ/mol. The K-J recompute on calibrated DFT now returns finite values for all twelve chem-pass leads (the pre-fix Q<0 oxygen-rich-undefined branch for L1, L5, L19 was a direct artefact of the ZPE bug), and L1 lands at K-J calibrated D = 9.52 km/s, which agrees with the 3D-CNN surrogate to within 0.04 km/s. The paper has also gained a free-route Cantera ideal-gas CJ recompute (commit 149852a) which is honest about its own limits: absolute numbers are roughly 3x too low because no covolume EOS is used, but the relative ordering kills the K-J L4 = 13.27 km/s claim as a regression artefact rather than a thermochemically consistent prediction. Per-lead h50 columns, an AiZynth nine-lead extension, and a reconciled retro narrative are also new since v1.0.

## 2. Strengths

The methodology contributions identified in the v1.0 review (alpha-anneal classifier-guidance bug, four-tier label-trust mask, hard-negative cheat-mining loop, multi-head sample-time gating) are unchanged and remain the strongest reasons to publish. The revision adds three further strengths.

**The ZPE bug fix is itself a useful reproducibility contribution.** A 42.7x silent unit error in PySCF's freq_au field, propagated through atomization-energy HOF, produced the implausible 16,763 kJ/mol calibration intercept that both v1.0 reviewers flagged. The authors found it by following the reviewer chain of reasoning (the intercept was off by two orders of magnitude relative to the GMTKN55 per-atom band, as my v1.0 minor issue noted), localised it, patched it, and re-derived all 17 cached per-lead JSONs while preserving the pre-fix values as *_pre_unit_fix audit fields. This is exactly the right way to handle a numerical-bug discovery during peer review.

**The post-fix calibration is no longer a 2-anchor zero-residual fit.** The HOF intercept is now -205.7 kJ/mol with a 2-anchor LOO residual of plus or minus 62.5 kJ/mol, which is in the plausible range for a B3LYP/6-31G(d) atomization-energy bias on small CHNO heterocycles. The authors are honest in Appendix D that this is still a 2-anchor fit and a 6-anchor extension remains deferred, but the headline number is no longer absorbing two orders of magnitude of unit-error.

**The Cantera CJ confirmation is a free, honest, partial alternative to EXPLO5.** Rather than gate the paper on a paid EXPLO5 licence, the authors built a Cantera/NASA9 ideal-gas CJ root-finder and reported what it can and cannot say: the absolute D values are too low because no BKW/JCZ3 covolume EOS is used at 30-100 GPa product pressures, but the relative L4/RDX ratio (1.01 from Cantera vs 1.61 from K-J calibrated) demonstrates that the high-N regime breakdown is a regression artefact, not a thermochemical truth. This is the right level of honesty about a free thermochemistry replacement; it does not substitute for EXPLO5/Cheetah-2 but it reframes the K-J L4 = 13.27 km/s number from "implausible result the authors should not have published" to "regression artefact that the population-level r = +0.43 evidence and the Cantera ratio independently confirm."

## 3. Walk-through of v1.0 Major weaknesses

**v1.0 weakness 1: Two-anchor DFT calibration is statistically unjustifiable for the headline number.**
- v1.0 quote: "the slope of 4.275 is implausibly steep ... the headline rho = 2.00 g/cm3 ... is the 3D-CNN surrogate output, not the DFT calibrate ... a 6-anchor calibration ... is exactly what should have been done before submission."
- v1.1+: The HOF intercept of -16,763 kJ/mol was the visible symptom of the ZPE unit bug, not a calibration choice; it is now -205.7 kJ/mol with a LOO residual of plus or minus 62.5 kJ/mol. The density slope of 4.275 is, however, unchanged, and the L1 calibrated rho of 2.53 g/cm3 still appears in Table D.1c. The authors have not added the four extra anchors (HMX, PETN, FOX-7, NTO) and Appendix D.4 still defers the 6-anchor extension.
- Verdict: PARTIALLY ADDRESSED. The HOF half of the calibration is now defensible as a screening tool. The density half is not; rho_cal = 2.53 g/cm3 for L1 is still above any known CHNO solid and still rests on two density points. The abstract is also still using the 3D-CNN rho = 2.00 as the headline number, which is now defensible (K-J calibrated D = 9.52 km/s confirms 3D-CNN D = 9.56 km/s), but the calibrated rho should be foregrounded as the more conservative number with its uncertainty band, not the surrogate.

**v1.0 weakness 2: K-J recompute is undefined for L1; headline D rests on 3D-CNN surrogate only.**
- v1.0 quote: "the calibrated DFT thermochemistry returns 'K-J undefined' for the headline lead ... once raw DFT is used, K-J gives D = 13.18 km/s, a number neither the authors nor any reader would defend."
- v1.1+: This was the single most damaging finding of v1.0 and it has been resolved cleanly. The Q<0 branch was a direct artefact of the inflated HOF; under the post-fix calibration L1 is K-J D = 9.52 km/s, P = 48.1 GPa (Table D.2), agreement with the 3D-CNN surrogate to 0.04 km/s. All twelve chem-pass leads are now defined on the K-J branch. L4's K-J D = 7.01 km/s post-fix replaces the v1.0 13.27 km/s claim, which is what the regime-aware story predicted.
- Verdict: RESOLVED. The headline lead now has a DFT-anchored K-J detonation velocity that matches the 3D-CNN surrogate and is no longer in the "K-J undefined" branch. The remaining gap is EXPLO5/Cheetah-2 absolute confirmation, which the Cantera scaffold does not provide.

**v1.0 weakness 3: Retro prose contradicts aizynth_results.json.**
- v1.0 quote: "the narrative description in the paper does not match the JSON. This is the kind of mismatch that gets a paper retracted from a chemistry venue."
- v1.1+: §5.6 has been rewritten against the actual JSON content (commit 7482920): n_routes = 9, state_score = 0.50, step_count = 4, with explicit acknowledgement that the four energetic-domain intermediates are NOT in stock and only tert-butanol and DPPA are. The wet-lab framing was downgraded from "most actionable" to "most template-reachable". An AiZynth extension run on the other 9 leads is reported in Table 9b (commit b98771f) and shows zero productive routes within the 200-MCTS / 300-s budget, which is the right framing of "USPTO-template MCTS finds nothing for energetic-domain ring closures".
- Verdict: RESOLVED.

**v1.0 weakness 4 (reproducibility check): m6_postprocess_bundle empty.**
- v1.0 quote: "m6_postprocess_bundle/results/ is empty (the m6_post.json referenced repeatedly ... is not present in the bundle the reviewer can inspect)."
- v1.1+: The bundle is now populated with the 9-file 3-seed grid (m1_3seed_C0_unguided_seed0..2.txt, C1_viab_sens_seed0..2.txt, C2_viab_sens_hazard_seed0..2.txt, C3_hazard_only_seed0..2.txt) plus chem_filter.py, chem_redflags.py, feasibility_utils.py, and labelled_master.csv. m6_post.py and m6_post_run.sh ship in the bundle root.
- Verdict: RESOLVED for the on-disk artefact. The Zenodo deposit live-status remains an editorial issue at submission time.

**v1.0 weakness 5: Validity = 1.000 across all MOSES conditions warrants audit.**
- v1.1+: I do not see an explicit validity-audit paragraph in the diff. The MOSES table still reports validity = 1.000 across all four conditions.
- Verdict: STILL OPEN. This is a low-effort fix (run RDKit MolFromSmiles with sanitize=True on the raw decoder outputs and report the sanitization fail-rate separately from the post-canonicalisation rate). It should be added before resubmission.

**v1.0 weakness 6: Tier-A row counts disagree (~3000 vs 575).**
- v1.1+: The 575 number is the subset of Tier-A rows carrying SIMULTANEOUS experimental rho, HOF, AND D, which is now stated explicitly in §5.5 / Appendix D.7. The "approximately 3000" Tier-A figure is the count with at least one experimental property. The reconciliation is now self-consistent in the prose although a single sentence in §3.1 making it explicit ("Tier-A spans approximately 3000 rows with at least one experimental property; 575 of these carry rho, HOF, and D simultaneously") would close the issue.
- Verdict: PARTIALLY ADDRESSED. The numbers are now reconcilable on careful reading; one cross-reference sentence in §3.1 would resolve it.

**v1.0 weakness 7: §5.2 prose claims viability >= 0.99 contradicts Table 1.**
- v1.1+: I do not see this fixed in the diff (no commit message references the §5.2 viability claim). Table 1 still shows row-3 viab = 0.83, row-4 = 0.93, row-5 = 0.86; if §5.2 still says "all five top leads have viability >= 0.99" it is still inconsistent.
- Verdict: STILL OPEN, low-effort. Either the prose is corrected to "viability >= 0.83" or the threshold is dropped.

## 4. Minor issues from v1.0

Of the 17 minor items in §4 of v1.0, the following were addressed in the revision:
- ZPE / atomization-energy intercept (the underlying root cause of multiple numerical complaints): RESOLVED via 4abe43d.
- Per-lead h50 columns and a chemotype-class BDE comparison (a7dace2): RESOLVED.
- Fig 8 "credible" editorialising (e0af58e): RESOLVED.
- Table D.1 splitting to fit page margin (e67dc80): RESOLVED.
- Fig 4 split into 4(a) training and 4(b) sampling (30d2867): RESOLVED.
- abstract \rho rendering (b98771f): RESOLVED.

Items still open: validity = 1.000 audit (weakness 5 above), §5.2 viability prose (weakness 7), L9/L20 minimum real frequency = 0 cm-1 (chemist reviewer minor; same fix), the duplicate ref-kamlet1968detonation entry, the Table D.2 RDX raw-DFT K-J = 12.14 km/s vs experimental 8.75 km/s open-form K-J disclosure.

## 5. Specific questions for the authors (updated)

1. The K-J calibrated D for L1 (9.52 km/s) now matches the 3D-CNN (9.56 km/s) to 0.04 km/s. Is this convergence robust to the choice of K-J product distribution (i.e., the open-form Q used here vs the Becker-Kistiakowsky-Wilson form), or would a different product-set choice put L1 back in the Q<0 branch?
2. The 6-anchor density calibration (HMX, PETN, FOX-7, NTO added) is still deferred. Why was the 6-anchor extension not run during this revision cycle, given that the four additional anchors are routine DFT optimisations that take roughly 6-24 CPU-hours each?
3. The Cantera ideal-gas CJ root-finder gives L4/RDX = 1.01. Has any sensitivity analysis been done on the Cantera result (e.g., NASA9 vs NASA7 polynomials, the 13-species CHNO product list vs an extended set including NO2, N2O, HCN), and would a covolume-augmented EOS (e.g., a simple BKW-flavoured correction in Cantera) close the absolute-value gap to within a factor of 1.5 or 2?
4. The abstract still uses the 3D-CNN rho = 2.00 g/cm3 as the headline density. Now that the K-J calibrated D = 9.52 km/s is available, would the authors consider replacing the abstract numbers with the calibrated DFT pair (rho_cal = 2.53 g/cm3, K-J D_cal = 9.52 km/s) plus the LOO uncertainty bands? The current setup conflates two estimators where the calibrated DFT is now the more rigorous one.
5. Has any across-seed denoiser-training spread been measured, or is the headline still single-denoiser-training plus 3-seed sampling? If single, please run two more denoiser trainings.

## 6. Reproducibility check (updated)

- m6_postprocess_bundle/results/ now ships the 9-file 3-seed grid (RESOLVED).
- m2_summary.json reflects the post-bug-fix HOF/rho cal values; pre-fix values preserved as *_pre_unit_fix for audit (good).
- aizynth_results.json now matches the §5.6 prose (RESOLVED).
- Cantera CJ scaffold (compute_cj_detonation.py, results/cj_cantera.json) ships in the repo (good).
- Per-lead h50 inference driver (scripts/h50_predict_leads.py) ships in the repo (good).
- Zenodo deposit live-status: still an editorial issue at submission time. NMI policy requires reviewer-accessible artefact at submission.
- MOSES validity = 1.000 audit: still missing.
- Single-denoiser-training caveat: unchanged from v1.0.

## 7. Decision recommendation to the editor

**Minor revision.**

The two most damaging v1.0 findings (calibration intercept of -16,763 kJ/mol; K-J undefined for the headline lead) were both downstream consequences of one ZPE unit bug, which the authors found, patched, and re-derived all dependent numbers from. The post-fix L1 K-J D = 9.52 km/s now agrees with the 3D-CNN surrogate to 0.04 km/s; this is the DFT-anchored detonation-velocity number whose absence I flagged in v1.0. The retro prose reconciliation (commit 7482920), the AiZynth nine-lead extension (commit b98771f), and the populated m6_postprocess_bundle (the 9-file 3-seed grid is now on disk) close three further v1.0 weaknesses cleanly.

What remains is small. The 6-anchor density calibration is still deferred, and the L1 calibrated rho = 2.53 g/cm3 is still above any known CHNO solid; this is the single most important item the authors should still do. The MOSES validity = 1.000 audit and the §5.2 viability prose are 30-minute fixes that were missed in the revision pass. An EXPLO5/Cheetah-2 absolute-D recompute on L1, L4, L5 remains the gold-standard close, and the authors are open about Cantera being only a partial alternative.

If the authors come back with (a) a 6-anchor density calibration, (b) the validity-audit table, and (c) the §5.2 viability prose corrected, I would recommend acceptance. If they come back with all of the above plus an EXPLO5 recompute on L1/L4/L5, this is a clear accept; the methodology contribution is well above the NMI bar and the chemistry case is now defensible at the screening-grade level.

## 8. Confidential comments to the editor

The v1.0 to v1.1+ revision is a clean response. The ZPE bug discovery is the kind of root-cause find that turns multiple-of-the-reviewer-complaints into a single fix, and the authors did the right thing by preserving the pre-fix values as audit fields rather than silently overwriting. The remaining 6-anchor density extension is the one chemistry weakness I would not waive; everything else is at minor-revision cost. Compared to what NMI typically prints in this space (Hoogeboom EDM, MolMIM, DiGress), the methodology bar is met and the chemistry-validation bar is now at "candidate-grade with a known density-calibration uncertainty band", which is acceptable for an NMI methodology paper provided the abstract is honest about the uncertainty bands. I would lean accept-with-minor-revision.
