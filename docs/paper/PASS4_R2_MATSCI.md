# PASS4 Reviewer 2 (Materials Science) Report

Final line-by-line check. Sections inspected: §2.4, §2.5, §3 (incl. §3.1), §5.2 (5.2.1 and 5.2.2), §5.3 (5.3.1, 5.3.2, 5.3.3), §6, §7, §8, Appendix D.1 to D.7. Reading was line-by-line, not skimmed. Reviewer reports of passes 1, 2, 3 were not opened.

## 1. Chemistry concerns

- **E2 violates a Stage-1/Stage-2 hard filter that §5.2.1 itself describes.** Table 12 (line 542) lists E2 as `O=[N+]([O-])NC([N+](=O)[O-])[N+](=O)[O-]`, formula C1H2N4O6, an N-nitramine with gem-trinitromethyl on a single carbon centre (one C bearing two C-NO2 groups; the third nitro is the N-NO2 on the amine). §5.2.1 line 375 explicitly states the production hard filter rejects "four with three nitros on a one-carbon centre." Either the gem-trinitromethyl pattern in E2 should have been blocked at Stage 1/2, or the filter description in §5.2.1 needs a clarifying parenthetical (e.g., the rule applies only when the carbon also lacks the N-NO2 partner). The footnote "E2 D and P are flagged: OB = +28.9% exceeds the +25% K-J reliability limit" already caveats the K-J number but does not address the SMARTS-gate inconsistency. This was raised in some form in earlier passes; the fix is either to drop E2 from the E-set headline or add a sentence in §5.3.3 acknowledging that E2 was promoted under a relaxed Stage-1 setting because the Tanimoto-cap mining used a separate SMARTS gate.

- **L1 nitrogen-fraction reported value drifts between §5.2.2 (line 414) and Table 7c (line 456).** Body text: "L1 has N-fraction ≈ 0.27 and oxygen balance ≈ +8 %." Table 7c column "K-J formula bias": "f_N ≈ 0.29." For C3N4O7 the heavy-atom N-fraction is 4/14 = 0.286, so 0.29 is correct and 0.27 is a rounding error on the same molecule. Since the 0.27 number is used to argue L1 sits "near PETN's K-J reliability regime (PETN, f_N = 0.17)," the gap to PETN is larger than stated; the qualitative argument still holds but the body should say 0.29.

- **NTO N-fraction reported as 0.40 on line 414** but NTO is C2H2N4O3, giving heavy-atom f_N = 4/9 = 0.44 (or all-atom 4/11 = 0.36 if H counted). Neither convention gives 0.40 cleanly; convention should be stated and value harmonised with the PETN line just above ("f_N = 0.17") which uses the heavy-atom convention (PETN C5H8N4O12, 4/21 with H = 0.19, or 4/21 = 0.19 either way; 0.17 is borderline). Pick one convention and apply uniformly.

- **K-J regime band in Table 1 line 189** quotes the canonical Kamlet-Jacobs oxygen-deficient regime as `2a + b/2 ≥ d ≥ b/2`. This is correct (a = #C, b = #H, c = #N, d = #O). E2 OB = +28.9 %, which corresponds to d > 2a + b/2; flagging E2 D as upper-bound is the right move. Consistent with the §5.3.3 footnote.

## 2. DFT / calibration

- **Functional / basis justification (D.1) is adequate.** B3LYP/6-31G(d) for geometry, ωB97X-D3BJ/def2-TZVP single-point, GMTKN55-grounded. The added sentence about B3LYP overestimating N-heteroatom bond lengths in oxatriazole/tetrazole rings now provides the chemotype-specific caveat the earlier passes asked for.

- **Thermal correction (D.2) is internally consistent.** 0 K HOFs are reported throughout; the 10-25 kJ/mol shift to 298 K is acknowledged and shown to sit inside the ±64.6 kJ/mol LOO RMS. The TATB-specific note in D.4 (line 884) about the 17 kJ/mol residual from the 0 K to 298 K thermal correction is a clean explanation of what would otherwise look like a calibration gap.

- **6-anchor calibration numbers are stable across §5.2.2, Table 7b, Appendix D.4, Table D.1c, Table D.2, and Table C.5 row 869.** Slope 1.392, intercept -0.415 on density; HOF offset -206.7 kJ/mol; LOO RMS ±0.078 g/cm³ on ρ and ±64.6 kJ/mol on HOF. Cross-run delta ≤0.0002 g/cm³. No drift detected.

- **The PETN HOF anchor footnote in Table 7b (line 434)** picks -538 kJ/mol from the LLNL handbook with a literature range of -504 to -539 kJ/mol; the spread is acknowledged to fit inside ±64.6 kJ/mol. This is appropriate.

- **L1 calibrated ρ extrapolation caveat (line 678):** "L1's ρ_cal = 2.09 g/cm³ involves chemotype extrapolation: no nitroisoxazole anchor is present in the 6-anchor set; a packing factor of 0.65 (lower end for aromatic compounds, vs 0.69 used here) would give ρ ≈ 1.97 g/cm³, shifting D_KJ by roughly ±0.3 km/s." This is the right caveat but it competes with the §6 item 1 headline that L1 clears the ρ ≥ 1.85 threshold. Even at 1.97 the threshold is cleared; safe.

## 3. Sensitivity prediction

- **h50 BDE estimate vs score-model estimate disagree by 2.7× on L1 (Table D.1c caption, line 910).** 30.3 cm (model) vs 82.7 cm (BDE). The caption now explicitly flags this as a chemotype-extrapolation effect and says "neither route is authoritative for a novel chemotype; experimental impact-sensitivity testing is required." This is the correct stance. Anchor sanity check (RDX 26.3 model vs 25-30 lit; TATB 89.2 model vs 140-490 lit; both routes underpredict TATB) is shown, with the H-bond-cushioning argument given.

- **Politzer-Murray BDE class typicals are stated as Ar-NO2 70, R2N-NO2 47, R-CH-NO2 55, R-O-NO2 40 kcal/mol (Table D.1c caption).** These are the canonical values from Politzer-Murray 2014 and applied consistently to L1 (Ar-NO2, h50=82.7) and the E-set (Table 13). Internally consistent.

- **Crystal-packing caveat is applied consistently.** §7 line 672 names crystal packing as the dominant unquantified error source; line 678 puts the L1-specific bracket; line 674 (T2 cross-check) gives Bondi-vdW ρ ∈ [1.69, 1.87] for L1 and [1.65, 1.83] for E1 across pk ∈ {0.65, 0.69, 0.72}; the polymorph absence is stated. The §8 conclusion (line 692) recommends synthesis-and-characterisation of L1 with this caveat in the body, not in the recommendation paragraph; could be tightened by adding "subject to crystal-packing and polymorph confirmation" to the recommendation, but not blocking.

## 4. Synthesisability

- **AiZynthFinder framing in §5.3.2 (lines 497-518) is correctly conservative.** L1 returns 9 routes, top route 4 steps, state score 0.50; the proposed terminal disconnection (one-step electrophilic ring-nitration of 4,5-dinitro-1,2-isoxazole with HNO3) is correctly flagged as improbable under HNO3 alone given ring deactivation, with literature references to fuming HNO3 / oleum / N2O5 routes. The DPPA-Curtius hazard call-out is appropriate for a primary-explosive-class acyl azide intermediate.

- **The 11-of-12 negative result is correctly attributed to USPTO drug-domain template bias, not to unsynthesisability of L4-L20.** This is the right read.

- **§8 recommendation (line 692) calls L1 a "candidate-for-synthesis" with the 4-step AiZynth route as evidence.** The body of §5.3.2 (line 509) already flags that the energetic-domain intermediates (4,5-dinitroisoxazole, Boc-protected amine, etc.) are `in_stock: false` against ZINC. The recommendation paragraph should repeat this caveat; as written it could be read as overreaching. Minor.

## 5. Numerical drift

- **L1 N-fraction 0.27 (§5.2.2) vs 0.29 (Table 7c).** Same molecule. See Chemistry concerns.
- **NTO f_N = 0.40 (§5.2.2 line 414).** Doesn't match either heavy-atom (0.44) or all-atom (0.36) convention. See Chemistry concerns.
- **L1 calibrated ρ:** 2.09 (Table 7c row L1, Table D.1c, §5.2.2 body, §6 item 1, §8). Consistent.
- **L1 D_K-J,cal = 8.25 km/s** (Table 7c, Table D.2, §6 item 1, §8). Consistent.
- **6-anchor calibration constants** (1.392, -0.415, -206.7) stable across all six locations checked.
- **K-J anchor residuals (Table D.2 lines 968-973):** RDX -1.32, TATB -0.97, HMX -1.58, PETN -0.06, FOX-7 -1.17, NTO -0.44 km/s vs experiment. §5.2.2 body line 414 lists "RDX -1.32, HMX -1.58, FOX-7 -1.17" and on PETN says "-0.06 km/s consistent with low N-fraction (0.17)." The body text earlier in §5.2.2 line 414 reads "the four added anchors give a population K-J residual against experiment spanning -0.06 to -1.58 km/s across the six anchors; for the three highest-N anchors (RDX, HMX, FOX-7) the residuals are -1.2 to -1.6 km/s (RDX exp 8.75 vs K-J 7.43; HMX exp 9.10 vs K-J 7.52; FOX-7 exp 8.87 vs K-J 7.70)." These match Table D.2 exactly. No drift.
- **E1 numbers (Table 13 line 559 vs §6 item 8 line 666 vs §7 line 678 vs body line 572):** ρ_cal 2.04 / 2.043, D_K-J,cal 9.00, P_K-J,cal 38.6 GPa, h50_BDE 82.7. Consistent.
- **Population K-J residual N-fraction Pearson r = +0.43 (p = 4×10⁻²⁷, n = 575)** quoted at lines 415, 665, and Table D.4 caption. Consistent.

## 6. Cross-tier label-trust

- §3.1 says Tier-A and Tier-B drive the conditional gradient; Tier-C and Tier-D train the unconditional prior via classifier-free dropout. §4.1 line 206 repeats this. No contradicting claim found in §5 or §6.

- §3.1 footnote on line 189 ("Tier C: K-J derived from quoted ρ & HOF; regime-limited; under-predicts D for high-N low-H compounds where the assumed gas-product distribution is wrong") matches the §5.2.2 / Appendix D.6 / Appendix D.7 narrative exactly.

## Verdict: **Accept (with two minor edits suggested)**

Pass-3 verdict was Accept; pass-4 confirms the manuscript is submission-ready. The numerical scaffolding is internally stable across §5.2.2, §6, §7, §8, and Appendix D. The two suggested edits are:

1. **Reconcile the L1 N-fraction value** (0.27 in §5.2.2 body vs 0.29 in Table 7c) and **state the N-fraction convention** for NTO (0.40 doesn't match heavy-atom or all-atom). Five-minute fix.
2. **Either drop E2 from §5.3.3 or add one sentence** explaining how E2's gem-trinitromethyl pattern survives a Stage-1 SMARTS gate that §5.2.1 explicitly says rejects "three nitros on a one-carbon centre." Either action is fine; the current text reads as inconsistent.

Both are non-blocking. The DFT methodology, K-J regime caveats, h50 disagreement framing, crystal-packing bracket, and AiZynthFinder negative-result attribution are all appropriately scoped. The headline L1 / E1 picks are defended honestly. Submit.
