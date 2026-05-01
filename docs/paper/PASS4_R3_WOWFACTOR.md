# PASS 4 Reviewer 3: Wow Factor / Impact Audit

Reviewer scope: Abstract, §1, §6, §8, Figs 1, 19, 22b, 23 captions.

## Where it lands well

- **Abstract, line 81** — "DGLD is the only method with consistent novel productive-quadrant coverage" is a sharp, defensible single-sentence headline. The paired "3.5 km/s surrogate artefact" framing is genuinely memorable; few generative papers anchor their lead claim against a quantified failure-mode of a competitor.
- **Abstract, line 79** — leading with "12 DFT-confirmed novel leads" plus the L1 numbers (rho_cal = 2.09, D_K-J,cal = 8.25 km/s, Tanimoto 0.27) is a strong triple-hook (count + property + novelty). The parenthetical K-J caveat is honest without burying the result.
- **§1 Introduction, line 89** — "the field has not surfaced a new HMX-class compound in the last fifteen years" is the strongest motivational sentence in the entire paper; it gives the reader an immediate stake.
- **§1, line 93** — "DGLD is the only method with consistent novel productive-quadrant coverage" repeated verbatim from the abstract; this is good (intentional reinforcement of the headline).
- **Fig 1 caption (line 99)** — the SELFIES-GA surrogate-to-DFT collapse panel description ("9.73 km/s drops to 6.28 km/s ... 3.5 km/s artefact") is the single most compelling visual narrative in the paper; it converts a methodology point into a story.
- **§6 item 3 (line 661)** — direct, well-structured comparison. The "75/100 exact corpus rediscoveries" and "18.3% memorisation, seed-stable" are punchy memorable numbers.
- **§6 item 8 (line 666)** — "tip of a credible distribution, not an isolated peak" is a very effective framing; it pre-empts the "lucky one molecule" criticism.
- **§8 Conclusion, line 690** — opens with the methodology triad (training-time gate, sample-time guidance, validation funnel) and lands on "DGLD is the only method with consistent productive-quadrant coverage confirmed at DFT level." Clean closing hook.
- **§8 line 692** — the "Recommendation for synthesis-and-characterisation" paragraph is unusually concrete for an ML paper and gives the reader a sense of agency.

## Where impact is muted (with rewrite suggestions)

- **Abstract, line 79** — "placing it within the HMX/CL-20 performance band (by Kamlet–Jacobs relative ranking; absolute D requires a thermochemical equilibrium solver, see §7)" buries the headline inside a 20-word inline caveat. The K-J caveat is needed but should follow a sentence break, not interrupt the headline. Suggested: "...placing it within the HMX/CL-20 performance band, structurally dissimilar from all 65 980 training molecules (nearest-neighbour Tanimoto 0.27). The K-J value is relative-ranking-grade; absolute D requires a thermochemical-equilibrium solver (§7)." Same fact, but the lead survives intact.

- **§1, line 93** — E1's strong number (D_K-J,cal = 9.00 km/s from a chemically distinct family) is currently buried in clause 4 of a long sentence and shielded by "pending thermal stability confirmation and an oxatriazole-class DFT anchor." This is the second-strongest single number in the paper and deserves its own sentence. Suggested: "A second lead, E1 (4-nitro-1,2,3,5-oxatriazole), reaches D_K-J,cal = 9.00 km/s from a chemotype family disjoint from L1 (oxatriazole-class anchor still pending; §7)."

- **§6 item 1 (line 659)** — the headline discovery paragraph spends roughly a third of its text on the K-J/3D-CNN gap discussion before stating the novelty number. Recommend leading with: "L1 is novel (max-Tanimoto 0.27 to all 65 980 labelled rows; absent from PubChem) and HMX-class on calibrated DFT-K-J (rho_cal 2.09 g/cm^3, D 8.25 km/s)." Then the calibration discussion. As written, the reader has to wade through caveats before reaching the "absent from PubChem" punch line.

- **§6 item 5 (line 663)** — the sentence "post-filter yield rises nearly five-fold (4 639 vs 966 candidates, +5x) and scaffold diversity rises to 24 Bemis-Murcko scaffolds" is impressive but reads as a parenthetical to the trinitro-isoxazole point. This is itself a strong scaling result and could be promoted into its own bold sub-claim.

- **§8 Conclusion, line 690** — does not name E1 anywhere. Given E1 is referred to as a "provisional co-headline lead" in §6.8, its complete absence from the closing paragraph mutes the "two-lead, two-chemotype-family" story. Suggested addition: "A second lead, E1 (oxatriazole family), reaches D_K-J,cal = 9.00 km/s pending an oxatriazole-class anchor."

- **Fig 19 caption (line 378)** — does not state how many of the 12 leads pass electronic stability (green vs red borders). The reader has to count. A single sentence ("11 of 12 pass electronic stability at gap >= 1.5 eV") would convert ornament into payload.

- **Fig 22b caption (line 603)** — "Only DGLD Hz-C2 sits in the green-tinted productive quadrant" is the best impact-statement caption in the paper. Already strong; no change needed.

- **Fig 23 caption (line 615)** — extensive (and useful) "note on score conventions" footnote at the end overshadows the actual visual claim. Recommend moving the convention note to the body text or to a smaller caption-footnote rendering.

## Headline-claim drift

The paper uses three slightly different framings of L1's status. Reconciling them is the single most important wow-factor lift:

| Section | Framing on L1 |
|---|---|
| Abstract line 79 | "within the HMX/CL-20 performance band (by K-J relative ranking)" |
| §1 line 93 | "within the HMX/CL-20 performance band at max-Tanimoto 0.27" |
| §6 item 1 line 659 | "falls below the nominal D >= 9.0 km/s threshold; it is within the HMX-class band by K-J relative ranking" + "P_K-J,cal = 32.9 GPa also falls below the nominal 35 GPa target threshold" |
| §8 line 690 | "DGLD is the only method with consistent productive-quadrant coverage confirmed at DFT level" |
| Fig 1 caption | "DGLD ... satisfies novelty > 0.45 and HMX-class on every axis" |

**The drift**: §6 item 1 explicitly admits L1 falls below both the D >= 9 km/s and the P >= 35 GPa nominal thresholds, while the abstract, §1, §8 and Fig 1 caption all assert "HMX-class" or "HMX/CL-20 performance band" without that qualifier. Fig 1 in particular says "HMX-class on every axis" which is incompatible with §6 item 1's explicit two-threshold miss. A reviewer who reads §6 carefully will spot this and treat the abstract framing as an over-reach.

**Reconciliation suggestion**: pick one consistent framing across all five locations. The defensible one is "within the HMX/CL-20 *relative-ranking* band" (which §6 supports). Strip "HMX-class on every axis" from Fig 1 caption; replace with "HMX-class density and relative D ranking; absolute D requires CJ recompute (§7)."

A second smaller drift: the count "12 DFT-confirmed leads" appears uniformly, but §6 item 8 reveals that of the E-set extension only "4 of 9 with a defined K-J recompute clear D_K-J,cal >= 7.0 km/s" — a relatively low bar. The "12 leads" claim is technically about being DFT local minima, not about all 12 reaching HMX-class D. Recommend explicitly distinguishing "12 DFT-confirmed local minima, of which N reach D_K-J,cal >= 8 km/s" so a reviewer cannot read the 12-count as "12 HMX-class leads."

A third smaller drift: the abstract mentions "REINVENT 4 generates genuinely novel heterocycles but peaks at D = 9.02 km/s" while §1 line 93 and §8 line 690 repeat the same framing. This is consistent and good. However, this number being *higher* than L1's calibrated 8.25 km/s is a latent narrative tension that the paper never explicitly resolves: REINVENT 4 hits D = 9.02 (UniMol surrogate, not DFT-audited) while L1 sits at 8.25 (DFT-audited). The reader is invited to assume REINVENT's 9.02 would also collapse under DFT audit, but this is not stated. A single sentence acknowledging the apples-to-apples gap would defuse this.

## Verdict

**Accept with minor revisions.** The wow-factor signal is real and the paper now clearly stakes a defensible ground in productive-quadrant coverage with an honest baseline graveyard. The single most important fix is reconciling the "HMX-class" vs "below D = 9 / below P = 35" framings between Fig 1, the abstract, and §6 item 1 (currently inconsistent). The second is promoting E1 into the abstract and §8 with its own sentence rather than as a clause-3 afterthought. The third is the small but high-leverage edit of moving the K-J caveat in the abstract out of the headline sentence so the lead survives intact.
