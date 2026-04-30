# NMI Reviewer Committee Report (Pass 3)
## DGLD: Domain-Gated Latent Diffusion for the Discovery of Novel Energetic Materials

_Prepared for the Editor, Nature Machine Intelligence_
_Date: 2026-04-30_
_Document covers: revised `short_paper.html` after the Pass-2 fix pass (six edits landed: composite-direction footnote, pool-fusion lift number, round-0 reconciliation, T1 BDE integration, 4 bibliography entries, Fig 22b inset, T7 removal)._

---

## Executive Summary (Editor)

Five reviewers reread the manuscript after the six Pass-2 fixes landed. **All three Pass-2 P1 items are closed.** Two minor textual residuals from the edit pass remain (one mild numerical phrasing tension; one cosmetic Fig 23 caption length). Neither is a blocker. The §7 T1 BDE integration is a genuine strengthening and was cleanly executed. The Fig 22b productive-quadrant inset is a useful visual companion to Table 6a. T7 removal is invisible in the prose. Bibliography is now complete on the four placeholders called out in Pass 2.

**Recommendation: Accept.** Editorial polish only; no further reviewer round required.

---

## Reviewer 1: AI / ML Expert

**Pass-2 closures verified.**

- Pass-2 P1 #1 (composite-direction inconsistency): Fig 23 caption (line 615) now contains the explicit "Note on score conventions" paragraph distinguishing the higher-is-better Stage-1+2 reranker score (y-axis) from the lower-is-better Pareto-reranker penalty in Table 6a / Fig 22 / Table G.4. The two scales are flagged as operating at different pipeline stages with cross-figure ranking consistency. **Closed.**
- Pass-2 P1 #2 (pool-fusion lift number drift): §6 item 5 (line 663) now reads "post-filter yield rises nearly five-fold (4 639 vs 966 candidates, +5×)". Matches Fig 17 caption (line 354: 5.1×) and Table 8 (line 639: +5×). **Closed.**
- Pass-2 P1 #3 (self-distillation round-0 ambiguity): §1 contributions list (line 105), §4.12 (line 357), and Appendix C.1 (line 807) all use the same definition: "round 0 = score-model trained on corpus only with Random-Forest-derived viability labels and 0 hard negatives, round 1 = corpus + 137 mined hard negatives, round 2 = corpus + 918 cumulative hard negatives + aromatic-heterocycle boost." §4.8 (lines 287-291) is fully consistent with this. **Closed.**

**New residual (P3 only).**

§5.5 ablation summary prose (line 649) says "Pool fusion across 5 lanes more than quadruples post-filter yield". 4 639 / 966 = 4.80×, which is technically "more than quadruples", but §6 item 5, Table 8, and Fig 17 all use "nearly five-fold / +5× / 5.1×". The ablation summary now reads slightly more conservative than the headline. **P3.** Optional unification to "nearly quintuples" or "rises ~5×" for cross-section consistency. Not a blocker; both phrasings are factually correct.

T3 multi-seed denoiser still flagged as in-flight in §7 (line 684); honest disclosure stands. No new ML concerns.

---

## Reviewer 2: Materials Science Expert

**Pass-2 closures verified.** Open-form vs closed-form K-J residual concern (Pass-2 R2.6) was a P2, not a Pass-2 P1, and remains a P2 carry-over (Table D.4 footnote already covers it; pulling to §5.2.2 prose is still a one-sentence polish that did not land in this pass).

**T1 BDE integration assessment.** The new §7 paragraph on GFN2-xTB weakest-bond BDE (line 676) is exactly the right addition. L1 weakest = 86.0 kcal/mol (C-NO2, dominant initiation channel); E1 = 92.7 kcal/mol (exocyclic C-NO2); both above the ~70 kcal/mol Politzer-Murray Ar-NO2 typical, with a correctly-stated caveat about unrelaxed-fragment xTB over-binding. The paragraph honestly notes "the unrelaxed-fragment xTB scan over-binds slightly relative to a fully relaxed wB97X-D3BJ recompute, but the chemistry assignment and the weakest-bond ranking are correct." This is the appropriate level of disclosure and substantively strengthens the L1 / E1 sensitivity story.

**No new concerns.** Six-anchor calibration, packing-factor bracket, h50 BDE table, oxatriazole anchor caveat, DNTF 7th-anchor honest-failure paragraph, and the L1 chemotype-extrapolation note all read as before. Materials-science narrative is now complete at submission grade.

---

## Reviewer 3: Wow Factor / Impact

**Pass-2 closures verified.** Two-headline (L1 + E1) story remains consistent across abstract / §1 / §6 / §8 / §7. SELFIES-GA collapse arrow remains in Fig 1 caption.

**Fig 22b inset assessment.** The new Figure 22b (line 601-604) showing the productive-quadrant snapshot of all six methods, with the SELFIES-GA surrogate to DFT collapse inset, is a valuable second visual landing of the central baseline-comparison message. It complements Fig 22 (forest plot) by giving readers a single-glance view of which methods occupy the green-tinted productive quadrant. Marker-area encoding (1 - memorisation rate) is a clean way to surface the LSTM memorisation issue alongside the novelty / D axes. **Strengthened.**

**No new concerns.** Impact narrative unchanged.

---

## Reviewer 4: Abstract Reader

**Pass-2 closures verified.** Abstract jargon-decompression survives intact. CHNO is expanded; productive-quadrant defined inline; max-Tanimoto contextualised; HMX-class qualifier present.

**No new concerns.** Pass-2 P3 about the para-1 "sparse-label, multi-objective generation problem" opening clause was optional and remains; not a blocker.

---

## Reviewer 5: Figures and Tables

**Pass-2 closures verified.**

**Fig 23 caption complexity (Pass-2 P3).** The new "Note on score conventions" paragraph (line 615) has *added* roughly 80 words to the caption to land the Pass-2 P1 #1 fix. Caption is now ~280 words. The fix was necessary and the prose is well-targeted, so this is not a regression so much as a deliberate trade: caption length grew to close a more important factual concern. Acceptable. **P3 carry-over.** Optional split into a 2-line plot description plus a "Source / encoding" footer remains a polish item.

**Fig 22b (new).** Caption (line 603) is concise, properly sourced ("Source values: Table 6a"), and the inset is well-labelled. Marker-area encoding for memorisation rate is explained. No issues.

**Table 6a memo rate cell (Pass-2 P3).** The REINVENT 4 footnote ("0.04% exact match, seeds 1-2; <1% novelty-window, seed 42") still mixes two memorisation categories without an inline definitional gloss. Pass-2 P3 polish item; not addressed in this round. **P3 carry-over.**

**No new figure / table issues introduced by the edit pass.**

---

## Editor Verdict

**Accept.** All three Pass-2 P1 items are closed; the six new edits (composite-direction footnote, pool-fusion lift number, round-0 reconciliation, T1 BDE paragraph, Fig 22b, bibliography fills) all land cleanly without introducing new factual errors or broken cross-references. The remaining items are P3 polish that the production editor can handle at proof stage.

### Remaining items (none P0 or P1)

| # | Priority | Issue | Location |
|---|---|---|---|
| 1 | P3 | §5.5 ablation summary says "more than quadruples" while §6 item 5 / Table 8 / Fig 17 say "nearly five-fold / +5× / 5.1×" (both technically correct; 4 639/966 = 4.80×). Optional unification. | line 649 |
| 2 | P3 | Fig 23 caption now ~280 words after composite-direction footnote added. Optional split into plot description + "Source / encoding" footer. | line 615 |
| 3 | P3 | Table 6a REINVENT 4 memo cell mixes "exact match" and "novelty-window" categories without inline gloss. | line 595 |
| 4 | P2 carry | Open-form vs closed-form K-J residual scope note in §5.2.2 prose (Pass-2 R2.6 carry-over; Table D.4 footnote already covers it). | §5.2.2 |
| 5 | P3 carry | Abstract para 1 "sparse-label, multi-objective generation problem" opening clause optional softening. | abstract |

### In-flight (not blockers)

- T3 multi-seed denoiser (~10 hr remaining): currently honestly flagged as in-flight in §7 (line 684); response-to-review territory.
- T1 BDE on additional E-set leads beyond E1: future-work scope, not required for headline.

### Final recommendation

**Accept.** The paper is ready for the Production editor. The five remaining items are stylistic polish and can be handled at proof stage without a further reviewer round.
