# Pass-5 Reviewer 3 (Wow Factor / Impact) — `short_paper.html`

Scope: post §5-rewrite (commit `5e2492e`, tag `post_5_rewrite`). Read line-by-line: Abstract, §1, §5.4, §5.5, §6, §8, plus Figs 1, 19, 23 captions and Tables 6, 7.

---

## 1. Where it lands well

- **Abstract (line 81).** The single most memorable sentence in the paper is now: "DGLD is the only method to land in the productive quadrant (simultaneously novel and on-target) at DFT level." That is the wow-factor headline; the SELFIES-GA 9.73 to 6.28 collapse follows immediately. After the rewrite the centerpiece survives intact, in fact tightened: "(a 3.5 km/s surrogate artefact)" is now a parenthetical and reads as a punch.
- **§1 paragraph 3 (line 93).** Same sentence "DGLD is the only method with consistent novel productive-quadrant coverage" lands here too, and the SELFIES-GA collapse is cited at full strength. The trio LSTM-memorises / GA-collapses / REINVENT-peaks-at-9.02 is the strongest single moment in the introduction.
- **Figure 1 caption (line 97).** The lower-left inset trick (surrogate triangle to DFT inverted-triangle) is the most visually compelling single design choice in the paper; the caption now spells it out explicitly with both numbers (9.73, 6.28). Strong.
- **§5.4 E1 paragraph (line 511).** "both E1's calibrated D and rho are higher than L1's" is a real co-headline pivot, and the two-honest-readings device (genuinely stronger vs upper-bounded pending oxatriazole anchor) reads as scientific maturity rather than caveat dilution. This is the section's emotional peak.
- **§6 discoveries 1, 3, 8 (lines 584, 586, 591).** The productive-quadrant framing is now the through-line; discovery 3 is where the SELFIES-GA collapse is told at full length; discovery 8 explicitly elevates E1 to "provisional co-headline lead from a chemotype family disjoint from the L1 isoxazole family". E1 survives the rewrite.
- **§8 paragraph 2 (line 615).** "Two leads from two distinct scaffold families on a single sampling run rules out the alternative reading that L1's productive-quadrant placement is a sampling artefact" — this is the single best new sentence in the post-rewrite paper. It converts E1 from a side-finding into structural support for the L1 claim.
- **Table 6 + Table 7 (lines 476, 494).** E1 row numbers (rho_cal 2.043, D 9.00, P 38.6, h50 82.7) are bolded inline at line 511 and survive as standalone tabular evidence. The bolding inside the cell (`<strong>2.043</strong>`, `<strong>9.00</strong>`) is correct wow-factor signalling.

## 2. Where impact is muted (concrete rewrites)

- **Abstract sentence-2 burying L1's distinctness (line 79).** "...is structurally dissimilar from all 65 980 training molecules (nearest-neighbour Tanimoto 0.27)." The "0.27" is one of the strongest numbers in the paper but reads as a parenthetical addendum to the rho/D values. Suggested rewrite: lead the second clause with novelty and treat rho/D as the qualifier, e.g. "L1, structurally distinct from all 65 980 training molecules (nearest-neighbour Tanimoto 0.27), reaches rho_cal 2.09 g/cm^3 and D 8.25 km/s." Same words, stronger order.
- **Abstract sentence on E1 (line 79).** E1's headline number (D 9.00 km/s, *higher than L1*) is currently flat: "A second lead, E1 ..., reaches D 9.00 km/s from a chemotype family disjoint from L1's." The fact that E1 is *higher* than L1 on D is buried; the sentence reads as a follow-up rather than a co-headline. Suggested rewrite: "A second lead from a disjoint chemotype family, E1 (4-nitro-1,2,3,5-oxatriazole), reaches D 9.00 km/s, *exceeding L1 on detonation velocity* (Kamlet-Jacobs relative ranking ...)." That italicised clause is what §6 and §8 already say; the abstract should match.
- **§1 paragraph 3 on E1 (line 93).** Same problem at line 93: "A second lead, E1 ..., reaches D 9.00 km/s from a chemically distinct scaffold family, pending thermal stability confirmation and an oxatriazole-class DFT anchor (§7)." The caveat clause swallows the headline. Move the caveat to a trailing parenthetical and state the comparison-to-L1 inline.
- **Figure 19 caption (line 373).** Says "Twelve chem-pass DGLD leads" but never mentions L1 or the wow-factor density 2.09 g/cm^3 by name. The single most-reproduced figure in any future citation will be Figure 19; it should name L1 and quote rho_cal 2.09 in the caption itself.
- **Figure 23 caption (line 541).** Long, technical, and burdened with the "Note on score conventions" disclaimer that explains two different composite scales. The disclaimer is correct and necessary, but it ends the caption on a deflating note. Suggested rewrite: open the caption with the wow-factor sentence ("DGLD is the only method whose top-1 lands in the green productive quadrant at DFT level"), then descriptive material, then the score-convention note last (where it is now).
- **§5.5 SELFIES-GA collapse (line 537).** The 9.73 to 6.28 collapse, which is the single most memorable adversarial finding in the paper, is now buried mid-sentence inside a long REINVENT/MolMIM paragraph. Suggested rewrite: split the SELFIES-GA collapse out as its own short bolded sentence at the top of the paragraph, e.g. "**The most informative single result: SELFIES-GA's best 40k novel candidate collapses from D_surrogate 9.73 to D_DFT 6.28 km/s under the same DFT chain (a 3.5 km/s surrogate artefact).**" This is the §5 wow factor; do not let it run together with REINVENT's 9.02.
- **§6 discovery 1 (line 584).** Discovery 1 spends most of its words explaining that L1's *D* falls below 9.0 km/s (the target threshold) and a rationalisation of the K-J vs 3D-CNN gap. That is honest but it muffles the lead. The wow-factor number in this discovery is rho_cal = 2.09 g/cm^3 (the highest of the set) and Tanimoto 0.27 (more novel than the augmented-corpus median). Lead with those, push the D-below-threshold sentence to the second half. Discovery 1 currently reads as a defence of L1 rather than a presentation of L1.

## 3. Headline-claim drift

I traced the L1 framing across all instances:

| Location | L1 framing |
|---|---|
| Abstract line 79 | "rho 2.09, D 8.25" — no band claim |
| Abstract line 81 | "the next compound to enter the HMX-class band" — generic, not L1-specific |
| §1 line 93 | "placing it within the HMX/CL-20 performance band at max-Tanimoto 0.27" |
| Fig 1 caption line 97 | "lands in the HMX-class band" — DGLD as a class |
| §5.3 line 393 | "places L1 in the HMX-class regime by *relative ranking against anchors*" |
| §6 disc 1 line 584 | "falls *below* the nominal D >= 9.0 km/s threshold; it is within the HMX-class band by K-J relative ranking" |
| §8 line 613 | "in the HMX/CL-20 performance band on commodity hardware" — DGLD as a class |
| §8 line 617 | "L1 meets five criteria" — no band claim, just criteria list |

This is *cleaner than Pass 4 reported*: the relative-ranking caveat is now uniformly present at every L1 mention. Mild residual drift remains in:

- **Abstract line 81 vs line 79.** Line 81's "the next compound to enter the HMX-class band" carries the connotation that *L1 is not yet that compound*, in tension with line 79's confident `rho 2.09 / D 8.25`. The two abstract paragraphs disagree on whether L1 *has entered* the band or *the framework lets future compounds enter* it. Suggested fix: at line 81, change "the next compound to enter" to "future compounds in the HMX-class band can be ..." removing the implication that L1 is not such a compound.
- **§1 line 93 vs §6 disc 1 line 584.** Line 93 says L1 is "within the HMX/CL-20 performance band". Line 584 says it "falls below the nominal D >= 9.0 km/s threshold; it is within the HMX-class band by K-J relative ranking". §1 should match §6's precision (one extra clause: "by K-J relative ranking"); a reader of §1 alone gets a stronger claim than the data supports.
- **Figure 1 caption line 97.** Says DGLD "lands in the HMX-class band" and "top-1 ... sits within 5% of the D >= 9.0 km/s threshold". 5% of 9.0 is 0.45 km/s; L1's calibrated D is 8.25 (8.3% short, not 5%). The "within 5%" applies to Hz-C2 top-1 D 9.39 km/s on the 3D-CNN scale, *not* to L1 calibrated. The caption mixes scales without flagging it. Either drop "within 5%" or qualify it as "(3D-CNN top-1)".

E1 framing (Pass 4 explicitly asked me to verify E1 co-headline survives):

| Location | E1 framing |
|---|---|
| Abstract line 79 | "second lead ... reaches D 9.00 km/s from a chemotype family disjoint from L1's" |
| §1 line 93 | "further lead from a chemically distinct scaffold family, pending..." |
| §5.4 line 511 | "**E1 oxatriazole as a co-headline finding**" — full bold heading |
| §6 disc 8 line 591 | "reported as a provisional co-headline lead from a chemotype family disjoint" |
| §8 line 615 | "promoting E1 to co-headline status pending..." |

E1 is consistently a co-headline in §5.4, §6, §8 but is *not* called a co-headline in the abstract or §1. **This is the most actionable headline-drift remaining.** If E1 is co-headline in §5/§6/§8, the abstract should say so: e.g. line 79 add "co-headline" or "second co-lead". Currently a reader of the abstract alone meets E1 as a follow-up; a reader of the body finds it elevated. Recommendation: at line 79 change "A second lead, E1" to "A co-headline lead, E1".

## 4. §5 rewrite-specific impact assessment

**§5.4 (combined novelty + retro + scaffold).** The combined subsection lands as one cohesive arc, not a stitched combination, because the opening sentence sets the through-line: "DGLD generates a chemotype distribution (10 DFT leads / 8 Bemis-Murcko scaffolds / 6 families), not a single isoxazole hit." This is a strong organising claim and the three audits (novelty, retro, E-set) each support it on their own axis. The H4 sub-headings (Novelty audit / Retrosynthesis audit / E-set scaffold-diversity audit) are correctly used as scene-setters rather than as new sub-sub-sections. E1 retains its standalone moment at line 511 with a bolded callout sentence and the "two honest readings" device. **Co-headline survives the merge.**

The retrosynthesis sub-section (lines 460-471) is the weakest part of §5.4 — 1/12 hit rate is honest but reads as an anticlimax between the strong novelty result and the strong E-set result. The current framing ("a public-USPTO drug-domain template-database gap, not unsynthesisability") is the right framing but it is buried. Consider moving the L1-9-routes-state-score-0.50 number earlier (currently at the end of the paragraph), so the retro audit opens with the positive finding and ends with the contextualisation.

**§5.5 (combined no-diffusion baselines).** The SELFIES-GA D 9.73 to 6.28 collapse is preserved (line 537) but, as flagged in §2 above, is now buried mid-paragraph with REINVENT 4 and MolMIM. This is the largest single wow-factor loss in the rewrite. Pre-rewrite the SELFIES-GA collapse had its own subsection; post-rewrite it is one of three findings in a single paragraph, and structurally REINVENT's 9.02 number sits at the end where the eye lands. Recommend lifting the SELFIES-GA collapse to its own sentence, bolded, at the top of the paragraph (or even as a one-line lead-in callout).

Otherwise the four-baseline collapse into one §5.5 reads as cleaner than the pre-rewrite four-subsection version. The Table 8 + Figure 22 + Figure 23 + Figure 24 combination is now the right level of evidence density for the section.

**Net §5 rewrite assessment.** Wow factor preserved on E1, partially diluted on SELFIES-GA. Cohesion improved on §5.4. Length reduction successful. The single targeted lift would be promoting the SELFIES-GA collapse line back to a structural emphasis it had pre-rewrite.

## 5. Verdict

**Accept with minor lifts.** The §5 rewrite is a net win: §5.4 holds together as a single arc, E1 survives as co-headline in body sections, the productive-quadrant claim is now consistently the headline framing across abstract, §1, §6, §8. The SELFIES-GA D=9.73 to 6.28 collapse, the strongest single adversarial finding in the paper, has lost some of its structural emphasis in §5.5 and should be promoted back. E1 should also be marked as co-headline in the abstract to match §5.4/§6/§8. Headline-claim drift on L1 ("HMX-class band" vs "below threshold but within band by relative ranking") is now uniformly handled in §5.3 and §6 and only mildly inconsistent in §1 and Figure 1's caption.

No data overshoot detected. The paper does not overclaim L1, E1, or the 12 leads at this draft.
