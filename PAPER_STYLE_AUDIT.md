# DGLD Paper, Style Audit

Scope: rhetorical / scientific-writing style of `docs/paper/index.html` at HEAD. Past audit categories (Unicode, cross-refs, table numbering, 50-word splits, caveat unification) are excluded. All line numbers refer to `docs/paper/index.html`.

---

## 1. Top 5 highest-impact style issues

### Issue 1, repetitive "absent from the open literature does not exclude restricted databases" parenthetical
The exact phrase
> "absence from the open literature does not exclude restricted databases or unpublished synthesis attempts"
appears verbatim **four times**: in the abstract (line 81), §1 intro (line 102), §1.5 contributions (line 112), §6 summary (line 558), and §8 conclusion (line 582). The hedge is appropriate the first time, but reading it five times in one paper makes the prose sound defensive rather than careful.

Suggested fix: keep the full clause once (in §5.3.1 where the literature search is actually described), and elsewhere replace with a short pointer, e.g., "absent from PubChem and from the open energetic-materials literature (caveat in §5.3.1)".

### Issue 2, "chemotype rediscovery via property-targeted ML" repeated as a slogan
The same italicised contribution claim is restated near-verbatim at lines 81 (abstract), 102 (§1), 112 (§1.5), 558 (§6), 582 (§8). It reads as if the paper is anticipating a reviewer who hasn't been convinced and re-asserts the framing every time the lead is mentioned.

Suggested fix: state once in the abstract with the strong wording, then refer back in the body sections as "the L1 chemotype-rediscovery result" or simply "the L1 lead".

### Issue 3, §6 bullet on architectural choices is unmeasured
Line 560:
> "CFG-scale ablation, per-head guidance-scale grid, and hard-negative augmentation of the viability head each shift the productive-quadrant rate by **a measurable margin**; the alpha-anneal+clamp fix to multi-head guidance is **necessary** for the headline results to hold."

"a measurable margin" is empty; the appendices have actual numbers. "necessary for the headline results to hold" is also overstated, the headline 96/97 PubChem-novel result comes from an unguided pool (line 386, 89/100 from the unguided run), so the alpha-anneal fix is necessary for guided runs to differ from unguided, not for the headline. This bullet should either cite the numbers or be split.

Suggested fix: replace with concrete numbers: "CFG sweep w∈{5,7,9} moves filter pass-through from 528→983→427 (Appendix E.8); the alpha-anneal fix is necessary for per-head guidance to produce non-bit-identical samples (§5.3.2)."

### Issue 4, abstract paragraph 2 oversells "fix"
Line 79:
> "we identify and fix a noise-anneal misconfiguration that silently disables guidance in 40-step latent regimes."

The paper actually shows that disabling the anneal restores per-head differentiation, but it does **not** show that this improves headline numbers, the merged top-100 is 89% from an unguided pool. The abstract claim implies the fix is load-bearing for the lead candidate, which §5.3.4 contradicts.

Suggested fix: "we identify and document a noise-anneal misconfiguration that silently disables guidance in 40-step latent regimes; disabling the anneal restores per-head steering (§5.3.2). Headline candidates are produced predominantly by the unguided sampler reranked by the Phase-A composite (§5.3.4)."

### Issue 5, paragraph rhythm in §5.5 (longest paragraph in the body)
The single paragraph at line 499 runs roughly 350 words and crams together: DFT method, calibration coefficients, leave-one-out residual, GMTKN55 sanity check, the bug-fix disclosure, the corrected intercept, the density slope discussion, and an EXPLO5 forward-pointer. Even after the prior audit-fold pass, this paragraph reads as a list of disconnected facts.

Suggested fix: split into three paragraphs, (i) method and calibration result, (ii) bug-fix disclosure, (iii) caveats and forward-work pointers.

---

## 2. Promotional / editorialising language inventory

| Term | Line(s) | Verdict |
|---|---|---|
| "strong baseline ladder" | 113 | **Replace.** "Strong" is a self-judgement. → "two-baseline comparison". |
| "strongest available baselines" | 83 | **Keep** (qualified by "two", which is a factual count). |
| "strongest explosives in regular use" | 81 | **Replace** with citation, e.g. "HMX/CL-20-class". |
| "strongest joint perf-and-stability candidate" | 493 | **Keep** (factual within the table). |
| "strongest novelty bound" | 391 | **Replace.** "the most stringent of the three novelty checks" is more honest. |
| "robust production recipe" | 328, 467, 842 | **Replace.** "Robust" is unearned. → "the recommended production recipe" (drop "robust"). Used 3x. |
| "well-calibrated" | several (e.g. 471, 570) | **Keep** (factual descriptor for xTB on CHNO; explicit). |
| "well-precedented" | 318 | **Keep** (factual chemistry claim). |
| "carefully" | not found | n/a |
| "rigorously / rigorous" | not found | n/a |
| "principled" | not found | n/a |
| "novel" (as value judgement) | many | **Mixed.** Most uses are factual ("novel against PubChem"); a few drift, e.g. line 102 "produces genuinely novel chemistry" → drop "genuinely". |
| "chemically reasonable" | 527, 582 | **Keep** (factual, qualified by the reaction class). |
| "reasonable" (line 502) | 502 | **Replace.** "the headline D claim for L1 is reasonable" → "the headline D claim for L1 is consistent with both K-J and Cantera ranking". |
| "alarming" (density slope) | 499 | **Replace.** Editorialising. → "the density slope of 4.275 is large and reflects only two anchors with one packing coefficient". |
| "comprehensive" | not found | n/a |
| "extensive" | not found | n/a |
| "in-depth" | not found | n/a |
| "compelling" | not found | n/a |
| "remarkable" | not found | n/a |
| "striking" | not found | n/a |
| "promising" | not found | n/a |
| "powerful" (per gram) | 949 | **Keep** (in glossary, used in physical sense). |
| "state-of-the-art" | not found | n/a |
| "competitive" | 458, 555 | **Keep** but qualify, line 555 "competitive with the HMX/CL-20 reference class on predicted detonation performance" is fair given Table 1 numbers; line 458 "is competitive given the constraint" is hedged enough. |
| "productive" / "productive-quadrant" | many | **Keep** (defined at first use as a factual region). |
| "headline" | many | **Keep** (factual, marks the lead claim). |
| "clean future-work path" | 499 | **Replace.** "Clean" is a value word. → "an EXPLO5/Cheetah-2 recompute is the natural follow-up". |
| "self-consistent" | 887 | **Keep** (factual descriptor of the FCD/composite anti-correlation). |
| "honest generalisation estimate" | 902 | **Replace.** "more honest" is editorialising. → "An out-of-fold AUROC of 0.91 ± 0.05 (5-fold CV) indicates the head generalises beyond memorised positives". |
| "qualitatively correct" (sensitivity-head) | 666 | **Keep.** Standard usage. |
| "well-understood novelty bound" | 163 | **Replace** "well-understood" → drop or replace with "standard" (which is what it is). |
| "long-standing standard" | 673 | **Keep.** Bibliographically defensible. |
| "particularly relevant" | 133 | **Keep.** Editorial framing in related work, acceptable. |

Net: about a dozen unearned soft adjectives, only "alarming" / "robust" / "honest" / "clean" / "strongest novelty bound" / "well-understood" actively detract.

---

## 3. Hedging audit

### Over-hedged claims (the data support a stronger statement)

- **Line 386, novelty headline.** "96 of 97 PubChem-evaluable candidates are unknown to PubChem". The paper reports 0/100 within Tanimoto 0.70 to the augmented corpus and 0 exact matches (Table 2). The current sentence under-uses the augmented-corpus result, which is the harder novelty test. Suggest leading with the Tanimoto-stratified number ("0/100 within Tanimoto 0.70 of any of 694 518 training rows") and folding the PubChem number after.

- **Line 419, "performance is preserved while the per-head scales swap between scaffold families".** This is a clean finding; no need for the hedged final clause "the per-head gradients steer the sampler into different chemistry classes, validating the methodological claim". Drop "validating the methodological claim", the table validates it.

- **Line 902, "indicating the head genuinely discriminates hazardous vs safe latent-space chemistry rather than overfitting to memorised positives"**. With AUROC 0.91 ± 0.05 over 5-fold CV this is a defensible assertion; the word "genuinely" is unnecessary hedging on the assertion, drop it.

### Under-hedged claims (the data do **not** support the wording)

- **Line 81 abstract, "Predicted properties match the strongest explosives in regular use".** "Match" is too strong; the paper's own §5.5 reports a -0.04 km/s anchor-calibrated K-J residual on L1 but a 1.5–2 km/s population residual elsewhere, and §7 explicitly tags 3D-CNN D as relative-ranking-grade. Suggest "Predicted properties are within HMX/CL-20-class on the surrogate, with the L1 lead independently corroborated by an anchor-calibrated DFT–K-J recompute (D = 9.52 km/s)."

- **Line 502, "L1 calibrated K-J D = 9.52 km/s now corroborates the 3D-CNN-surrogate prediction of 9.56 km/s within 0.05 km/s, an independent quantum-chemistry recompute that lands within rounding of the surrogate's headline number".** This is a 2-anchor fit with leave-one-out residual ±62.5 kJ/mol on HOF; the K-J residual table (D.2) shows other leads with 0.78–2.16 km/s residuals. The L1 0.05 km/s agreement could be partly an anchor-calibration artefact (RDX is one of two anchors). Suggest qualifying: "L1 lands within 0.05 km/s of the surrogate; this is the closest agreement of the 12-lead set, where residuals range from −0.04 to −2.16 km/s (Table D.2)".

- **Line 580 conclusion**, "produce hundreds of novel CHNO candidates with predicted detonation properties at or above HMX". "Hundreds" is loose; cite the actual number, e.g., "the merged top-100 plus the 200-row Pareto front" or just "the merged top-100".

- **Line 102, "Only DGLD lands in the novel-and-on-target quadrant"**. This is true at the top-1 level but there are only three top-1 points compared (DGLD, SMILES-LSTM, MolMIM). "Only DGLD among the three baselines compared" would be honest.

- **Line 446, "DGLD top-1 candidates are the only ones that simultaneously satisfy novelty and high D"**. Same issue as above; n=3 is a small comparison set. Soften to "among the three method families compared".

---

## 4. Tense / voice / register issues

- **"We" usage is consistent.** No instances of "the authors" found.

- **Tense drift.** Mostly clean. One drift in §5.5 line 499: "Earlier drafts of this section reported a HOF intercept of −16 763 kJ/mol that both reviewers correctly flagged" mixes past historical (correct), with present-tense "the intercept collapses to −205.7 kJ/mol" (also correct as a permanent fact). This is fine, but note that "Earlier drafts ... reported" sits awkwardly inside a methods section, this is a meta-disclosure that probably belongs in a numbered Errata note or an appendix bug-disclosure box rather than mid-paragraph (§5.5 line 499; same content also at line 715 in Appendix D).

- **Passive constructions where active improves.**
  - Line 215 (alt-text + caption tag): not relevant.
  - Line 286, "validation token-accuracy is 64.5 %", consider "the fine-tuned encoder achieves 64.5 % token accuracy on validation".
  - Line 318, "Each pool is canonicalised, deduplicated, and stripped of charged species and over-large molecules". Reads cleanly as passive (no clear human agent), keep as is.
  - Line 320, "Hard filters reject (i) ... (ii) ...". Reads cleanly, keep.

- **Register, chemistry side.** No instances of informal "blow up", "explode", "bond breaks". The paper uses "decompose" and "shock-sensitive" appropriately.

- **Register, ML side.** No "the model just memorises" or "kitchen sink". The paper uses "memorisation" and "rediscovery" precisely.

- **Numbers/units.**
  - "65 980" with thin-space, used consistently (lines 81, 102, 106, 122, ...). No "65,980" drift detected.
  - "9.56 km/s" with space, used consistently. No "9.56km/s" drift detected.
  - "694 k" and "694 518" with thin-space, consistent.
  - "326 k", "40 k", "10 k" with thin-space, consistent.
  - One mild inconsistency: line 286 has "64.5 %" with thin-space, but several other places have "65 %", "77 %", "85 %" without explicit thin-space markup (raw " %"). The HTML uses `&nbsp;%` consistently, so this is just rendering, no fix needed.

---

## 5. Sentence-rhythm issues

### Over-long paragraphs

- **Line 102 (§1, paragraph 2 of intro after the four-families paragraph).** ~330 words. Crams the trinitro-isoxazole story, the SMARTS audit pointers, the labelled master neighbour, the baseline-ladder summary, and the Fig 1 callout. Same content is restated in Issue 2 above. Suggest splitting into (a) the lead candidate and its literature context, and (b) the baseline-ladder paragraph.

- **Line 499 (§5.5).** ~350 words; see Issue 5 in the Top-5 list.

- **Line 386 (§5.3.1).** ~250 words but reads cleanly because it is a single argument (PubChem novelty + xTB filter + labelled-master rediscoveries + the chemotype interpretation). Borderline, leave as is.

### Repeat openings

- §1 line 92 starts "High-energy-density materials..."; line 94 starts "We pose..."; line 96 starts "Reliable labelled data..."; line 98 starts "Four families of methods..."; line 100 starts "We propose...". Good rhythm.

- §6 summary bullets all start with bold lead phrases ("**Yield scales**", "**Top-ranked leads reach**", "**DGLD is the only family**", ...). Reads cleanly.

- §5.3.3 has three paragraphs in a row starting "**Headline.**", "**Result.**", "**Multi-seed pattern across DGLD conditions.**" (lines 445–449). Good.

- §5.6, three paragraphs in a row that each open by referring to the previous routes/result ("For L1, AiZynthFinder returns...", "L4 and L5 returned no productive routes...", "For the present paper this means..."). Connectives are present; rhythm is fine.

### Disconnected lists

- §6 (lines 556–565) is a bullet list of seven items under one header sentence. The header at line 555 promises a "consistent signal", and the bullets each report a different finding, but there is no closing transition; line 566 then tries to summarise. The list works, but the closing sentence at 566 ("The combined message is that...") is functional rather than synthetic, it restates the abstract. Optional improvement: replace with a sentence that ties the seven bullets to the one-line claim of §1.5.

---

## 6. Jargon-without-first-use-definition list

The glossary (Appendix G) defines: HEDM, HMX/CL-20/RDX/TATB/PETN, ρ, HOF, D, P, K-J, DFT, GFN2-xTB, SMARTS, Tanimoto/Morgan FP, Bemis–Murcko, SELFIES, cfg-dropout, FCD, FiLM, Chapman–Jouguet, atomization-energy HOF, Bondi vdW, ETKDGv3, gem-tetranitro, oxygen balance, Phase-A composite.

**Used in body without first-use definition AND missing from glossary**:

- **DDIM** (lines 309, 333, 406, 785). Used as a sampler name. Not in glossary. Suggest a one-line glossary entry.
- **DDPM** (lines 207, 211, 289, 555, 580). Defined ad-hoc at line 289 ("variance-preserving DDPM"), but expansion not given. Add to glossary.
- **MOSES** (lines 161, 833, 849, 851). Used multiple times. Not in glossary. Add a one-line entry ("MOSES, the Molecular Sets benchmark suite for distribution-learning metrics on molecular generators").
- **SA / SCScore** (lines 163, 318). Defined contextually ("synthetic accessibility... drug-like reaction corpora"), but the abbreviations themselves are not expanded at first use. Borderline; add to glossary for chemistry-reviewer-without-ML-background.
- **EMA** (line 291, "EMA decay 0.999"). Exponential moving average. Not defined. ML reviewers know it; chemistry reviewers may not. One-liner in glossary or first use.
- **CFG** (lines 295, 333, 380, 419, 830). Used as shorthand for "classifier-free guidance"; defined as such at line 140 with reference, but the abbreviation expansion is implicit. Add to glossary for completeness.
- **AUROC** (lines 511, 902). Standard ML term, but the paper has chemistry readers. Glossary would help.
- **PAINS** (line 851). Pan-Assay Interference compounds; used in Table E.4 column header without expansion. Add to glossary.
- **SNN** (Table E.4 column header, line 852). Single-Nearest-Neighbour Tanimoto, defined in the table caption at line 851. Acceptable.
- **Bemis–Murcko**, **DFT**, **GFN2-xTB**, **SELFIES**, **FiLM**, **classifier-free guidance**, **DDIM** (this last one is missing) → DDIM is the only one of these without a glossary entry.
- **MMFF94** (line 471), **B3LYP/6-31G(d)**, **ωB97X-D3BJ/def2-TZVP** (lines 471, 499). Functional and basis names; chemistry reviewers know them, ML reviewers will not. The body cites GMTKN55 / Goerigk for context but does not say "B3LYP and ωB97X-D3BJ are commonly used hybrid density functionals; 6-31G(d) and def2-TZVP are atom-centred Gaussian basis sets". One sentence at first use in §5.5 would help.
- **ELBO** (line 285, the LIMO loss). Glossary entry would help chemistry reviewers.
- **Pareto front** (line 326, 346). Used heavily. Not in glossary. Chemistry reviewers will mostly know it, ML reviewers definitely will. Optional add.
- **Bondi vdW** (line 499). In glossary, ok.
- **freq_au** (line 499, code-internal field name). Internal artefact; ok inline since it explicitly says "PySCF's `freq_au` field (angular frequency in atomic units)".

**Cleanest fix**: add 5 short glossary entries (DDIM, DDPM, MOSES, EMA, AUROC) plus one inline gloss in §5.5 for B3LYP/ωB97X-D3BJ.

---

## 7. Title / section / figure titles

- **Main title**: "DGLD: Domain-Gated Latent Diffusion for the Discovery of Novel Energetic Materials". Length OK; "for the Discovery of" is slightly over-formal, "for Discovering" would be tighter, but "for the Discovery of" matches journal-paper register. Keep.
- **§ titles**: "1. Introduction", "2. Related work", "3. Dataset", "4. Methodology", "5. Experiments", "6. Summary of results", "7. Limitations", "8. Conclusion". Clean and consistent.
- **Subsection titles**: mostly one short noun phrase (§4.1 LIMO fine-tuning, §5.4 Semi-empirical stability validation (xTB-GFN2)). Consistent.
- **Figure titles**: italicised "Takeaway" lines are present on most figures (Fig 1, 2, 3c, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15). Two figures lack a takeaway: **Fig 3(a)** at line 175 and **Fig 3(b)** at line 180 (the labelled-corpus EDA panels), and **Fig A.1** at line 624. For consistency, either add a takeaway or drop them from the others, the current state is mixed. I would add a short takeaway to Figs 3(a)/(b)/A.1 for consistency.
- **Table captions**: consistently start with "**Table N.**". Consistent.

---

## Section 7 (audit), Overall verdict on writing style

The paper's writing has been audit-folded enough times that gross issues (run-on sentences, broken hedges, mixed tense) are gone. The dominant remaining issue is **defensive over-repetition**: the same chemotype-rediscovery hedge and the same "absent from the open literature does not exclude restricted databases" disclaimer appear four to five times across abstract, intro, contributions, summary and conclusion. This makes the paper feel anxious about its own headline rather than confident in it. A second, smaller issue is unmeasured editorialising (the words "robust", "alarming", "honest", "strongest", "clean" are doing rhetorical work that the actual numbers already do better). The §5.5 paragraph on DFT calibration is the one place where information density still outpaces paragraph rhythm and would benefit from a three-way split. Glossary coverage is excellent for chemistry-side jargon and should be extended with ~5 short entries on the ML-side abbreviations (DDIM, DDPM, MOSES, EMA, AUROC). Tense and voice are consistent, "we" is used uniformly, and the abstract's quantitative claims are mostly hedged correctly with the one exception that "match the strongest explosives in regular use" is slightly stronger than the K-J residual table supports. Overall the paper is well-written and reviewer-ready; the suggested edits are tightening passes rather than structural rework.
