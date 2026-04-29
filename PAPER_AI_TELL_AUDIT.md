# Paper AI-Tell Audit — `docs/paper/index.html`

Scope: read-only audit of `docs/paper/index.html` (1015 lines) against the 15-pattern checklist for AI-generated prose tells. Effort cap respected.

---

## Section 1 — Hard tells (HIGH risk)

The paper is, on the whole, surprisingly low on AI-tells. Most of the canonical hallmarks (transition-adverb cascades, padding constructions, scope-claim closers, "Furthermore/Moreover" openers, "It is important to note that") are entirely absent (`grep` returned zero matches across the whole file). The hard tells that do remain are mostly structural.

1. **Line 81 (abstract, last sentence) — "Takeaway" sentence form recurring in every figure caption.**
   - Pattern: empty-summary closing, repeated as a template.
   - The italic *Takeaway:* line at the end of nearly every figure caption (Fig 1, 2, 3(c), 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15+) is a repeating rhetorical device that no human writer would impose with that uniformity. By Fig 9 it reads like a slot-filler.
   - Rewrite: pick three or four figures where a takeaway genuinely needs underlining and drop the device elsewhere. Do not reuse the literal word "Takeaway" twelve times.

2. **Line 102 — long appositional pile-up around the trinitro-1,2-isoxazole result.**
   - Pattern: long appositional clauses chained with semicolons doing the work of em-dashes ("X (Y); the closest published Z; the W proposes here is not located..."). Single sentences carrying three to five subordinate clauses.
   - This sentence appears verbatim or near-verbatim in the abstract (line 81), §1 intro (line 102), the contributions list (line 112), §6 summary (line 490), and §8 conclusion (line 514). Reviewers will notice the repetition.
   - Rewrite: state the chemotype-rediscovery framing once, in §5.2 or §6, and reference it by tag elsewhere.

3. **Line 92 — opening "Yet the canonical performance anchors of the field, HMX and CL-20, were developed decades ago" + tricolon.**
   - Pattern: tricolon ("smaller charges, higher specific impulse, and more efficient pyrotechnics") in the very first paragraph, immediately after "translate directly into."
   - This is the kind of triple-parallel that AI uses as a default rhetorical move. The third item ("more efficient pyrotechnics") is the weakest and shows the structure is doing the work, not the content.
   - Rewrite: "translate directly into smaller charges and higher specific impulse." Drop the third.

4. **Line 96 — "Reliable labelled data is small (a few thousand experimental measurements), heterogeneous in provenance (...), and expensive to extend."**
   - Pattern: tricolon with parenthetical decoration on each leg. Classic AI cadence.
   - Rewrite: "Reliable labelled data is small and expensive to extend; what exists is heterogeneous in provenance (DFT, K-J, 3D-CNN)."

5. **Line 98 — four-element parallel "Empirical formulas... Discriminative ML surrogates... Generative SMILES language models... the standard image-domain classifier-free-guidance recipe..."**
   - Pattern: enumerated parallel clauses with the same verb-and-limitation cadence ("X are fast and interpretable but regime-limited..."; "Y score candidates accurately but do not propose them"; "Z memorise"; "W ports poorly"). Four sentences in a row built to the same rhythm.
   - Rewrite: vary at least two of the four. Break one out into its own short sentence; collapse another into a clause of the previous.

6. **Line 100 — single sentence of 167 words ("DGLD... performs latent diffusion over a frozen, fine-tuned LIMO encoder... while Tier-C... and Tier-D... train only the unconditional prior via classifier-free-guidance dropout. At inference time it is gated by a three-layer chemistry-rule credibility filter...").**
   - Pattern: the one-sentence-paragraph encyclopaedia entry. AI loves to compress a system description into one runaway sentence with dependent clauses bracketed by em-dash-equivalents.
   - Rewrite: split into three sentences (architecture; training-time gating; inference-time gating).

7. **Lines 109–115 — six numbered contributions, each a standalone bolded bullet with a self-contained micro-paragraph of 3–6 sentences.**
   - Pattern: over-symmetric paragraph structure. Every contribution is built to the same template (bold lead-phrase ending in period, then two to four supporting sentences ending with a metric).
   - Rewrite: leave the six bolded leads, but compress contributions 3 and 6 to two lines each. The 230-word contribution 3 (bullet on the trinitro-isoxazole) duplicates the abstract and the §1 intro paragraph.

8. **Line 118 — "The remainder of the paper is organised as follows. §2 reviews the literature; §3 describes the dataset; §4 describes the methodology; §5 reports experiments and validation; §6 summarises results; §7 discusses limitations; §8 concludes."**
   - Pattern: the boilerplate roadmap sentence. Eight conjoined parallel clauses with identical "§N <verb>s the X" shape.
   - This is the single most obvious AI-tell in the paper. NMI does not require it; *Nature* family papers usually omit it altogether.
   - Rewrite: delete the paragraph entirely, or compress to "§2 reviews related work; the rest of the paper develops DGLD and its validation."

---

## Section 2 — Soft tells (MED risk)

**Tricolons / triple-parallel structures.** Frequent. Representative occurrences (line numbers):

- Line 77 abstract: "DFT-derived, formula-derived from Kamlet–Jacobs, surrogate-model-derived" (three-item gloss of provenance)
- Line 79: "chemist-curated SMARTS, semi-empirical electronic-structure triage, first-principles DFT"
- Line 92: "smaller charges, higher specific impulse, and more efficient pyrotechnics"
- Line 94: "crystal density, oxygen balance, heat of formation, and detonation kinetics" (four-element)
- Line 96: "small (...), heterogeneous in provenance (...), and expensive to extend"
- Line 98: four-element "fast and interpretable but regime-limited and lossy"
- Line 100: "FiLM modulation... cfg-dropout... credibility filter"
- Line 102: "polynitroisoxazole chemotype family... (sabatini2018) (tang2017)... 3,4,5-trinitro-1H-pyrazole..."
- Line 110 contrib 1: "chemist-audited SMARTS rules, semi-empirical electronic-structure triage, and first-principles DFT minimum confirmation"
- Line 117 contrib 6: "photovoltaic acceptors, redox-flow electrolytes, opioid-sparing analgesics" (this exact tricolon appears verbatim three separate times: abstract line 85, contrib 6 line 115, and conclusion neighbourhood; the *opioid-sparing analgesics* third leg is conspicuously off-domain and is a tell)
- Line 128 §2 intro: "(LIMO encoder, classifier-free guidance, FiLM conditioning, GFN2-xTB and PySCF for validation)" four-element
- Line 152: "nitrogen-rich heterocycles, nitramines, azides"

Total: 30+ tricolons across the body. This is the dominant soft tell. Many are load-bearing (genuine three-element technical lists), but the *opioid-sparing analgesics* domain-transfer tricolon and the *smaller charges, higher specific impulse, more efficient pyrotechnics* opener are decorative.

**"Not just X, but Y" / "rather than X, this Y" pivots.** "Rather than" appears 10+ times. Most are legitimate technical contrasts ("3D-CNN \(D\) should be read as relative-ranking-grade, not absolute-value-grade", line 504; "rediscovery, not discovery", line 379). Not flagged.

**Em-dash-equivalent comma-pause structures.** Present and load-bearing. Long appositional clauses are heaviest in the abstract (lines 77, 79, 81, 83, 85), the §1 intro paragraph at line 100 (167-word sentence), and the §5.5 DFT paragraph at line 431 (the one running ~430 words with multiple clause-pause structures separated by semicolons). These are the most AI-feeling stretches in the paper.

**Repeated transition adverbs.** **Not present.** A grep for `Moreover|Furthermore|Notably|Importantly|Interestingly|Remarkably|Specifically|Additionally|In particular|In essence|In other words` returned zero matches. This is the single strongest signal that the paper has had a careful editor pass over it.

**Hedge stacks.** Not present at three-or-more density. "may", "could", "approximately", "roughly" each appear, but each instance is single-hedge.

**Padding constructions.** Not present. `grep` for "It is important to note", "It should be emphasised", "It is worth mentioning", "As we have already discussed", "As mentioned earlier" returned zero matches. This is unusual for AI text and indicates editorial discipline.

**Generic introductory frames.** Not present. No paragraph begins with "In this section we", "In this paper we", or "We now turn to". The §5 opening at line 265 is the closest ("The experimental section answers four questions.") and it earns its frame because it actually enumerates the four questions and binds each to a subsection.

**Empty-summary closings.** Two flavours present:
- The figure-caption *Takeaway:* device (Section 1, item 1) is the most visible.
- Paragraph-ending "this is X, not Y" closers ("rediscovery, not discovery"; "real local minimum, not a numerical artefact"; "ranking-grade rather than absolute-value-grade"). These are technically informative and read as researcher prose, not as AI summary-padding.

No paragraph ends with "Taken together..." or "These results demonstrate that..." or "In summary, we have shown..." (`grep` returned zero matches).

**Confident scope-claim closers.** **Not present.** Zero "paves the way for", "opens new avenues", "represents a significant step toward", "we believe this opens". The §8 conclusion at lines 511–518 is impressively restrained ("DGLD demonstrates that...", "lowering the barrier to entry"); for an NMI-target conclusion this is the right register.

**Bullet-list explosions in flowing-prose sections.** Two acceptable bullet uses (§1 contributions list lines 109–115; §6 summary lines 488–497). Both earn their bullet form. No prose-paragraph-as-bulleted-list anti-pattern.

**Dictionary-bot connectors.** Not present. Zero matches for "In essence", "In other words", "That is to say", "To put it another way".

**Vocabulary-as-thesaurus.** Mostly absent. The trinitro-1,2-isoxazole lead is consistently called "L1" or "the trinitro-isoxazole" or "the headline lead"; this is acceptable variation in a 1015-line paper. No paragraph that I scanned restates the same noun three different ways within a few sentences.

**Over-uniform sentence length.** A few stretches show this:
- Lines 144–155 (§2.3 paragraph and §2.4 opening): consecutive sentences in the 28–40 word band.
- Line 431 (§5.5 mega-paragraph): the sentences are long but variable (35–95 words), which actually reads as researcher prose explaining methodology, not as AI uniformity.
The paper as a whole alternates short and long sentences far more than typical AI output.

**Over-symmetric paragraph structure.** The contributions list (line 109–115) follows the [bold lead phrase + 2–4 explanatory sentences + final-metric closer] template six times in a row. Otherwise paragraph structure is genuinely varied; many paragraphs are two sentences, several are seven or eight, some run nearly a full screen.

---

## Section 3 — Section-by-section verdict

**Abstract (lines 75–86).** Reads as careful-editor-over-AI-draft. The four-paragraph structure with bold pull-quotes ("96 of 97 PubChem-evaluable... are unknown to PubChem"), the long appositional sentences, and the "domain-agnostic; only the inference-time credibility filter changes per domain" closer are AI-flavoured. But there are no transition adverbs and no scope-claim flourish; the abstract states what was done and stops. Most AI-feeling: line 81 "polynitroisoxazole chemotype family ... isomeric 3,4,5-trinitro-1H-pyrazole" appositional pile-up.

**§1 Introduction (lines 89–123).** Mixed. The opening paragraph (line 92) reads as authentic researcher prose with one decorative tricolon. Line 100 is a 167-word single sentence that screams AI. The contributions list (109–115) is over-symmetric. The roadmap sentence at line 118 is the most obvious tell in the whole paper. Most representative tell: "§2 reviews the literature; §3 describes the dataset; §4 describes the methodology; §5 reports experiments and validation; §6 summarises results; §7 discusses limitations; §8 concludes."

**§2 Related work (lines 126–166).** Reads as authentic researcher prose. Citations are densely woven, the prior-work positioning is specific and fair, and there are no AI tells beyond the §2 opening tricolon at line 128. The "DGLD differs from this family in two respects" structure (line 147) is the kind of explicit comparison-and-contrast a methodical researcher writes. Most AI-feeling: the tight parallelism of the four-method comparison cadence in §2.1 (line 131).

**§3 Dataset (lines 169–203).** Authentic researcher prose. The four-tier hierarchy table earns its place; the prose around it is informative rather than decorative. No tells.

**§4 Methodology (lines 206–260).** Authentic, with one mild tell: §4.4 (lines 237–247) uses bolded inline subheadings ("**Architecture and training labels.**", "**Sample-time gradient.**", "**Role of the viability head and the hazard head.**", "**Configuration caveat for small-step latent diffusion.**") which is a slightly AI-y organising device but is well-suited to dense methodology prose and is not over-used.

**§5 Experiments (lines 263–483).** Authentic researcher prose throughout. The §5.5 DFT paragraph at line 431 is long and dense but everything in it is technical content. The bug-fix disclosure at line 433 ("Bug-fix disclosure. Earlier drafts of this section reported a HOF intercept of -16,763 kJ/mol that both reviewers correctly flagged as physically implausible; the root cause was a unit-conversion bug...") is the most human-feeling moment in the paper: it credits reviewers, names the file and pre-commit, and quantifies the wrong number. AI does not write paragraphs like this without prompting.

**§6 Summary of results (lines 486–498).** Editor-over-AI. The bolded-lead bullet template is uniform; the §6.2 trinitro-isoxazole bullet at line 490 duplicates abstract / §1 intro / §8 conclusion language. Most representative: the chemotype-rediscovery boilerplate appears here for the third time in essentially the same form.

**§7 Limitations (lines 501–508).** Authentic researcher prose. Specific, hedged appropriately, names what is not in the paper rather than gesturing at it.

**§8 Conclusion (lines 511–521).** Editor-over-AI. The "Several extensions are natural next steps" enumeration (line 516) is AI-default scaffolding; the (i)/(ii)/(iii)/(iv)/(v) enumeration of follow-ups is NMI-typical and probably acceptable, though it could be shortened by half. The closing paragraph at line 518 ("Energetic-materials discovery has historically advanced one molecule per decade, with HMX (1942), CL-20 (1987), and ONC (1999)...") is good human-flavoured prose with concrete dates and names; no tell.

---

## Section 4 — Verdict (per-section bucketing)

**(a) Authentic researcher prose:** §2 Related work; §3 Dataset; §4 Methodology; §5 Experiments (especially §5.4, §5.5, §5.6); §7 Limitations.

**(b) Careful editor pass over AI draft:** Abstract; §6 Summary of results; §8 Conclusion.

**(c) Raw AI:** None. There is no section that reads as if it left the model unedited.

The single weakest stretch (most AI-flavoured) is the §1 introduction, specifically the 167-word system-description sentence at line 100, the over-symmetric contributions list at lines 109–115, and the §1 roadmap at line 118.

---

## Section 5 — Top-3 changes that would most de-AI the writing

**1. Delete the §1 roadmap sentence (line 118).**
Before: "The remainder of the paper is organised as follows. §2 reviews the literature; §3 describes the dataset; §4 describes the methodology; §5 reports experiments and validation; §6 summarises results; §7 discusses limitations; §8 concludes."
After: delete entirely. The reader has the table of contents above and section headers below; the roadmap sentence is the most reliable AI-tell in the paper.

**2. Drop the per-figure "Takeaway:" italic device from at least 8 of the 13 figure captions.**
Before (Fig 6, line 274): "*Takeaway: productive-quadrant yield scales with pool size, with headroom remaining at 40k.*"
After: end the caption at "Both curves are still moving with pool size at 40 000." The takeaway is obvious from the body sentence preceding the figure.
Keep the device on Fig 1 (the headline figure), Fig 8 (Pareto), Fig 13 (DFT dumbbell) where the takeaway is genuinely non-obvious. Remove from the other ten.

**3. Split the 167-word system-description sentence (line 100) into three sentences and remove one of the four near-verbatim restatements of the trinitro-isoxazole framing (abstract line 81, intro line 102, contrib 3 line 112, §6 line 490, §8 line 514).**
Before (line 100): "We propose Domain-Gated Latent Diffusion (DGLD), which performs latent diffusion over a frozen, fine-tuned LIMO encoder that maps SMILES to a 1024-d Gaussian latent, with the unlabelled half-million-row backbone of the augmented corpus drawn from ZINC. At training time the model is gated by a four-tier label-trust hierarchy: Tier A experimental and Tier B DFT labels drive the conditional gradient through FiLM modulation, while Tier C Kamlet-Jacobs and Tier D 3D-CNN surrogate labels train only the unconditional prior via classifier-free-guidance dropout. At inference time it is gated by a three-layer chemistry-rule credibility filter (chemist-curated SMARTS, semi-empirical GFN2-xTB electronic-structure triage, and first-principles B3LYP/6-31G(d) plus wB97X-D3BJ/def2-TZVP confirmation). A noise-conditional score model with one head per target property and per safety axis adds per-head classifier-guidance gradients at sample time, exposing the per-head scales as on/off switches over chemistry classes without retraining the diffusion model."
After: keep the three-sentence split (first sentence stops at "Gaussian latent."; second covers training-time gating; third covers inference-time gating; the noise-conditional score-model sentence becomes a fourth sentence or is moved to §4.4 where it already lives). Then in §6 and §8, replace the chemotype-rediscovery boilerplate with a one-line tag ("L1 trinitro-1,2-isoxazole, the polynitroisoxazole chemotype-rediscovery lead from §5.2") and let §5.2 carry the long-form description.

These three changes would shift §1 from bucket (b) toward bucket (a) and remove the most reliable AI-fingerprint in the document.
