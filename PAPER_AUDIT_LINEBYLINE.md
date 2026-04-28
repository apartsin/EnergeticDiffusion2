# Paper line-by-line editorial audit

Audit of `docs/paper/index.html` (1031 lines, post-restructure). Read-only review against on-disk artefacts in `m2_bundle/`, `results/`, `experiments/`, `aizynth_bundle/`. Dimensions: (a) correctness, (b) clarity, (c) presentation flow, (d) duplication, (e) impact preservation, (f) reader comprehension.

---

## 1. Top 10 highest-impact fixes

### F1. L337 contradicts L324 on rank-1 properties (a, HIGH)
Line 337 says "Lead #1 is a four-membered nitrogen-rich ring with an N-oxide and a gem-dinitro, predicted at rho=1.89, D=9.69 km/s, P=40.17 GPa". Table 1 row 1 (L324) and the abstract (L75) state L1 is trinitro-1,2-isoxazole at rho=2.00, D=9.56, P=40.52. The §5.2 body paragraph appears to describe the rank-1 of an *earlier* run that has since been replaced; this is a direct numerical contradiction in the same section.
Fix: rewrite L337 to describe the actual table row 1 (trinitro-1,2-isoxazole, aromatic isoxazole with three ring-nitros, rho=2.00, D=9.56 km/s, P=40.5 GPa).

### F2. L690 vs Table D.2 row L4: incompatible K-J residuals (a, HIGH)
The §D.6 prose at L690 says "L4 (predicted D=8.98 km/s, anchor-calibrated K-J=6.63 km/s, residual = 2.35 km/s)". Table D.2 row L4 (L676) reports K-J calibrated D=13.27, residual=+4.29 km/s. The sensitivity-decomposition narrative in D.6 (and the body §5.6 line 476) is built on the 2.35 km/s number; the table is built on the 4.29 km/s number. The paper cannot keep both.
Fix: pick whichever K-J convention (closed-form-with-Q-subtraction vs open-form) is canonical and recompute the sensitivity table; explicitly document the two conventions in one place (D.7 already mentions an "open-form K-J" caveat; consolidate).

### F3. L466 says L1 trinitro-isoxazole has rho=2.00 g/cm^3, but raw DFT and the 3D-CNN disagree (a, HIGH)
Line 466 reports "the trinitro-isoxazole at rho=2.00 g/cm^3, D=9.56 km/s, P=40.5 GPa". Table D.1 row L1 (L642) gives raw DFT rho=1.80, calibrated rho=2.53. Table 1 (L324) gives 3D-CNN rho=2.00. The 2.00 number is the surrogate; the body should say "predicted (3D-CNN) rho=2.00; raw DFT rho=1.80 (calibrated 2.53)" and pick one consistently when the same lead is described as the "headline" candidate. Same issue in abstract (L75) and intro (L96, L108).
Fix: state once, in §5.6, that the headline triple {rho=2.00, D=9.56, P=40.5} is the surrogate prediction and that DFT-recomputed values for L1 are listed in Tables D.1/D.2 (with K-J undefined on the calibrated branch). Remove "rho=2.00 g/cm^3" from §5.4 line 466 or qualify it as surrogate.

### F4. xTB graph-survives flag is false for 7 of 8 leads but paper says only ranks 4 and 6 fail (a, HIGH)
`experiments/xtb_topN.json` records `graph_survives=false` for ranks 1, 3, 4, 5, 6, 7, 8 (only rank 2 survives connectivity). Rank 1 (trinitro-isoxazole) fails because the xTB-optimised structure is read back at charge -1 (`"Final molecular charge (0) does not match input (-1)"`). Table 5 (L450-458) does not surface this; only the gap gate is reported, and the verdict "most stable; aromatic, large gap" is given for rank 1 despite the connectivity test failing. Fig 15 caption (L462) references the survival flag explicitly. Either the JSON's graph-survives convention is too strict (charge round-trip artefact) or the body claim is wrong.
Fix: explicitly state in §5.4 that graph-survives in `xtb_topN.json` failed for L1 due to a charge-perception round-trip artefact in the parser, not a real bond-breaking event, and add a column to Table 5 (or a footnote) documenting this; otherwise readers comparing the figure caption against the table will lose trust.

### F5. L526 says §5.6 reports a 1.5-2 km/s K-J vs 3D-CNN residual; Table D.2 residuals are +0.73 to +4.31 km/s (a, MED)
Limitations §7 line 526 anchors on "1.5-2 km/s" but Table D.2 shows leads with residuals of +4.29, +4.31, +3.48, +2.83 km/s and three "K-J undefined" rows. The 1.5-2 km/s number does not appear in the per-lead table.
Fix: replace "1.5-2 km/s" with the actual median/IQR observed in Table D.2, or label it as the population-level residual on Tier-A rows (Table D.4) and re-derive.

### F6. d9_kj_nfrac_table.json is referenced but missing from disk (a, MED)
Fig 14 caption (L479) and §5.6 line 476 cite `d9_kj_nfrac_table.json` for the Pearson r=+0.43, p<10^-26 result. The file is absent at `experiments/d9_kj_nfrac_table.json`; the numbers reproduce in Table D.4 in the appendix. Reviewers checking the artefact will fail.
Fix: rename the in-text citation to point at Table D.4 (the canonical home of those bins) or commit the JSON.

### F7. The "DGLD vs no-diffusion baselines" lift is buried in §5.3.3 prose (e, HIGH)
The headline contribution narrative (only DGLD lands in the novel-and-on-target quadrant) is made strongly in the abstract and intro, but §5.3.3 (L394-425) buries the punchline behind a "How to read this table" paragraph (L418), six caveats, and a deferred reference to Fig 1. A reviewer skimming §5.3.3 reads it as "guidance configurations differ on composite", not "no-diffusion baselines fail".
Fix: open §5.3.3 with one sentence: "Both no-diffusion baselines fail the productive-quadrant test: SMILES-LSTM top-1 is an exact training-row reproduction (Tanimoto=1.000), MolMIM top-1 is novel but at sub-HMX D=7.70 km/s. DGLD is the only family that simultaneously satisfies novelty>0.45 and HMX-class D, rho, P (Fig 1)." Then introduce the table as evidence.

### F8. Productive-quadrant Fig 2 caption has SVG-rendering breakage (b, MED)
Lines 118-120: literal `\(
ho \ge 1.85\) g/cm\(^3\)` — the LaTeX `\rho` macro got line-broken across the source. The reader gets `ho >= 1.85 g/cm 3`. Same at L120 ("100 clear ho >= 1.85").
Fix: rejoin the `\rho` token on a single line; spot-check for the same line break elsewhere.

### F9. "Fig 11" is referenced before it is introduced; appears mid-§5.2 but is a §5.6 deliverable (c, MED)
Figure 11 (lead-cards grid, L331) is rendered immediately after Table 1 in §5.2, but its caption says "anchor-calibrated DFT/Kamlet-Jacobs triple", which is a §5.6 concept. Readers hit DFT calibration vocabulary in §5.2 with no §5.6 context. §5.6 then refers back to "Fig 11" (L462) by figure number, which works HTML-side but is structurally inverted.
Fix: move Fig 11 (lead cards grid) to §5.6 right after Table D.1, or relabel it as Fig D.x. Renumber Fig 12, 13, 14, 15 accordingly. Update §5.6 line 462 cross-reference.

### F10. Cross-section duplication of the headline-lead triple {rho=2.00, D=9.56, P=40.5} (d, MED)
The exact triple "rho=2.00 g/cm^3, D=9.56 km/s, P=40.5 GPa" appears verbatim in: abstract L75, intro L96, intro L108 (contributions list), §5.4 L466, summary §6 L512, conclusion §8 L535. Six restatements with no cross-reference.
Fix: state it once in §5.2 (Table 1) and §5.6 (DFT validation), and replace the other four occurrences with "(rank-1 triple, §5.2)".

---

## 2. Per-section detailed audit

### Abstract (L69-L80)
- L71-L73 [b] [LOW] First sentence is 32 words and stacks four parallel clauses; second paragraph leads with "Naive generative models trained on this data either memorise the training distribution or extrapolate without calibration" which is true but unsupported until §2; a reviewer pauses here.
- L75 [a/d] [MED] "trinitro-1,2-isoxazole, is a previously unreported molecule: it appears in no public chemistry database and lies far from any molecule in our 694 k-molecule training corpus" — "previously unreported" is stronger than the §5.3.1 claim; PUG REST returned transient errors, and the labelled-master sanity-check rediscovery (1-nitro-1H-tetrazol-1-amine at rank 56) is the only known overlap. State precisely: "absent from PubChem (PUG REST query) and at maximum Tanimoto 0.32 (median) to the 694 k augmented corpus".
- L77 [e] [LOW] "DGLD outperforms the two strongest available baselines, which fail in complementary ways" — strong but the lead-in immediately dilutes by listing failure modes; the sentence currently buries the lift inside a "fail in complementary ways" qualifier. Reorder to lead with "DGLD is the only method that produces novel HMX-class candidates on every property axis" then enumerate.

### §1 Introduction (L83-L114)
- L86 [b] [LOW] Sentence at L86 is 47 words and contains three parallel clauses ("the canonical performance anchors..., the cadence..., the space..."); split into two.
- L88 [c] [LOW] Discovery-task framing ("we pose the discovery task as conditional molecular generation") fires here, but §2 still revisits the framing; consider trimming §2's repeat.
- L92 [b] [MED] L92 mixes four families of methods, six citations, and the headline guidance bug into one paragraph (78 words); the bug deserves a dedicated sentence not a tail clause. Split.
- L94 [b] [MED] L94 is 100+ words, three parenthesised expansions; this is the single hardest sentence in the introduction. Split into (i) what DGLD is structurally, (ii) the training-time gating, (iii) the inference-time gating.
- L96 [d] [MED] Repeats the rho=2.00, D=9.56, P=40.5 triple already given in the abstract; better to cross-reference rather than restate.
- L102 [a] [LOW] "two 4-condition matrices yielding 7 distinct guidance settings × 3 seeds" is a §5.3.3 detail; in Fig 1 caption it lands without §5.3.3 context. Move detail to body.
- L106-L111 (numbered contributions list) [d] [MED] Contribution 3 restates the rho/D/P triple a third time in the same section.

### §2 Related work (L124-L164)
- L126 [c] [LOW] First sentence is "DGLD sits at the intersection of three lines of work" but the section then enumerates eight subsections (2.1-2.8). State "eight" or list the actual line-up.
- L161 [c] [MED] §2.7 says "We adopt MOSES-style metrics in Appendix E.11 and treat FCD as a chemistry-class-transfer signal" — this preview fires before §3 even starts. Move the sentence to §5.3 / E.11 or shorten.
- §2 is otherwise tight; no correctness issues found.

### §3 Dataset (L167-L201)
- L169 [b] [HIGH] L169 is one paragraph of 11 sentences and 280 words covering corpus assembly, tier hierarchy preview, deduplication, oversampling preview, and figure forward-references. Reviewers re-read this to find the 694k vs 66k vs 326k vs 380k numbers. Split into three paragraphs by topic.
- L173, L178, L195 [c] [MED] Fig 3 is split across three `<figure>` blocks (3a, 3b, 3c) with three captions, but only Fig 3 (a) gets the bold figure number. Fig 3 (c) caption (L195) refers to "Figure 3(c)" but Fig 3(b) caption (L178) does not say "Figure 3(b)" in a parallel way. Make the labelling consistent.
- L191 [a] [LOW] Tier C count "~25 000" rows. Tier D "~30 000". Tier B "~9 000". Tier A "~3 000". Sum is 67k; labelled master is stated as 65,980 (L364). The "~" is consistent but the sum should equal the labelled master after dedup. State the dedup overlap once.
- L201 [a] [MED] Body says "Tier-A/B rows above the 90th percentile are oversampled by 5x-10x"; Appendix C.1 (L609) says oversample factor 10x at quantile 0.05 (top 5%). Numbers conflict (90th percentile vs top 5%; 5x-10x vs 10x).

### §4 Methodology (L204-L301)
- L206 [b] [LOW] "Per-component hyperparameter values are tabulated in Appendix C; per-experiment compute budgets and version codes are in Appendix E (per-experiment costs in Appendix E.2)" — the same pointer fires twice. Pick one.
- L257 [a] [LOW] "We fine-tune all parameters for ~8.5k steps on 326 k energetic-biased SMILES" — 326k matches §5.3.3 SMILES-LSTM corpus. Fine.
- L259 [b] [LOW] "we discuss in §6 how this mismatch affects diffusion-time decoding" — §6 is the summary section and does not contain the discussion. The sentence is a stale forward pointer.
- L284 [c] [MED] "(the cluster pattern of Appendix E.6)" pointer fires before the cluster pattern is motivated; Appendix E.6 is reproducibility-detail section, not a "look here for the headline".
- L288 [b] [HIGH] "documented in Appendix C.1-B.3" — appendix labels are out of order (C.1 then B.3); should read "Appendix C.1-C.3" or "Appendix B.1-B.3" depending on which label group is canonical. The actual subsections in C are labelled B.1, B.2, B.3 (L617, L619, L621), which do not match the parent C heading. **The Appendix C subsection labels (B.1-B.3) are wrong.**

### §5 Experiments (L304-L505)
- L306 [b] [HIGH] L306 is a 10-sentence wall introducing four research questions, seven section pointers, and three appendix pointers (200+ words). Split.
- L319 [a] [MED] §5.2 says "of the top-400 candidates from the single-objective rerank, 45 are rejected by the hard filters... the 355 survivors define a Pareto front of 34 candidates"; sentence count addition (11+4+28+2 = 45) checks. Pareto-front 34 mentioned again at L337. Fine.
- L324-L329 [a] [HIGH] **F1 above: L337 vs Table 1 contradiction.**
- L334 [c] [HIGH] **F9 above: Fig 11 placed in §5.2 but described as a §5.6 deliverable.**
- L337 [a] [HIGH] "rho=1.89, D=9.69, P=40.17" — does not match Table 1 row 1.
- L341 [c] [LOW] Anchors-drop-out paragraph (dinitramide etc.) appears mid-§5.2 with no transition. Move to §5.3.1 alongside the labelled-master rediscovery list.
- L356-L374 (§5.3.1 Novelty) [a/d] [MED] L359 cites "97/100 are absent from the labelled master"; Table 2 (L368-369) says fraction exact match 1% labelled-master and 0% augmented. The 1% = 1/100 = the 1-nitro-1H-tetrazol-1-amine sanity-check; the body separately says "3/100 labelled-master rediscoveries (dinitramide, 1,2-dinitrohydrazine, N,N'-dinitrocarbodiimide)". 1% exact match vs 3 rediscoveries is inconsistent unless the latter three are not exact-Tanimoto matches; clarify the difference between "exact match" (Tani=1) and "rediscovery" (closely matching).
- L367 [a] [LOW] Table 2 fraction columns: labelled-master "fraction>0.55 = 3%, fraction>0.70 = 1%, exact = 1%". For augmented: "1%, 0%, 0%". Reproduces `experiments/novelty_stratification_summary.json`. Good.
- L379 [b] [MED] "outputs are bit-identical across clamp values within each anneal setting (md5 e26b43... for anneal=0; cc917f... for anneal=2)" — the md5 prefix `cc917f` for anneal=2 is given but does not match Table E.1 cluster D `e26b43`; readers tracking the bug will be confused. Pre-patch md5 was different from post-patch md5. State explicitly.
- L395 [b] [HIGH] §5.3.3 lead paragraph has a 78-word "Note on labels" parenthetical (L395) explaining that C0/C1/C2/C3 codes mean different things in §5.3.2 vs §5.3.3. This is exactly where a reviewer stops and asks "what does this mean?". Fix: rename §5.3.3 conditions to remove the C-letter collision (e.g. always say Hz-Cn / SA-Cn even at first mention, drop the "C0/C1/C2/C3" abbreviation entirely in §5.3.3).
- L401 [a] [MED] Table 4 SMILES-LSTM row top-1 D=9.58, rho=1.96, P=40.0 — `results/novelty_top1.json` says 9.576, 1.956, 39.985. OK rounding. SMILES-LSTM is "(no diffusion)". Good.
- L407 [a] [LOW] "**0.27 ± 0.03** (most novel; 0/3 memorised)" for Hz-C2 — `results/novelty_top1.json` confirms 0/3 memorised in C2_viab_sens_hazard. Good.
- L412-L415 (Fig 12) [c] [MED] Fig 12 forest plot is inserted between the table and the "How to read" prose; the figure precedes a paragraph that explains why the table cannot be read at face value. Either move the explanation in front of the figure or move the figure after L420.
- L418 [e] [HIGH] **F7 above: the punchline is buried.**
- L429-L433 (§5.3.4) [a] [MED] L429 says "The top-100 by composite v2 has range [0.564, 0.831]" — Fig 10 caption (L796) confirms the same range. Good.
- L431 [a] [LOW] "24 unique Bemis-Murcko scaffolds over the merged top-100" — denominator stated. Also gives "5-14 scaffolds per condition top-100" and "369-503 in full pool=10k". The three numbers explicitly disclaimed as non-comparable; clear.
- L433 [a] [MED] "diffusion sampler... reaches top-1 D in the 8.75-9.46 km/s band" — but §5.3.3 reports DGLD top-1 D as 9.32-9.54 km/s and best 9.72; the 8.75 lower bound seems to come from Appendix E.6 conditions B and C (heuristic-vs-literature ablation, L760-762: condition C=8.75). Cross-section consistency wobbles; clarify whether the 8.75 is the production sampler or the literature-grounded sensitivity ablation.
- L444-L466 (§5.4) [a] [HIGH] **F4 above (graph-survives mismatch).** Also L466 third occurrence of {rho=2.00, D=9.56, P=40.5}.
- L466 [a] [LOW] "85/100 stable" cross-references §5.3.4; §5.3.4 does not contain the 85/100 number explicitly (it's at L359). Cross-pointer is to the wrong subsection; should be §5.3.1.
- L470 (§5.5) [a] [LOW] "the larger unguided pool achieved a substantially higher physical-stability rate. The interpretation is that classifier guidance can drive the sampler into a mode where individual top-of-funnel candidates score high in our learned proxies but fail at the level of frontier-orbital electronic stability" — this is a plausible interpretation but the §5.4 set is "the gated top-8" which the body elsewhere says is from a guided run; specify which run.
- L472-L486 (§5.6) [a] [HIGH] **F2, F3, F5 above.** Also L474 mixes raw DFT, calibrated DFT, surrogate, and 3D-CNN in one 200-word run-on. Split.
- L486 [c] [LOW] "The per-lead 3D-CNN-vs-DFT-K-J-cal dumbbell, plus the N-fraction residual scatter, is in Fig 14" — fires after the Fig 14 figure has already been rendered (L478). Forward pointer is post-hoc. Move the sentence above the figure or remove.
- L489-L505 (§5.7) [a] [LOW] AiZynth state score for L1 = 0.50 (Table 9, L495) — confirmed in `aizynth_bundle/results/aizynth_results.json` ("state score": 0.5). Good.

### §6 Summary of results (L508-L519)
- L512 [a] [LOW] "Twelve chem-pass leads were validated by DFT" matches Table D.1 (twelve chem-pass + 2 anchors). Good.
- L512 [d] [MED] Restates {rho=2.00, D=9.56, P=40.5}; cross-reference instead.
- L515 [a] [LOW] "Semi-empirical xTB triage (§5.4) retains 96/100 merged-top-100 candidates that converge to ground-state geometries; 85/100 also pass the HOMO-LUMO >= 1.5 eV gate" — the 96/100 figure is not derived in §5.4 (which only reports the gated top-8 = 6/8 surviving); §5.3.1 says 85/100 pass the gap gate but does not give the convergence-only 96/100 figure. Add a sentence in §5.3.1 / §5.4 introducing the 96/100.

### §7 Limitations (L522-L529)
- L525 [a] [HIGH] **F5 above (1.5-2 km/s contradiction).**
- L527 [c] [LOW] "tables elsewhere use a single sampling seed" — vague antecedent; specify "Table 1, Table 3, Table D.1".

### §8 Conclusion (L532-L542)
- L535 [d] [MED] Restates {rho=2.00, D=9.56, P=40.5} for the sixth time and the 9.72 km/s for the second time. Cross-reference §5.2 / §5.3.3.

### Appendix A (L547-L578)
- L562 [b] [LOW] "Augmented CHNO/CHNOClF SMILES dump assembled from public energetic-chemistry compilations" — what compilations? Klapotke is one citation but the body says "Klapotke reviews, patent-extracted SMILES, energetic-materials review papers"; flag as low specificity.
- L570 [a] [LOW] Labelled master "~65 980 unique molecules" — matches `experiments/novelty_stratification_summary.json` `labelled_master_corpus_size: 65980`. Good.

### Appendix B (L581-L587)
- L586 [c] [LOW] Fig B.1 caption is dense; reader does not need 12 bullet labels in caption text. Move to a small table.

### Appendix C (L589-L621)
- L617-L621 [a] [HIGH] **F.10 / Appendix labels are wrong: subsections under "C. Model architecture" are labelled B.1, B.2, B.3.** Either rename the section to B.1-B.3 (matching the body's "Appendix C.1-B.3" reference at L288) or rename the subsections to C.1-C.3.

### Appendix D (L623-L727)
- L630 [b] [MED] "5-15 kJ/mol per atom, which scales to ~100-250 kJ/mol at the molecular level for our 15-25-atom leads, consistent with the +170 to +224 kJ/mol offsets we observe at the RDX/TATB anchors" — the +170/+224 numbers (RDX: -884.2, +70 exp; TATB: +796.2, -141 exp give offsets of -954 and +937 kJ/mol on the calibrated values, not +170/+224). Recheck arithmetic.
- L632 [a] [LOW] "The 2-anchor calibration offsets rho_factor = 1.136 and HOF_offset = -197 kJ/mol fall inside the literature bias bands" — these numbers are not the slopes/intercepts in `m2_calibration.json` (a_rho=4.275, b_rho=-5.172, c_hof=-16763). Two different calibration parametrisations are being mixed; clarify.
- L686-L687 [a] [HIGH] Table D.2 RDX row K-J calibrated D=8.25 (vs experimental 8.75, residual not shown), TATB row K-J calibrated D=5.64 (vs experimental 7.90, residual ~ -2.3 km/s). The anchor compounds themselves show large K-J calibrated residuals; this contradicts the framing that calibration recovers D for anchors. Add a footnote.
- L711 [a] [LOW] Table D.4 last row "[0.55-1.00], n=9, mean residual +0.54". Pearson r=+0.43 reproduces in `experiments/novelty_stratification_summary.json` neighbours; OK.
- L714 [a] [MED] "Pearson r(f_N, residual) = +0.43 (p = 4 × 10^-27, n=575)" — Fig 14 caption (L479) cites the same r from `d9_kj_nfrac_table.json`; that file is **missing on disk**.

### Appendix E (L729-L854)
- L740 [b] [LOW] §E.5 four-test diagnostic protocol is named but not described; reviewers will want one-sentence summaries inline.
- L748-L750 [a] [LOW] Table E.1 cluster md5 prefixes match the body §5.3.2 (L379) cluster D `e26b43`. Good.
- L760-L762 [a] [LOW] Table E.2 condition C "literature-grounded sensitivity head" top-1 D=8.75 — this is the same 8.75 cited at §5.3.4 (L433). Cross-link.
- L805-L809 [a] [MED] Table 6 (MOSES metrics): C0 unguided uniqueness 0.728. But `results/pool_metrics.json` row `m1_3seed_C0_unguided_seed0` reports uniqueness 0.67375 (different sample/seed/post-processing); Table 6 is at pool=10k while pool_metrics is at pool=20k or different pipeline. Document which sampling/seed produced Table 6.
- L827-L835 [a] [LOW] Table 7 FCD numbers reproduce in `results/fcd_results.json` (LSTM 0.517, C0 unguided 24.99±0.05 etc.). Good.
- L848 [a] [LOW] "energetic-SMARTS chemistry filter reject 23/100" matches §5.3.1. Good.

### Appendix F, G (L856-L964)
- No correctness issues. F is a lookup table; G is well-written. Could shorten G by one third without loss.

---

## 3. Cross-section duplication map

| Claim / number | First appearance | Second | Third+ | Recommendation |
|---|---|---|---|---|
| Headline triple {rho=2.00, D=9.56, P=40.5} | Abstract L75 | Intro L96 | L108, L466, L512, L535 | Keep at L75 + §5.2 Table 1; cross-reference elsewhere |
| 9.72 km/s best top-1 | §6 L512 | §8 L535 | - | Keep at §5.3.3 (currently absent there!), cross-ref in §6 + §8 |
| SMILES-LSTM Tani=1.000 / "exact memorisation" | Abstract L77 | Intro L96 | L109, L401, L420, L513, §6 L512, §8 L533 | Keep at §5.3.3 lead, cross-ref everywhere else |
| MolMIM D=7.70 km/s | Abstract L77 | Intro L96 | L109, L402, L420, L513 | Keep at §5.3.3; cross-ref |
| 96/97 PubChem-novel | Abstract L75 | Intro L96 | §5.3.1 L359 | Keep §5.3.1; cross-ref |
| 77/100 chem-pass | Abstract L75 | §5.3.1 L374 | §5.3 L515, App E.7 L779 | Keep §5.3.1 + E.7 |
| 85/100 xTB-pass | Abstract L75 | §5.3.1 L359 | §5.4 L466, §6 L515 | Keep §5.3.1; cross-ref §5.4 |
| Pearson r=+0.43 N-fraction | §5.6 L476 | Fig 14 caption L479 | §6 L517, §7 L525, App D.7 L714 | Keep App D.7 + §5.6; trim §6 |
| Domain-agnostic / "transfers to other domains" | Abstract L79 | Intro contribution 6 L111 | §8 L539 | Pick one |
| Tier hierarchy description (4 tiers, A/B drive cond grad) | §1 L94 | §3.1 L182 | §3.1 L198, §4 L205, §3.1 figure caption L195 | §3.1 is canonical; intro should be one-line |
| The 326 k corpus | §1 L96 | §4.1 L257 | §5.3.3 L394 | OK as-is |

---

## 4. Reader-friendly summary: top 5 changes for a non-specialist reviewer

1. **Decisively split the §5.3.3 baseline lift from the "guidance configurations" narrative.** The single most damaging clarity issue is that the punchline (DGLD is the only family in the productive quadrant) is buried inside a "How to read this table" caveats paragraph. A non-specialist should see the lift in one sentence at the top of §5.3.3 and only then encounter the per-condition variance.
2. **Pick one canonical statement of the rank-1 triple {rho=2.00, D=9.56, P=40.5} and cross-reference everywhere else.** Six restatements with internally inconsistent numbers (L337 says rho=1.89, D=9.69, P=40.17, contradicting Table 1) destroys reader trust. Make §5.2 / Table 1 the single source.
3. **Reconcile the K-J calibrated numbers between the §D.6 prose and Table D.2.** A reviewer who tracks the L4 sensitivity argument will hit a hard contradiction: 2.35 km/s residual in prose, 4.29 km/s in the table. Decide which K-J convention the paper uses, recompute, restate.
4. **Surface the xTB graph-survives column in Table 5.** The figure caption (Fig 15) tells the reader to check graph-survives, but the table does not. A reviewer comparing JSON to body will flag this immediately. Either add the column or footnote that the JSON's connectivity check is fooled by a charge-perception round-trip and is not a real failure.
5. **Fix the Appendix C subsection labels.** The §C.1-C.3 hyperparameter, hard-negative, sensitivity-head, and rebalancing subsections are labelled B.1-B.3 in the source; the body cross-references them as "Appendix C.1-B.3". Renumber consistently. Same renumbering exercise should fix the Fig 11 placement (currently in §5.2, but its caption belongs to §5.6).

---

## On-disk vs body-claim mismatches found

- L1 xTB graph-survives flag in `experiments/xtb_topN.json` is `false` for ranks 1, 3, 4, 5, 6, 7, 8 (only rank 2 survives connectivity). Paper Table 5 implies only ranks 4 and 6 fail the joint gate. **Real mismatch** (likely benign in rank 1's case due to a charge-perception artefact, but the paper does not say so).
- §D.6 text claim "L4 K-J calibrated D = 6.63 km/s, residual 2.35 km/s" vs Table D.2 row L4 (also from `m2_summary.json`'s `kj_dft_cal` field) which gives D=13.27, residual=+4.29 km/s. **Real mismatch** within the paper; both cannot be sourced from `m2_summary.json` simultaneously.
- `experiments/d9_kj_nfrac_table.json` cited in Fig 14 caption and §5.6 prose is **missing on disk**. The numerical results reproduce in Table D.4 but the JSON artefact does not exist.
- Body L201 says "5x-10x oversampling above the 90th percentile"; Appendix C.1 (L609) says "10x at quantile 0.05" (top 5%). **Internal inconsistency**; on-disk hyperparameter set is presumably 10x / top-5%.
- Appendix subsection labels under section "C. Model architecture" are B.1, B.2, B.3 — the body cross-reference at L288 reads "Appendix C.1-B.3", which is non-sensical. **Source-text labelling bug.**

All other body claims spot-checked against `m2_summary.json`, `m2_calibration.json` (slopes 4.275 / -5.172, intercept -16763), `results/m6_post.json` (top1 D = 9.720 for Hz-C1 viab seed 1), `results/molmim_post.json` (top1 D = 7.70 km/s, rho 1.756, P 25.5), `results/fcd_results.json` (LSTM 0.517, DGLD conditions 24-26), `results/novelty_top1.json` (LSTM is_memorized=true Tani=1.000), `experiments/novelty_stratification_summary.json` (median NN-Tani 0.32 augmented / 0.36 LM, frac>0.70 = 0%/1%), and `aizynth_bundle/results/aizynth_results.json` (L1 routes=9, score=0.5, top route 4 reactions) reproduce correctly.
