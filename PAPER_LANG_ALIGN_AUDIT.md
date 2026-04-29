# DGLD paper: language / alignment / inconsistency audit

Scope: read-only audit of `docs/paper/index.html` (1040 lines).
Date: 2026-04-28. State: post-revision (prereview-v1.1 + ZPE bug-fix follow-ups).

---

## Section 1: Language audit

### HIGH

- **L81, L122-123 (abstract, Fig 2 caption): broken `\rho` math.** The source contains `\(
ho` instead of `\(\rho\)`; the literal carriage-return interpretation of `\r` has stripped the `\r`. Lines 81 (abstract, "Predicted properties match... \(\rho = 2.00\)") and 122-123 (Fig 2 caption, two occurrences: "\(\rho \ge 1.85\)" and "\(\rho \ge 1.85\) AND novelty") will render as `(ho = 2.00)` and `(ho >= 1.85)` in KaTeX or simply broken. This is the single most embarrassing rendering bug remaining in the document.
- **L325 (Table 1 caption): internal-tag leak `score_model_v3e_h50`.** A code-internal artefact name (`v3e`) appears in body prose where the user has explicitly flagged that internal tags should not leak. It also appears in L284 (`score_model_v3e_h50` referenced as a name) and at L738 (Appendix E.1, where it is appropriate). Body-prose mention should be replaced with "the literature-grounded sensitivity head (§4.4 / Appendix C.2)".
- **L310 (§5 intro): "post-hoc" hyphenation drift.** "hazard-head post-hoc filtering" appears here and at L488, L846. Earlier user feedback flagged "post-hoc" usage; the term is fine technically but is used inconsistently with body register elsewhere (which prefers "applied as a re-rank weight" or "as a soft re-rank").

### MED

- **British/American mix.** The paper consistently uses British "labelled", "synthesise", "characterise", "canonicalise". Drift detected:
  - L298 "minimised" / L260 "modulation" (fine, both spellings agree).
  - L295 "synthesisability" appears alongside L527 "synthesisability" — consistent. No US "labeled" or "synthesizable" found. Good.
- **Sentence > 50 words.** Several offending sentences:
  - L81 last sentence ("DGLD outperforms ... 4 physical properties (Fig 1).") and L83 first sentence (~70 words): split into two would help readability.
  - L100 (Methodology paragraph 4 of Intro): the `<strong>Domain-Gated Latent Diffusion (DGLD)</strong>` sentence runs 95+ words; consider splitting at the FiLM modulation comma.
  - L102 the trinitro-isoxazole megasentence is ~140 words. Break at "is absent from PubChem and from our 65 980-row labelled-master corpus".
  - L168 (§2.8 GFN2-xTB) ~75 words, single sentence.
  - L284 viability-head sentence (~70 words).
  - L292 (configuration caveat) ~80 words.
  - L378 strengthened-SMARTS paragraph: "Crossing the strengthened chemistry filter ..." ~55 words.
  - L466 graph-survival false-positives paragraph (~85 words).
  - L504 AiZynth result paragraph for L1 (~110 words).
- **Inconsistent terminology for the labelled-master corpus.** The paper uses three forms:
  - "labelled master" (no hyphen), L341 ("our internal 66 k labelled corpus"), L515.
  - "labelled-master" (hyphenated) when used as adjective: L102, L363, L376, L515.
  - "labelled master corpus" without hyphen even when adjectival: L368, L376.
  Most consistent rule would be: hyphenate when used as a compound modifier ("labelled-master corpus"), no hyphen when standalone noun ("the labelled master"). Currently inconsistent.
- **`energetic-SMARTS-pass` vs `energetic-chemistry SMARTS pass` vs `chem_redflags`.** Three tags for the same screen:
  - L809 column header "energetic-SMARTS-pass" (Table 6).
  - L77 abstract: "energetic-chemistry SMARTS safety screen".
  - L378, L738 prose: "strengthened SMARTS catalog" / "chem_redflags".
  Pick one body-prose form; reserve `chem_redflags` for code references.
- **L1 / trinitro-isoxazole / trinitro-1,2-isoxazole.** Three forms:
  - "trinitro-isoxazole" (L83, L341, L364, L378, L454, L515).
  - "trinitro-1,2-isoxazole" (abstract L81, L102, L112, L325, L492).
  - "L1" (Table 1, Table D.1, Fig 7, Fig 12, Fig 13).
  - "3,4,5-trinitro-1,2-isoxazole" (L81, L102 abstract — full IUPAC).
  Recommend: introduce as "3,4,5-trinitro-1,2-isoxazole (hereafter L1, or trinitro-isoxazole when chemotype is salient)" once in §1, then use "L1" or "trinitro-1,2-isoxazole" consistently.
- **L292 "v3e" reference in body prose.** "score_model_v3e_h50" leaks an internal tag. Same as HIGH item above.
- **L284 unbalanced parenthetical "RF" abbreviation.** "P_{\text{RF}} is a RandomForest classifier ... validation AUC 0.999" — "RF" abbreviation expanded inconsistently (L301 says "Random-Forest", L416 "Random-Forest", L284 "RandomForest" all-one-word).

### LOW

- **L466 long figure-caption note** runs 5 lines and contains the phrase "round-trip"; readable but heavy.
- **L260-263 §4.1 first paragraph:** abrupt shift from past tense ("We start") to present ("The encoder maps").
- **L466, L488** use the in-house phrase "graph-survival" without prior definition (defined inline at L466, but Fig 12 caption uses it before reading the definition box).
- **Unicode minus literals.** A spot-check shows `&minus;` and `-` are used; I did not find raw `−` (U+2212) literals in body prose. Good.
- **Em-dashes / double-dashes.** Spot-check: `&mdash;` not used; `&ndash;` is used for ranges (correct). No bare `--` found in prose. One long-dash-style phrasing at L466 ("xTB-optimised structure is read back at a different formal charge from the input") uses commas; OK.
- **Math vs prose `D`.** Almost every body mention uses `\(D\)`. Exceptions: L320 ("at 40k", informal), L519 ("DGLD novelty at 1.000"), L702 ("\(\Delta\rho = -1.97 / 2.47 = -0.797\) g/cm^3"). These are acceptable; no policy-level drift.
- **L515 "Twelve chem-pass leads were validated by DFT".** Tense shift (passive past) breaks a run of present-tense bullets in §6.

---

## Section 2: Alignment audit

### Abstract <-> §1 contributions <-> §6 summary <-> §8 conclusion

The four locations agree on the core L1 narrative and now consistently use the **chemotype-rediscovery** wording. Specifically:
- Abstract L81: "chemotype rediscovery in the polynitroisoxazole family" + Hervé attributed to **isomeric pyrazole** — correct.
- §1 L102, L112: same wording, same Hervé/Sabatini/Tang citation block.
- §6 L515: same wording.
- §8 L539: same wording.
- All four cite `[sabatini2018][tang2017]` for the isoxazole family and `[herve2010]` for the pyrazole isomer. Consistent.

**One minor inconsistency:** §1 L102 says "is not located in the open energetic-materials literature, with the caveat that absence from the open literature does not exclude restricted databases or unpublished synthesis attempts"; §6 L515 says "absent ... from the open energetic-materials literature, modulo the standard restricted-database caveat"; §8 L539 says "absent ... from the open energetic-materials literature (with the standard restricted-database caveat)". Three different phrasings of the same caveat. Recommend a standard form.

The L1 numerical triple (ρ=2.00, D=9.56, P=40.5) is identical in all four locations. The DFT-K-J recompute D=9.52 km/s is reported in §5.5 L478, §6 L521, §8 (implicit via §5.5 reference) — consistent.

The Cantera-CJ caveat is mentioned in §5.5 L479 and §6 L521 ("a thermochemical-equilibrium Cantera CJ recompute additionally confirms the relative ordering"), but is **not in the abstract or §8**. This is appropriate (it is a sanity check, not a headline result), and the §5.5 prose explicitly downgrades it to "relative ranking only". Defensible.

### §5 intro promises vs §5.x first-sentence delivery

§5 intro (L308-310) promises four research questions and maps them to subsections:
- (i) Does the pipeline produce candidates? → §5.1, §5.2. **Delivered.** §5.1 L314 first sentence is about pool-size scaling; §5.2 L322 is about gated reranker. ✓
- (ii) Does it generalise / compare to baselines? → §5.3. **Delivered.** §5.3 L359 first sentence covers all three subquestions cleanly.
- (iii) Are candidates physically plausible? → §5.4 (xTB), §5.5 (DFT), §5.6 (retrosynthesis). **Delivered.** Each subsection's first sentence is on-topic.
- (iv) Do architectural choices matter? → "reported in Appendix E (E.8-E.12)". **Misalignment:** the §5 intro promises this is in Appendix E, but §5.3.4 (L431) and §5.3.3 (L398) both report architectural-choice numbers (per-head guidance grid, hard-negative ablation, multi-seed). The pointer should read "summarised in §5.3 and §5.5; full numbers in Appendix E.6, E.8, E.9, E.11, E.12".

### Figure captions vs body prose

- **Fig 1 (L106):** body L83 says "DGLD is the only method that produces candidates simultaneously novel and on-target" — caption matches. ✓
- **Fig 2 (L122-123):** body L322 cites Fig 2 implicitly via "the top-1 candidates from this sweep are visualised in (D, P) plane in Fig 2" L360. Caption mentions "Top-200 generated leads from the pool=40k joint rerank" — body says "top-1 candidates of each method" (from §5.3.3 L360, "visualised in the predicted (D, P) plane in Fig 2"). **Misalignment:** Fig 2 actually shows top-200 leads from a single 40k pool, not "top-1 of each method". The §5.3.3 cross-reference at L360 is incorrect.
- **Fig 3 (L177):** "(a) Joint distribution of density and detonation velocity". Body L173 says "Joint property distributions and per-property histograms ... are shown in Fig 3(a) and Fig 3(b)". ✓
- **Fig 4 (L257, inline SVG):** caption matches the §4 intro. ✓
- **Fig 5 (L276):** body L272 cites Fig 5 ("as Figure 5 shows"). ✓
- **Fig 7 (L337) lead cards:** caption says "Twelve chem-pass DGLD leads (L1, L2, L3, L4, L5, L9, L11, L13, L16, L18, L19, L20)". Body L335-345 around Table 1 references "twelve chem-pass leads" at L476. ✓
- **Fig 11 (L441) merged provenance:** caption breakdown 89/5/3/3 matches body L433. ✓
- **Fig 12 (L466) xTB strip:** caption is consistent with the §5.4 narrative; the long graph-survival false-positive note is appropriate.
- **Fig 13 (L482) DFT dumbbell:** caption claims "the residual trends positive with rising N-fraction"; body L479 reports Pearson r=+0.43 (positive). ✓

### Appendix tables vs body §5.5 references

- **Table D.1 (L641):** body §5.5 L476 quotes "calibrated densities of the 12 chem-pass leads into ρ_cal ∈ [1.75, 2.53] g/cm^3 and calibrated HOFs into [-372, +331] kJ/mol". Table D.1 column ρ_cal min row is L18 (1.75), max is L1 (2.53) ✓; HOF_cal min is L19 (-372.3), max is L9 (+330.8) — body says +331, table says +330.8. ✓ rounded.
- **Table D.1b (L663):** body §5.5 L476 references "three additional reference-class scaffolds (R2, R3, R14) ... reported separately in Appendix D.5"; table is in Appendix D.5. ✓
- **Table D.2 (L676):** body L478 says "K-J detonation velocities (Table D.2) span 6.84-9.52 km/s across the 12 chem-pass leads"; Table D.2 confirms L1=9.52 (max), L18=6.84 (min). ✓
- **Table D.3 (L698):** body L479 references the "sensitivity decomposition (slopes dD/dρ ≈ +2.47 km/s per g/cm^3; dD/dHOF ≈ -0.22 km/s per 100 kJ/mol)"; Table D.3 confirms these slopes. **Post-bug-fix recompute consistent.** ✓

### Appendix F (baseline-method examples) vs §5.3.3 reference

§5.3.3 does not actually point readers to Appendix F directly. Instead, Appendix F (L862) self-introduces by saying "the headline novelty-vs-property scatter (Fig 1) plots only the single top-1 SMILES of each baseline ... To complement that point-estimate view, the tables below list the ten most-novel CHNO-neutral candidates from each baseline pool". **Alignment gap:** §5.3.3 should add a forward pointer "(tail-of-pool examples in Appendix F)" near L424.

---

## Section 3: Inconsistency audit

### Headline numbers

- **ρ=2.00, D=9.56, P=40.5 for L1.** Verified identical at: abstract L81, §1 L102, §1 L112, §1 L341, §5.2 L341, §5.4 L470, §6 L515, §8 L539, Fig 7 caption L337, Table 1 row 1 L328, Fig B.1 L590. ✓
- **DFT-K-J L1 D=9.52 km/s.** Verified at §5.5 L478 ("9.52 km/s within 0.05 km/s"), Table D.2 row L1 (9.52), §6 L521 ("9.52 km/s, within 0.05 km/s"), abstract L81 ("\(D = 9.52\) km/s, agreeing ... within 0.05 km/s"). ✓
- **MolMIM 70M D=7.70 km/s.** Verified abstract L83 ("7.70 km/s"), §1 L102 ("D = 7.70 km/s"), §5.3.3 L406 (Table 4: 7.70), L422 ("top-1 D=7.70 km/s"), L424 ("D=7.70 km/s"), §6 L516, §8 (implicit). ✓
- **Best DGLD top-1 D=9.72 km/s (Hz-C1 viab seed 1).** Mentioned only in §6 L515 and §8 L539. **Inconsistency:** this number does not appear anywhere in §5.3.3 or in Table 4. Table 4 row "DGLD SA-C1 viab-only" reports "9.54 ± 0.04" mean across 3 seeds — a single seed at 9.72 is plausible but is not surfaced in the experimental section. Either add the per-seed maximum to §5.3.3 prose, or remove the 9.72 km/s claim from §6/§8.

### Calibration coefficients

- **a_rho=4.275, b_rho=-5.172.** Verified §5.5 L476 ("ρ_cal = 4.275·ρ_DFT - 5.172"), Table D.1 caption L641, Appendix D L674. ✓
- **c_hof intercept.** Three different precisions appear:
  - §5.5 L476: "HOF_cal = HOF_DFT - 205.7 kJ/mol" and "intercept of -205.7 kJ/mol".
  - §5.5 L476: "the intercept collapses to -205.7 kJ/mol".
  - Appendix D.4 L636: "**HOF_offset = -197 kJ/mol**". **INCONSISTENCY.** This is a real discrepancy: -205.7 vs -197. Either D.4 was not updated when the bug fix re-derived the intercept from -16763 to -205.7, or it refers to a different two-anchor variant. Appendix D.4 L636 also says "ρ_factor = 1.136" which contradicts the slope of 4.275 stated everywhere else (1.136 looks like a multiplicative factor, 4.275 is the linear-regression slope; these are different parameterisations). Resolution: D.4 uses a different (older?) calibration; reconcile the wording.
- **LOO residual ±62.5 kJ/mol.** Verified §5.5 L476 and L674. ✓

### Sample-count discrepancies

- **96/100 vs 96/97 vs 85/100 vs 77/100.**
  - Abstract L77: "96 are unknown to PubChem" (denominator implied 100).
  - §1 L102: "Across the merged paper top-100, 96/97 candidates are PubChem-novel".
  - §5.3.1 L363: "**96 of 97 are absent from PubChem**" (3 PubChem REST errors excluded, denominator 97).
  - §6 L518: "Semi-empirical xTB triage (§5.4) retains **96/100** merged-top-100 candidates that converge to ground-state geometries".
  - §5.4 L470: "the full extended xTB triage (top 1-100 of the merged paper top-100) reaches **85/100 stable**".

  **The two "96" numbers refer to two different things:** one is PubChem novelty (denominator 97 because of 3 REST errors), the other is xTB convergence (denominator 100). The abstract L77 phrasing "96 are unknown to PubChem" omits the "/97" denominator and elides the REST-error caveat. **Recommended fix:** abstract should say "96 of 97 candidates queryable on PubChem are absent" or "≥96/100 are novel against PubChem (3 REST timeouts excluded)". §6 L518 is fine.

- **77/100 chem-pass.** Verified abstract L77, §5.3.1 L378, Appendix E.7 L785. ✓
- **85/100 xTB-pass at 1.5 eV gate.** Verified abstract L77, §5.4 L470, §6 L518. ✓

### "Two 4-condition matrices yielding 7 distinct guidance settings"

- L399 §5.3.3 prose: "Together the two matrices yield seven distinct guidance settings".
- L401-402 Table 4 caption: "two overlapping 4-condition matrices ... yielding seven distinct DGLD guidance settings".
- L106 Fig 1 caption: "two 4-condition matrices yielding 7 distinct guidance settings".
- L515 §6: "the two-matrix 7-setting × 3-seed sweep".
- L539 §8: "the two-matrix 7-setting × 3-seed sweep".

All consistent. ✓

### Citation IDs (cross-check)

The following cite-targets appear in the body. Resolution status against `<li id="ref-...">` in the bibliography (L975-L1037):

- All `ref-eckmann2022limo`, `ref-gomez2018automatic`, `ref-jin2018junction`, `ref-reidenbach2023molmim`, `ref-ross2022molformer`, `ref-bengio2021gflownet`, `ref-hoogeboom2022edm`, `ref-vignac2023digress`, `ref-irwin2022chemformer`, `ref-peng2023moldiff`, `ref-mathieu2017sensitivity`, `ref-kamlet1968detonation`, `ref-elton2018applying`, `ref-casey2020prediction`, `ref-zhou2023unimol`, `ref-huang2021applying`, `ref-herve2010`, `ref-sabatini2018`, `ref-tang2017`, `ref-ho2022cfg`, `ref-dhariwal2021diffusion`, `ref-song2019score`, `ref-song2021sde`, `ref-ho2020ddpm`, `ref-rombach2022ldm`, `ref-krenn2020selfies`, `ref-ertl2009sa`, `ref-coley2018scscore`, `ref-rogers1960computer`, `ref-landrum2024rdkit`, `ref-sterling2015zinc`, `ref-kim2023pubchem`, `ref-jaegle2021perceiver`, `ref-olivecrona2017reinvent`, `ref-yang2017chemts`, `ref-bagal2022molgpt`, `ref-winter2019cddd`, `ref-maziarka2020molcyclegan`, `ref-schneuing2022diffsbdd`, `ref-guan2023targetdiff`, `ref-corso2023diffdock`, `ref-peng2022pocket2mol`, `ref-nefati1996ann`, `ref-klapotke2017nitrogen`, `ref-griffiths2020constrained`, `ref-yang2019chemprop`, `ref-schutt2018schnet`, `ref-qiao2020orbnet`, `ref-brown2019guacamol`, `ref-polykovskiy2020moses`, `ref-preuer2018fcd`, `ref-reymond2015gdb`, `ref-emdb`, `ref-cameo`, `ref-bruns2020watson`, `ref-bannwarth2019gfn2`, `ref-goerigk2017benchmark`, `ref-bondi1964vdw`, `ref-genheden2020aizynth`, `ref-sun2020pyscf`, `ref-perez2018film` — **all resolve** in the bibliography. ✓
- **Duplicate `id="ref-kamlet1968detonation"`**: L986 and L1028 both have the same `id`. HTML id duplication; browsers will jump to the first occurrence only. Remove the duplicate at L1028.

### Figure / table numbering

Figures 1-15 in monotone document order: Fig 1 (L105), Fig 2 (L121), Fig 3 abc (L176, L181, L189), Fig 4 SVG (L213), Fig 5 (L275), Fig 6 (L318), Fig 7 (L336), Fig 8 (L348), Fig 9 (L353), Fig 10 (L417), Fig 11 (L440), Fig 12 (L465), Fig 13 (L481), Fig 14 (L800), Fig 15 (L818), Fig A.1 (L580), Fig B.1 (L589). **Numbering is monotone** ✓.

Tables: T1 (L324), T2 (L368), T3 (L385), T4 (L402), T5 (L451), T6 (L808), T7 (L831), T8 (L850), T9 (L495). **T9 appears in §5.6 between T5 and T6** — out-of-order numbering. T9 should be T6 (and the others renumbered T7, T8, T9, T10), or the AiZynth table should be renumbered to fit between T5 and T6. **HIGH-impact inconsistency for a referee.**

Appendix tables: Table A.1, C.1, D.1, D.1b, D.2, D.3, D.4, D.5, E.1, E.2, E.3, F.1, F.2 — appendix-prefixed numbering is consistent.

---

## Section 4: Top 5 highest-impact fixes

### 1. Restore broken `\rho` math at L81, L122, L123 (HIGH)

**Problem.** The HTML source at line 81 (abstract paragraph 3) and at lines 122-123 (Fig 2 caption) contains literal newlines after a backslash where `\rho` was intended; the file reads `\(
ho` instead of `\(\rho\)`. KaTeX will fail to render or render this as `(ho = 2.00)`. This is the single most user-visible defect in the document.

**Proposed fix.** Replace each occurrence with `\(\rho\)` on a single line. Three occurrences total: abstract L81 ("Predicted properties match the strongest explosives in regular use: \(\rho = 2.00\)..."), Fig 2 caption L122 ("\(\rho \ge 1.85\) g/cm^3"), Fig 2 caption L123 ("100 clear \(\rho \ge 1.85\) AND novelty"). Search for `(\n ho` (literal) in the file.

### 2. Reconcile HOF intercept -205.7 vs -197 in §5.5 / Appendix D.4 (HIGH)

**Problem.** §5.5 L476 reports the post-bug-fix HOF calibration intercept as "−205.7 kJ/mol" and references `m2_calibration.json`. Appendix D.4 L636 still says "HOF_offset = −197 kJ/mol" and "ρ_factor = 1.136" (note the different parameterisation: a multiplicative density factor rather than a linear-regression slope). This appears to be stale text from before the ZPE bug-fix re-derivation.

**Proposed fix.** Update Appendix D.4 to use the same regression form as Table D.1 caption: `ρ_cal = 4.275 ρ_DFT − 5.172` and `HOF_cal = HOF_DFT − 205.7 kJ/mol`. If the −197 / 1.136 numbers refer to a separate older 4-anchor or 6-anchor extension, label them clearly as such; if they are leftovers, replace.

### 3. Renumber Table 9 (AiZynth) to T6, shift T6/T7/T8 (MED-HIGH)

**Problem.** Table 9 (AiZynth retrosynthetic search) appears at L495 in §5.6, before Tables 6 (MOSES), 7 (FCD), 8 (hazard alignment) which appear in Appendix E. A referee scanning the table list will mark this as obvious carelessness.

**Proposed fix.** Rename Table 9 → Table 6 (since it appears first after Table 5), and renumber the appendix tables to T7 (MOSES), T8 (FCD), T9 (hazard). Update the cross-reference in §5.6 prose at L495 ("Table 9").

### 4. Reconcile Fig 2 caption with §5.3.3 cross-reference at L360 (MED)

**Problem.** §5.3.3 L360 says "The top-1 candidates from this sweep are visualised in the predicted (D, P) plane in Fig 2"; Fig 2 caption (L122) actually shows "Top-200 generated leads from the pool=40k joint rerank" — these are not "top-1 of each method" but "top-200 of one method".

**Proposed fix.** Either (a) change L360 prose to "The top-1 candidates from this sweep are visualised across method families in Fig 1; the top-200 of the best DGLD pool is shown in the (D, P) plane in Fig 2.", or (b) recompose Fig 2 to display top-1 per method (which would duplicate Fig 1).

### 5. Unify "labelled-master" / "labelled master" hyphenation, and remove `score_model_v3e_h50` from body prose (MED)

**Problem.** Inconsistent hyphenation of "labelled master" / "labelled-master" across L102, L341, L363, L368, L376, L515, etc. Internal-tag leak `score_model_v3e_h50` in the Table 1 caption at L325 and at L284.

**Proposed fix.** Apply the rule "labelled-master" when used as a compound modifier (before a noun) and "labelled master" otherwise, replace globally. In the Table 1 caption, replace "code <code>score_model_v3e_h50</code> (Huang and Massa)" with "the literature-grounded sensitivity head (§4.4 / Appendix C.2)".

---

## Section 5: Sections that read smoothly

- **§2 Related work** is in good shape. Each subsection is on-topic, the citations all resolve, and the L168 paragraph on validation tooling reads as a confident, well-organised positioning of the framework against neighbouring lines. Only minor: long sentences flagged in §1 above.
- **§3.1 Four-tier label hierarchy.** The four-tier table at L191-198 is the cleanest structural element of the paper. Tier definitions, row counts, and the conditioning policy are presented as a single self-contained unit. The bridge prose at L186 and L202 makes the gating policy concrete without overstating it.
- **Appendix C (model architecture and hyperparameter values).** The Table C.1 hyperparameter table at L595-619 plus the C.1-C.3 self-distillation / sensitivity-head / gradient-norm prose is dense, on-topic, and answers exactly the reproducibility questions a methods-focused referee would ask. C.2 (literature-grounded sensitivity head) in particular reads as well-defended.

The §5.4 xTB triage prose (L446-472) is also in good shape: the Table 5 columns map cleanly to the prose, the verdict logic (xTB convergence AND HOMO-LUMO ≥ 1.5 eV) is consistent with the table column "verdict", and the false-positive note at Fig 12 caption is necessary disclosure rather than gold-plating.
