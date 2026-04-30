# Section 5 (Experiments) Rewrite Plan

Audit target: `docs/paper/short_paper.html`, lines 364-647.
Audit date: 2026-04-30. Verdict: NMI committee already at Accept across four passes; this audit is for "central-message clarity and focus", not new findings.

Current §5 word count (paragraph + caption + table text, HTML-stripped): **~7,200 words**.
Proposed §5 word count target: **~3,400-3,700 words** (a ~50% reduction in body, with detail relocated to existing appendix subsections).

---

## 1. Per-subsection diagnosis

### §5.1 Overview (line 366-367)

- **Central message**: "§5 reads results-first: leads (5.2), feasibility audits (5.3), baselines (5.4), ablations (5.5); headline targets are rho >= 1.85, D >= 9.0, P >= 35, Tanimoto <= 0.55; validation chain is SMARTS -> Pareto -> xTB -> DFT (§4.10)."
- **Supporting evidence**:
  - Subsection roadmap (L367)
  - Headline targets (L367)
  - Validation-chain pointer (L367)
  - Production hyperparams (CFG=7, pool >= 40k, alpha-anneal off; L367)
- **Body content that could move**: hyperparameter restatement (CFG=7 etc.) is duplicated from §4.12 and Appendix B.5; cut.
- **Name issue**: "Overview" is fine but generic; could be "Roadmap and headline targets".
- **Ordering**: fine where it is.

### §5.2 Filter pipeline outputs (Stages 1-4) (L369-468)

This is the heart of the body. Current structure: §5.2.1 (Stage 1+2 Pareto) and §5.2.2 (Stage 3+4 xTB+DFT). Combined ~2,400 words.

#### §5.2.1 Stage 1+2 (L372-391)

- **Central message**: "After hard filters, 355/400 single-objective top candidates survive; the gated Pareto front carries 34 candidates with viab >= 0.83; lead L1 (trinitro-1,2-isoxazole) sits at rho_pred=2.00, D_pred=9.56."
- **Supporting evidence**:
  - 355 survivors / 34-candidate Pareto front (L373)
  - L1 properties (L380) -- KEEP, headline
  - Naive single-objective candidate cautionary (L382) -- shows gates earn their keep
  - 89/100 source-pool finding (L370) -- KEEP, headline (Reviewer 3 hook)
  - Figure 19 (12 lead cards) and Figure 20 (Pareto)
- **Movable to appendix**: detail on the rejected polynitro-on-C2 candidate (L382) including its predicted D/P/rho (move to D.10 source-pool / D.7 SMARTS); the "20-of-20 absent from PubChem; 17/20 absent from internal corpus" sentence at L384 (overlaps §5.3.1, fold there).
- **Name issue**: "Stage 1+2 outputs: Pareto top leads" is descriptive but heavy. Proposed: **"Top leads from the gated Pareto reranker"**.
- **Ordering**: fine.

#### §5.2.2 Stage 3+4 (L394-467)

- **Central message**: "All 12 chem-pass leads are real DFT minima; 6-anchor calibration places L1 at rho_cal=2.09 and D_KJ,cal=8.25, putting L1 in the HMX-class band by relative ranking, with the 1.31 km/s residual to the 3D-CNN attributed to surrogate over-prediction."
- **Supporting evidence**:
  - "85/100 survive xTB; 12/100 survive DFT" (L397)
  - "13/15 stable in unguided pool=80k" pool-size finding (L404) -- KEEP (headline)
  - L1 calibrated rho_cal=2.09, D_KJ,cal=8.25 (L410, L412) -- KEEP (spine)
  - Figure 21 (DFT dumbbell + N-fraction residual)
  - Table 2 (6-anchor calibration)
  - Table 3 (per-lead K-J propagation)
  - SMARTS-rejected DFT cross-check (L420) -- principle of "DFT and SMARTS are complementary"
  - Hazard-head post-hoc (L422) -- one-sentence pointer is enough
- **Movable to appendix**:
  - The 73 -> 28 -> Stage-4-fail breakdown sub-paragraph (L397 second half) -> **C.5 or D.7**
  - The xTB-recipe paragraph (RDKit ETKDGv3 + MMFF94 + xTB --opt tight, L399) -> **C (DFT methodology)** as a "stage-3 recipe" subsection; body needs only one sentence
  - The full K-J residual decomposition prose (L412 second half: PETN outlier, NTO milder residual, Pearson r calculation) -> **C.6 / C.7** (already exists)
  - The "Independent CJ-detonation cross-check" paragraph (L413, ~250 words on Cantera ideal-gas being 3.5x off) -> **C** as a methodology cross-check; body needs one sentence
  - The composite-method HOF parenthetical (L420 last sentence about G4/CBS-QB3) -> drop (or move to §7 Limitations)
  - Table 3 (per-lead K-J propagation) is currently in body; consider moving to **C** since the body claim ("L1 D_KJ,cal=8.25") only cites the L1 row -- BUT: keep in body since paper's spine cites multiple rows visually for comparison; trim Table 3 footnote prose
  - The "Recipe and motivation" paragraph (L408) is methodological recap; one sentence remains, the rest -> **C.1**
- **Name issue**: "Stage 3+4 outputs: physics-validation chain" is accurate. Proposed: **"Physics validation: xTB triage and DFT confirmation"**.
- **Ordering**: combine §5.2.1 and §5.2.2 into a single §5.2 narrative? Both are about top leads; the Stage 1+2 / Stage 3+4 split is methodology-flavoured rather than reader-flavoured. Recommend keeping the two-subsection split but with cleaner names (above) and tighter prose.

### §5.3 Validation and feasibility audits (L470-577)

#### §5.3.1 Novelty audit (L473-490)

- **Central message**: "96/97 of the merged top-100 are absent from PubChem and 0/100 lie within Tanimoto 0.70 of the 694k augmented training corpus -- the model interpolates near energetic chemistry, it does not memorise."
- **Supporting evidence**:
  - 96/97 PubChem (L475)
  - 97/100 labelled-master absent (L475)
  - Table 4 (stratified novelty against labelled master + augmented corpus) -- KEEP
  - 0/100 within Tanimoto 0.70 of augmented corpus (L488) -- KEEP
- **Movable to appendix**:
  - The "strengthened SMARTS catalog" paragraph (L490, ~150 words: 77/100 retention, dual-pass 60-65/100 estimate) -> **D.7** (already has the per-class breakdown table); body only needs one sentence pointing to D.7. This block is currently inside §5.3.1 but is about chem-filter motif additions, not novelty -- it should move out.
- **Name issue**: "Novelty audit" is correct.
- **Ordering**: this subsection should arguably come ahead of §5.2.2 (novelty is the cheaper signal; physics is the expensive one), but the production reader expects performance-first. Keep current order.

#### §5.3.2 Retrosynthesis audit (L493-516)

- **Central message**: "AiZynthFinder reaches L1 in 4 steps via USPTO templates; 11/12 chem-pass leads return no productive routes within budget, exposing an energetics-domain template-database gap rather than candidate unsynthesisability."
- **Supporting evidence**:
  - L1: 9 routes, top-route 4 steps, state score 0.50 (Table 5, L501)
  - L4/L5: zero productive routes, reproduced at 5x budget (L509)
  - Extension to remaining 9 leads -- 0/9 productive routes (L513)
  - 1/12 hit rate as the headline (L516)
- **Movable to appendix**:
  - The detailed disconnection narrative for L1 (L507, ~280 words on DPPA Curtius, Boc protection, ZINC catalog gap, hazard note on the acyl azide intermediate) -> **D (Reproducibility) as a new subsection D.14 "AiZynthFinder route detail for L1"** OR fold into existing D.10/D.7 area. This level of detail is for a reader who wants to reproduce the synthesis; not for the central-message reader.
  - The drug-domain-template-bias discussion (L509) is two paragraphs and one sentence captures it; the rest moves to **D.14** alongside the L1 detail.
- **Name issue**: fine.
- **Ordering**: fine.

#### §5.3.3 Distributional audit (E1-E10) (L520-577)

- **Central message**: "A 500-candidate scaffold-distinct extension pool yields 10 DFT-confirmed leads spanning 8 Bemis-Murcko scaffolds and 6 chemotype families; E1 (1,2,3,5-oxatriazole) reaches D_KJ,cal=9.00 from a chemotype family L1 does not cover, demonstrating DGLD generates a distribution rather than a single chemotype."
- **Supporting evidence**:
  - 500-pool methodology (L522)
  - 10 leads, 8 scaffolds, 6 families (L533, L577)
  - Table 6 (E-set structures + xTB)
  - Table 7 (DFT + K-J + h50)
  - E1 D_KJ,cal=9.00, rho_cal=2.04 (L570) -- KEEP, spine
  - 1,2,3,5-oxatriazole stability caveat (L570) -- KEEP one sentence; details move
  - "two honest readings" of E1 vs L1 (L571-575)
- **Movable to appendix**:
  - The full 1,2,3,5-oxatriazole literature stability discussion (L570 first half: BDE 130-155 kJ/mol, ring-opening to azide-nitrile-oxide zwitterion, DSC/TGA recommendation) -> **C.5** or new caveat block; body says only "E1 has known thermal/Lewis-acid ring-opening pathways; an oxatriazole anchor extension is scoped as future work in §7"
  - The "two honest readings" enumerated list (L571-575) is the right caveat structure; trim from ~200 words to ~80 words ("either E1 is genuinely above L1, or the K-J residual is chemotype-dependent; an oxatriazole-anchor recompute is needed to discriminate, see §7")
  - The E2 OB-overflow footnote (L552) -> **E.13** (already exists)
  - The "8 of 9 have h50 >= 30 cm" stats sentence (L577) is informative, keep
- **Name issue**: "Distributional audit (E1-E10)" works but slightly buries the message. Proposed: **"Scaffold diversity: the E-set extension"**.
- **Ordering**: this subsection currently sits inside §5.3 ("Validation and feasibility audits"). It is not really a feasibility audit; it is a second discovery batch. Consider promoting it to its own §5.3 ("Scaffold diversity") and demoting "Validation audits" to §5.3.1 novelty + §5.3.2 retrosynthesis as a single §5.3.

### §5.4 Comparison with no-diffusion baselines (L580-617)

#### §5.4.1 SMILES-LSTM and MolMIM 70M (L583-609)

- **Central message**: "Four no-diffusion baselines on the same corpus expose the diffusion-prior contribution: SMILES-LSTM memorises (18.3% exact-match rate); MolMIM 70M is uncalibrated for the energetic regime; SELFIES-GA collapses from D_surrogate=9.73 to D_DFT=6.28 under the same DFT audit; REINVENT 4 is novel but optimises N-fraction not the DGLD composite -- DGLD Hz-C2 is the only condition with novel productive-quadrant coverage at DFT level."
- **Supporting evidence**:
  - Table 8 (top-1 head-to-head) -- KEEP
  - Figure 22 (forest plot) -- KEEP
  - Figure 23 (productive-quadrant scatter) -- KEEP
  - SMILES-LSTM 18.3% memorisation (L590, L604) -- KEEP, spine
  - SELFIES-GA D_surrogate->D_DFT collapse 9.73 -> 6.28 (L604) -- KEEP, spine (Reviewer 3 hook)
  - REINVENT 4 near-zero memorisation, D=9.02 vs DGLD 9.39 (L593, L604)
- **Movable to appendix**:
  - The Figure 23 footnote on score-conventions (~80 words on "composite S vs top-1 composite penalty being different scales") (L608) is critical for figure correctness but not for narrative; move to figure caption only or to **E** appendix
  - The MolMIM "uncalibrated" detail in the Table 8 caption is fine
- **Name issue**: "SMILES-LSTM and MolMIM 70M" undersells: REINVENT and SELFIES-GA are also there. Proposed: **"Head-to-head against no-diffusion baselines"** (or simply move the subsection up to be §5.4 directly without an internal §5.4.1).
- **Ordering**: §5.4 is short enough that the §5.4.1/§5.4.2 split is overkill; fold into a single §5.4.

#### §5.4.2 Distribution-learning metrics (L611-617)

- **Central message**: "DGLD has FCD=24-26 vs SMILES-LSTM FCD=0.52: the diffusion sampler is performing targeted search off the prior, not mimicking the corpus -- and within DGLD, FCD and Pareto-reranker composite are anti-correlated, exactly as the methodology is designed to do."
- **Supporting evidence**:
  - DGLD FCD 24-26, SMILES-LSTM FCD 0.52 (L612)
  - IntDiv1 0.818-0.838, scaffold uniqueness 659-1262 (L612)
  - Figure 24 (MOSES small multiples)
- **Movable to appendix**: most of L612 prose can compress to one sentence + one figure pointer; full table is already at **E.11** (existing).
- **Name issue**: "Distribution-learning metrics" is fine.
- **Ordering**: this is a 1-paragraph subsection. Fold into §5.4 as the closing paragraph.

### §5.5 Ablation summary (L620-646)

- **Central message**: "Seven ablations isolate per-component contribution; Tier-gate is the largest single contributor (4.6% vs 53.9% keep-rate reversal); diffusion prior over Gaussian-latent contributes +0.45 km/s D-lift; multi-head guidance trades scaffold count for novelty (Hz-C2 max-Tani 0.27, 5 scaffolds)."
- **Supporting evidence**:
  - Table 9 (master ablation table)
  - Figure 25 (forest plot)
  - Three summary paragraphs (L642, L644, L646)
- **Already consolidated**: detailed ablations live in **F.1-F.6**. This subsection works as-is. Do not move further.
- **Name issue**: fine.
- **Ordering**: fine.

---

## 2. Proposed §5 outline

| # | Title | One-line role |
|---|-------|---------------|
| **5.1** | Roadmap and headline targets | Pipeline overview, four-stage chain SMARTS->Pareto->xTB->DFT, and headline targets (rho >= 1.85, D >= 9.0, P >= 35, Tanimoto <= 0.55). |
| **5.2** | Top leads from the gated Pareto reranker (Stages 1+2) | 355/400 hard-filter survivors; 34-candidate Pareto front; L1 trinitro-isoxazole at rho_pred=2.00, D_pred=9.56; 89/100 from smallest unguided pool. |
| **5.3** | Physics validation and DFT confirmation (Stages 3+4) | xTB: 85/100 stable, 13/15 in unguided pool=80k; DFT: 12 chem-pass leads are real minima; L1 calibrated rho_cal=2.09, D_KJ,cal=8.25 places L1 in HMX-class band by relative ranking. |
| **5.4** | Novelty, synthesisability, and scaffold diversity | Combined feasibility-audit subsection: novelty (96/97 PubChem-absent, 0/100 within Tani 0.70 of augmented corpus); retrosynthesis (1/12 reachable via USPTO -- L1); E-set scaffold diversity (10 DFT leads, 8 scaffolds, 6 families; E1 oxatriazole D=9.00). |
| **5.5** | Comparison with no-diffusion baselines | SMILES-LSTM 18.3% memorisation; SELFIES-GA D_surrogate=9.73 -> D_DFT=6.28 collapse; REINVENT 4 D=9.02 (N-frac proxy); DGLD Hz-C2 the only DFT-confirmed productive-quadrant condition; FCD=24-26 confirms targeted-search interpretation. |
| **5.6** | Ablation summary | Seven ablations, master Table 9 + forest plot Figure 25; tier-gate (4.6% vs 53.9% keep-rate), diffusion prior +0.45 km/s, multi-head guidance scaffold-vs-novelty trade. |

**Net structural change**: Current §5 has 5 numbered subsections + 7 sub-subsections = 12 navigation nodes. Proposed §5 has 6 numbered subsections + 0 sub-subsections (or 2 if §5.4 is split into a/b/c). The flattening removes a layer of indirection.

**Alternative outline (more conservative, keeps two-tier hierarchy)**:
- 5.1 Roadmap
- 5.2 Top leads (Stages 1-4) -- combines current 5.2.1 + 5.2.2 with subheadings inside the prose
- 5.3 Feasibility audits -- combines current 5.3.1 + 5.3.2 + 5.3.3 with subheadings
- 5.4 No-diffusion baselines -- combines current 5.4.1 + 5.4.2
- 5.5 Ablation summary

Either outline works; the conservative one is easier to land without renumbering downstream cross-references.

---

## 3. Consolidation table

Destination notation: **C.x** = Appendix C (DFT methodology); **D.x** = Appendix D (Reproducibility, contains D.7 SMARTS, D.10 source-pool, D.11 distribution-learning metrics renamed E.11 in current numbering); **E.x** = current numbering (E.10/E.11/E.12/E.13 are inside Appendix D in the file but labelled E. -- see lines 1093/1107/1159/1175); **F.x** = Appendix F (Detailed ablations).

Note: the file has a labelling inconsistency where D.8/D.9 are followed by E.10/E.11/E.12/E.13 inside the same `<h3>D` block. Treat E.10-E.13 as the existing reproducibility-block subsections; do not propose new "I.x" subsections.

| Move | Current location | Word count est. | Destination | Disposition |
|------|------------------|-----------------|-------------|-------------|
| Naive single-objective polynitro-on-C2 detail | §5.2.1 L382 | ~140 | **D.7** (SMARTS catalog) | Replace body with one sentence. |
| 20-of-20 PubChem absence (3 internal anchors drop out) | §5.2.1 L384 | ~80 | **D.7** | Fold into existing strengthened-SMARTS prose. |
| Stage-3 recipe (RDKit ETKDGv3+MMFF94+xTB --opt tight) | §5.2.2 L399 | ~180 | **C** as new C.x "Stage-3 xTB triage recipe" | Body keeps one sentence pointing here. |
| 73 -> 28 -> Stage-4-fail breakdown | §5.2.2 L397 second half | ~120 | **C.5** or **D.7** | Body says "73 candidates fail Stage 4 by K-J undefined / imaginary frequency / composition gate; breakdown in C.5." |
| Calibration intercept derivation (GMTKN55 wB97X-D band, slope physical range, packing 0.69 explanation) | §5.2.2 L410 | ~120 | **C.1** or **C.3** | Body keeps "calibration: rho_cal = 1.392 rho_DFT - 0.415; HOF_cal = HOF_DFT - 206.7; LOO ±0.078 / ±64.6; full anchor derivation in C.1-C.3." |
| K-J residual decomposition prose (PETN outlier, NTO milder, Pearson r=+0.43) | §5.2.2 L412 second half | ~250 | **C.6** / **C.7** (already exist) | Body keeps the L1 1.31 km/s residual + N-fraction interpretation. |
| Cantera ideal-gas CJ cross-check paragraph | §5.2.2 L413 | ~250 | **C** as new C.x "Independent CJ cross-check" | Body keeps one sentence: "Ideal-gas CJ ranks L1, L4, L5 as RDX-class; absolute D values require BKW/JCZ3 covolume corrections (§7)." |
| Hazard-head post-hoc filtering paragraph | §5.2.2 L422 | ~50 | **E.12** (already exists) | Body keeps one sentence pointer. |
| Composite-method HOF (G4/CBS-QB3) sentence | §5.2.2 L420 last | ~30 | Drop or move to §7 Limitations | Drop. |
| Strengthened-SMARTS catalog paragraph | §5.3.1 L490 | ~150 | **D.7** (already has table) | Body says "the strengthened SMARTS catalog (D.7) retains 77/100 candidates after rejecting N-nitroimines, polyazene chains, and azo-amino-NO2 motifs; rank-1 trinitro-isoxazole survives." |
| L1 retrosynthesis disconnection narrative | §5.3.2 L507 | ~280 | **D** as new D.14 "AiZynthFinder route detail for L1" | Body says "AiZynth returns 9 routes for L1, top-route 4 steps, state score 0.50; the disconnection sequence walks back via Boc protection and a Curtius rearrangement of 4,5-dinitro-1,2-isoxazole-3-carboxylic acid (full route + ZINC catalog audit + hazard note in D.14)." |
| Drug-domain-template-bias paragraph | §5.3.2 L509 | ~140 | **D.14** | Body keeps one sentence on negative-result reproducibility at 5x budget. |
| 1,2,3,5-oxatriazole literature stability discussion | §5.3.3 L570 first half | ~180 | **C.5** (chem caveat block) | Body keeps "E1 has known thermal/Lewis-acid ring-opening pathways; oxatriazole anchor extension scoped as future work (§7)." |
| "Two honest readings" enumerated list | §5.3.3 L571-575 | ~200 | Trim in body to ~80 words | Compress, do not move. |
| Distribution-learning metrics prose | §5.4.2 L612 | ~220 | **E.11** (already exists) | Body shrinks to two sentences: FCD ratio (DGLD 24-26 vs SMILES-LSTM 0.52) + targeted-search interpretation. |
| Figure 23 score-convention footnote | §5.4 L608 | ~80 | Stays in figure caption | Trim caption; keep the substance. |

**Top-3 highest-ROI moves (cited in reply)**:
1. **Cantera CJ cross-check paragraph -> Appendix C** (250 words; pure methodology).
2. **L1 retrosynthesis disconnection detail -> Appendix D.14 (new)** (280 words; reproducibility-grade).
3. **K-J residual decomposition prose -> Appendix C.6/C.7** (250 words; already has the home, just push the explanatory prose).

---

## 4. Per-subsection rewrite recipes

### §5.1 (target: ~120 words; current ~180)

- **Lead sentence**: "Section 5 is structured results-first: §5.2-5.3 walk the four-stage validation chain (SMARTS -> Pareto -> xTB -> DFT, §4.10) on the merged top-100; §5.4 audits feasibility and scaffold diversity; §5.5 contrasts DGLD against no-diffusion baselines; §5.6 is the ablation summary."
- **Narrative arc**: chain -> targets -> production-config pointer.
- **Body**: state the four headline targets; cite §4.10 for the chain; cite §4.12 for hyperparameters; do not restate them.
- **Drop**: hyperparameter restatement.

### §5.2 (target: ~480 words; current ~700)

- **Lead sentence**: "Of the top-400 single-objective candidates, the §4.10 hard-filter gates reject 45 (poly-N-on-C2, MW < 130, OB > +25%); the 355 survivors define a 34-candidate Pareto front (Figure 20)."
- **Narrative arc**: hard-filter pass -> Pareto front -> top-5 properties (with L1 highlighted) -> the gates earn their keep -> 89/100 source-pool finding.
- **Body**: Table not needed (Figure 19 carries it); Figure 19 + 20.
- **Optional caveats**: one sentence on naive single-objective top being a model-cheat; full detail moves to D.7.

### §5.3 (target: ~700 words; current ~1,400)

- **Lead sentence**: "All 12 chem-pass leads are real DFT minima; under 6-anchor calibration L1 reaches rho_cal=2.09 g/cm^3 and D_KJ,cal=8.25 km/s, placing it in the HMX-class band by relative ranking against anchors."
- **Narrative arc**: xTB triage (85/100, 13/15 unguided pool=80k) -> DFT real-minima (12/12 + 2 anchors) -> 6-anchor calibration headline numbers -> K-J recompute on calibrated inputs -> the 1.31 km/s residual is surrogate over-prediction (one sentence; details in C.6) -> SMARTS-vs-DFT complementarity.
- **Body**: Figure 21 (dumbbell + N-fraction); Table 2 (anchor calibration); Table 3 (per-lead K-J propagation, possibly trim to top 5 + L9 + L20).
- **Optional caveats**: K-J fixed-product-distribution caveat (one sentence + §7 pointer).

### §5.4 (target: ~700 words; current ~1,900)

Combines current §5.3.1 + §5.3.2 + §5.3.3.

- **Lead sentence**: "Three feasibility audits frame the leads against PubChem, public USPTO retrosynthesis templates, and a scaffold-distinct E-set extension; the headline is that DGLD generates a chemotype distribution (10 DFT leads / 8 scaffolds / 6 families), not a single isoxazole hit."
- **Narrative arc**: novelty (96/97 absent from PubChem; 0/100 within Tani 0.70 of augmented corpus -- the strongest available reference) -> retrosynthesis (1/12 USPTO-reachable -- L1; 11/12 negative results expose energetics-domain template gap) -> E-set distributional audit (E1 D=9.00 from oxatriazole family, distinct from L1 chemotype).
- **Body**: Table 4 (stratified novelty); Table 5 (AiZynth); Tables 6+7 (E-set structures + DFT).
- **Optional caveats**: oxatriazole stability (one sentence); SMARTS catalog strengthening (one sentence pointing to D.7).

### §5.5 (target: ~450 words; current ~700)

Combines current §5.4.1 + §5.4.2.

- **Lead sentence**: "Four no-diffusion baselines on the same training corpus expose the diffusion-prior contribution: SMILES-LSTM memorises at 18.3%; SELFIES-GA collapses from D_surrogate=9.73 to D_DFT=6.28 under the same audit; REINVENT 4 is novel but at D=9.02 (N-fraction proxy); DGLD Hz-C2 is the only condition with novel productive-quadrant coverage at DFT level."
- **Narrative arc**: baselines list with one sentence per method -> productive-quadrant scatter (Figure 23) -> distribution-learning sanity check (FCD=24-26 vs 0.52 confirms targeted-search interpretation).
- **Body**: Table 8 + Figure 22 + Figure 23 + Figure 24.
- **Optional caveats**: composite-vs-penalty score-convention note in Figure 23 caption only (do not repeat in body prose).

### §5.6 (target: ~350 words; unchanged)

Already consolidated. Keep as-is.

---

## 5. Content to drop entirely

- The hyperparameter restatement in §5.1 (CFG=7, pool >= 40k, alpha-anneal off): already in §4.12 and B.5.
- The Figure 23 score-convention sentence in body prose (L608): keep in figure caption only.
- The "20 of 20 absent from PubChem; 17/20 absent from internal corpus; three internal-known anchors drop out" sentence at L384: redundant with §5.3.1's 96/97 claim (and the 3-rediscoveries set).
- The composite-method HOF pointer at L420 last: not load-bearing; reader who cares is at G4-tier and beyond paper scope.
- The Figure 22 caption sentence on MolMIM scale (currently both in caption and body).
- "We note up front that..." opener at L370: replace with the Pareto narrative directly.
- The "Source-pool breakdown" repeat at L391 (already covered earlier in same subsection).

---

## 6. Estimated word counts

| Subsection | Current (words) | Proposed | Delta |
|------------|-----------------|----------|-------|
| 5.1 Roadmap | 180 | 120 | -60 |
| 5.2 Top leads (5.2.1 collapsed) | 700 | 480 | -220 |
| 5.3 Physics validation (5.2.2 renamed) | 1,400 | 700 | -700 |
| 5.4 Feasibility (combined 5.3.1+5.3.2+5.3.3) | 1,900 | 700 | -1,200 |
| 5.5 Baselines (combined 5.4.1+5.4.2) | 1,900 | 700 | -1,200 |
| 5.6 Ablations (unchanged) | 700 | 700 | 0 |
| **Total** | **~6,800** | **~3,400** | **-3,400** |

(Above counts exclude tables and figure captions, which add ~400 words and are largely unchanged.)

Body word reduction: **~3,400 words / 50%**, with all reduction routed to existing appendix subsections (C.1, C.5, C.6, C.7, D.7, D.14 [new], E.11, E.12, E.13). Headline numbers (12 DFT leads; L1 rho=2.09, D=8.25; E1 D=9.00; SELFIES-GA 9.73 -> 6.28 collapse; 18.3% LSTM memorisation; 4.6% vs 48% keep-rate reversal; 89/100 source-pool) all remain in body. Four-stage validation funnel remains visible in body. §5.4 baseline comparison remains in body in collapsed form.

---

## 7. Risks and counter-arguments

- **Risk**: collapsing §5.2.1+§5.2.2 into §5.2+§5.3 (renumbered) breaks downstream cross-references to "§5.2.2". Mitigation: use the conservative outline in §2 (keep 5.2/5.3/5.4/5.5 hierarchy with internal sub-subsections) if cross-reference churn is a concern.
- **Risk**: the "two honest readings" of E1 vs L1 is a discriminating epistemic move; trimming it from 200 to 80 words may lose the nuance. Mitigation: keep the structure (option 1 / option 2) but shorter; do not collapse to one sentence.
- **Risk**: Table 3 (per-lead K-J propagation, 12 rows) currently in body could be moved to **C** to reduce visual weight. Counter-argument: visual-weight is a feature here; the table is the spine of the L1=8.25 km/s claim. Recommendation: keep in body, trim caption.
- **Risk**: Figure 23 (productive-quadrant scatter) is the Reviewer-3 hook; do not move or shrink. Recommendation: keep, only trim its 80-word footnote in the caption to ~30 words.
