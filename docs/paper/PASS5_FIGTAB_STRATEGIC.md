# Pass-5 Figures + Tables Strategic Review
_Generated 2026-05-01_

Source: `docs/paper/short_paper.html` (1333 lines, audited 2026-05-01).
Note: §6 was renumbered to "Limitations" (the prior "Discoveries and summary" §6 has already been migrated). Figure numbering 1 to 26 (with appendix figs A.1, F.1) and tables 1 to 9 (body) plus A.1, B.1a/b/c, B.5, C.1, C.1b, C.1d, C.2 to C.5, D.1 to D.6, D.4b, E.1 to E.2, F.1 to F.6.

## 1. Inventory

### Figures (body + appendix; numbering as printed in current HTML)
- Figure 1 (line 95). Headline panel: novelty vs (D, rho, P) for top-1 per method.
- Figure 2 (line 108). Top-200 leads in (D, P) plane, two-panel (rho-coloured, novelty-coloured).
- Figure 3 (line 163). EDA: density vs detonation-velocity scatter for labelled corpus.
- Figure 4 (line 168). EDA: per-property histograms (rho, D, P, HOF).
- Figure 5 (line 176). Tier-A/B/C/D composition pies per property.
- Figure 6 (line 206). Pipeline overview (encode/generate/guide/filter row).
- Figure 7 (line 217). LIMO VAE training panel.
- Figure 8 (line 230). Per-step training-mask construction.
- Figure 9 (line 243). Denoiser training panel.
- Figure 10 (line 265). Labeling pipelines for guidance heads.
- Figure 11 (line 278). Multi-task score-model training.
- Figure 12 (line 291). Self-distillation loop.
- Figure 13 (line 304). Sampling / generation panel.
- Figure 14 (line 317). 4-stage filtering funnel.
- Figure 15 (line 334). Pool fusion.
- Figure 16 (line 345). CFG scale w sweep curve.
- Figure 17 (line 350). Pool-size scaling: top-1 score and pass count.
- Figure 19 (line 371). 12 lead cards, 3x4 grid.
- Figure 20 (line 379). Pareto front (S vs viability scatter).
- Figure 21 (line 394). DFT dumbbell (3D-CNN to K-J cal) + N-fraction residual scatter.
- Figure 22 (line 533). Forest plot of top-1 composite penalty across methods + DGLD conditions.
- Figure 23 (line 540). Productive-quadrant scatter for the 12 leads vs baselines.
- Figure 24 (line 547). MOSES small-multiples (validity, scaffold uniq, IntDiv, FCD).
- Figure 25 (line 570). Guidance-ablation forest plot.
- Figure 18 (line 1019). Composite-score distribution over top-200 (appendix D.10).
- Figure 26 (line 1026). Source-pool contributions to merged top-100 (appendix D.10).
- Figure A.1 (line 641). Atom-composition statistics.
- Figure F.1 (line 1184). Guided vs unguided composite-distribution histogram.

Total figures: 28 (Fig 18 sits in appendix D.10 by current numbering, although its original assignment looks legacy).

### Tables (body + appendix)
- Table 1 (line 181). Four-tier label hierarchy.
- Table 2 (line 400). 6-anchor DFT calibration values.
- Table 3 (line 413). Per-lead K-J propagated uncertainty.
- Table 4 (line 452). Stratified novelty (top-100 vs LM, vs augmented corpus).
- Table 5 (line 464). AiZynthFinder retrosynthesis on L1/L4/L5.
- Table 6 (line 477). E-set leads structure + pre-screen.
- Table 7 (line 495). E-set DFT, calibration, K-J, h50.
- Table 8 (line 520). Top-1 baseline comparison (LSTM, MolMIM, GA, REINVENT, DGLD).
- Table 9 (line 556). Ablation summary.
- Table A.1 (line 617). Per-source provenance.
- Table B.1a (line 654). LIMO hyperparameters.
- Table B.1b (line 672). Denoiser + score-model hyperparameters.
- Table B.1c (line 704). External label sources + reranker config.
- Table B.5 (line 763). Production recipe.
- Table C.1 (line 799). Per-lead DFT structural validation.
- Table C.1b (line 842). Reference-class scaffold structures.
- Table C.1c (line 820). DFT-derived properties for 12 leads.
- Table C.1d (line 852). DFT properties for reference scaffolds.
- Table C.2 (line 863). K-J recompute on 6-anchor calibration.
- Table C.3 (line 892). K-J residual decomposition for L4.
- Table C.4 (line 903). N-fraction stratification of K-J residual (575 Tier-A rows).
- Table C.5 (line 918). GuacaMol scaffold-Tanimoto per condition.
- Table D.1 (line 954). 12-cell head-state grid clustering.
- Table D.2 (line 967). Heuristic vs literature h50 head ablation.
- Table D.3 (line 979). SMARTS reject per-class breakdown.
- Table D.4 (line 1033). MOSES-style metrics, single-seed.
- Table D.4b (line 1047). MOSES multi-seed.
- Table D.5 (line 1067). FCD per condition.
- Table D.6 (line 1085). Hazard-head vs SMARTS catalog overlap.
- Table E.1 (line 1116). Top-10 most-novel MolMIM candidates.
- Table E.2 (line 1133). Top-10 most-novel SMILES-LSTM candidates.
- Table F.1 (line 1156). Hard-negative budget anchor/cheat probe.
- Table F.2 (line 1170). Diffusion vs Gaussian-latent control.
- Table F.3 (line 1191). Four-way head-to-head top-1.
- Table F.4 (line 1210). DGLD multi-seed Hz/SA matrix.
- Table F.5 (line 1231). M7 five-lane 100k pool fusion.
- Table F.6 (line 1245). Tier-gate ablation.
- Plus Table B.5 anchor block at line 763 and the unnumbered CFG-quantile table at line 997 (D.8).

Total tables: 37 numbered (plus 1 unnumbered CFG-quantile table at line 997 in D.8).

## 2. Per-figure assessments

### Figure 1 (line 95) - novelty vs (D, rho, P), three-panel
- accuracy: numbers consistent with prose (Hz-C2 D = 9.39, P = 38.7; Hz-C2 max-Tani 0.27; SELFIES-GA collapse 9.73 to 6.28). Inset on D panel that conveys the surrogate-to-DFT collapse for SELFIES-GA is inventive.
- clarity: dense (3 panels, 5 method markers, marker-area encoding for memorisation, inset for surrogate collapse). Strong message but borderline cognitive load.
- redundancy: D-panel partially overlaps Fig 2 (top-200 in D plane) and Fig 22 (forest plot top-1 composite). Fig 1 is the only one that puts novelty on the y-axis though, so the angle is distinct.
- viz opportunity: a small "method legend with marker-area key" sub-callout (currently buried in caption) would land the memorisation-area encoding faster.
- placement: keep in body (this is the headline; it earns its slot).
- recommended action: keep, but trim the caption (currently 200+ words) to a 3-sentence lead and put the SELFIES-GA inset detail in a one-line footnote under the figure.

### Figure 2 (line 108) - top-200 in (D, P) plane, rho/novelty colourings
- accuracy: anchors and target lines consistent with §5.
- clarity: the two-panel split (rho colour, novelty colour) is a good "same data, two questions" pattern.
- redundancy: substantial overlap with Fig 1 D and P axes; Fig 1 is single-method-best, Fig 2 is full top-200 distribution.
- viz opportunity: combine Fig 1 D and Fig 2 panel A into a single richer plot ("top-200 cloud in (D, P) with anchors and target lines, coloured by novelty") and demote Fig 2 panel B to appendix. This is the cleanest single-figure summary of the productive-quadrant claim.
- placement: marginal body slot - currently earns its place but only because Fig 1 abstracts to top-1. If §1 narrative tightens, Fig 2 could move to §5.2 prelude.
- recommended action: keep in body but consider merging panel A with Fig 1 D-panel cloud overlay; demote panel B (novelty-coloured) to appendix.

### Figure 3 (line 163) - density vs det-velocity EDA
- accuracy: matches §3 prose.
- clarity: clear and efficient.
- redundancy: complements Fig 4 (per-property histograms) - jointly tell the "high-tail is sparse" story.
- viz opportunity: anchor labels (CL-20, HMX, RDX) on the scatter would let readers locate the high tail without consulting §3 prose.
- placement: keep in body.
- recommended action: keep; add anchor labels.

### Figure 4 (line 168) - per-property histograms
- accuracy: ok.
- clarity: ok but shows 4 separate histograms; HOF heavy tail message is the only load-bearing fact.
- redundancy: small overlap with Fig 3.
- placement: marginal. Could fold the "HOF tail" panel into Fig 3 as inset and demote remaining 3 panels to appendix.
- recommended action: consolidate Fig 3+4 into one EDA figure with HOF inset; or leave both but trim caption.

### Figure 5 (line 176) - tier composition pies
- accuracy: tiers match Table 1.
- clarity: small pies are weak for "Tier-A is small fraction" story; bar chart would be sharper.
- redundancy: with Table 1 (also tier counts).
- viz opportunity: replace with a single horizontal stacked bar (rho, HOF, D, P rows, Tier-A/B/C/D segments) - single figure conveys the trust-gating story.
- placement: keep in body but redesign.
- recommended action: redraw as horizontal stacked bar; pies are the wrong chart for proportion-comparison-across-properties messaging.

### Figure 6 (line 206) - pipeline overview
- accuracy: matches Figs 7-15 panels and §4 narrative.
- clarity: high-value entry point.
- redundancy: subsumes Figs 7-15 at one zoom level; the per-stage panels are the deeper view.
- placement: body, anchor of §4.
- recommended action: keep.

### Figures 7-15 (lines 217-336) - per-stage method panels
- accuracy: panels match §4 prose.
- clarity: nine separate diagrams is heavy. Most are method-recipe panels that would suit appendix once Fig 6 lands the high-level picture.
- redundancy: each is a zoom into one stage; collectively they exceed what most NMI methodology figures carry.
- viz opportunity: keep Fig 6 + Fig 9 (denoiser, the load-bearing core) + Fig 14 (filtering, the load-bearing audit) in body; demote Figs 7, 8, 10, 11, 12, 13, 15 to appendix B.
- placement: split. Fig 6, Fig 9, Fig 14 in body; rest to appendix.
- recommended action: demote Figs 7, 8, 10, 11, 12, 13, 15 to appendix B with new B.x numbering.

### Figure 16 (line 345) - CFG-w sweep
- accuracy: matches §4.12 / D.8 numbers.
- clarity: ok line plot.
- redundancy: with the unnumbered table at line 997 (D.8 CFG-quantile rel-err) and §5.6 ablation row.
- placement: appendix candidate. The "w=7 is the sweet spot" story is one number; the ablation summary table already records it.
- recommended action: demote to appendix D.8 next to the rel-err table.

### Figure 17 (line 350) - pool-size scaling
- accuracy: ok; also referenced by Table 9 ablation summary.
- clarity: efficient two-curve panel.
- redundancy: with Table F.5 (M7 five-lane 100k confirmation).
- placement: borderline body. The pool-size lever is methodologically central; recommend keep in body, but co-place Table F.5 immediately under it for the reader.
- recommended action: keep.

### Figure 19 (line 371) - 12 lead cards 3x4 grid
- accuracy: card labels match Tables 6/7 formulas; gap-pass borders match Stage-3 xTB triage.
- clarity: best at-a-glance summary in the paper. Border-colour-by-stability is a strong design.
- redundancy: with Tables 6/7 (which give SMILES + numerics) and with Table C.1c (DFT properties).
- viz opportunity: the cards already aggregate structure + chemotype + key numbers. The redundancy is justified - the figure is for impact, the tables for citation.
- placement: keep in body.
- recommended action: keep. Possibly add a small E-set companion grid (E1-E10) at smaller size in §5.4, which would make Tables 6+7 demote-able to appendix.

### Figure 20 (line 379) - Pareto front scatter
- accuracy: ok.
- clarity: efficient.
- redundancy: with Fig 23 (productive-quadrant scatter): both are S-vs-viability scatters with stars.
- placement: appendix candidate. Fig 23 supersedes it with the baseline-marker overlay and Pareto-floor strip; the bare Pareto front is a sub-set view.
- recommended action: demote to appendix D (next to Table 9 ablation row for reranker), or merge with Fig 23 as a single richer figure.

### Figure 21 (line 394) - DFT dumbbell + N-fraction residual scatter
- accuracy: matches Tables C.2, C.4.
- clarity: dumbbell on left is excellent; right panel adds population context.
- redundancy: with Table C.2 (numerical) and Table C.4 (population stratification).
- placement: keep in body. This is the headline DFT-validation figure.
- recommended action: keep.

### Figure 22 (line 533) - top-1 composite forest plot across methods
- accuracy: matches Table 8.
- clarity: clean forest layout.
- redundancy: heavy overlap with Fig 25 (ablation forest) and Fig 23 (productive-quadrant baseline markers). Three forest-shaped views in §5.5-§5.6 is one too many.
- viz opportunity: merge Fig 22 + Fig 25 into a single two-panel forest: panel A across methods (Fig 22 content), panel B across DGLD ablation conditions (Fig 25 content). One reading, one mental model.
- placement: keep in body but as merged panel.
- recommended action: combine Fig 22 + Fig 25 into a single two-panel forest figure.

### Figure 23 (line 540) - productive-quadrant scatter
- accuracy: prose flag (note on score conventions) is honest about the S-axis vs Table-8 composite mismatch. Useful caveat but signals a deeper problem: two different scoring conventions across the same paper section. Consider unifying.
- clarity: dense (12 leads, 4 baselines, 2 dashed thresholds, marker-area encoding).
- redundancy: with Fig 22 (same baselines, similar message). Fig 23 is the single best summary if score conventions are reconciled.
- placement: keep in body but only if score-axis is harmonised with Fig 22 / Table 8.
- recommended action: keep, but unify the y-axis convention across Fig 22, Fig 23, Table 8 in one direction (lower = better, or higher = better) and remove the "Note on score conventions" caveat.

### Figure 24 (line 547) - MOSES small-multiples
- accuracy: matches Tables D.4, D.4b, D.5.
- clarity: 4-panel small-multiples; informative but appendix-grade content (validity 100% across all conditions etc.).
- redundancy: with Tables D.4 / D.4b / D.5; Table D.4b is the same data multi-seed.
- placement: appendix. The headline "DGLD trades FCD for property targeting" is interesting but not body-grade for an NMI submission.
- recommended action: demote to appendix D.11.

### Figure 25 (line 570) - guidance-ablation forest plot
- accuracy: matches Table F.4.
- clarity: ok.
- redundancy: with Fig 22 (other forest) and Table F.4 multi-seed.
- placement: see Fig 22 - merge.
- recommended action: merge with Fig 22 as panel B of a unified two-panel forest.

### Figure 18 (line 1019) - top-200 composite distribution
- accuracy: matches D.10 prose.
- clarity: ok histogram.
- placement: appendix (already there). No change.
- recommended action: keep in appendix.

### Figure 26 (line 1026) - source-pool contributions
- accuracy: 89/5/3/3 split matches §5.2 prose.
- clarity: clear bar.
- placement: appendix (already there).
- recommended action: keep.

### Figure A.1 (line 641) - atom-composition statistics
- accuracy: ok, 30k subsample.
- clarity: ok.
- placement: appendix (correct).
- recommended action: keep.

### Figure F.1 (line 1184) - guided vs unguided composite distribution
- accuracy: ok.
- clarity: small histogram, paragraph could carry the message instead.
- redundancy: Table 9 ablation row + Tables F.2/F.3 already convey the result.
- placement: appendix; could even be dropped.
- recommended action: drop or downsize.

## 3. Per-table assessments

### Table 1 (line 181) - tier hierarchy
- accuracy: row counts match §3 prose.
- clarity: 4 rows, very readable.
- redundancy: with Fig 5 pies.
- placement: keep in body.
- recommended action: keep; if Fig 5 is removed, this table fully carries the tier story.

### Table 2 (line 400) - 6-anchor calibration values
- accuracy: matches §5.3.
- clarity: 6 rows, OK but only 2 columns (rho, HOF) are loaded; the slope/intercept lives in caption.
- redundancy: with §5.3 prose and Appendix C.4.
- placement: keep in body but consider folding into a richer table that also shows DFT-side values + LOO residuals for the anchor.
- recommended action: keep, but expand to include rho_DFT, HOF_DFT, and LOO residual per anchor (currently those are in C.5 prose only).

### Table 3 (line 413) - per-lead K-J uncertainty
- accuracy: 12 rows correspond to chem-pass leads.
- clarity: 9 columns, dense but well-ordered (ID first, derived sensitivities right).
- redundancy: with Table C.1c (DFT-side numerics) and Table C.2 (raw vs cal K-J).
- placement: keep in body.
- recommended action: keep. This is the propagated-uncertainty table; load-bearing.

### Table 4 (line 452) - stratified novelty
- accuracy: matches §5.4.
- clarity: 2 rows, 7 columns; reads efficiently.
- redundancy: with Fig 1 (novelty axis at top-1) and §5.4 prose.
- viz opportunity: 2 rows is on the small side for a body table; could be a footnote bullet pair instead.
- placement: keep in body but consider compressing.
- recommended action: keep; or convert to inline statistic summary in §5.4 prose.

### Table 5 (line 464) - AiZynthFinder retro
- accuracy: 3 rows match §5.4 prose.
- clarity: 5 columns, easy.
- redundancy: with §5.4 prose and Appendix D.14.
- placement: keep in body.
- recommended action: keep.

### Tables 6, 7 (lines 477, 495) - E-set structure + DFT
- accuracy: match §5.4 E-set prose.
- clarity: Table 6 has 7 columns + SMILES (10 rows; SMILES code is wide); Table 7 has 8 columns. Together they're a wall.
- redundancy: with each other (same 10 leads, two property cuts) and partially with Fig 19 (which only shows L-set, not E-set).
- viz opportunity: build an "E-set lead cards" companion grid mirroring Fig 19; demote Tables 6+7 to appendix.
- placement: marginal body. The E-set is co-headline (E1 oxatriazole), so it deserves a body presence; an E-set card grid would carry the message at a glance.
- recommended action: build E-set card grid as Fig 19b in body; move Tables 6+7 to appendix C as "E-set DFT detail."

### Table 8 (line 520) - baseline comparison top-1
- accuracy: ok with footnote caveats.
- clarity: 6 rows + 2 DGLD rows, 7 columns; footnotes are dense.
- redundancy: with Fig 22 forest, Fig 23 productive-quadrant.
- placement: keep in body. This is a baseline-comparison table.
- recommended action: keep; but harmonise the "top-1 composite" convention with Figs 22/23 (see Fig 23 entry).

### Table 9 (line 556) - ablation summary
- accuracy: 7 rows match §F.x.
- clarity: 4 columns (ablation, what varies, headline, detail). Readable.
- redundancy: each row is intentionally a pointer to F.x; the "headline result" column is the body-grade summary.
- placement: keep in body.
- recommended action: keep. Strong table.

### Table A.1 (line 617) - source provenance
- placement and content: appendix-correct. Keep.
- recommended action: keep.

### Tables B.1a/b/c (lines 654, 672, 704) - hyperparameters
- placement and content: appendix-correct. Keep.
- recommended action: keep.

### Table B.5 (line 763) - production recipe
- accuracy: matches §4.12 production-knob references.
- clarity: 17 rows, 3 columns; reads as a config dump.
- redundancy: each row is duplicated by the §4.12 prose. This is intentional ("source of truth"), and reviewers want exactly this.
- placement: appendix-correct.
- recommended action: keep. This is a strength of the paper; reproducibility-positive.

### Tables C.1, C.1b, C.1c, C.1d (lines 799-852) - DFT detail
- accuracy: 14 + 3 + 14 + 3 rows; structural and property splits.
- clarity: split into two tables per "structure vs property" is good. The reference-class split (C.1b/d) sits in a slightly awkward place (between two main-set tables).
- redundancy: Tables C.1, C.1b are structural twins; C.1c, C.1d are property twins.
- placement: appendix; consider re-ordering as C.1 -> C.1c -> C.1b -> C.1d so the main-set cluster reads end-to-end before the reference-class detail.
- recommended action: re-order in appendix C.

### Table C.2 (line 863) - K-J recompute on calibrated values
- placement and content: appendix-correct. Strong supporting table for §5.3.
- recommended action: keep.

### Table C.3 (line 892) - K-J residual decomposition for L4
- accuracy: ok.
- clarity: 2-row table to make a single rhetorical point. Could be a paragraph.
- placement: appendix.
- recommended action: optional - convert to a 2-line bullet pair in the C.6 prose to save table-overhead.

### Table C.4 (line 903) - N-fraction stratification (575 rows)
- accuracy: matches population residual analysis.
- clarity: 6-bin table; a small bar chart would communicate "residual flips sign at high-N" faster than reading 5 numbers.
- viz opportunity: convert to small bar chart figure C.x (residual vs N-fraction bin).
- placement: appendix.
- recommended action: convert to figure (residual vs N-fraction); keep table values in caption.

### Table C.5 (line 918) - GuacaMol scaffold-Tanimoto
- accuracy: ok.
- placement: appendix.
- recommended action: keep.

### Tables D.1-D.6 (lines 954-1085) - reproducibility / ablation detail
- accuracy: each backs a §F.x or §5.6 claim.
- clarity: appendix-grade detail; some (D.1 cluster table, D.6 hazard-vs-SMARTS overlap) are particularly load-bearing.
- redundancy: D.4 vs D.4b is acceptable (single-seed vs multi-seed; both are needed for the reviewer who wants seed variance). D.4b vs Fig 24 is a real overlap (same data; figure is small-multiples view, table is multi-seed numbers).
- placement: keep in appendix. Recommend dropping Fig 24 (see Fig 24 entry) and keeping Table D.4 + D.4b.
- recommended action: drop Fig 24, keep D.4 + D.4b.

### Tables E.1, E.2 (lines 1116, 1133) - baseline example outputs
- accuracy: top-10 each.
- clarity: SMILES are wide; readable in narrow column.
- placement: appendix-correct.
- recommended action: keep.

### Tables F.1-F.6 (lines 1156-1245) - ablation detail
- accuracy: each backs a Table-9 row.
- clarity: appendix-grade.
- redundancy: by design (Table 9 summarises, F.x details).
- placement: appendix-correct.
- recommended action: keep.

### Unnumbered table (line 997) - CFG-quantile rel-err
- accuracy: ok.
- clarity: 4 rows, 5 columns - small.
- placement: appendix D.8.
- recommended action: number it (call it D.8a) for citation hygiene.

## 4. Cross-cutting recommendations

### Redundancies to resolve
- **Fig 22 + Fig 25 (forest plots)**: merge into a single two-panel forest (panel A across methods, panel B across DGLD ablation conditions). Saves a body figure and unifies the "method comparison" vs "internal ablation" framing.
- **Fig 20 + Fig 23 (S-vs-viability scatters)**: Fig 23 supersedes Fig 20 with baseline overlay; demote Fig 20 to appendix.
- **Fig 24 + Tables D.4 / D.4b (MOSES)**: drop Fig 24; the tables are sufficient for appendix.
- **Tables 6+7 + Fig 19 (E-set vs L-set)**: build E-set card grid (Fig 19b); demote Tables 6+7 to appendix C.
- **Fig 5 + Table 1 (tier composition)**: Fig 5 is the wrong chart type; redraw as horizontal stacked bar OR drop Fig 5 in favour of Table 1.
- **Figs 7, 8, 10, 11, 12, 13, 15 (per-stage method panels)**: demote to appendix B; keep Fig 6 (overview), Fig 9 (denoiser core), Fig 14 (filtering core) in body.

### Proposed new figures
- **E-set lead cards grid (Fig 19b)**: mirrors Fig 19 layout for E1-E10. Lifts the co-headline E1 claim into a body figure and lets Tables 6+7 move to appendix.
- **Residual-vs-N-fraction bar chart (Fig C.4 visual)**: convert Table C.4 into a small bar chart in §5.3 or appendix C - the "residual flips sign at high N-fraction" message is a chart, not a table.
- **Anchor-and-lead timeline / scaffold genealogy (optional, ambitious)**: a small figure showing the established energetic scaffolds (HMX, CL-20, RDX) mapped against L1's positionally-novel substitution pattern would anchor the "fifteen years since HMX-class" narrative if §1 makes that claim. Speculative; skip if §1 doesn't lean on the historical-context framing.

### Promotions / demotions
- **Promote**: none. The body figure budget is already at the upper limit for an NMI submission.
- **Demote to appendix**:
  - Fig 5 (or redesign in place)
  - Figs 7, 8, 10, 11, 12, 13, 15 (per-stage method panels)
  - Fig 16 (CFG-w sweep)
  - Fig 20 (Pareto scatter)
  - Fig 24 (MOSES small-multiples; possibly drop entirely)
- **Move within body**: Fig 22 + Fig 25 to merged two-panel forest.

### Visualization upgrades (table to figure conversions)
- Fig 5 redrawn as horizontal stacked bar.
- Table C.4 converted to bar chart.
- Tables 6+7 augmented by an E-set card grid (figure form).

## 5. Prioritised action list (top 10)

1. **Merge Fig 22 (line 533) + Fig 25 (line 570) into single two-panel forest plot.** Currently three forest-shaped views (Figs 22, 25; plus Fig 23 markers) compete for the same mental model. One figure, two panels (across-methods, across-ablations) is cleaner and saves a body slot.

2. **Demote per-stage method panels Figs 7, 8, 10, 11, 12, 13, 15 (lines 217-336) to appendix B.** Keep Fig 6 (overview), Fig 9 (denoiser core), Fig 14 (filtering core). NMI methodology figures rarely run to nine panels in body.

3. **Build E-set lead-cards grid (new Fig 19b) at line ~510 and demote Tables 6+7 (lines 477, 495) to appendix.** E1 is co-headline; a card grid lifts it from buried-in-table to glanceable-figure.

4. **Demote Fig 24 MOSES small-multiples (line 547).** Tables D.4 + D.4b carry the same data with multi-seed variance; the figure is appendix-grade.

5. **Demote Fig 20 (line 379) Pareto scatter to appendix.** Fig 23 (line 540) supersedes it with the baseline-marker overlay and Pareto-floor strip.

6. **Harmonise score conventions across Fig 22, Fig 23, Table 8 (lines 533, 540, 520).** Currently Fig 23 carries a 5-line caveat-paragraph about "composite S" vs "top-1 composite" - same word, opposite directions. Pick one convention (suggest higher = better for the productive-quadrant view, since "novelty + on-target = upper-right" matches that direction) and rewrite axis labels and table column headers to match. Removes a serious reviewer trip-hazard.

7. **Redraw Fig 5 (line 176) as horizontal stacked bar.** Pies are the wrong chart type for "Tier-A is small fraction across multiple properties" - a stacked bar lets the reader compare proportion-across-properties at a glance.

8. **Convert Table C.4 (line 903) into a bar chart (or histogram-with-residual-overlay).** The "K-J residual flips sign at high N-fraction" finding is a chart, not a table - it's the headline of §5.3's K-J argument and it's currently buried in 6 numerical bin rows.

9. **Demote Fig 16 (line 345) CFG-w sweep to appendix D.8.** The "w=7 is sweet spot" story is one number; the unnumbered rel-err table at line 997 already records it. Fig 16 doesn't earn a body slot.

10. **Number the unnumbered CFG-quantile table (line 997) as D.8a or similar.** Citation hygiene; reviewers refer to tables by number.

### Net effect
Body figure count: 26 (in current count, including Fig 18/26/A.1/F.1 which are already appendix; body proper is ~22-24) -> approx 14-15 body figures after action items 1, 2, 4, 5, 9 (merge, demote 7, demote 1, demote 1, demote 1 = -10, +1 new = 14-15).
Body table count: 9 -> 7 after action item 3 (demote Tables 6+7).

The paper retains every quantitative claim; the redistribution moves recipe-grade content into appendix where reviewers expect it and keeps the body figure-budget focused on headline results (productive quadrant, DFT validation, ablation forest, lead grids, pipeline).
