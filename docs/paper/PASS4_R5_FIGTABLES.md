# PASS 4 Reviewer 5 Report: Figures and Tables

Scope: every figure caption + table caption + the prose sentence that introduces each. File reviewed: `docs/paper/short_paper.html` (1399 lines).

## Figure Inventory

| Fig | Line | Caption length (words) | Completeness |
|---|---|---|---|
| 1 | 99 | ~140 | ok (long but headline) |
| 2 | 112 | ~50 | ok |
| 3 | 167 | ~50 | ok |
| 4 | 172 | ~30 | ok |
| 5 | 180 | ~45 | ok |
| 6 | 210 | ~55 | ok |
| 7 | 221 | ~45 | ok |
| 8 | 234 | ~55 | ok |
| 9 | 247 | ~55 | ok |
| 10 | 269 | ~55 | ok |
| 11 | 282 | ~75 | ok |
| 12 | 295 | ~55 | ok |
| 13 | 308 | ~55 | ok |
| 14 | 321 | ~30 | short (relies on §5.2.2 ref) |
| 15 | 338 | ~75 | ok |
| 16 | 349 | ~20 | short |
| 17 | 354 | ~50 | ok |
| 18 | 1098 | ~25 | short |
| 19 | 378 | ~80 | ok |
| 20 | 390 | ~20 | short |
| 21 | 418 | ~45 | ok |
| 22 | 608 | ~80 | ok |
| 22b | 603 | ~70 | ok |
| 23 | 615 | ~165 | long (justified, includes scoring caveat) |
| 24 | 623 | ~25 | short |
| 25 | 646 | ~60 | ok |
| 26 | 1105 | ~30 | ok |
| A.1 | 732 | ~35 | ok |
| G.1 | 1252 | ~30 | ok |

Total: 26 main-numbered figures (1-26) + Fig 22b + Fig A.1 + Fig G.1 = 29 figures.

## Table Inventory

| Table | Line | Caption length (words) | Completeness |
|---|---|---|---|
| 1 | 184 | ~45 | ok |
| 6a | 589 | ~95 (with footnotes) | ok |
| 7 | 482 | ~30 | ok |
| 7b | 428 | ~40 | ok |
| 7c | 441 | ~85 | ok |
| 8 | 631 | ~30 | ok |
| 9 | 500 | ~30 | ok |
| 12 | 537 | (h4 heading only, no full caption) | short |
| 13 | 554 | (h4 heading only, no full caption) | short |
| A.1 | 707 | ~30 | ok |
| C.1a | 744 | ~20 | short |
| C.1b | 762 | ~20 | short |
| C.1c | 794 | ~40 | ok |
| C.5 | 853 | ~40 | ok |
| D.1 | 889 | ~120 | long (but valid) |
| D.1b | 932 | ~35 | ok |
| D.1c | 910 | ~210 | long (justified, h50 sub-explanation) |
| D.1d | 942 | ~20 | short |
| D.2 | 953 | ~75 | ok |
| D.3 | 980 | ~70 | ok |
| D.4 | 991 | ~60 | ok |
| D.5 | 1006 | ~40 | ok |
| E.1 | 1032 | ~55 | ok |
| E.2 | 1045 | ~50 | ok |
| E.3 | 1057 | ~40 | ok |
| E.4 | 1111 | ~115 | long (justified, defines all column meanings) |
| E.4b | 1125 | ~50 | ok |
| E.5 | 1145 | ~95 | ok |
| E.6 | 1163 | ~50 | ok |
| F.1 | 1183 | ~25 | ok |
| F.2 | 1200 | ~10 | short ("Columns as Table F.1") |
| G.1 | 1223 | ~40 | ok |
| G.2 | 1237 | ~65 | ok |
| G.3 | 1258 | ~15 | short |
| G.4 | 1277 | ~30 | ok |
| G.5 | 1298 | ~30 | ok |
| G.6 | 1312 | ~30 | ok |

## Skipped or duplicate numbers

### Tables: SUBSTANTIAL NUMBERING GAPS
The main-text table sequence is **non-contiguous and contains anomalies**:

- **Table 1** (line 184) → next main-numbered table is **Table 6a** (line 589). **Tables 2, 3, 4, 5, 6 are completely missing** from the document. None of these numbers are referenced in prose either (verified by grep), so this is unused/abandoned numbering rather than broken cross-refs.
- **Table 7b** (line 428) appears *before* **Table 7** (line 482). Reverse order; readers expect 7 → 7b → 7c, but here it is 7b → 7 → 7c. Table 7c is at line 441 (between 7b and 7).
- **Table 9** (line 500) appears, but **Table 10 and Table 11 are missing**, and the next table is **Table 12** (line 537), then **Table 13** (line 554), then back to **Table 8** (line 631).
- Table 8's textual order is therefore: 1, 7b, 7c, 7, 9, 12, 13, 6a, 8 — disordered for a print reading.
- **Tables 12 and 13 use `<h4>` headings instead of `<caption>`** (lines 537, 554), so they don't have the same caption styling as the rest. Also: their caption text is only the title-line "Extension-set leads: structure and pre-screen." / "...DFT, 6-anchor calibration, K-J, h50." with no completeness on what columns mean (insufficient self-containment).

### Figures: clean, no skips
- Figs 1–26 are present. Pass-3 verdict noted that the "25-26 gap" was closed; verified — Fig 25 (line 646) and Fig 26 (line 1105) both present.
- Fig 22b sits between Fig 22 (line 608) and Fig 23 (line 615). Note: Fig 22b appears at line 603, *before* Fig 22 at line 608 — minor reverse order.
- Appendix figures: A.1 (line 732) and G.1 (line 1252) present. No B.x, C.x, D.x, E.x figures exist (all are tables in those appendices).

### Cross-reference consistency check
- "Fig 26" is referenced in prose at line 1101 ("Fig 26") with the figure at line 1105 — consistent.
- "Fig 18" referenced at line 1095 with the figure at line 1098 — consistent.
- All Fig 1–25 prose refs match the captioned figure number.

## Caption issues per figure / table

- **Line 14 (Fig 14):** caption depends on §5.2.2 cross-ref to explain Stages 3-4; standalone reading is incomplete. Acceptable for a methods-section pipeline figure but at the edge.
- **Line 16 (Fig 16):** "Classifier-free guidance scale w sweep at pool=8 000 per setting, ranked by the two-denoiser pool. w=7 is the empirical sweet spot." — caption does not describe what the y-axis is. Reader must enter the prose.
- **Line 20 (Fig 20):** "Filtered candidates (post hard-gate): saturating-performance score (x) vs. viability classifier output (y), coloured by sensitivity proxy. Stars mark the Pareto front." Self-contained but does not say *which* pool size or which run. Stale/ambiguous numbering risk: the caption is silent on the pool size from which 34/355 are drawn (the prose at line 375 says pool=40 000); caption should include "from pool=40 000".
- **Line 24 (Fig 24):** "Distribution-learning small-multiples..." — no quantitative anchor (e.g. "n=10k per condition"); reader cannot interpret without the §5.4.2 prose.
- **Line 18 (Fig 18):** caption is fine; minor — the caption text says "range [0.564, 0.831]" which is the same range the prose at line 1101 quotes for the merged top-100 ("range [0.564, 0.831]"). Confirm: this is the *unguided pool=40k* range vs the *merged top-100 range*; if both equal [0.564, 0.831] this is suspicious. The caption says it is the unguided pool=40k top-200 range; the prose says it is the merged top-100 range. **Possible stale-number duplication** between Fig 18 and the line 1101 statement; needs author check.
- **Table 12 / Table 13 (lines 537, 554):** use `<h4>` not `<caption>`; caption text is title-only and does not explain columns or units. Recommend converting to `<table><caption>...</caption>` form to match the rest of the paper, and adding column-level definitions.
- **Table F.2 (line 1200):** "Columns as Table F.1." — too terse for self-containment. Should restate columns.
- **Table 7b vs 7c vs 7 ordering (lines 428, 441, 482):** Tables 7b and 7c appear in §5.2.2 (DFT calibration block); Table 7 appears later in §5.3.1 (novelty audit). The numbering implies 7 should precede 7b/7c; the actual ordering is reversed. Suggests Table 7 should be re-numbered (e.g., to Table 7d) or moved earlier.
- **Tables 12 and 13:** despite their h4 headings ("Table 12. Extension-set leads: structure and pre-screen." / "Table 13..."), they sit **between** Table 9 and the rest; if numbering followed strict order they should have been Table 10 and 11. The current numbering (12, 13) is inconsistent with the missing 10, 11 slot. Either retire 10/11 by renaming 12→10 and 13→11, or document why the leap exists.

### Terminology consistency in captions

- **K-J vs Kamlet–Jacobs:** Fig 14 caption uses "Kamlet–Jacobs" (line 321 implicit via §5.2.2 reference; the term appears in §4.10 prose, but several captions use "K-J" alone (Tables D.3, D.4, D.5, 7c, Fig 21). Caption-level consistency: Fig 21 caption uses "DFT–K-J" (hyphenated). Tables D.2 caption uses "Kamlet–Jacobs". Mixed in captions; minor.
- **max-Tani vs max-Tanimoto:** Table 6a header column says "max-Tani to LM" (line 590, also Table G.4 at line 1278), while caption text uses "max-Tanimoto" (line 615 Fig 23 caption, line 1183 Table F.1). Acceptable shorthand for headers but flag for consistency.
- **Pareto reranker / Pareto-reranker / Pareto scaffold-aware ranker:** all three forms appear across captions; recommend pick one.

### Source-data citations

Most captions point to the underlying table (e.g., Fig 21 → Table D.4; Fig 22b → Table 6a; Fig 23 → Tables 6, 6a, D.1c). Good practice. Exceptions:

- **Fig 16 / Fig 17:** no pointer to the underlying CSV/table for the swept numbers.
- **Fig 18:** no pointer (the line 1101 pool-fusion provenance lists the same range, suggesting the figure could cite the source pool-fusion result file).
- **Fig 24:** no source-table pointer (numbers come from Table E.4 / E.4b implicitly).

### Stale-number check

Spot-checked headline numbers in regenerated figures (1, 22, 22b, 23):

- **Fig 1 (line 99):** REINVENT 4 D=9.02, novelty 0.51 — matches Table 6a (line 595, D=9.02, max-Tani 0.57). Slight discrepancy: caption says "novelty 0.51" while table says "0.57 (aminotetrazine); 0.32–0.38 (seeds 1-2)". 1 − 0.57 = 0.43, not 0.51; 1 − 0.49 = 0.51. **Fig 1 caption "novelty 0.51" is not directly recoverable from Table 6a's max-Tani 0.57**; either the figure used a different lead, or this is a stale residual from before the Table 6a update. Author should reconcile.
- **Fig 22b / Fig 23:** consistent with Table 6a numbers.
- **Fig 22:** mentions "REINVENT 4 (N-fraction proxy)" and "SELFIES-GA 2k" — matches table 6a footnotes.

## Verdict: Minor revisions

The figure inventory is clean (Figs 1-26 plus 22b, A.1, G.1; no skips after the pass-3 fix). Captions are mostly self-contained. The substantive issues are all in the table-numbering layer, not the figures.

### Top 3 issues

1. **Table numbering is broken in §5 body.** Tables 2, 3, 4, 5, 6, 10, 11 are entirely absent from the document (and never referenced in prose). The actual sequence in §5 is 1, 7b, 7c, 7, 9, 12, 13, 6a, 8, then appendix tables. This is the most reader-jarring inconsistency; the simplest fix is to renumber 6a → 2, 7b → 3, 7c → 4, 7 → 5, 9 → 6, 12 → 7, 13 → 8, 8 → 9, restoring contiguous order. Alternatively, document the gaps as deliberate (matching some earlier numbering convention).

2. **Fig 1 REINVENT novelty number (0.51) is not recoverable from Table 6a's max-Tani (0.57)** — apparent stale number after Fig 1 regeneration. Author should either correct the caption or reconcile which lead supplied the figure marker.

3. **Tables 12 and 13 use `<h4>` headings instead of proper `<caption>` elements** with single-sentence titles that do not document columns. They fail the self-containment check; both need full captions matching the rest of the paper, and ideally renaming to fill the 10/11 gap.

Minor: Fig 16, Fig 20, Fig 24 captions are too short for self-containment. Table F.2 "Columns as Table F.1" is similarly under-specified. Captions mix "K-J" / "Kamlet–Jacobs" and "max-Tani" / "max-Tanimoto"; pick one per token type.
