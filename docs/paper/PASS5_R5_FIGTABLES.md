# Pass-5 Reviewer 5 (Figures and Tables) report

Audit of `docs/paper/short_paper.html` (1345 lines). Body figures Fig 1-26, body tables 1-9, plus appendix figs (A.1, F.1) and appendix tables (A.1; B.1a/b/c; B.5; C.1, C.1b, C.1c, C.1d, C.2, C.3, C.4, C.5; D.1-D.6; E.1, E.2; F.1-F.6).

## 1. Figure inventory (line, ~caption length, completeness)

| Fig | Line | Words (approx) | Completeness |
|---|---|---|---|
| 1 | 97 | ~210 | ok (very long but necessary; multi-method legend) |
| 2 | 110 | ~50 | ok |
| 3 | 165 | ~55 | ok |
| 4 | 170 | ~40 | ok |
| 5 | 178 | ~60 | ok |
| 6 | 208 | ~55 | ok |
| 7 | 219 | ~50 | ok |
| 8 | 232 | ~55 | ok |
| 9 | 245 | ~60 | ok |
| 10 | 267 | ~60 | ok |
| 11 | 280 | ~85 | ok |
| 12 | 293 | ~55 | ok |
| 13 | 306 | ~70 | ok |
| 14 | 319 | ~40 | short, could note that the upstream is the §4.10 chain |
| 15 | 336 | ~75 | ok |
| 16 | 347 | ~25 | short (lacks units of "two-denoiser pool" - works in context) |
| 17 | 352 | ~55 | ok |
| 18 | **1033** (D.10 sub-section) | ~25 | short; placement is in appendix while numbering is body-style "Figure 18" |
| 19 | 373 | ~85 | ok |
| 20 | 381 | ~22 | short (does not state pool size, source data) |
| 21 | 396 | ~50 | ok |
| 22 | 534 | ~95 | ok |
| 23 | 541 | ~210 | ok (long, conventions block is needed) |
| 24 | 548 | ~25 | short (could state pool size, seed count) |
| 25 | 571 | ~55 | ok |
| 26 | **1040** (D.10 sub-section) | ~30 | short; same out-of-order body-numbering issue as Fig 18 |
| A.1 | 657 | ~30 | ok |
| F.1 | 1198 | ~30 | ok |

Body figure count: 26 (Figs 1-26 contiguous, no gap). Note: Figs 18 and 26 sit physically inside the appendix (lines 1033, 1040) but carry body-style numbers; the earlier "25-26 gap" is closed because Fig 25 now lives at line 571 and Fig 26 at line 1040. See section 4 issue [body-figure-in-appendix].

## 2. Table inventory

| Tbl | Line | Words | Completeness |
|---|---|---|---|
| 1 | 182 | ~50 | ok |
| 2 | 401 | ~40 | ok |
| 3 | 414 | ~95 | ok |
| 4 | 452 | ~35 | ok |
| 5 | 464 | ~40 | ok |
| 6 | 477 | ~40 | ok |
| 7 | 495 | ~75 | ok |
| 8 | 520 | ~115 | ok |
| 9 | 556 | ~30 | ok |
| A.1 | 632 | ~35 | ok |
| B.1a | 669 | ~20 | short (could note steps, batch) |
| B.1b | 687 | ~20 | short (no specific steps reference) |
| B.1c | 719 | ~35 | ok |
| B.5 | 778 | ~40 | ok |
| C.1 | 814 | ~115 | ok |
| C.1b | 857 | ~30 | ok |
| C.1c | 835 | ~170 | ok (long, with anchor sanity check & L1 discrepancy) |
| C.1d | 867 | ~25 | ok |
| C.2 | 878 | ~85 | ok |
| C.3 | 905 | ~75 | ok |
| C.4 | 916 | ~70 | ok |
| C.5 | 931 | ~40 | ok |
| D.1 | 967 | ~55 | ok |
| D.2 | 980 | ~50 | ok |
| D.3 | 992 | ~45 | ok |
| D.4 | 1046 | ~115 | ok |
| D.4b | 1060 | ~55 | ok |
| D.5 | 1080 | ~75 | ok |
| D.6 | 1098 | ~55 | ok |
| E.1 | 1129 | ~30 | ok |
| E.2 | 1146 | ~10 | very short ("Columns as Table E.1") - acceptable cross-reference |
| F.1 | 1169 | ~50 | ok |
| F.2 | 1183 | ~75 | ok |
| F.3 | 1204 | ~20 | short |
| F.4 | 1223 | ~30 | ok |
| F.5 | 1244 | ~40 | ok |
| F.6 | 1258 | ~35 | ok |

Body Tables 1-9 contiguous and monotonic. Pass-4 fix landed (Tables 12/13 -> 6/7).

Appendix table sequences:
- A: A.1 only.
- B: B.1a, B.1b, B.1c, then B.5. **Skipped: B.2, B.3, B.4** (no captioned tables for those subsection numbers in the visible HTML; the subsections B.2-B.4 are prose-only). Either rename B.5 -> B.2 for contiguity or document the skip.
- C: C.1, C.1b, C.1c, C.1d, C.2, C.3, C.4, C.5 (no C.6+, no C.7-C.13 tables - those are prose subsections only; ok).
- D: D.1, D.2, D.3, D.4, D.4b, D.5, D.6 (no D.7+, but D.7-D.14 are prose subsections).
- E: E.1, E.2.
- F: F.1, F.2, F.3, F.4, F.5, F.6 contiguous.

## 3. Skipped or duplicate numbers

- **Body figures**: Figs 18 and 26 sit inside Appendix D.10 (lines 1033, 1040). Fig numbering is contiguous 1-26 across the document but the visual order in the rendered DOM is non-monotonic (Fig 17 at L352, then Fig 19 at L373, Figs 18 and 26 only render after L1030). PDF/HTML readers will see an out-of-sequence jump. Pass-4 closed the previous "25-26 gap" by promoting Fig 26 here, but did not move it next to Fig 25.
- **Appendix B**: tables jump B.1c -> B.5 (no B.2/B.3/B.4 tables). Subsection numbers in the prose go B.0, B.1 ... B.5 with no captioned tables for B.2-B.4, but the table numbering inherits the subsection number, leaving a visible gap. Recommend renumbering B.5 -> B.2 or relabelling as Table B.2 to remove the gap.
- **No duplicates** detected.

## 4. Caption issues per figure/table (line-numbered)

1. **Line 664** (prose in §B.0): "Tables C.1a (LIMO VAE), C.1b (denoiser + score model), and C.1c (label sources + reranker)". These tables are actually labelled **B.1a, B.1b, B.1c** at lines 669/687/719. The prose was not updated when the table block migrated. **Stale cross-reference, must fix.**
2. **Line 1126** (prose in §E): "Tables F.1 and F.2 list the ten most-novel CHNO-neutral candidates from the MolMIM 70 M and SMILES-LSTM pools". The actual captions at 1129/1146 are **Table E.1 and Table E.2**. **Stale cross-reference, must fix.**
3. **Line 367** (§5.1 roadmap, body prose): "§5.5 contrasts DGLD against no-diffusion baselines; §5.6 is the ablation summary." The §5.6 prose at line 552 is consistent. The roadmap mentions "validated through the four-stage chain ... documented in §4.10 (Fig 14)" -- ok. But line 552 says "Table 9 lists the headline result"; consistent.
4. **Line 511** (§5.4 prose): "see Appendix C.5 caveat block" - C.5 is the "Per-lead DFT-recomputed property table" (line 811). The "caveat block" is conceptual and may not match a single anchor. Minor: the C.5 anchor in HTML id terms covers a multi-paragraph block, but the prose phrase "C.5 caveat block" may confuse readers because Table C.5 (line 931) is the GuacaMol scaffold-Tanimoto table - **two distinct things share label "C.5"** (subsection prose vs separate table). Recommend renaming Table C.5 to C.6 (since its content is C.8 scaffold-similarity, the table is logically under the C.8 prose subsection at line 929; Table C.5 being labelled "Table C.5" while sitting inside subsection "C.8" is itself confusing).
5. **Fig 22 caption** (line 534) ends "see Table 8 footnotes" - Table 8 is at line 520, ok.
6. **Fig 23 caption** (line 541): "Source: S/viab for L1-L5 from Table 6" - Table 6 (line 477) is the E-set lead table (extension SMILES + xTB). The S and viability columns for L1-L5 are not in Table 6 - they are described in §5.2 prose. This sourcing reference points at the wrong table. **Likely needs to be Table 7 or Table C.1c**. Verify.
7. **Fig 14 caption** (line 319): "Four-stage funnel on the Figure 15 Pool fusion output" -- but the prose ordering is filtering (§4.10) BEFORE pool fusion (§4.11). Reading the caption in isolation suggests Pool fusion feeds Filtering, while §4.11 says fusion is post-decode and *inside* Stage 1 of the reranker. The dependency arrow is technically correct but the wording is confusing.
8. **Fig 26 caption** (line 1040): "The unguided pool=40 000 supplies 89 of the top-100" - figure caption number is fine but does not cite a source table; D.10 prose at line 1036 has the breakdown.
9. **Table 8 caption** (line 520): mentions "from Table 9" -- Table 9 at line 556 is the ablation summary, not the multi-condition baseline source. Table 8 itself contains the DGLD Hz-C0/Hz-C2 rows, so the cross-ref is loose; consider "from §5.6" instead.
10. **Caption number consistency vs §5 rewrite**: All §5.X.Y patterns in captions and prose checked - **zero stale §5.2.1, §5.3.1, §5.4.1, or §5.5.x found** (grep negative). Plain §5.5 references at lines 775 and 1055 are factual ("§5.5 hosts ..." and "promoted into §5.5 as Fig 24") and refer to current §5.5 (Comparison with no-diffusion baselines). These are consistent.

## 5. Cross-reference audit results

- **Stale §5.X.Y**: none.
- **§5.5 (ablation) leftovers**: none. Current §5.5 is "Comparison with no-diffusion baselines"; current §5.6 is "Ablation summary". All caption mentions of §5.5 / §5.6 match the new structure.
- **C.12 / C.13 / D.14 references in captions**: none of the figure/table captions reference C.12, C.13, or D.14 directly. The new appendix subsections are referenced only in prose (line 461 mentions D.14; lines 803-948 internal). No caption rewrites needed.
- **Body terminology**: "K-J" appears 36 times (in many captions), "Kamlet-Jacobs" appears in Tables 1 caption, Fig 21 caption ("DFT-K-J"), Fig 19 caption ("DFT/Kamlet-Jacobs"), Table 7 caption. Both forms used; consistent with the K-J shorthand convention. "max-Tanimoto" / "max Tanimoto" both used (Fig 1 "max-Tanimoto"; Fig 2 "max Morgan-FP-2 Tanimoto"; Table E.1 "max-Tanimoto"). Acceptable.

## 6. Verdict

**Minor revisions.** Numbering is now monotonic for body figures 1-26 and body tables 1-9 (Pass-4 issues closed). The §5 rewrite did not leave stale §5.X.Y caption refs. New C.12/C.13/D.14 do not require caption updates.

Two stale prose-to-caption cross-references must be fixed:

1. **Line 664**: prose says "Tables C.1a, C.1b, C.1c" but actual captions are **B.1a, B.1b, B.1c**.
2. **Line 1126**: prose says "Tables F.1 and F.2" but actual captions are **E.1 and E.2**.

Plus three lower-priority items:
- **Fig 23 caption (line 541)**: "Source: S/viab for L1-L5 from Table 6" likely points at the wrong table (Table 6 is E-set xTB pre-screen; S/viab live in §5.2 prose or Table C.1c).
- **Appendix B**: tables jump B.1c -> B.5 without captioned B.2/B.3/B.4. Consider renaming B.5 to B.2 to close the gap.
- **Fig 18 (line 1033) and Fig 26 (line 1040)**: physically render inside Appendix D.10 while carrying body-numbered "Figure 18 / Figure 26" labels - the document's figure-number sequence is monotonic but the rendered visual order jumps from Fig 17 directly to Fig 19, then re-enters at 18/26 well into the appendix.

No verdict-changing structural problems; revise the two stale cross-refs and consider the three minor items.
