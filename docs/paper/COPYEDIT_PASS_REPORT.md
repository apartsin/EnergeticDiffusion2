# Copy-edit pass report
_Generated 2026-04-30_

## Fixes applied (categorised count)

- Typos: 5
  - L572: "Katritizky" → "Katritzky"
  - L412: "the strengthened SMARTS SMARTS gate" → "the strengthened SMARTS gate"
  - L918: "the strengthened SMARTS SMARTS gate of §5.3.1" → "the strengthened SMARTS gate of §5.3.1"
  - L1154: "energetic-SMARTS SMARTS gate reject" → "energetic-SMARTS gate reject"
  - L412: stray space before semicolon ("kJ/mol ;" → "kJ/mol;")
- Em-dash removal: 1
  - L611: " — " → "; " (only literal em-dash in prose; the &mdash; in tables L1289/L1303 are placeholders for empty cells and were left intact)
- Cross-reference fixes: 4
  - L831 (App C.4): "§5.5.4 SA-axis matrix" → "§G.4 SA-axis matrix"
  - L833 (App C.4): "re-running §5.5.2 to confirm" → "re-running §G.4 to confirm"
  - L835 (App C.4 cross-references list): "§5.5.4 Multi-seed pattern" → "§G.4 Multi-seed pattern"
  - L572 (§5.3.3): "applied to L1 in §5.3.2" → "applied to L1 in §5.2.2" (the 6-anchor calibration is in §5.2.2, not §5.3.2)
- Number drift / internal-consistency fixes: 2
  - L603 (Fig 22b caption): SELFIES-GA D_surrogate "9.74" → "9.73" to match the dominant value used in the abstract, Fig 1, §6, §7, §8 (5 occurrences of 9.73 vs 2 occurrences of 9.74; standardised on 9.73).
  - L611 (§5.4.1 prose): SELFIES-GA D_surrogate "9.74" → "9.73" (same reconciliation).
- Terminology consistency: 6
  - L535, L579: "Bemis-Murcko" (regular hyphen) → "Bemis&ndash;Murcko" (en-dash, the dominant form, 11 prior matches)
  - L89, L91, L180, L184: "Kamlet-Jacobs" (regular hyphen) → "Kamlet&ndash;Jacobs" (en-dash, the dominant form, 17 prior matches)
  - L572: "Politzer-Murray" → "Politzer&ndash;Murray" (only 1 hyphen-form against 5 en-dash-forms)
  - L441 (Table 7c caption): "K&ndash;J" (en-dash) → "K-J" (regular hyphen, dominant form: 107 vs 2)
  - L725 (App A.1): "(Tier A ∩ Tier B" → "(Tier-A ∩ Tier-B" (only 2 unhyphenated occurrences against 41 hyphenated)
  - L659 (§6 item 1): "65 980-row labelled master" → "65&nbsp;980-row labelled master" (sole regular-space; rest of paper uses non-breaking space)
- HTML hygiene: 1
  - L863: section title "D. first-principles audit" → "D. First-principles audit" (only lowercase section heading among A/C/D/E/F/G; standardised to title case)
- Cite-tag mismatches: 0 (every cite tag has a matching `<li id="ref-XXX">`)

Total: ~19 mechanical edits.

## Judgment items NOT applied (need author review)

- L1294 (§G.5): "The five-lane fusion more than doubles the keep rate (4.6 % vs 2.4 %)" — actual ratio is 4.6 / 2.4 ≈ 1.92, which is *less* than double, not "more than doubles." Recommend rewording to "nearly doubles" or "almost doubles"; the cleaner alternative is to drop the claim and let "quintuples the absolute number of passing candidates" (4639 / 966 ≈ 4.80, accurately phrased) carry the lift on its own.
- L639 (Table 8 Pool-fusion row): "Post-filter yield 966 → 4 639 (+5×)" is rounded up from 4.80; consistent with "quintuples" elsewhere but the §6 prose calls it "nearly five-fold (4 639 vs 966 candidates, +5×)". The "+5×" tag throughout is rounding-up; no cleaner number exists. Cosmetic only — not auto-edited.
- Figure-25 gap: the figure-numbering sequence is 1–24 then 26, 27 — Figure 25 is undefined in the file. No prose currently references Figure 25, so it does not break a cross-reference, but the gap is unusual. Recommend renumbering 26 → 25 and 27 → 26 (purely cosmetic; fixes the visible numbering jump readers will notice). This was NOT auto-fixed because it is a multi-edit renumber across the whole HTML and risks collateral cross-reference breakage.
- §5.1 absent: §5 jumps from §5.0 Overview straight to §5.2. No prose references §5.1 anywhere, so this is harmless, but a reader scanning the TOC will notice. Recommend renumbering §5.0 → §5.1, OR keeping §5.0 as "Overview" and ignoring the gap. Cosmetic only — not auto-edited.
- L412: long sentence with mixed semicolons and clauses ("\(...\) and HOF\(_{\text{cal}}\) = ...; cross-run consistency was verified at \(\Delta\rho \le ...\); ... brings the calibrated densities ..."). Reads as a parenthetical-within-sentence that could be split for clarity. Authorial style decision; not auto-edited.
- L77 (Abstract): "by Kamlet–Jacobs relative ranking; absolute D requires a thermochemical equilibrium solver, see §7" — the parenthetical aside is technically correct but interrupts the flow. Author may prefer to move to a footnote or end-of-sentence position. Not auto-edited.
- L380 alt-text uses "by" while figcaption uses "Twelve DGLD lead cards in a 3-by-4 grid" — the alt-text encoding is fine; flag only if the figure dimensions changed.

## Internal-consistency findings

- **SELFIES-GA D_surrogate value (9.73 vs 9.74)**: 5 occurrences of 9.73 (abstract, Fig 1, §G.4, §6, §8) vs 2 occurrences of 9.74 (Fig 22b caption, §5.4.1 prose). Reconciled to **9.73** in the two minority occurrences. (Reasoning: abstract and §6 should govern; the underlying experiment numbers are well-known to round to 9.73, and the 9.74 instances appear to be an earlier draft rounding.)
- **L1 ρ_cal (2.09 vs 2.093)**: prose throughout reports 2.09; Table 7c, Table D.1c, Table D.2 (per-lead) report 2.093. The two-decimal "2.09" is the correct headline rounding of the three-decimal table value; no discrepancy.
- **L1 raw DFT ρ_DFT (1.80 vs 1.801)**: prose reports 1.80, Table D.1c reports 1.801. Acceptable rounding.
- **Memorisation rate (18.3%)**: consistent throughout (abstract, §1, §5.4.1, §6, §8).
- **Tanimoto novelty 0.27**: consistent for L1 across abstract, §1, §5.2 (implied), §5.3.1, §6, §8, Fig 1, Fig 22b, Table 6a, Table G.4, and Table 7 stratification. No drift.
- **Viability 1.00 vs 0.83-1.00**: Fig 23 dashed line is at 0.83 (top-5 threshold); §5.2.1 says L1-L5 "have viability 0.83-1.00". Fig 23 places L1 at the top of the productive quadrant (consistent with the higher end of that band, ≈1.00 implied). No contradiction; the range and the plotted L1 marker are mutually consistent.
- **Pool-fusion ratio (4639/966)**: actual ratio is 4.80, rounded to "+5×" or "nearly five-fold" everywhere. The Table 8 entry, §6 item 5, §G.5, and Fig 17 all phrase this as "+5×" or "five-fold." Internally consistent. The "more than doubles the keep rate" phrasing in §G.5 (4.6%/2.4% = 1.92) is the only minor numerical-language slippage and is flagged above for author review.
- **Figure references**: All `Figure NN` and `Fig NN` prose mentions resolve to existing figures (with the exception that Figure 25 is never defined in the file but is also never cited; harmless gap).
- **Cite-tag completeness**: Every `<a class="cite" href="#ref-XXX">` resolves to a matching `<li id="ref-XXX">`; no orphans, no unused references.
- **Section cross-references**: After the §5.5 → §G migration fixes above (4 stragglers in App C.4), every `§X.Y` reference points to an existing section.
