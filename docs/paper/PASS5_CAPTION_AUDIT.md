# Pass-5 Caption Audit (R-style)

Surface-read complement to R5 (numbering / cross-refs). Mandate: voice, length, terminology, footnote chain, dash style, source attribution.

Total caption blocks audited: **41** (Figures 1-26 plus A.1 and F.1 = 27; Tables 1-9 plus A.1, B.1a-c, B.5, C.1-C.5 (incl. C.1b/C.1d), D.1-D.6 (incl. D.4b), E.1-E.2, F.1-F.6 = 32 captions). Numbering re-counted from the file: 14 figcaptions + 27 captions = **41**.

---

## Per-caption rows

### Figure 1 (line 97)
- length: 168 words
- stand-alone readable: yes (with effort; dense)
- opening style: noun phrase, OK ("Top-1 candidate per method against novelty...")
- terminology issues: "max-Tanimoto" used in body but caption writes "Tanimoto = 1.0" / "max-Tanimoto"; ok, consistent. "SMILES-LSTM" / "SELFIES-GA" hyphenation OK.
- footnotes: none; uses inline (Hz-C2 top-1 ...) parentheticals; OK
- voice/tense: present, OK
- em-dash count: 0
- recommended changes:
  - Trim: this caption tries to be the headline narrative; the (3.5 km/s artefact) clause and the "(Hz-C2 top-1 D = 9.39 km/s, P = 38.7 GPa)" clause both belong in body prose. Cut to ~110 words.
  - Move REINVENT (Table 8) inline citation to source-line at end: "Source: Table 8 + Table F.4 + DFT recompute."
  - Add explicit source line.

### Figure 2 (line 110)
- length: 49 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "Morgan-FP-2" should be "Morgan-FP, radius 2" or match Table 4's wording ("Morgan FP (radius 2, 2048 bits)"); minor.
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes:
  - Standardise "Morgan-FP-2" -> "Morgan FP r=2".
  - "panel A" / "panel B" capitalisation differs from "Panel A" elsewhere; pick one.

### Figure 3 (line 165)
- length: 53 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none material.

### Figure 4 (line 170)
- length: 41 words
- stand-alone readable: partial (does not say what x/y are)
- opening: noun phrase, OK
- terminology issues: "high-tail-oversampling recipe" is jargon; OK with section ref
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: clarify axes once (small "x: density / y: count" tag).

### Figure 5 (line 178)
- length: 50 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "Kamlet-Jacobs" with en-dash via &ndash; OK; "3D-CNN" consistent. "Trust-gating" first use is here; defined in §4.3.
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none material.

### Table 1 (line 182)
- length: 47 words
- stand-alone readable: yes
- opening: noun phrase ("Four-tier label-trust hierarchy"), OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Figure 6 (line 208)
- length: 65 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: ASCII arrow `->` in "encode -> generate -> guide -> filter"; replace with rightward-arrow entity or "&rarr;" for consistency with later captions that use &rarr;.
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: replace ASCII `->` with &rarr;.

### Figure 7 (line 219)
- length: 47 words
- stand-alone readable: yes
- opening: bold-noun-phrase pattern ("Figure 7. LIMO training.") — slightly different style from Figs 1-6 (which lack a sub-title). Consistent with Figs 8-15.
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none material; flag opening-style switch globally (see action list).

### Figure 8 (line 232)
- length: 60 words
- stand-alone readable: partial (only with §4.3)
- opening: bold-noun-phrase, consistent with Fig 7
- terminology issues: "FiLM input in Figure 9" should match later naming "FiLM-ResNet"
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: minor.

### Figure 9 (line 245)
- length: 56 words
- stand-alone readable: yes (technical reader)
- opening: bold-noun-phrase, OK
- terminology issues: math-heavy; acceptable for §4.5
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Figure 10 (line 267)
- length: 60 words
- stand-alone readable: yes
- opening: bold-noun-phrase, OK
- terminology issues: "3D-CNN/Uni-Mol" — note: body uses "Uni-Mol" with hyphen and "UniMol" without elsewhere. Decide one.
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: standardise "Uni-Mol" vs "UniMol"; choose one (Uni-Mol matches the upstream paper).

### Figure 11 (line 280)
- length: 88 words
- stand-alone readable: yes
- opening: bold-noun-phrase, OK
- terminology issues: "FiLM-MLP trunk" consistent
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none material.

### Figure 12 (line 293)
- length: 54 words
- stand-alone readable: yes
- opening: bold-noun-phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Figure 13 (line 306)
- length: 65 words
- stand-alone readable: yes
- opening: bold-noun-phrase, OK
- terminology issues: "epsilon_theta" reference is consistent
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Figure 14 (line 319)
- length: 44 words
- stand-alone readable: yes
- opening: bold-noun-phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Figure 15 (line 336)
- length: 67 words
- stand-alone readable: yes
- opening: bold-noun-phrase, OK
- terminology issues: "Stage-1 reranker (Figure 14)" - the cross-reference is to "Filtering" caption which itself points to Figure 15; a tight loop. R5's job to flag, but worth noting.
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none material.

### Figure 16 (line 347)
- length: 25 words
- stand-alone readable: partial (does not say what y-axis is)
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: add y-axis tag, e.g. "y: post-filter survivors" or "y: top-1 composite".

### Figure 17 (line 352)
- length: 53 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "5.1×" Unicode times symbol; consistent with rest of paper
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Figure 18 (line 1035)
- length: 22 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "Pareto scaffold-aware ranker" matches §4.10 terminology
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Figure 19 (line 373)
- length: 81 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "Kamlet/Jacobs" hyphenation: caption uses "Kamlet&ndash;Jacobs" (en-dash) — OK
- footnotes: marker "?" explained inline (good)
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none material.

### Figure 20 (line 381)
- length: 23 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: add data source: "Source: pool=40k unguided run, post Stage-2 reranker."

### Figure 21 (line 396)
- length: 50 words
- stand-alone readable: yes
- opening: noun phrase ("Left: ...; Right: ..."), unconventional but OK
- terminology issues: "DFT-K-J" hyphenation is en-dash via &ndash; in source; OK
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none material.

### Table 2 (line 401)
- length: 35 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "rho_cal" in plain text vs `\rho_{\text{cal}}` in MathJax elsewhere; caption uses MathJax, OK
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table 3 (line 414)
- length: 87 words
- stand-alone readable: partial (term "K-J formula bias" requires §5.3 context)
- opening: noun phrase, OK
- terminology issues: none
- footnotes: implicit; should explain inline what the "δD" propagation does
- voice/tense: mixed: "are obtained" (passive) and "reports" (active); minor
- em-dash count: 0
- recommended changes: tighten to ~50 words; move "3D-CNN surrogate error is a third source not included here" to a footnote marker rather than caption body.

### Table 4 (line 452)
- length: 30 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table 5 (line 464)
- length: 33 words
- stand-alone readable: partial (depends on familiarity with AiZynthFinder)
- opening: noun phrase, OK
- terminology issues: '"target node only" baseline' uses straight quotes; HTML elsewhere uses curly &ldquo;&rdquo; (Table B.5 does). Standardise to curly quotes.
- footnotes: none
- voice/tense: past ("was applied" implied), OK
- em-dash count: 0
- recommended changes: replace straight-quote pair with &ldquo;target node only&rdquo;.

### Table 6 (line 477)
- length: 47 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: column-list style ("Columns: lead ID, ...") is a fragment, not a sentence; consistent with other captions in this paper (Tables 4, 6, 7). House-style choice.
- footnotes: marker &dagger; in cells but caption itself does not mention &dagger;. The footnote text is in a separate `<p>` after the table (line 493). Trace: the caption should briefly point to the footnote ("&dagger; flagged values: see note below table.").
- voice/tense: present implicit
- em-dash count: 0
- recommended changes: add a single line in the caption pointing at the footnote paragraph.

### Table 7 (line 495)
- length: 75 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "h<sub>50,BDE</sub>" uses subscript chain; consistent
- footnotes: marker &dagger; (E2 row) — caption does not mention &dagger;; footnote is below table (line 493 covers both Table 6 and Table 7). Confusing because the footnote sits between Table 6 and Table 7. **Trace fail**: a reader scanning Table 7 alone would not know the &dagger; refers to the OB > 25% caveat.
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: add a one-line "&dagger; OB-bound; see note above" or merge the two-line footnote paragraph with each caption.

### Figure 22 (line 534)
- length: 88 words
- stand-alone readable: partial (acronyms: SMILES-LSTM, MolMIM, REINVENT 4, SELFIES-GA all assumed-known)
- opening: noun phrase ("Forest plot of..."), OK
- terminology issues: "MolMIM 70 M" — paper uses "MolMIM 70 M" and "MolMIM 70&nbsp;M" inconsistently; caption uses the spaced form, OK
- footnotes: refers to "Table 8 footnotes" (good cross-reference)
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: this caption mostly recapitulates Table 8 prose; trim by 30 words. Move the "MolMIM is a drug-domain reference and its composite is on a different scale" sentence into Table 8 caption since that's where the footnote chain lives.

### Figure 23 (line 541)
- length: 197 words
- stand-alone readable: partial (very long; readers will skip)
- opening: noun phrase, OK
- terminology issues: "L9-L20 (hollow blue)" hyphen vs "L9&ndash;L20" en-dash inconsistency
- footnotes: trace: ✗ (the "*Note on score conventions*" italic block is essentially a footnote merged into the caption; should be a separate caption-foot block)
- voice/tense: present, OK
- em-dash count: 0
- recommended changes:
  - Excessive length (>100 word threshold). Trim aggressively to ~70 words.
  - Move the entire "*Note on score conventions*" sentence into a footnote/asterisk under the figure.
  - Source attribution is good ("Source: S/viab for L1-L5 from Table 6; h50 from Table C.1c; baselines from Table 8").

### Figure 24 (line 548)
- length: 24 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "FCD" used without first definition in caption (defined in §5.5)
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: expand FCD on first use: "FCD (Fréchet ChemNet Distance)".

### Table 8 (line 520)
- length: 132 words
- stand-alone readable: partial (heavy footnotes)
- opening: noun phrase, OK
- terminology issues: footnote markers &dagger; and &ddagger; both used. Both explained inline. Good.
- footnotes: traced: yes for both
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: long but earned; consider hoisting the &ddagger; REINVENT explanation into a separate "Notes" `<p>` under the table to free the caption.

### Table 9 (line 556)
- length: 32 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Figure 25 (line 571)
- length: 56 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Figure 26 (line 1042)
- length: 30 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table A.1 (line 632)
- length: 33 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: "and pre-cross-source-deduplication" awkward — read as "and post... and pre...". Rewrite: "post-canonicalisation and post-charge-filter; pre-cross-source-deduplication."

### Figure A.1 (line 657)
- length: 51 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table B.1a (line 669)
- length: 16 words
- stand-alone readable: partial (under-explained: only labels what the table is, not what its source data tells the reader)
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: under-length but acceptable for hyperparameter dump.

### Table B.1b (line 687)
- length: 18 words
- as above; identical comments to B.1a.

### Table B.1c (line 719)
- length: 39 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "Stage-1 reranking" hyphen ok
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table B.5 (line 778)
- length: 34 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: uses curly quotes &ldquo;&rdquo; — preferred house style; matches the request.
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table C.1 (line 814)
- length: 110 words
- stand-alone readable: partial (very long with two embedded *Note* blocks)
- opening: noun phrase, OK
- terminology issues: none
- footnotes: trace: the two italic *Note on L3/L16* / *Note on L9/L20* blocks are footnotes embedded in the caption.
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: hoist the two *Note on...* blocks into a `<p>` immediately after the table; trim caption to ~50 words.

### Table C.1b (line 857)
- length: 33 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table C.1c (line 835)
- length: 161 words
- stand-alone readable: partial (very long; "L1 discrepancy" subblock belongs in body or note)
- opening: noun phrase, OK
- terminology issues: en-dash vs hyphen on "Politzer&ndash;Murray" via &ndash; OK
- footnotes: trace: "L1 discrepancy:" is an embedded note paragraph
- voice/tense: present mostly
- em-dash count: 0
- recommended changes: this caption is the longest in the paper (>100 words); move the "L1 discrepancy" 4-sentence block to a `<p>` below the table. Trim caption to ~50 words.

### Table C.1d (line 867)
- length: 26 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table C.2 (line 878)
- length: 65 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "K-J\(_\text{cal}\)" math markup OK
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table C.3 (line 907)
- length: 67 words
- stand-alone readable: yes (after rough familiarity with K-J)
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none material.

### Table C.4 (line 918)
- length: 51 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "Pearson r(f_N, residual)=+0.43" - good
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table C.5 (line 933)
- length: 39 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table D.1 (line 969)
- length: 53 words
- stand-alone readable: yes (technical)
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table D.2 (line 982)
- length: 45 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table D.3 (line 994)
- length: 35 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "azo&minus;NO<sub>2</sub>" uses &minus; which is mathematically a minus, not a hyphen; visually fine.
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: replace `&minus;` with regular hyphen in chemical-bond compound names ("azo-NO2") — &minus; is for math, not chem nomenclature.

### Table D.4 (line 1048)
- length: 102 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: column-list-as-fragment style; OK
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: long but readable; trim by 20 words by removing the per-column duplicates ("MOSES IntDiv1 internal-diversity score" can drop "MOSES").

### Table D.4b (line 1062)
- length: 67 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table D.5 (line 1082)
- length: 92 words
- stand-alone readable: partial; embeds methodological argument
- opening: noun phrase, OK
- terminology issues: "Fr&eacute;chet" (HTML entity) consistent
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: the "SMILES-LSTM no-diffusion baseline... is distributionally indistinguishable..." sentence is body-prose; move out. Trim caption to ~40 words.

### Table D.6 (line 1100)
- length: 49 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table E.1 (line 1131)
- length: 28 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table E.2 (line 1148)
- length: 12 words
- stand-alone readable: yes (defers to Table E.1)
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table F.1 (line 1171)
- length: 33 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table F.2 (line 1185)
- length: 53 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Figure F.1 (line 1200)
- length: 31 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table F.3 (line 1206)
- length: 13 words
- stand-alone readable: partial (very short)
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: under-length; expand by one sentence describing what columns C0/C1/C2/C3 mean (or point to F.4).

### Table F.4 (line 1225)
- length: 36 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table F.5 (line 1246)
- length: 25 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: none
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

### Table F.6 (line 1260)
- length: 35 words
- stand-alone readable: yes
- opening: noun phrase, OK
- terminology issues: "&thinsp;" thin-space in 20 000 — house-style consistent (used elsewhere)
- footnotes: none
- voice/tense: present, OK
- em-dash count: 0
- recommended changes: none.

---

## Cross-cutting findings

- **Em-dashes in captions**: zero literal em-dashes in any caption. All "—"-style breaks are en-dashes (&ndash;) used for compound names (Kamlet-Jacobs, Politzer-Murray, Bemis-Murcko, etc.) or hyphens. Pass on dash-style.
- **Opening style**: every caption opens with a noun phrase ("**Figure N.** Foo..."); some Methods captions add a sub-title (e.g., "**Figure 7. LIMO training.**") and others don't. Inconsistent across §3 / §4 figures (Figs 1-6 plain; 7-15 sub-titled; 16-26 plain again). Decide: either every caption has a sub-title, or none do.
- **Uni-Mol / UniMol / 3D-CNN / 3DCNN**: spelling drifts. "Uni-Mol" (B.1c) vs "UniMol" (Table 8 footnote, F.2) vs "3D-CNN" (consistent). Pick one.
- **"max-Tani" vs "max-Tanimoto" vs "Tanimoto"**: caption-internal use is mostly "max-Tanimoto" or "max-Tani"; both appear (Fig 22 uses "max-Tani"; Fig 1 uses "max-Tanimoto"; Table 8 uses "max-Tani"). Standardise to "max-Tanimoto" for first use, "max-Tani" thereafter, or commit to "max-Tanimoto" everywhere.
- **Footnote chain**: three trace failures (Table 6/7 dagger; Figure 23 Note on score conventions; Table C.1 Note on L3/L16 and L9/L20; Table C.1c L1 discrepancy block). All four are footnote-equivalents embedded in the caption; preferred fix is a small `<p>` below the table.
- **Stat density**: Figures 1, 22, 23, and Tables 8, C.1c are top-five stat-dense; Tables 6/7 are data tables (acceptable). Figure 23 (197 words) and Table C.1c (161 words) are clear hoist-targets.
- **Source attribution**: most captions lack an explicit "Source:" line. Figure 23 is the only one that does it correctly. Adding one-line source attributions to Figs 1, 16, 17, 18, 20, 21, 22, 24, 25 would close that gap.
- **Subjective adjectives**: zero "clearly", "obviously", "significantly". Pass.
- **Voice/tense**: present-tense throughout; no inappropriate past-tense in captions. Pass.
- **Acronym first-use**: FCD in Fig 24 caption is the only acronym not expanded inline (defined only in §5.5). Trivial fix.

---

## Top 10 prioritised action items

1. **Figure 23 (line 541)** — current 197 words. Cut the "*Note on score conventions*" block out of the caption and put it as a `<p>` immediately after `</figure>`. Also trim the L1-L5 / L9-L20 marker description by 30 words. Rationale: this is the single longest caption in the paper and the only one that visibly violates the >100-word threshold.

2. **Figure 1 (line 97)** — current 168 words. Move the "(Hz-C2 top-1 D = 9.39 km/s, P = 38.7 GPa)", the SELFIES-GA "(3.5 km/s artefact)" parenthetical, and the marker-area-encoding sentence into the §1.0/§5.5 prose. Target 110 words. Rationale: this is the headline figure; readers will skip a 168-word caption.

3. **Table C.1c (line 835)** — current 161 words. Move the "L1 discrepancy:" 4-sentence block ("The score model predicts h50,model=30.3 cm... experimental impact-sensitivity testing is required") into a `<p>` below the table. Rationale: 4-sentence per-row commentary belongs as a note, not a caption. Target 60 words.

4. **Table C.1 (line 814)** — current 110 words. Move both *Note on L3/L16* and *Note on L9/L20* blocks into a `<p>` below the table. Rationale: same pattern as #3; embedded *Note* blocks are footnote-equivalents.

5. **Table 6 / Table 7 footnote (lines 477 & 495)** — the &dagger; marker on E2 in both tables points to a single `<p>` between them, which is structurally invisible to a reader scanning Table 7 alone. Add inline "(&dagger; see footnote below)" or move the footnote paragraph adjacent to Table 7. Rationale: orphan footnote-trace failure.

6. **Figure 22 (line 534)** — current 88 words; recapitulates Table 8 footnote logic. Cut "MolMIM is a drug-domain reference and its composite is on a different scale (uncalibrated); the bar extends to ~4.79 and is shown for completeness rather than direct comparison" to ~10 words: "MolMIM bar is on a different scale (uncalibrated drug-domain reference)." Rationale: redundancy with Table 8 footnotes.

7. **Figures 7-15 versus 1-6/16-26 sub-title style** — Figs 7-15 use "**Figure N. Sub-title.**", while Figs 1-6 and 16-26 use "**Figure N.** Topic." Pick one and apply to all 26 figures. Rationale: consistency; reviewers notice this on copy-edit.

8. **Uni-Mol / UniMol unification** — find/replace globally to "Uni-Mol" (matches Zhou et al. 2023 upstream usage) in all captions: B.1c, Table 8, F.2 currently mix. Rationale: terminology consistency.

9. **Table A.1 caption (line 632)** — rewrite "post-canonicalisation and post-charge-filter and pre-cross-source-deduplication" to "post-canonicalisation and post-charge-filter; pre-cross-source-deduplication." (semicolon, not third "and"). Rationale: parsing.

10. **Figure 16 (line 347)** — add y-axis label inside caption. Current: "Classifier-free guidance scale w sweep at pool=8 000 per setting, ranked by the two-denoiser pool." Proposed: "Classifier-free guidance scale w sweep (y: post-filter survivors per 8000-sample pool, ranked by two-denoiser fusion). w=7 is the empirical sweet spot." Rationale: readers cannot interpret the y-axis from the caption alone.

---
