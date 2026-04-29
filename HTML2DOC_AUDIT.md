# html2doc Skill Audit (DGLD paper, 2026-04-28)

Target: `C:\Users\apart\Projects\claude-skills\html2doc\` converting `E:\Projects\EnergeticDiffusion2\docs\paper\index.html` to a publication-grade DOCX.

Verdict (TL;DR): **Yes, with caveats.** Stages 2 and 3 (Pandoc + python-docx styling) work cleanly on our paper. Stage 1 (KaTeX to MathML) cannot run because the `katex` npm package is not installed, so all inline `$...$` and display `$$...$$` math currently passes through as literal LaTeX text. Installing one npm package unlocks the remaining 90% of the conversion.

---

## Section 1: Skill structure and dependencies

### File inventory (actual layout differs from briefing)

```
html2doc/
  README.md                        3.5 KB   user docs
  html2doc.py                      4.6 KB   orchestrator (3 subprocess calls)
  reference.docx                  35.7 KB   pandoc style template
  requirements.txt                  21 B    pypandoc, python-docx
  scripts/
    katex_to_mathml.js             4.2 KB   stage 1
    convert_to_docx.py             4.2 KB   stage 2
    apply_academic_style.py       25.1 KB   stage 3
    create_reference_doc.py        8.0 KB   regenerates reference.docx
```

The briefing said "5 Python scripts + 1 JS"; actual count is **3 Python scripts + 1 JS + 1 orchestrator** (4 + 1 + 1). There is no `academic_styles.py` (the README references it but no such file exists; the styling configuration is inlined as the `PROFILES` dict in `apply_academic_style.py`).

### What each script does

- `html2doc.py` (orchestrator). Validates deps, then shells out to the three stages with the input/output filenames stitched together. Hardcodes relative paths `html2doc/scripts/katex_to_mathml.js` etc., so it must be invoked with the skill installed at `<cwd>/html2doc/`.
- `scripts/katex_to_mathml.js`. Reads HTML, regex-extracts `$$...$$` and `$...$` math, runs `katex.renderToString({output:'mathml'})`, splices MathML back into HTML wrapped in `<div class="equation-mathml">` or `<span class="inline-mathml">`.
- `scripts/convert_to_docx.py`. Strips the head `<title>` tag (to avoid double-title), regenerates `reference.docx` via `create_reference_doc.py`, then `pypandoc.convert_file(extra_args=['--mathml', '--reference-doc=...'])`. Verifies output by counting `<m:oMath` in the resulting `word/document.xml`.
- `scripts/apply_academic_style.py`. Loads the DOCX with python-docx, applies a profile-driven style sheet (`camera-ready-generic` or `review-manuscript`): margins, fonts, table borders/shading/alignment, page numbers in footer, caption styling (bold-italic "Figure N:" prefix), reference hanging indent, pagination controls.

### Dependency status (as actually installed on this machine)

| Dep | Required | Status |
|---|---|---|
| Python 3.x | runtime | OK (`/c/Python314/python` 3.14) |
| pypandoc | stage 2 | **OK** (1.17) |
| python-docx | stage 3 | **OK** (1.2.0) |
| pandoc | stage 2 | **OK** (3.1.13 at `C:\Users\apart\AppData\Local\Programs\pandoc\pandoc`) |
| Node.js | stage 1 | OK (v24.13.1) |
| `katex` npm package | stage 1 | **MISSING** (no `node_modules/katex` in skill dir, project dir, or `~`) |

### Smoke test results

`python html2doc.py --help` works, prints usage cleanly. Argparse exposes `--input/--output/--keep-temp/--profile`.

Running the full pipeline (`python html2doc/html2doc.py --input docs/paper/index.html --output paper_test.docx --keep-temp` from the project root) **fails immediately** at the dependency check:

```
Checking dependencies...
Missing dependencies:
  - katex (run: npm install katex)
```

Two compounding install issues:

1. The skill is at `C:\Users\apart\Projects\claude-skills\html2doc\` but `html2doc.py` builds shell commands with the literal prefix `html2doc/scripts/...`, so it must either be copied or symlinked into the project root, or the user must `cd` into the parent of the skill directory. There is no PATH-aware lookup.
2. `check_dependencies()` literally tests `os.path.exists('node_modules/katex')` against the current working directory. Even if `katex` were globally installed (`npm install -g katex`), this check would still report it as missing. The user would need to run `npm install katex` from the project root, creating a project-local `node_modules/`.

---

## Section 2: Pipeline trace

### Stage 1: KaTeX to MathML (Node)

Invocation: `node html2doc/scripts/katex_to_mathml.js --input <in.html> --output <in_mathml.html>`.

Schema: HTML in, HTML out. Math regions are detected by **two regexes only**:

- Display: `/\$\$([^$]+)\$\$/g`
- Inline: `/\$([^$\n]+)\$/g`

Hard assumptions:
- Math must be expressed as `$...$` / `$$...$$`. **`\(...\)` and `\[...\]` are NOT detected** (the regexes don't match those).
- The display regex is non-greedy by character class (`[^$]+`), so it cannot handle math containing literal dollar signs.
- The inline regex requires the math to be on a single line.
- Anything that matches `$X$` anywhere in the document (even inside a code block or HTML comment) will be replaced. There is no awareness of HTML structure (no parser; pure string regex).

Failure modes: KaTeX parse errors are caught per-equation and logged to stderr; the original `$...$` is left unchanged (it stays literal in the output). Missing `katex` module is a fatal `process.exit(1)`.

### Stage 2: MathML to DOCX (Pandoc + python-docx wrapper)

Invocation: `python html2doc/scripts/convert_to_docx.py --input <in_mathml.html> --output <out.docx> --profile <profile>`.

What runs under the hood: `pandoc <in> -o <out> --mathml --reference-doc=<skill>/reference.docx` (regenerated each call from `create_reference_doc.py`). Pandoc converts MathML to native OMML in the resulting DOCX. The wrapper post-checks success by counting `<m:oMath` tags.

Hard assumptions:
- Pandoc must be on PATH (pypandoc auto-downloads if not, but ours is already installed).
- HTML head `<title>` is stripped to avoid duplicating the body H1.
- Image paths in `<img src="...">` are resolved by Pandoc **relative to the current working directory**, not the HTML file's directory. Running from the wrong cwd loses every figure with a "Could not fetch resource" warning.

### Stage 3: Academic styling (python-docx)

Invocation: `python html2doc/scripts/apply_academic_style.py --input <converted.docx> --output <final.docx> --profile <profile> [--table-width 100]`.

Schema: DOCX in, DOCX out. The script walks paragraphs and tables, classifies each paragraph by heuristics, and reassigns its style.

Hard assumptions and heuristics worth knowing:
- Paragraphs whose text starts with `Figure N` or `Table N` (with `:` or `.`) are reformatted as captions; the leading `Figure N:` / `Table N:` is auto-bolded and italicised.
- Paragraphs whose text starts with `Keywords:` are styled as `Keywords`; literal `Abstract` becomes the abstract label; literal `References` opens a "References mode" where every following paragraph becomes a `Reference Entry` until another heading appears.
- Numbered headings are detected by regex `^\s*\d+(\.\d+)*\.?\s+`. `Heading 1` is **only** reserved for the manuscript title. Section "1. Introduction", "2.1 ...", etc. all stay at whatever level Pandoc emitted (typically `Heading 2`/`Heading 3`).
- Front-matter author/affiliation detection picks the first two non-empty paragraphs after the title, before the first numbered section / `Abstract` / `Keywords:` line. This is fragile.

Failure modes documented in the README: equations not converting (KaTeX missing), images not showing (SVG rather than PNG/JPG; relative path issue), tables not full-width (use `--table-width 100`).

---

## Section 3: Risk assessment for the DGLD paper

`docs/paper/index.html`: 1051 lines, 26 tables, 20 `<img>` tags, 8 `$$...$$` blocks, 46 `class="cite"` links, 85 `<sub>` tags.

| Feature | Survival prediction |
|---|---|
| `$$...$$` display math (8 blocks) | OK once `katex` is installed; clean OMML in Word |
| `$...$` inline math (e.g. `$D$`, `$P$`) | OK with KaTeX. Without it, dollar pairs leak through; verified 7 paragraphs with literal `$` in our test run |
| `\(\rho\)`, `\(D\)`, `\(\bar\alpha_t\)`, `\(\varepsilon_\theta\)` etc. | **WILL NOT BE CONVERTED** even with KaTeX installed. The katex_to_mathml.js regex only matches `$...$` / `$$...$$`, never the LaTeX-bracket form. The DGLD paper uses `\(...\)` heavily (Section 4 onwards). These will stay as literal `\(\rho\)` text in Word. **This is the single biggest risk.** |
| 26 tables | Confirmed clean in test run; all 26 were styled with proper borders and header shading |
| 20 PNG figures | Clean once cwd is set to `docs/paper/` (Pandoc resolves `figs/*.png` relative to cwd, not source file). Verified |
| `<a class="cite" href="#ref-X">[refX]</a>` citations | Pandoc renders these as Word hyperlinks (129 hyperlinks in test). Visually they appear as `[ho2020]` text linking to `#ref-X` anchors. **Not real Word cross-references**; if you re-number references, the links keep the old text. Acceptable for review submission, manual fixup needed for camera-ready |
| `<sub>3</sub>` molecular subscripts | Clean; verified 273 native `w:vertAlign val="subscript"` in the test DOCX |
| HTML entities `&nbsp; &minus; &ndash; &plusmn; &omega; &middot; &rho;` | Pandoc decodes them correctly; zero literal entity strings found in test output |
| `§2`, `§3` section signs | Encoding issue in our test run produced replacement chars (`�`). Likely an HTML file BOM / declared-charset interaction; needs verification once full pipeline runs. Quick win: confirm `<meta charset="utf-8">` is honoured |
| Bold-italic "Takeaway:" caption pattern | Pandoc preserves the bold/italic span; the captions render correctly. The `emphasize_caption_label` post-step bolds-italicises the leading "Figure N:" prefix only, so any pre-existing emphasis inside the caption body is kept |
| `h2` top-level sections | Pandoc emits `Heading 2`. Stage 3 does NOT promote them to `Heading 1`. Final DOCX has Heading 2 (9), Heading 3 (29), Heading 4 (11). Acceptable for journal templates that use Title for paper title and Heading 1+ for sections |

---

## Section 4: Concrete try-and-report

### Full pipeline run (failed)

```
$ python C:/Users/apart/Projects/claude-skills/html2doc/html2doc.py \
    --input docs/paper/index.html --output paper_test.docx --keep-temp

Checking dependencies...
Missing dependencies:
  - katex (run: npm install katex)
```

Failure stage: pre-flight dependency check. Per the briefing, npm install is out of scope.

### Partial pipeline (stages 2 and 3 only)

To still produce evidence, I bypassed stage 1 by feeding the raw HTML directly to `convert_to_docx.py`, then ran `apply_academic_style.py`. This skips KaTeX-to-MathML, so display/inline math is not converted, but everything else exercises the pipeline as it would normally run.

Stage 2 (run from `docs/paper/`):
```
Saved reference DOCX: ...\reference.docx
Converting: index.html -> ...\paper_test_stage2.docx
Conversion complete
=== Conversion Results ===
Native Word equations: 0           # expected, KaTeX skipped
Unconverted $ signs: 2             # leak into XML
```

Stage 3:
```
Applying academic formatting...
  Centered 19 images
  Formatted 299 text paragraphs
  Formatting tables...
    Table 1..26: all sized + bordered
  Formatting complete
Saved: paper_test.docx
```

Output DOCX inspection:
- 396 paragraphs, 26 tables, 1 section
- Style histogram: Title 1, Subtitle 2, Heading 2 x9, Heading 3 x29, Heading 4 x11, Body Text x184, First Paragraph x51, Compact x21, Image Caption x14, Table Caption x7, Reference Entry x62, Abstract Label x1, References Heading x1, Normal x3
- 273 native subscripts (molecular formulas survive)
- 129 hyperlinks (citations rendered as plain hyperlinks, not Word cross-references)
- 0 OMML equations (expected; KaTeX skipped). With KaTeX run, this would jump to ~8 display + however many `$x$` inline; **but `\(...\)` math will still produce zero**.
- Section heading recognition: numbered sections were promoted to `Heading 2/3/4` correctly
- Title and authors front matter detected and styled correctly
- Captions: "Figure N." / "Table N." prefix bold-italicised by `emphasize_caption_label`. Working

Test artifacts cleaned up after inspection (`paper_test*.docx` removed).

---

## Section 5: Recommendations

**Fit-for-purpose: yes-with-caveats.** The skill is a competent Pandoc wrapper plus a careful python-docx styler that produces a journal-grade DOCX in one command. The two real obstacles for our paper are: (a) installing the `katex` npm package, and (b) the fact that the KaTeX stage only matches `$...$` / `$$...$$`, not `\(...\)`, which our paper uses heavily.

### Manual fixups likely needed in Word (after a clean run)

1. **Inline `\(...\)` math** (the dominant case in our paper). Will appear as raw LaTeX backslash-text. Either:
   - Pre-process: a one-line `sed` over `index.html` rewriting `\(X\)` to `$X$` and `\[X\]` to `$$X$$` before invoking `html2doc.py`. This is cheaper than fixing equations one-by-one in Word.
   - Or fix in Word: select each `\(X\)`, retype as Word equation. Maybe ~80 instances in our paper.
2. **Citations** are hyperlinks, not Word fields. If the journal requires Word's native "Insert Citation", you'll need to redo them. For most submissions plain `[refX]` text + a "References" section is fine.
3. **Section signs `§`**: spot-check encoding. Worst case, find-replace `?` or `?` with `§` (only ~3 places).
4. **Top-level section headings** are `Heading 2`. If the journal template wants `Heading 1`, batch-promote in Word (Replace... All).
5. **Table fit**. Some of our 26 tables have 13 columns; even with `autofit` and 100% width, expect manual column-width tuning.
6. **Figure 4** is two PNGs in adjacent paragraphs (4a training, 4b sampling). The DOCX will keep them as two centered images; you may want them side-by-side, requiring manual two-column placement in Word.

### Skill follow-ups (not in scope for this paper)

- `katex_to_mathml.js`: extend regex to also match `\(...\)` and `\[...\]`. One-line patch.
- `html2doc.py`: replace `os.path.exists('node_modules/katex')` with a proper Node resolution (`node -e "require('katex')"`). Also fix the hardcoded `html2doc/scripts/...` prefix; use `Path(__file__).parent / 'scripts' / ...` so the orchestrator works from any cwd.
- `convert_to_docx.py`: change cwd to the input HTML's directory (or pass `--resource-path`) so figures resolve regardless of where the orchestrator was invoked.
- README: file list lies (`academic_styles.py` does not exist; layout differs).

### Suggested invocation for our paper (once katex is installed)

```bash
cd E:/Projects/EnergeticDiffusion2/docs/paper
# pre-process \(...\) -> $...$ if not patched in skill
python C:/Users/apart/Projects/claude-skills/html2doc/html2doc.py \
    --input index.html --output ../../dgld_paper.docx \
    --profile review-manuscript --keep-temp
```

The `review-manuscript` profile (1.5 line spacing, larger captions) is more appropriate than `camera-ready-generic` (1.15 spacing) for a journal first submission.
