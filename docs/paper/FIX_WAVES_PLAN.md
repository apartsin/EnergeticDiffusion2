# Fix Waves Plan
_All fixes from NMI committee review + bibliography audit + intro audit + figure audit._
_Target: `docs/paper/short_paper.html` (and supporting figure scripts)_

---

## Wave 1 — Prose fixes in `short_paper.html` (foreground, sequential edits)

These are targeted text edits with no external dependencies. Total: ~18 fixes.

### Critical structural fixes
- **§1 line 110**: delete the duplicate "remainder of the paper is organised" paragraph (already exists at line 95).

### Abstract fixes
- Add "carbon-hydrogen-nitrogen-oxygen (CHNO)" full form on first use.
- Add HMX-class K-J qualifier: "(by Kamlet–Jacobs relative ranking; absolute D requires a thermochemical equilibrium solver, see §7)".
- Define "productive quadrant" inline first use: "simultaneously novel and on-target for performance".
- Define "max-Tanimoto": "structurally dissimilar from all 65 980 training molecules (nearest-neighbour Tanimoto 0.27)".
- Add E1 qualifier "(pending thermal stability confirmation)".

### §1 Introduction fixes
- Add stakes sentence to para 1: motivation framed in propellant-mass / civilian gas-generator terms.
- Decompress "CFG silently misfires" jargon: "standard guidance methods fail silently when the generative trajectory is as short as molecular generation requires".
- Add domain-gate generalisability sentence in para 3.
- Move E1 oxatriazole to its own short paragraph between para 3 and the contributions list.
- Define "productive quadrant" inline at first use in §1.

### §5 / §6 / §7 / §8 fixes
- §5.2 or §5.3: state earlier that 89/100 of merged top-100 come from the unguided pool (currently buried in §6).
- §7: add denoiser seed-variance "unquantified gap" sentence.
- §7: add CSP-as-natural-next-step sentence with CGD 2023 citation reference.
- §7: move retrosynthesis 1/12 hit rate from §6 parenthetical here, expand to 2 sentences.
- §8: add E1 thermal stability qualifier.
- §8: add synthesis recommendation for L1 (absent from PubChem + AiZynthFinder 4-step + xTB pass + DFT minimum = recommend for synthesis).

### Other
- §4.8 / §5.5.1: clarify "three rounds" of self-distillation language to be consistent with the budget-137 / budget-918 comparison.
- Fix Gaussian 40k row in Table 3: replace "n/a" cells with explanatory notes.

---

## Wave 2 — Bibliography updates (background agent)

References to add/fix in the body and the `<ol class="refs">` reference list.

### Citations to add
1. **GeoLDM** — Xu et al., ICML 2023, arXiv:2305.01140. Add to §2.3 with a 2-sentence contrast.
2. **npj CompMat 2025** — DOI:10.1038/s41524-025-01845-6. Add to §2.5 and §6 as direct competitor.
3. **Choi et al. 2023** — Propellants, Explosives, Pyrotechnics 48(4), DOI:10.1002/prep.202200276. Add to §2.4.
4. **CSP for energetic molecular crystals** — CGD 2023, DOI:10.1021/acs.cgd.3c00706. Add to §7.
5. **ML crystal density** — Chem. Mater. 2024, DOI:10.1021/acs.chemmater.4c01978. Add to §2.4.

### Citations to fix
6. **MolMIM venue** — change to "MLDD Workshop, ICLR 2023".
7. **EMDB** — add URL, access date, institutional provenance.
8. **Kamlet–Jacobs 1968** — verify page range (23–35 vs 23–55) against AIP source.

---

## Wave 3 — Table updates (foreground after Wave 1)

- Update Figure 23 `<figcaption>` in HTML to reflect the regenerated PNG (now plots SELFIES-GA, REINVENT 4, MolMIM, LSTM as scatter points; spreads L9–L20 along Pareto floor).
- Add "K-J formula bias (typical)" column to Table 7c.
- Split Table C.1 into C.1a (LIMO VAE), C.1b (denoiser + score model), C.1c (pipeline configuration) — optional, defer if low impact.

---

## Wave 4 — Figure regeneration (foreground; scripts already exist)

- **`plot_baseline_forest.py`** (Figure 22): add REINVENT 4 and SELFIES-GA rows to the forest plot ordering. Sources: Table 6a values; existing `m6_post.json` style data files (or hardcode the values from the table).
- **`plot_novelty_scatter.py`** (Figure 1): add SELFIES-GA DFT-collapse annotation (D=9.73 surrogate -> D=6.28 DFT) and REINVENT 4 marker (D=9.02, novel).
- Update both figure captions in HTML.

---

## Execution order

1. **Wave 1** runs first (single-session, sequential edits) — produces clean prose baseline.
2. **Wave 2** runs as a background agent in parallel — research and bibliography wiring.
3. **Wave 3** runs after Wave 1 — minor caption / table touchups.
4. **Wave 4** runs after Wave 1 — figure script updates + regeneration.

Each wave commits to the same branch in a single commit on completion.
