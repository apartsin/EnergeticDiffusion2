# DGLD Paper Rewrite Plan
_Target output: `short_paper.htm` — streamlined version of `index.html`_

---

## Diagnosis

Four structural problems compound each other:

1. **Nested parenthetical caveats everywhere** — every finding embeds 2-3 qualifying numbers, cross-references, and "see §X" pointers inline, making sentences unreadable.
2. **Lab-notebook Q/F structure** — §5.x sections lead with a "Question:" box, narrate the investigation, then deliver a "Finding:" box. Reads as a progress report, not a results section.
3. **Development history in the method** — round-1/round-2, version names, retried runs tell the story of *making* the system rather than describing the system.
4. **Abstract overloaded** — 5 long paragraphs (~450 words) with inline error bars, compound-by-compound numbers, and caveats that belong in §5.

---

## Target structure

| Section | Current | Proposed |
|---------|---------|---------|
| Abstract | 5 paragraphs, ~450 words | 3 paragraphs, ~180 words |
| §1 Intro | ~8 paragraphs | 4 paragraphs |
| §2 Related work | keep | trim 20% |
| §3 Data / problem | keep | trim |
| §4 Method | keep structure, has dev history | same subsections, clean prose |
| §5.1 Setup | keep | keep |
| §5.2 DFT validation | Q/F format, 3 sub-investigations | results-first, 2 paragraphs + tables |
| §5.3 Scaffold diversity | Q/F + E2 audit + stage-4 rescue | 1 paragraph + table, audit to appendix |
| §5.4 Baselines | ~900-word per-method prose | 1 table + 2 paragraphs |
| §5.5 Ablations | 4x Q/F sections | 1 consolidated paragraph + table refs |
| §6 Limitations | keep | trim |
| §7 Uncertainty | keep | trim |
| §8 Conclusion | 3 paragraphs | 2 paragraphs |
| Appendix A-D | reference material | keep, trim |
| Appendix E (E.1-E.11) | mixed results + narration | tables only, strip narrative, add E.12 |
| Appendix F | baseline examples prose | keep table, cut prose to 2 sentences |

---

## Section-by-section changes

### Abstract (target: 3 paragraphs, ~180 words)
- **Para 1** (problem): sparse-label multi-objective generation for CHNO energetic materials; surrogate miscalibration and silent CFG degradation.
- **Para 2** (method + result): DGLD = domain-gated diffusion + multi-task score model + four-stage chemistry funnel; 12 DFT-confirmed leads; L1 at rho_cal=2.09 g/cm3, D_KJ_cal=8.25 km/s, novel (max-Tanimoto 0.27).
- **Para 3** (comparison): SELFIES-GA best novel candidate collapses from D_surrogate=9.73 to D_DFT=6.28 km/s (3.5 km/s surrogate artefact); SMILES-LSTM memorisation 18.3%; REINVENT reaches D=9.02 km/s. Code on Zenodo.
- **Remove**: all inline caveats, parenthetical gap-explanations, "(L3 and L16 are the same compound, see §5.2.1)", oxygen-balance regime notes, surrogate-DFT gap explanation — those move to §5.

### §1 Introduction (target: 4 paragraphs)
- Para 1: why energetic materials discovery is hard and slow.
- Para 2: four method families and their core limitations (2 sentences per family max).
- Para 3: what DGLD does differently (domain gate, guidance, funnel).
- Para 4: paper outline (one sentence per section).
- **Remove**: the paragraph explaining Kamlet-Jacobs regime limits (belongs in §5.2).

### §4 Method
- Remove all development-round narration (v2 model, v3d model, round-1, round-2 asides).
- Keep the final system description cleanly.
- §4.7 score-model training: state final hyperparameters only (peak LR 2e-4, batch 1024, ~40k steps, EMA 0.999).

### §5.2 DFT validation
- Remove all "Question." / "Finding." boxes.
- Lead with the result: "Table 7 lists DFT-calibrated density and detonation velocity for all 12 chem-pass leads. Surrogate rank order is preserved. L1 and E1 are headline leads."
- §5.2.2 calibration: 2 paragraphs max; strip per-anchor narration.
- Move CJ cross-check detail to Appendix D.

### §5.3 Scaffold diversity (E-set)
- Replace "stage-4 rescue", "E2 SMARTS audit", and "reading the table" sub-paragraphs with:
  - One short paragraph: 10 scaffold-distinct leads, 6 chemotype families, E1 as co-headline.
  - One sentence: E2 and E9 as filter-validation examples.
  - Then the table.
- Move the full E2 SMARTS audit paragraph to **Appendix E.12** (new subsection).

### §5.4 Baselines
- Compress ~900-word per-method prose to 3-4 sentences per method.
- Keep SELFIES-GA DFT audit finding in 3 sentences (key competitive differentiator).
- Move exact REINVENT per-seed memorisation counts ("107/245 379", "100/244 690") to Appendix F prose note.
- Reference the comparison table; let the table do the work.

### §5.5 Ablations
- Replace 4x Q/F ablation sections with one paragraph covering the three most important results: CFG schedule, self-distillation budget, domain gate.
- Reference Appendix E tables for full numerics.

### §8 Conclusion (target: 2 paragraphs)
- Para 1 (3-4 sentences): what DGLD showed — method, lead quality, comparison outcome.
- Para 2 (3 sentences): future work — condense 5 bullets to 3.
- **Remove entirely**: "Energetic-materials discovery has historically advanced one molecule per decade, with HMX (1942)..." paragraph.

### Appendix E
- Keep all tables exactly as-is.
- Replace multi-paragraph narrative intros with 1-2 sentence intros per subsection.
- **Add E.12**: E2 SMARTS audit text moved from §5.3.

### Appendix F
- Keep the examples table.
- Cut multi-paragraph prose commentary; replace with 2-sentence intro.

---

## Things to remove entirely

| Item | Location |
|------|----------|
| Internal file paths (m6_post.json, m2_bundle, moses_multiseed_summary.json, m1_sweep_C*) | Throughout — should be done already |
| "runpod RTX 4090", "vast.ai" cloud-compute specifics | E.11 — should be done already |
| "One container hit a CUSOLVER fault..." | §5.3 — should be done already |
| Development round names (v2 model, v3d model, v3f production run) | §4.7, §5.5.1 — should be done already |
| "(L3 and L16 are the same compound, see §5.2.1)" | Abstract |
| Exact REINVENT per-seed memorisation counts | §5.4 |
| Surrogate-DFT gap explanation beyond first occurrence | Multiple |
| HMX/CL-20 historical context paragraph in conclusion | §8 |
| K-J regime limit explanation paragraph in introduction | §1 |

---

## Estimated length reduction

| Part | Current (est. words) | Proposed |
|------|---------------------|---------|
| Abstract | ~450 | ~180 |
| §1 Introduction | ~700 | ~450 |
| §4 Method | ~2 000 | ~1 600 |
| §5 Experiments | ~3 500 | ~2 000 |
| §8 Conclusion | ~600 | ~300 |
| Appendix E narrative | ~1 500 | ~400 |
| **Total** | **~8 750** | **~4 930** |

~45% trim. All removed content is either cut (development narration) or moved to appendix (audit details, per-seed numbers). Scientific claims and tables are preserved in full.
