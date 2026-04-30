# NMI Reviewer Committee Report
## DGLD: Domain-Gated Latent Diffusion for the Discovery of Novel Energetic Materials

_Prepared for the Editor, Nature Machine Intelligence_
_Date: 2026-04-30_
_Document covers: `short_paper.html` (streamlined version) + bibliography audit_

---

## Executive Summary (Editor)

Five reviewers assessed the manuscript across orthogonal lenses: AI/ML methodology, materials-science validity, novelty and impact, abstract-level communication, and figures/tables quality. The consensus is that this paper makes a **genuine and clearly scoped contribution** to property-targeted molecular generation with a real DFT validation chain. The core claim (DGLD produces novel CHNO candidates that survive a 4-stage physics validation while strong baselines do not) is supported by the experimental evidence. Three issues currently prevent a recommendation of acceptance: (1) two high-priority missing citations that NMI reviewers in this space will certainly raise; (2) the headline detonation velocity claim (D_KJ,cal = 8.25 km/s, "HMX-class") rests on K-J relative ranking not on absolute DFT or equilibrium-code numbers, and the text acknowledges this only in §7; the abstract and conclusion should carry a single qualifying phrase; (3) the paper's own seed variance analysis (§5.5.4) shows per-condition SD comparable to the across-condition mean differences, which weakens the multi-head guidance claim - this needs one honest sentence in the conclusion. None of these require new experiments. The bibliography audit (separate section below) identified 3 high-priority and 3 medium-priority citation additions.

**Recommendation:** Minor revision.

---

## Reviewer 1: AI / ML Expert

**Summary of findings**

The method is technically sound. The latent-diffusion-in-a-frozen-VAE architecture is well-motivated: training the DDPM only in the LIMO 1024-d latent space sidesteps autoregressive differentiability and reduces per-step cost enough to make multiple guidance heads tractable. The tier-gate ablation (§5.5.6) is the paper's strongest internal result: removing perf-head masking collapses the sampler to degenerate poly-N chains (53.9% keep rate, 0.667 N-fraction, open-chain guanidinium-class top-1), directly demonstrating that label-quality gating is not cosmetic. This result is buried at the end of §5.5 and should be elevated.

**Strengths**

- Tier-gate ablation is a clean, falsifiable experiment; the effect is large and qualitative.
- Self-distillation hard-negative budget comparison (137 vs 918 negatives, -0.10 on gem-tetranitro) closes the loop between training-corpus statistics and sampling-time discriminative failure.
- The Gaussian-latent control (§5.5.2) is the right null baseline for a latent-diffusion claim; 48% keep rate vs 4.6% correctly shows that the diffusion prior concentrates the distribution, not just cleans it.
- Pool-fusion sampling is well-motivated as a diversity lever; the M7 five-lane 100k result is internally consistent.
- Appendix C (Table C.1) is an unusually complete hyperparameter record.

**Weaknesses and specific concerns**

1. **Seed variance on the denoiser itself.** §7 explicitly states: "Each denoiser is one ~6 hr training run; seed variance across the diffusion model itself is not reported." The multi-seed ablation in §5.5.4 varies the _sampling seed_ given a fixed trained checkpoint. For a paper claiming that the method is robust, the absence of denoiser seed variance is a gap reviewers will press on. At minimum, the limitations section should state how large this gap might plausibly be.

2. **GeoLDM is a direct unacknowledged peer.** Xu et al. (ICML 2023, arXiv:2305.01140) introduced geometric latent diffusion for 3D molecule generation using a 1D-in-3D architecture. DGLD uses a 1D latent; GeoLDM uses 3D coordinates. The architectural choice should be explicitly defended in §2.3 ("we use a 1D string latent because energetic-materials property conditioning does not require 3D geometry and avoids conformer preprocessing"). Currently GeoLDM is absent entirely from the manuscript.

3. **Guidance claim vs. seed variance.** Table 5 shows per-condition top-1 composite SD of 0.106-0.184 vs across-condition mean differences of ~0.05-0.25. The §5.5.3 single-seed pool=40k numbers are used to motivate C2 as the production default, but §5.5.4 shows the variance is comparable to the signal. One honest sentence in the conclusion should note this.

4. **CFG weight selection.** CFG weight w=7 appears as a fixed parameter (§4.7) with no ablation. The guidance-weight sensitivity analysis in §4.9 covers the score-model head weights, not CFG w. Since misfire at short trajectories is stated as a motivation for the whole paper, reviewers will ask how sensitive results are to w.

5. **Self-distillation described as "three rounds" in §4.8 but only two budgets compared.** The ablation compares 137 vs 918 negatives, corresponding to round 1 and "round 2 production checkpoint." A reader is left wondering what happened in round 3. Either clarify that the production checkpoint IS round 3 (137 -> intermediate -> 918), or simplify the language.

6. **89/100 of the merged top-100 from the unguided pool** is disclosed in §6 item 5 and §5.3 but not front-loaded. A reader of §5.2 thinks the 12 leads are primarily from the guided run; they are not. The contribution of multi-head guidance to the headline leads should be stated earlier and more precisely (e.g., "L1 originates from the guided Hz-C2 lane; 89/100 of the merged top-100 come from the unguided pool, confirming that the diffusion prior itself is the primary discriminator").

7. **MolMIM venue is wrong.** The paper cites MolMIM as "arXiv:2208.09016, 2023" with no venue. The actual venue is MLDD Workshop at ICLR 2023. This should be corrected in the references.

**Minor issues**

- §4.8 says "three rounds of self-distillation" but §5.5.1 describes "round-1 viability head (137 mined hard negatives) and round-2 production checkpoint (918 mined hard negatives)." Consistent terminology throughout.
- The FiLM conditioning in the denoiser is described in §4.6 and again in §2.2 (citing Perez et al. 2018); the description is slightly different in each location. Consolidate.

---

## Reviewer 2: Materials Science Expert

**Summary of findings**

The paper demonstrates real chemistry awareness: the four-tier label hierarchy correctly distinguishes K-J-derived surrogate labels from experimental measurements, and the 4-stage validation funnel escalating from SMARTS to DFT is appropriate for a first-pass pre-experimental screen. The limitations section (§7) is unusually honest and detailed for a machine-learning paper. Several concerns remain about the materials-science framing of the headline claim.

**Strengths**

- The DFT pipeline (B3LYP/6-31G(d) geometry + wB97X-D3BJ/def2-TZVP single-point + 6-anchor calibration) is a recognised pre-experimental recipe. The 6-anchor LOO calibration uncertainty (±0.078 g/cm³, ±64.6 kJ/mol) is correctly propagated through the K-J sensitivity slopes.
- The SELFIES-GA DFT audit (best novel candidate collapses from D=9.73 to D=6.28 km/s, a 3.5 km/s surrogate artefact) is a strong and important result, correctly framed as a surrogate over-prediction in a chemotype outside the 3D-CNN's training distribution.
- The xTB HOMO-LUMO gap triage as a pre-DFT electronic-stability screen is appropriate.
- The §7 crystal-packing caveat is honest: "a ±5% packing-factor error alone propagates to ±0.15 g/cm³ in density... ±0.4 km/s in D, roughly twice the 6-anchor calibration uncertainty."
- The DNTF 7th-anchor attempt and its honest failure (LOO residual +0.122 g/cm³, LOO-RMS rising from ~0.032 to 0.055 g/cm³) is a nice example of transparent science.

**Weaknesses and specific concerns**

1. **The headline "HMX-class" claim requires one qualifier in the abstract.** D_KJ,cal = 8.25 km/s is correctly stated to be a K-J relative-ranking result, not absolute D. HMX's experimental D is 9.10 km/s; L1's K-J calibrated D is 8.25 km/s, 0.85 km/s below. The §7 disclaimer is thorough, but the abstract says "placing it within the HMX/CL-20 performance band" without this qualifier. Add a parenthetical: "(K-J relative ranking; absolute D requires a thermochemical-equilibrium solver)".

2. **Missing direct competitor: npj Computational Materials 2025.** A paper published in March 2025 (DOI: 10.1038/s41524-025-01845-6) uses RNN + transfer learning + Pareto front + QM validation for energetic molecules, reporting 60 candidates beating CL-20 in heat of explosion. This is the closest published competitor to DGLD and must be cited and contrasted in §2.5. The differentiators are: DGLD uses latent diffusion (not RNN + RL), conditions on D/P/rho (not just heat of explosion), and DFT-validates density rather than just heat of explosion screening.

3. **Missing field review citation: Choi et al. 2023 (PEP).** "Artificial Intelligence Approaches for Energetic Materials by Design: State of the Art, Challenges, and Future Directions," Propellants, Explosives, Pyrotechnics 48(4), 2023. This is the standard field review. Every energetic-materials ML paper submitted to a multidisciplinary journal must cite it in §2.4. Its absence will be noticed by any materials-science reviewer.

4. **Sensitivity prediction limitations.** The hazard/sensitivity head uses h50 BDE-correlated values (Politzer-Murray method, §4.7) rather than experimental impact sensitivity. The §7 limitations acknowledges this but does not quantify how different BDE-correlation estimates are from experiment for the lead compounds' chemotype families. A one-sentence rough estimate (e.g., "Politzer-Murray BDE correlation underestimates h50 by ~20-40% for nitroaromatics vs experiment; nitroisoxazole-class calibration data are absent") would be appropriate.

5. **Retrosynthetic accessibility.** The paper reports AiZynthFinder gives a 4-step productive route for L1 and negative results for L4/L5, with "1/12 USPTO retro hit rate." This is framed as a limitation in §6 item 6. For a paper targeting Nature Machine Intelligence (a general ML venue with materials readers), this is an important result - the failure rate and its cause (drug-domain USPTO templates vs energetics-domain chemistry) should be in §7, not relegated to a parenthetical in §6.

6. **Oxatriazole ring stability of E1.** The §5.3 text correctly notes that the 1,2,3,5-oxatriazole ring has known ring-opening pathways under acidic or high-temperature conditions. This should be carried through more consistently: the E1 co-headline claim in the abstract and conclusion should include "(pending thermal stability confirmation)."

7. **Crystal structure prediction.** The paper never discusses CSP for energetic molecular crystals as a natural future step. Reviewers from the computational energetics community will ask. At minimum, one sentence in §7 should acknowledge that Bondi vdW packing with a fixed factor is a crude approximation and reference a CSP approach (e.g., CCDC Mercury crystal packing studies, or the 2023 Crystal Growth & Design CSP for energetic molecular crystals paper, DOI: 10.1021/acs.cgd.3c00706).

**Minor issues**

- The 6-anchor set (RDX, TATB, HMX, PETN, FOX-7, NTO) is reasonable but FOX-7 is sometimes written as 1,1-diamino-2,2-dinitroethene (DADNE); confirm the anchor is the same compound and state the IUPAC name in Appendix D.
- Table 7b caption states the calibration coefficients but the PETN HOF footnote is extremely long (covers measurement method range); this level of detail belongs in a table note, not a data cell.

---

## Reviewer 3: Wow Factor / Impact Assessment

**Summary of findings**

This paper has a clear, falsifiable headline: "We built a generative model for energetic materials, ran it against three baselines, and ours is the only one that produces novel candidates confirmed by DFT." The 3.5 km/s SELFIES-GA surrogate artefact is the best single number in the paper for memorability - a competitor that looks 37% better than the real world is a striking illustration of why surrogate-only evaluations are insufficient.

**What lands well**

- The 3.5 km/s surrogate collapse is a clear "moment of truth" result that will be cited independently of the full paper.
- 18.3% LSTM memorisation rate, seed-stable across 3 seeds, is a credible data-contamination smoking gun.
- The 4-stage validation funnel (SMARTS -> Pareto -> xTB -> DFT) as a "chemistry-credibility escalator" is a concept that generalises to other molecular domains; this is the methodological contribution that has legs beyond energetics.
- Self-distillation hard negatives mined from the model's own failures is elegant and under-explored.
- Releasing checkpoints and sampling scripts on Zenodo is the right move for reproducibility at NMI.

**What limits impact**

1. **The headline numbers require careful reading to land correctly.** D_KJ,cal = 8.25 km/s is the headline, but HMX's experimental D is 9.10 km/s. The paper qualifies this in §7 and §6, but not in the abstract or conclusion. A first-pass reader sees "HMX/CL-20 performance band" and "D=8.25 km/s" and may not realise that the gap to HMX is ~0.85 km/s, which at K-J sensitivity could narrow or widen depending on the EOS. Add a single qualifying phrase in the abstract.

2. **The two-headline narrative is underexploited.** E1 (4-nitro-1,2,3,5-oxatriazole) has D_KJ,cal = 9.00 km/s from a chemically distinct scaffold family and is mentioned in the abstract and §5.3 but does not appear prominently in the conclusion. Two DFT-confirmed leads from two distinct scaffold families is a stronger story than one.

3. **Figure 1 (novelty vs D three-panel) should be the paper's most memorable graphic.** Currently it shows "7 settings x 3 seeds" = 21 blue points + a handful of baseline markers. The SELFIES-GA DFT collapse should be overlaid explicitly (a red arrow from surrogate D to DFT D for the best novel SELFIES-GA candidate). This would make the main visual argument self-contained.

4. **The connection to practical deployment is underemphasised.** The paper never says explicitly: "L1 is absent from PubChem and from the 66k labelled master, has a 4-step retrosynthetic route via AiZynthFinder, passes xTB electronic stability, and is confirmed as a B3LYP local minimum. These are sufficient conditions to recommend it for synthesis." A short, declarative "synthesis recommendation" sentence in the conclusion would strengthen the practical impact claim.

5. **The domain-gate concept generalises.** The introduction frames DGLD as specific to energetic materials, but the tier-gate recipe (label-trust masking by data source quality) is applicable to any domain with heterogeneous label quality: drug discovery (experimental vs. computational IC50), materials science (DFT vs. force-field lattice energies). A single sentence in §1 para 3 noting this generality would broaden the audience without diluting the specific contribution.

---

## Reviewer 4: Abstract Reader

**Summary of findings**

The abstract (3 paragraphs, ~180 words after the rewrite) is a substantial improvement over the prior version. It has a clear problem-method-result structure and avoids most of the inline caveats. Three targeted improvements remain.

**Paragraph 1 (problem)**

> "Designing novel high-energy-density CHNO compounds is a sparse-label, multi-objective generation problem: of roughly 66 k labelled rows only ~3 k derive from experiment or first-principles DFT..."

- "CHNO compounds" will be opaque to half the NMI readership (AI audience). Consider "carbon-hydrogen-nitrogen-oxygen (CHNO) high-explosive compounds" on first use.
- "sparse-label, multi-objective generation problem" is jargon-first. Consider "generation under sparse, heterogeneous labels with multiple competing design targets" for readability.
- The phrase "classifier-free guidance (CFG) silently misfires under the short sampling trajectories required for molecular latent diffusion" is jargon-dense. A reader must already understand CFG and latent diffusion to follow this. Consider: "standard guidance methods fail silently when the sampling trajectory is as short as molecular generation requires."

**Paragraph 2 (method + headline result)**

> "...12 DFT-confirmed novel leads; the headline lead, trinitro-1,2-isoxazole (L1), reaches ρ_cal = 2.09 ± 0.15 g/cm³ and D_K-J,cal = 8.25 km/s, placing it within the HMX/CL-20 performance band at max-Tanimoto 0.27 to the 65 980-row labelled corpus."

- "placing it within the HMX/CL-20 performance band" needs one qualifier: "(by Kamlet-Jacobs relative ranking; absolute detonation velocity requires a thermochemical equilibrium solver, see §7)". Without this, reviewers will flag the abstract as overclaiming.
- "max-Tanimoto 0.27 to the 65 980-row labelled corpus" - "max-Tanimoto" needs defining for an NMI audience (it is the closest structural neighbour in the training corpus); consider "structurally dissimilar from all 65 980 training molecules (nearest-neighbour Tanimoto 0.27)."

**Paragraph 3 (comparison)**

> "DGLD is the only method with consistent novel productive-quadrant coverage across seeds."

- "productive-quadrant" is undefined in the abstract. Define inline: "simultaneously novel and on-target for performance (the productive quadrant)."
- The Zenodo DOI in the abstract is listed as a draft reservation ("reserved DOI"). This is fine before publication but should be flagged for the production editor to verify the DOI is minted before final typesetting.

**Overall abstract verdict**

Three targeted phrase-level fixes plus the HMX qualifier would make this abstract complete. No structural changes needed; the 3-paragraph form is correct.

---

## Reviewer 5: Figures and Tables

**Summary of findings**

The manuscript has 26+ figures and 15+ tables across the main text and appendix. The visualisation strategy is sound: dumbbell plots for DFT vs surrogate comparison, forest plots for ablation effect sizes, scatter plots for productive-quadrant coverage. Four specific issues need attention.

**Figure-level issues**

1. **Figure 1 (novelty_vs_D.png, three panels) - primary argument figure, needs improvement.**
   - The caption states "7 settings × 3 seeds" = 21 blue DGLD points. On the current figure, the 21 points form a dense cluster; individual settings are not distinguishable. Add a legend with 7 marker shapes or colours for the 7 guidance conditions (Hz-C0 through Hz-C3, SA-C1 through SA-C3).
   - The SELFIES-GA DFT collapse (D_surrogate=9.73 to D_DFT=6.28 km/s) is described in prose and §5.4 but is not shown in Figure 1. A red arrow or annotated point showing the DFT position of the best novel SELFIES-GA candidate would make the central argument self-contained in the figure.
   - The MolMIM 70M gold diamond is labelled in the caption as "novel but at D=7.70 km/s"; it should be annotated on the figure itself.

2. **Figure 21 (DFT dumbbell + N-fraction scatter) - strongest figure in the paper.**
   - This is well-designed. The only change: add a horizontal line at D=8.25 km/s (L1 DFT-calibrated) alongside the existing D=9.0 HMX threshold, so readers can see where the K-J result sits relative to the threshold without cross-referencing text.
   - The N-fraction scatter (right panel) should label RDX, HMX, FOX-7 as named points to anchor the reader.

3. **Figure 19 (lead cards grid, 3x4) - good visual design, minor issue.**
   - The border colour scheme (green = xTB pass, red = fail) is useful but red/green is inaccessible to ~8% of male readers with red-green colour blindness. Consider blue/orange or add a shape indicator.
   - The "?" marker for L20 (80k extension set, no top-100 rank) should be explained in the caption more explicitly.

4. **Figure 22 (baseline forest plot) - clear.**
   - No changes required.

**Table-level issues**

1. **Table 6a (baseline comparison) - footnotes are doing too much work.**
   - The double-footnote (†, ‡) system for SELFIES-GA and REINVENT4 comparability caveats is necessary but the footnote text is long. Consider splitting into a separate "Comparability notes" paragraph after the table.

2. **Table C.1 (hyperparameter dump) - correct but overwhelming.**
   - This table spans many pages. Consider splitting into three sub-tables: (a) LIMO VAE, (b) denoiser + score model, (c) pipeline configuration. Label them C.1a/C.1b/C.1c.

3. **Table 7c (uncertainty propagation) - good table, missing column.**
   - Add a column "K-J formula bias (typical, km/s)" showing the expected K-J underprediction at this compound's N-fraction (from §5.2.2). This makes explicit that the calibration uncertainty and the K-J formula bias are separate error sources of different magnitudes.

4. **Table 3 (Gaussian control) - note column mismatch.**
   - The "Gaussian 40k (N-frac proxy)" row has "n/a" for top-1 D and top-1 composite, which is technically correct but looks like missing data. Replace with a note cell: "Not ranked by D (N-fraction proxy only; see §5.5.2)".

**General figures/tables comment**

The paper's appendix tables (E.1 through E.13) are structured correctly: one metric per table, minimal narrative. The only appendix table needing attention is E.12 (hazard-head AUROC threshold sweep): the table lacks axis labels on the threshold sweep plot that accompanies it (if that figure exists).

---

## Improvement Plan (Priority-Ordered, No New Experiments Required)

| Priority | Action | Location | Effort |
|---|---|---|---|
| P1 | Add qualifier in abstract: "(K-J relative ranking; absolute D requires thermochemical equilibrium solver)" | Abstract para 2 | 1 sentence |
| P1 | Add GeoLDM (Xu et al. ICML 2023) to §2.3 with one sentence on the 1D vs 3D architectural distinction | §2.3 | 2-3 sentences |
| P1 | Add npj CompMat 2025 energetics generation paper as direct competitor in §2.5 and §6 | §2.5, §6 | 3-4 sentences |
| P1 | Add Choi et al. 2023 PEP field review to §2.4 | §2.4 | 1 citation + 1 sentence |
| P2 | Define "productive quadrant" inline in abstract ("simultaneously novel and on-target for performance") | Abstract para 3 | 1 phrase |
| P2 | Define "max-Tanimoto" in abstract ("nearest-neighbour Tanimoto to training corpus") | Abstract para 2 | 1 phrase |
| P2 | Correct MolMIM venue to "MLDD Workshop, ICLR 2023" | References | 1 edit |
| P2 | Add "productive-quadrant" definition on first use in §1 | §1 | 1 sentence |
| P2 | Move retrosynthetic accessibility result (1/12 AiZynthFinder hit rate, drug-domain bias) from §6 parenthetical to §7 with 2 sentences | §6 -> §7 | 2 sentences |
| P2 | Add E1 co-headline "(pending thermal stability confirmation)" qualifier in abstract and conclusion | Abstract, §8 | 1 phrase ×2 |
| P3 | Add one sentence on denoiser seed variance as unquantified gap | §7 | 1 sentence |
| P3 | Clarify that 89/100 merged top-100 come from the unguided pool earlier in §5.2, not buried in §6 | §5.2 or §5.3 | 1 sentence |
| P3 | Clarify self-distillation "three rounds" language to be consistent with the 2-budget comparison | §4.8, §5.5.1 | Terminology fix |
| P3 | Add red/orange DFT-collapse arrow for best novel SELFIES-GA candidate to Figure 1 | Figure 1 | Regenerate figure |
| P3 | Annotate Figure 21 (dumbbell) with D=8.25 km/s line and named anchors (RDX, HMX, FOX-7) | Figure 21 | Minor figure edit |
| P3 | Add domain-gate generalisability sentence in §1 para 3 | §1 | 1 sentence |
| P3 | Add synthesis recommendation for L1 in conclusion (absent from PubChem + AiZynthFinder 4-step route + xTB pass + DFT minimum = sufficient for synthesis recommendation) | §8 | 1-2 sentences |
| P3 | Add one sentence in §7 on CSP as the missing step after DFT density | §7 | 1 sentence + CGD 2023 citation |
| P4 | Fix Figure 19 lead-card colours for red-green colour blindness | Figure 19 | Recolour |
| P4 | Add "CHNO" full form on first use in abstract | Abstract | 1 word change |
| P4 | Split Table C.1 into C.1a/C.1b/C.1c | Appendix C | Formatting |
| P4 | Add "K-J formula bias (typical)" column to Table 7c | Table 7c | 1 column |
| P4 | Fix Gaussian 40k row in Table 3 (replace "n/a" with explanatory note) | Table 3 | 1 cell |
| P4 | Verify Kamlet-Jacobs 1968 page range (23-35 vs 23-55) against AIP source | References | 1 check |
| P4 | Upgrade EMDB citation with proper URL, access date, and institutional provenance | References | 1 citation |

---

## Experiments Required List (Separate from Prose Fixes)

These are additional experiments that would strengthen specific claims. **None are required for a minor revision recommendation**, but each would upgrade a conditional claim to an unconditional one:

| Experiment | Upgrades claim | Estimated effort |
|---|---|---|
| **E1 (highest priority): Thermochemical equilibrium CJ recompute (EXPLO5 or Cantera SDT) on L1 DFT-calibrated inputs** | Converts D_KJ,cal = 8.25 km/s from "K-J relative ranking" to "thermochemical-equilibrium ranking"; resolves the 1.31 km/s 3D-CNN vs K-J gap attribution | ~1 day, existing DFT inputs |
| **E2: Denoiser training seed variance (train 3 checkpoints, report top-1 D distribution)** | Closes the "seed variance on trained model itself is not reported" gap in §7 | ~18 hr RTX 4090 |
| **E3: CFG weight ablation (w in {3, 5, 7, 10} at pool=10k)** | Demonstrates that w=7 is not a lucky pick; closes the "CFG weight selection" gap raised by Reviewer 1 | ~4 hr RTX 4090 |
| **E4: Oxatriazole DFT anchor for E1** | Allows E1 to be reported as a full co-headline lead at the same confidence level as L1 | 1-2 DFT jobs (~6 hr) |
| **E5: xTB BDE for L1 O-N bond (weakest bond screen)** | Provides quantitative thermal stability bound for L1 isoxazole; complements E1 thermal stability concern | ~30 min, commodity CPU |
| **E6: ML crystal density model (Chem. Mater. 2024, DOI:10.1021/acs.chemmater.4c01978) applied to L1 and E1** | Cross-validates Bondi vdW density estimate against an independent ML prediction | ~1 hr, existing SMILES |
| **E7: REINVENT4 run conditioned on D/rho/P targets via UniMol scoring (not N-fraction proxy)** | Tests whether the REINVENT4 vs DGLD gap narrows when REINVENT4 is given the same optimization target; makes the comparison cleaner | ~4 hr, existing setup |

---

## Bibliography Audit Summary

_Conducted by literature review agent on 2026-04-30_

### High priority (reviewers will flag)

| Ref | Issue | Action |
|---|---|---|
| GeoLDM (Xu et al., ICML 2023, arXiv:2305.01140) | Missing entirely; direct architectural peer for latent-space diffusion | Add to §2.3 with 2-sentence contrast |
| npj CompMat 2025 (DOI:10.1038/s41524-025-01845-6) | Missing; closest published competitor (RNN+RL+QM validation for energetics) | Add to §2.5 and §6 as direct competitor |
| Choi et al. 2023, PEP 48(4), DOI:10.1002/prep.202200276 | Missing; standard field review every energetic-materials ML paper must cite | Add to §2.4 |

### Medium priority

| Ref | Issue | Action |
|---|---|---|
| MolMIM (arXiv:2208.09016) | Venue missing: should be "MLDD Workshop, ICLR 2023" | Fix reference |
| CSP for energetic molecular crystals, CGD 2023, DOI:10.1021/acs.cgd.3c00706 | Not cited; crystal structure prediction is the natural next step after DFT density | Add to §7 |
| ML crystal density, Chem. Mater. 2024, DOI:10.1021/acs.chemmater.4c01978 | Not cited; directly overlaps with density surrogate role in DGLD | Add to §2.4 |

### Low priority (verify, not urgent)

| Ref | Issue | Action |
|---|---|---|
| Kamlet-Jacobs 1968, J. Chem. Phys. 48:23 | Page range cited as 23-35; some sources say 23-55; verify against AIP original | Check AIP |
| EMDB citation | No DOI, URL, or access date; grey literature | Add URL + access date |

### Confirmed correct

LIMO (Eckmann et al., PMLR 162:5777, ICML 2022), MOSES (Polykovskiy et al., Front. Pharmacol. 11:565644), Uni-Mol (Zhou et al., ICLR 2023), CFG (Ho and Salimans arXiv:2207.12598), REINVENT4 (Loeffler et al., J. Cheminformatics 16:20, 2024) all verified correct.

---

## Verdict

**Recommendation: Minor revision.**

The paper makes a real contribution. The headline is honestly scoped and the ablations support the method. The changes needed are prose-level qualifiers (1 sentence in the abstract, 1 sentence in §2.3, 1 sentence in §2.5), 3 citation additions, and a figure update. No new experiments are required for the revision itself; the experiments list is for a potential follow-up or response to Reviewer 2 if they push on absolute D values.
