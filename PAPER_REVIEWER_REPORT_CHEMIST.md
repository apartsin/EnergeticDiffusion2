# Referee Report: Domain-Gated Latent Diffusion for Energetic-Materials Discovery

**Reviewer persona:** Senior energetic-materials chemist (synthesis, DSC/TGA, h50/ESD, B3LYP/G2/CBS-QB3 thermochemistry, EXPLO5/Cheetah).

---

## Section 1: Summary in chemist's words (under 200 words)

The authors train a small conditional latent diffusion model over a frozen MolMIM VAE space, conditioned on four bulk explosive descriptors (crystal density, heat of formation, detonation velocity D, detonation pressure P) and gated at sample-time by per-property "score heads" (viability, sensitivity, hazard, performance). They then rerank with SA/SCScore, an energetics-SMARTS chemistry filter, and a strengthened hazard SMARTS layer. The top-100 merged pool is triaged by GFN2-xTB (HOMO-LUMO gap gate at 1.5 eV), and twelve "chem-pass" leads are subjected to gas-phase DFT (B3LYP/6-31G(d) geometry, omega-B97X-D3BJ/def2-TZVP single-point, atomization-energy HOF, Bondi van-der-Waals density at packing 0.69), with a 2-anchor (RDX, TATB) linear calibration. Three leads (L1 trinitro-1,2-isoxazole, L4 tetrazoline-N-nitramine, L5 acyl-oxime nitrate) are submitted to AiZynthFinder; only L1 returns routes. The headline candidate is L1 at predicted rho=2.00 g/cm3, D=9.56 km/s, P=40.5 GPa. No experimental data, no h50, no DSC, no thermal-decomposition prediction, no EXPLO5/Cheetah recompute is reported.

## Section 2: Strengths from a chemistry perspective

1. **Honest layered surrogate disclosure.** §5.5 and Appendix D are unusually candid about the systematic atomization-energy bias of B3LYP/6-31G(d), the Bondi packing-coefficient ambiguity, and the well-known Kamlet-Jacobs (K-J) breakdown at high N-fraction. The population-level reproduction of the K-J failure mode (Pearson r=+0.43 across 575 Tier-A rows) is genuine evidence of regime awareness, not common in ML-for-energetics papers.

2. **Hard-negative cheat-mining and SMARTS hazard layer.** The self-distillation loop that demotes gem-tetranitro-on-small-rings, polyazene chains, trinitromethane and N-nitroimine motifs is a real chemistry-side contribution. The literature-grounded sensitivity head fitted to 306 Huang-Massa h50 pairs (Pearson r=+0.71 on 30 held-out rows) is the right instinct, even if the dataset is small.

3. **xTB HOMO-LUMO gap gate as a triage signal.** The 1.5 eV cut-off applied after GFN2-xTB tight optimisation correctly demotes the spiro-tetranitro chain candidates (ranks 4 and 6 of the original top-8). This is consistent with my experience: small-gap CHNO species with cumulated nitros tend to be open-shell-like and impact-sensitive.

4. **Anchor probe set is reasonable.** Anchoring against RDX, HMX, TNT, FOX-7, PETN, TATB, NTO during score-model training is the right anchor list for a CHNO generator and reflects familiarity with the field.

5. **Retrosynthesis attempt and honest negative result.** Acknowledging that AiZynthFinder's USPTO+ZINC stack is drug-discovery-biased and returns null on L4/L5 is more honest than most ML papers manage; the authors do not claim L4/L5 are unsynthesisable.

## Section 3: Major chemistry concerns

### 3.1 Atomization-energy HOF at B3LYP/6-31G(d) is not fit for absolute purpose

(a) The raw atomization-energy HOFs in `m2_summary.json` are catastrophic in absolute terms: RDX +15879 kJ/mol, TATB +17559 kJ/mol, L1 +7305 kJ/mol, L9 +17066 kJ/mol. The experimental RDX HOF is approximately +66 to +70 kJ/mol (the paper's own glossary cites +70). The 16763 kJ/mol calibration intercept is doing all the work, and it is forced through only two anchors with zero residual diagnostic. In my experience, atomization-energy HOFs at B3LYP/6-31G(d) without thermal correction, basis-set extrapolation, and atomic-state spin-orbit corrections are not usable as an absolute property predictor for a paper that ranks candidates against HMX/CL-20.

(b) What is missing: a composite-method (G4, CBS-QB3, W1) recompute on at least the top three leads; an isodesmic or atom-equivalent scheme that absorbs the systematic per-atom bias internally rather than via a 2-anchor regression; thermal correction to 298 K (the paper acknowledges this but defers it).

(c) What would convince me: G4 or CBS-QB3 HOFs at 298 K for L1, L4, L5, RDX, TATB. If the calibrated DFT HOFs in Table D.1 land within roughly 50 kJ/mol of the composite-method values, the methodology is defensible as a screening tool; if the disagreement is hundreds of kJ/mol (likely for the strained heterocycles), the headline numbers must be downgraded to ranking-grade only and the predicted D values stripped from the abstract.

### 3.2 Kamlet-Jacobs in the high-N CHNO regime is not a publishable absolute-property predictor

(a) The paper recomputes K-J on the 2-anchor-calibrated rho and HOF and obtains D=13.27 km/s for L4 (a tetrazoline-N-nitramine), which exceeds CL-20 by approximately 4 km/s and lies above any experimentally measured CHNO explosive. For L1, L5, L19, the calibrated thermochemistry crosses the K-J product-distribution sign convention and the velocity is reported as undefined. The §5.5 attribution to "regime breakdown" is correct in direction but does not absolve the paper from publishing the 13.27 km/s number in Table D.2.

(b) What is missing: an EXPLO5 or Cheetah-2 thermochemical-equilibrium recompute on the same calibrated rho and HOF, with full Chapman-Jouguet product-distribution treatment. The K-J formula has been understood to fail above f_N ~ 0.4 since the 1990s; modern energetic-materials literature uses K-J only as a sanity check.

(c) What would convince me: EXPLO5/Cheetah D and P for at least L1, L4, L5 alongside the K-J columns of Table D.2. If EXPLO5 places L1 at 9.0-9.5 km/s with positive Q, the headline claim survives; if the K-J undefined branch is replicated by EXPLO5 (i.e. the calibrated thermochemistry really is unphysical), the lead is not publishable as an HMX-class candidate.

### 3.3 Bondi van-der-Waals density with packing 0.69, calibrated by 4.275*rho - 5.172, is alarming

(a) The 2-anchor density slope of 4.275 is far outside the range I would expect from a Bondi vdw + packing approach. Real CHNO crystal densities span roughly 1.6-2.0 g/cm3; vdw-volume packed densities at fixed coefficient 0.69 typically span 1.5-1.9 g/cm3; the slope of a sane regression should sit near 1.0, not 4.3. A slope this large means tiny noise in raw rho_DFT (which has its own ~0.1 g/cm3 packing-coefficient uncertainty band, as the paper notes) is amplified onto the final calibrated density. L1 raw 1.80 -> calibrated 2.53 g/cm3 is implausible for an aromatic CHNO with five heavy substituents on a five-ring; CL-20 sits at 2.04, FOX-7 at 1.89, and 2.53 would put L1 above any known CHNO solid.

(b) What is missing: a crystal-structure prediction (CSP) workflow such as USPEX, GRACE, or even a simple Polymorph Predictor sweep for the top three leads; a multi-anchor (n>=6) density calibration that separates the Bondi additive bias from the packing-coefficient bias; an explicit estimate of crystal-density uncertainty per lead.

(c) What would convince me: any one of (i) a CSP-derived rho estimate for L1, (ii) a Hofmann-Stine group-additivity rho estimate as an independent check, (iii) a 6-anchor calibration that yields a slope below 1.5 with a standard error.

### 3.4 No impact, friction, ESD, or thermal-decomposition data or prediction for the leads

(a) The paper has a literature-grounded sensitivity head (h50-trained) used as a guidance signal during sampling, but no per-lead h50, ESD, friction or DSC-onset prediction is reported in the results tables. The xTB HOMO-LUMO gap is admitted by the authors to be a weak proxy. For a chemist asked to invest a graduate student's time in synthesising L1, the absence of any sensitivity prediction is a deal-breaker.

(b) What is missing: a per-lead inference of the literature-grounded sensitivity head (a number per lead, in cm of h50 or in [0,1] sigmoid units); a Mathieu-style impact-sensitivity model, a Politzer ESP-based sensitivity correlation, or a Rice-Hare-Byrd thermal-decomposition estimate.

(c) What would convince me: a Table 5b extension where each chem-pass lead carries a predicted h50 (with uncertainty), a predicted DSC onset, and a Mathieu impact-sensitivity score. The trinitro-isoxazole L1 should be flagged or cleared by this layer before any synthesis recommendation.

### 3.5 Trinitro-1,2-isoxazole (L1) specifically

(a) 3,4,5-trinitro-1,2-isoxazole is not a novel scaffold. The aminodinitroisoxazole precursor invoked in the AiZynth route is from the Herve group's published work on isoxazole-based energetics (Herve et al., 2010-2013, J. Energ. Mater., New J. Chem.). 3,4-dinitro-1,2,5-oxadiazole and isoxazole-fused energetics have been studied for over fifteen years; the chemotype is known to be moderately impact-sensitive due to the weak N-O ring bond.

(b) The predicted rho=2.00 g/cm3 from the 3D-CNN is plausible at the upper end for an aromatic five-ring CHNO with three nitros (compare 3,4,5-trinitropyrazole at ~1.87, dinitrofurazan ~1.92). The calibrated rho=2.53 g/cm3 from Table D.1 is not plausible for any aromatic CHNO; it would beat CL-20 by 0.5 g/cm3, and there is no precedent for this in the experimental literature.

(c) The novelty claim of this paper rests partly on the productive-quadrant rate. Given that L1 is essentially an extension of a well-known Herve-group scaffold, the "novel" framing is overstated; the contribution should be presented as rediscovery of a sensible chemotype, not as a new energetic candidate.

### 3.6 Retrosynthetic auditing: L1 route quality and L4/L5 nullity

(a) The proposed L1 route is: tert-butanol + 4,5-dinitro-1,2-isoxazole-3-carboxylic acid -> Curtius via DPPA -> Boc-protected aminodinitroisoxazole -> deprotection -> ring nitration. The final mixed-acid nitration of an electron-poor amine on an already di-nitrated isoxazole is harsh: HNO3/oleum at low temperature is the standard, but the N-O bond of the isoxazole ring is known to cleave under prolonged exposure to fuming acid (Sheremetev et al., Russ. Chem. Bull.). The route is plausible but the final step is the dangerous one and is exactly where bench yield typically collapses.

(b) The 4,5-dinitro-1,2-isoxazole-3-carboxylic acid starting material is listed as "ZINC in-stock" which is misleading; the ZINC catalog records virtual catalog availability, not bench availability. Most of these are not actually in stock at common suppliers (Sigma, TCI, AK Scientific) and would require custom synthesis.

(c) For L4 and L5, the AiZynth null result is correctly attributed to USPTO drug-discovery bias, but the authors should not present this as "uninformative." For an N-nitramine ring closure, the expected reaction is N2O5 in CH2Cl2 with a strong dehydrating agent or NO2BF4; for an O-nitrate of an oxime, this is typically Ac2O/HNO3. These are textbook energetics-domain reactions; their absence from the USPTO templates is a known limitation, but their presence in the actual literature is not in dispute. A targeted literature search by the authors (rather than a USPTO-template MCTS) would have produced credible routes for L4 and L5.

## Section 4: Minor chemistry issues

- The glossary (line 901) cites RDX D=8.75 km/s, which refers to the steady-state rate at rho=1.80 g/cm3 (the standard, fine; correct).
- The glossary entry for atomization-energy HOF cites 5-15 kJ/mol per atom for B3LYP/6-31G(d). This is the correct GMTKN55 band for general organics, but per-atom errors on strained nitrogen-rich heterocycles run higher (15-25 kJ/mol per atom for tetrazoles per Goerigk's CHNO subset).
- The functional name "wB97X-D3BJ" is not a stock combination in most quantum chemistry packages; the canonical wB97X-D includes its own short-range damping. Confirm that the gpu4pyscf implementation is wB97X-V-D3BJ or wB97X-D3(BJ); the former and latter are different functionals.
- Table D.1 reports L9 minimum real frequency as "0 cm-1." Either there is a true zero-frequency mode (in which case the geometry is not a minimum and the result is wrong) or this is a rounding/printing artefact. This must be fixed.
- L20 also reports min real freq = 0 cm-1, same issue.
- "tetrazoline" is non-standard; the canonical IUPAC for the L4 SMILES (O=[N+]([O-])C1=NCN=NN1[N+](=O)[O-]) is a 1H-tetrazoline-N-nitramine but the structure as drawn has unusual valence; please verify.
- Table D.2 reports raw-DFT K-J D for RDX as 12.14 km/s vs experimental 8.75. This is a 38% over-prediction at the anchor itself; the calibration is therefore being asked to absorb a factor-of-1.4 error at RDX. The paper should explicitly state that this is a Q-without-product-enthalpy implementation of K-J (the "open form"), which is non-standard.
- The paper cites Casey et al. 2020 as a B3LYP/6-31G* benchmark (corrected to 6-31G(d) elsewhere). 6-31G* and 6-31G(d) are formally identical but the asterisk notation is older; consistency in the manuscript would help.
- Trinitro-1,2-isoxazole is misnamed; at C3,C4,C5 substitution it is 3,4,5-trinitro-1,2-isoxazole or trinitro-isoxazole; the SMILES is unambiguous but the body text alternates between "trinitro-isoxazole" and "trinitro-1,2-isoxazole."
- The "h50=40 cm pivot" choice for the sigmoid sensitivity proxy is reasonable but should cite the Storm-Stine threshold of 40 cm explicitly.

## Section 5: Specific questions for the authors

1. Provide G4 or CBS-QB3 HOFs at 298 K for L1, L4, L5, RDX, and TATB. Report the difference between composite-method HOF and the calibrated B3LYP/omega-B97X-D3BJ atomization-energy HOF as a per-molecule residual.

2. Provide an EXPLO5 or Cheetah-2 detonation recompute on the calibrated rho and HOF for L1, L4, L5. Report D and P from the thermochemical-equilibrium code alongside the K-J columns of Table D.2.

3. Justify the Bondi-vdw + 0.69-packing density estimator quantitatively: report the slope and standard error of an n>=6 anchor regression (HMX, PETN, FOX-7, NTO added). If the slope remains above 2.0, replace the method.

4. Report a per-lead predicted h50 from the literature-grounded sensitivity head, with the train/test residual band as uncertainty. Also report Mathieu impact-sensitivity scores as an independent estimate.

5. Confirm that L9 and L20 minimum real frequencies are not actually zero. If they are, the geometry is not a minimum and these leads must be removed.

6. Comment on the prior literature on 3,4,5-trinitro-1,2-isoxazole and its precursor 3-amino-4,5-dinitroisoxazole (Herve, Sheremetev). Reframe L1 as a "rediscovery validation" rather than a novel candidate.

7. Specify the precise functional implemented in gpu4pyscf: wB97X-D, wB97X-D3, wB97X-D3BJ, or wB97X-V-D3BJ. These are different functionals with different parameterisations.

8. Provide at least an order-of-magnitude DSC-onset or thermal-decomposition prediction (Rice-Hare-Byrd, Mathieu, or simple BDE of weakest bond) for the top three leads.

9. Repeat AiZynthFinder for L4 and L5 with an energetics-domain template set (e.g., ASKCOS-energetics or a hand-curated nitramine/nitrate template list), or at minimum cite hand-drawn forward routes from the energetics literature.

10. Provide single-crystal density via crystal-structure prediction (Polymorph Predictor, GRACE, USPEX) or via a Stine-Hofmann group-additivity estimate for the top three leads, as an independent check on the Bondi-packed value.

## Section 6: Suggested experimental and computational follow-ups

1. **Composite-method HOF.** Run G4 or CBS-QB3 on L1, L4, L5, plus RDX and TATB, with thermal correction to 298 K. Approximately 6-24 CPU-hours per molecule on standard hardware. This is the single most-impactful chemistry follow-up and would either rescue or kill the absolute HOF claim.

2. **EXPLO5 or Cheetah-2 recompute.** Both codes accept calibrated rho and HOF and output D, P, T_CJ with full product-distribution thermochemistry. Free academic licences are available. Run on the same five compounds. If L1 lands at 9.0-9.5 km/s with positive Q, the headline survives; if it falls to 7-8 km/s, the abstract claim must be revised.

3. **Per-lead sensitivity prediction.** Run the literature-grounded h50 head as inference on every chem-pass lead and report the number. Add a Mathieu sensitivity score (open-source code is available). Add a Politzer ESP imbalance index from the existing DFT geometries (free, ~5 minutes per molecule). Three independent sensitivity proxies will give a credible safety triage.

4. **Sub-mg synthesis of L1.** A real chemistry collaboration would do this on roughly 100 mg scale with proper containment behind an HPDE shield, characterise by NMR, IR, mp, DSC, and impact (BAM fall hammer). The 4-step Herve route is well-precedented for the precursor, and the final mixed-acid nitration is the single new step. This is the experiment that would convert the paper from a methodology contribution to a discovery contribution.

5. **Multi-anchor density calibration.** Include HMX (1.91), PETN (1.77), FOX-7 (1.89), NTO (1.91), and CL-20 (2.04) as additional anchors. Refit the Bondi-packed density transform with at least n=7 points and report slope, intercept, residual standard deviation. If the slope drops to ~1.0-1.5, the methodology is defensible.

## Section 7: Decision recommendation to the editor

**Major revision.**

The ML methodology contribution (sample-time gating via per-property score heads in a frozen-VAE latent space, with hard-negative cheat mining and a literature-grounded h50 sensitivity head) is genuinely interesting and is likely publishable in a methods venue. The chemistry case for the top lead is, in its current form, not publishable as an absolute-property prediction. Specifically, §3.1 (atomization-energy HOF without composite-method validation), §3.2 (K-J at high N-fraction without EXPLO5/Cheetah recompute), §3.3 (Bondi-packed density with a slope-of-4.275 calibration), and §3.4 (no per-lead sensitivity prediction in the results) collectively render the headline number "L1 at D=9.56 km/s, rho=2.00 g/cm3" indefensible without a substantial revision. The §3.5 reframing of L1 as a Herve-group rediscovery and the §3.6 honest treatment of the AiZynth null on L4/L5 are achievable in revision. With the items in §6 addressed, particularly EXPLO5/Cheetah and composite-method HOF on the top three leads, this paper would be a credible NMI-level submission. As submitted, it overstates the chemistry confidence, and a chemist asked to synthesise L1 on the strength of these numbers would have to spend an additional week reproducing the thermochemistry before committing.

## Section 8: Confidential comments to the editor

This paper is more chemistry-honest than the median ML-for-energetics submission I have seen in the past three years; the §5.5 acknowledgement of K-J regime failure and the Appendix D bias-band discussion are unusual and praiseworthy. However, the absolute-number claims still outrun the chemistry that backs them, in a way that is typical of the genre. I would not reject outright; the methodology and the honesty of the limitations sections both deserve a chance at revision. If the authors come back with EXPLO5 numbers and a composite-method HOF for the top three leads, this becomes a publishable paper.
