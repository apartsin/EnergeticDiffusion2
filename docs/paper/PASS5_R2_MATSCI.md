# Pass-5 Reviewer 2 (Materials Science) — short_paper.html

Scope: §2.4-2.5, §3, §5.1-5.4, §6, §7, §8, Appendix C (incl. new C.9-C.13), Appendix D.14.
Source-of-truth tag: `post_5_rewrite` (commit 5e2492e).

---

## 1. Chemistry plausibility

- **L1 SMILES, formula, oxygen balance.** L1 is "3,4,5-trinitro-1,2-isoxazole", C3N4O7 (no H, since all three ring carbons carry NO2). Stoichiometric OB to CO2 with a=3, b=0, c=4, d=7 is OB% = 1600·(d - 2a - b/2)/MW = 1600·(7 - 6)/232 ≈ +6.9%. The paper reports OB ≈ +8% (line 393) and HOF_cal = +22.9 kJ/mol with N-fraction f_N = 4/(3+0+4+7) = 0.286, matching the quoted f_N ≈ 0.29 in §5.3. Internally consistent.
- **L1 chemistry caveat.** A 1,2-isoxazole bearing C-NO2 at C3, C4 *and* C5 is an extremely electron-deficient aromatic; it is not unreasonable for §6 to flag it as a "chemotype rediscovery within the polynitroisoxazole family" (Sabatini 2018). The paper concedes this honestly.
- **E1 (4-nitro-1,2,3,5-oxatriazole, C1N4O3).** Plausible CHNO motif; the §7/§5.4 caveat about thermal/Lewis-acid ring-opening pathways is appropriately explicit. f_N = 4/8 = 0.50 places E1 in the high-N regime where Table C.4 reports the K-J residual sign-flips to over-prediction (+0.54 km/s in the [0.55-1.00] bin); E1's f_N at 0.50 actually sits in the [0.40-0.55) bin (mean residual -1.60 km/s, K-J under-predicts), which is *not flagged* as a caveat on E1's headline D = 9.00 km/s. This is mildly favourable bias toward E1.
- **E2 (CH2N4O6, gem-dinitro nitramine, OB +28.9%).** Correctly flagged as upper-bound only (D.13). Arithmetic: OB% = 1600·(6 - 2 - 1)/(166) = 1600·3/166 ≈ +28.9% — confirmed.
- **E9 (bare 1H-tetrazole CH2N4, no NO2).** OB ≪ -200% — K-J undefined is correct.
- **L4 closure arithmetic (Table C.3).** ΔHOF closure = -2.27 km/s ÷ (-0.22/100 km/s per kJ/mol) = +1032 kJ/mol. This is correct, and the sign is right (D rises with lower (more negative) HOF when ∂D/∂HOF < 0, but the table is correctly framed as "raise HOF to *close* the residual" because the K-J D is below the 3D-CNN). Plausibility argument (above CL-20) is sound.

## 2. DFT methodology and calibration

- **B3LYP/6-31G(d) opt + ωB97X-D3BJ/def2-TZVP single-point** is a defensible recipe for CHNO with the GMTKN55 calibration cited in §2.8. Acceptable.
- **No explicit thermal correction discussion.** HOF_cal is reported as a single value; no statement on whether ZPE / 298K thermal corrections are included or whether they are absorbed into the linear -206.7 kJ/mol intercept. For a paper-grade HOF claim this should be explicit. Minor (the 6-anchor calibration empirically absorbs systematic bias; LOO RMS ±64.6 kJ/mol is large enough to mask the issue).
- **PETN HOF anchor (-538 kJ/mol, condensed-phase, Dobratz 1981).** The footnote (line 407) acknowledges the literature range -504 to -539 and frames it as within the LOO RMS. This is honest, but PETN is the *outlier* anchor (its own LOO residual is only -0.06 km/s vs experiment, while HMX is -1.58 km/s — see Table C.2). PETN is doing more anchor work than the paper acknowledges; one anchor's quality flatters the headline.
- **Bondi-vdW packing factor 0.69 fixed.** §7 covers this honestly. The C.9 bracket [pk = 0.65, 0.69, 0.72] gives ρ ∈ [1.69, 1.87] for L1 against the calibrated 2.09 — a 14% offset reproduced by the slope 1.392, which is consistent. **The 6-anchor calibration is doing the heavy lifting on density**, not the DFT itself, and the paper says so.
- **6-anchor reweighting from cubane to flat aromatics.** All six anchors (RDX, HMX, TATB, PETN, FOX-7, NTO) are nitro-decorated saturated heterocycles or planar aromatics; *none* are 1,2-isoxazoles or oxatriazoles. The headline ρ_cal for L1 and E1 is therefore extrapolative; this is acknowledged for E1 in §7 but only partially for L1 (the §7 packing-factor sentence does mention the nitroisoxazole gap).

## 3. Sensitivity (h50, BDE)

- **Politzer-Murray BDE-correlated h50.** Table 7 reports h50 = 24.8-82.7 cm across the E-set. The Politzer-Murray BDE correlation is gas-phase electronic; **crystal-packing effects on h50 (which can shift values by ±50% across polymorphs) are nowhere quantified**. §7 caveat is generic; no per-lead h50 uncertainty is propagated.
- **C.10 GFN2-xTB BDE = 86 kcal/mol on C-NO2 of L1.** The paper notes the unrelaxed-fragment xTB protocol over-binds vs Politzer-Murray Ar-NO2 ~70 kcal/mol; this is a sensible disclaimer. The "no sub-80 kcal/mol channel ⇒ no primary-explosive sensitivity" claim is *only* a frontier-channel argument and ignores friction/electrostatic-discharge sensitivity. The §6 headline-recommendation language ("L1 meets five criteria for an experimental campaign") is therefore slightly overreaching on the safety dimension.

## 4. Synthesisability (D.14 review)

- **Curtius rearrangement step (D.14, line 1117).** DPPA + carboxylic acid → acyl azide → isocyanate via Curtius is *chemically correct mechanism*. The disconnection is plausible from a 4,5-dinitro-1,2-isoxazole-3-carboxylic acid synthon.
- **Acyl azide hazard claim.** "Acyl azides on highly electron-poor polynitro heterocycles are primary-explosive-class compounds subject to Curtius rearrangement and thermal decomposition below ambient temperature." This is **correctly stated** (cf. picryl azide and analogues; nitro-aromatic acyl azides are notoriously sensitive). The mitigation language (≤ -10 °C, in-situ trapping) is appropriate. Strong.
- **Terminal HNO3 nitration step caveat.** The paper correctly flags that AiZynth's USPTO template fires on the C-H bond of 4,5-dinitro-1,2-isoxazole and that a third C-nitration on this strongly deactivated ring is not literal protocol; fuming HNO3/oleum or N2O5 is correctly cited as literature precedent. This is the right level of caution.
- **§8 Conclusion synthesis recommendation language.** "AiZynthFinder finds a 4-step productive route" with five criteria, then recommends synthesis-and-characterisation. The DPPA hazard and the "synthetic starting point rather than a literal protocol" caveats from D.14 do not propagate into the §8 recommendation paragraph. A reader of §8 alone gets a more confident L1-go-to-bench message than D.14 supports. **Minor concern (M1)**: §8 recommendation paragraph should reference D.14 hazard caveat.

## 5. K-J equation usage

- **Per-lead branch (Table C.2) vs population branch (Table C.4).** The paper carefully distinguishes "closed-form K-J on calibrated DFT inputs" (Table C.2, used for ranking) from "open-form K-J with Q ≈ ΔHf/MW" (Table C.4, used for f_N stratification only). The C.7 footnote about the open-form artefact in [0.55-1.00] bin is essential and correctly placed.
- **Regime-limit gate (line 187).** Tier-C definition specifies "reliable for oxygen-deficient CHNO with 2a + b/2 ≥ d ≥ b/2" — this is the standard Kamlet-Jacobs B regime (oxygen-deficient). For L1 (a=3, b=0, d=7): 2a + b/2 = 6, d = 7 — **L1 is *oxygen-positive* (d > 2a + b/2), formally outside the K-J B-regime that the dataset gates on**. The paper applies K-J to L1 anyway (Table C.2: D = 8.25 km/s). This is a standard approximation in the energetic-materials community (the K-J A-regime / oxygen-rich variant uses different product assumptions) but the paper does not cite which K-J variant it uses for L1 with d=7. **Concern (M2)**: explicit K-J variant statement (B-regime vs A-regime, or unified product-distribution scheme) is not in the body or in C.12/C.13.
- **L1 OB ≈ +8% claim placing L1 in "PETN-like, K-J-reliable regime" (line 393, 429).** OB = +8% is approximately PETN-class (PETN is OB ≈ -10%, slight deficit; L1 at +8% is slightly *over*-balanced, not PETN-like). The statement "PETN-like regime" is a loose characterisation; sharper would be "near-stoichiometric, near-zero-OB regime where K-J's gas-product distribution assumptions are best-justified". Not a numerical drift, but a regime-label inconsistency.

## 6. Numerical drift (cross-section consistency)

- L1 ρ_cal = 2.093 (Table 3), 2.09 (§5.3, §6, §8), 2.09 (§7 §C.9). Consistent.
- L1 D_cal = 8.25 km/s appears 6× (Table 3, §5.3, Table C.2, §6, §7, §8). Consistent.
- L1 raw DFT ρ = 1.80 (§5.3) vs 1.80 (C.9). Consistent.
- HOF_cal intercept -206.7 kJ/mol appears in §5.3, Table 2 caption, and C.5. Consistent.
- LOO RMS ±0.078 g/cm³, ±64.6 kJ/mol — consistent across §5.3, Table 2, C.5.
- E1 ρ_cal = 2.04 (§5.4, §6, §8), D_cal = 9.00 (§5.4, §6, §8). Consistent.
- C.11 reports 7-anchor LOO-RMS rises from "~0.032 to 0.055 g/cm³" — but §5.3 quotes 0.078 g/cm³ for the 6-anchor LOO. The 0.032 number in C.11 is on a *different metric* (per-anchor ρ residuals before regression) but the paper labels it "7-anchor LOO-RMS". **Minor concern (M3)**: 0.032 vs 0.078 disagreement in C.11 — likely a different-metric mislabel, but reads as numerical drift.
- C.13: "RDX predicts 2.50 km/s vs experimental 8.75 km/s" → 8.75/2.50 = 3.5× factor. Consistent with the headline "~3.5×". Confirmed.

## 7. NEW C.12 / C.13 / D.14

- **C.12 xTB triage recipe.** Standard ETKDGv3 + MMFF94 + GFN2-xTB --opt tight pipeline; 1.5 eV gap as soft proxy is conventional; treats sensitivity as "weakest-bond + crystal" not gap alone. Correctly written.
- **C.13 Cantera ideal-gas CJ cross-check.** "ρ ≈ 3.5× under-prediction" framing: it is the *velocity* under-prediction (RDX: 2.50 km/s computed vs 8.75 km/s experiment, factor 3.5×), not ρ. The text says "Cantera ideal-gas equation of state is ~3.5× too low in absolute terms (RDX predicts 2.50 km/s vs experimental 8.75 km/s)" — *velocity* low, not density. The wording is unambiguous on rereading; but the assignment cover-letter phrased it as "ρ ≈ 3.5× under-prediction", which is a misreading on the cover-letter side, not on the paper. **Paper is correct**.
- **C.13 L1, L4, L5 RDX-class ranking defensibility.** An ideal-gas CJ ranking with ~3.5× absolute error preserves *relative ordering only within the same product-gas family*. L1 (f_N=0.29, OB +8%, polynitroaromatic), L4 (f_N high, tetrazoline-N-nitramine), and L5 (acyl-oxime nitrate) are *not* the same product-gas family (L1 is CO2+H2O-rich, L4 is N2-rich, L5 has nitrate-ester product channels). Cross-family relative ranking under ideal-gas CJ is therefore weak evidence. The paper explicitly says "within the same product-gas composition family", but then ranks L1, L4, L5 together as if they qualify — they don't fully. **Concern (M4)**: cross-family relative-ranking claim is over-stated; should say "within composition-similar leads" or split the ranking by gas-product family.
- **D.14 disconnection narrative.** Reviewed in §4 above. Chemistry valid; hazard claim correctly stated; minor §8 propagation gap (M1).

## Verdict

**Accept with 3 minor edits.** The §5 rewrite improves clarity; the new C.12, C.13, D.14 subsections are honest and well-scoped. No P0/P1 chemistry errors. The four flagged issues (M1: §8-to-D.14 hazard caveat propagation; M2: K-J variant statement for oxygen-positive L1; M3: C.11 0.032 vs 0.078 LOO-RMS metric label; M4: C.13 cross-family ranking caveat) are all minor and addressable in copy-edit pass. None block acceptance.

---

### Three most important concerns (line numbers)

1. **(M2, line ~187 / line 393 / Table C.2 line 881):** L1 has OB +8% (a=3, b=0, d=7 ⇒ d > 2a+b/2), formally outside the oxygen-deficient K-J regime that Tier-C of §3.1 gates on; the paper applies K-J to L1 without naming which K-J variant (B-regime/A-regime/unified product scheme). One sentence in §5.3 or C.12 naming the variant would close this.
2. **(M4, C.13 line 950):** The Cantera ideal-gas CJ relative ordering claim across L1, L4, L5 mixes product-gas-composition families (CO2/H2O-dominant L1 vs N2-dominant L4); ideal-gas CJ preserves ordering only *within* a family, as the paragraph itself states two sentences earlier. The cross-family RDX-class lumping should be qualified.
3. **(M1, §8 line ~617):** The synthesis-and-characterisation recommendation paragraph for L1 lists five criteria but does not propagate D.14's DPPA acyl-azide primary-explosive-class hazard caveat or the "synthetic starting point rather than literal protocol" framing for the terminal HNO3 nitration. A one-clause cross-reference to D.14 in §8 would prevent the §8-only reader from getting a more confident go-to-bench message than D.14 supports.
