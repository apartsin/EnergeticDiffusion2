# PASS 5 Reviewer Report — Reviewer 4 (Abstract Reader)

Target: `docs/paper/short_paper.html` abstract + §1 paragraph 1.
Pass-4 verdict: Minor revision (length drift to ~258 w; sentence-level rewrites needed).

## 1. Word count

- Current abstract: **~217 words** (counted across the three `<p>` blocks).
- Pass-4 target: ~180 words. Pass-5 rewrite landed at ~210, current is ~217.
- **Verdict on length:** Acceptable. Within the +20% tolerance NMI typically permits. No further trimming required, but room to remove ~15 words if editor pushes back.

## 2. Three-paragraph structure

- **Para 1 (problem):** stakes + sparse-label framing. Holds.
- **Para 2 (method + L1 + E1):** DGLD definition, L1 headline, E1 co-headline. Holds; E1 inserted as requested.
- **Para 3 (comparison + Zenodo close):** baselines + DOI + forward-looking close. Holds.
- **Verdict on structure:** Clean. The three-act shape is now visible at a glance.

## 3. Sentence-by-sentence trace

| ID | Parse | Note |
|----|-------|------|
| S1 | medium | Stakes-first hook lands ("no new HMX-class compound has been disclosed in fifteen years"). The "propellant mass / warheads / civilian gas-generators" triplet front-loads three application domains in one breath, mildly slowing entry. |
| S2 | high | Sparse-label sentence: 66 k vs 3 k contrast is good but the second clause ("naive generative models trained on the full mixture either memorise the high-performance tail or extrapolate without calibration") is a 22-word run-on with two failure modes muddled. |
| S3 | medium | DGLD definition. Compact tricolon (gate / guidance / funnel). Ends on "first-principles DFT audit" — strong. |
| S4 | low | "The result is 12 DFT-confirmed novel leads." Clean, declarative, memorable. |
| S5 | high | L1 sentence: dense — name + ρ + D + Tanimoto + corpus-size in one sentence. Decompression from pass-4 partly successful but still 3 numerical claims. |
| S6 | high | E1 sentence: ends with parenthetical methodological caveat ("Kamlet–Jacobs relative ranking throughout; absolute D requires a thermochemical-equilibrium solver, §7") that *deflates* the headline number. Caveat belongs in §1, not the abstract. |
| S7 | medium | "DGLD is the only method to land in the productive quadrant…" — strong topic sentence for para 3. |
| S8 | high | Three-baseline comparison crammed into one sentence with semicolons. 18.3% / 9.73 / 6.28 / 3.5 / 9.02 = five numerals. Reader retention drops past number 3. |
| S9 | medium | Zenodo + forward-looking close. The final clause ("at the cost of a few GPU-days") lands well. The Zenodo DOI mid-sentence is a speed-bump. |

## 4. Jargon inventory

| Term | Defined inline? | Note |
|------|-----------------|------|
| HMX-class | no | Assumes reader knows HMX is the performance benchmark. Fine for NMI; an NMI generalist will still parse "class" from context. |
| CHNO | no | Standard in the energetics literature; opaque to non-specialists. Could be expanded once. |
| DFT | no | Acceptable for NMI audience. |
| Tanimoto | no | "nearest-neighbour Tanimoto 0.27" — non-cheminformaticians will not know the [0,1] scale or that 0.27 means "very dissimilar." |
| Kamlet–Jacobs | partial | Mentioned but not defined; the parenthetical "absolute D requires a thermochemical-equilibrium solver" makes it worse, not better. |
| SMILES-LSTM, SELFIES-GA, REINVENT 4 | no | Three baseline names with no inline expansion. NMI reader will tolerate. |
| D_K-J,cal, D_surrogate, D_DFT | partial | Subscripts are dense; reader has to triangulate three different D's in three sentences. |
| ρ_cal | no | Reader infers "calibrated density." |
| L1 / E1 | yes | Defined as "headline compound" / "second lead." |

## 5. Number-density per sentence

- S1: 1 (fifteen years).
- S2: 2 (66 k, 3 k).
- S3: 0.
- S4: 1 (12).
- S5: **3** (2.09, 8.25, 0.27) + 65 980. Over budget.
- S6: **2** (9.00, §7).
- S7: 0.
- S8: **5** (18.3%, 9.73, 6.28, 3.5, 9.02). Far over budget.
- S9: 1 (918) + DOI.

S5 and S8 are the two number-overload sentences.

## 6. First-sentence hook

The "no new HMX-class compound disclosed in fifteen years" lands. The triplet of applications (propellant / warheads / gas-generators) softens it slightly — a dual-use hook (defense + civilian) is editorially safer but trades urgency for breadth. **Hook works**.

## 7. Closing sentence

"the next compound to enter the HMX-class band can be discovered, validated, and recommended for synthesis at the cost of a few GPU-days" — **lands**. The Zenodo DOI mid-sentence is the only friction.

## 8. Phrase-level rewrites

### Rewrite A (S6 — E1 caveat)

- **Current:** "A second lead, E1 (4-nitro-1,2,3,5-oxatriazole), reaches D_K-J,cal = 9.00 km/s from a chemotype family disjoint from L1's (Kamlet–Jacobs relative ranking throughout; absolute D requires a thermochemical-equilibrium solver, §7)."
- **Proposed:** "A second lead, E1 (4-nitro-1,2,3,5-oxatriazole), reaches D_K-J,cal = 9.00 km/s on a chemotype disjoint from L1's, broadening the productive scaffold base."
- **Why:** The K-J vs thermochemical-solver caveat is the right scholarly note for §5.3, but in the abstract it deflates a co-headline. Move to §1 paragraph 1 (where the regime-limit of K-J is already stated).

### Rewrite B (S8 — three-baseline pile-up)

- **Current:** "SMILES-LSTM memorises 18.3% of its outputs exactly; SELFIES-GA's best novel candidate collapses from D_surrogate = 9.73 to D_DFT = 6.28 km/s under audit (a 3.5 km/s surrogate artefact); REINVENT 4 generates novel high-N heterocycles but peaks at D = 9.02 km/s."
- **Proposed:** "SMILES-LSTM memorises 18.3% of outputs; SELFIES-GA's best novel candidate loses 3.5 km/s under DFT audit; REINVENT 4 stays novel but peaks at 9.02 km/s."
- **Why:** Drops 9.73 and 6.28 (already implicit in "3.5 km/s under DFT audit"), removing two of five numerals. Reader retains the *pattern* (memorise / collapse / cap-out) instead of drowning in subscripts.

### Rewrite C (S2 — sparse-label decompression)

- **Current:** "Designing one is a sparse-label problem: of ~66 k labelled CHNO molecules only ~3 k carry experimental or DFT-quality measurements, and naive generative models trained on the full mixture either memorise the high-performance tail or extrapolate without calibration."
- **Proposed:** "Designing one is a sparse-label problem: of ~66 k labelled CHNO molecules, only ~3 k carry experimental or DFT-quality measurements. Naive generators trained on the mixture either memorise the high-performance tail or extrapolate uncalibrated."
- **Why:** Splits a 39-word run-on into 21 + 16. Same content, two parse-events instead of one. Drops "high-performance" from second mention to avoid echo.

### Optional rewrite D (S9 — DOI placement)

- **Current:** "Code, checkpoints, and 918 mined hard negatives are released on Zenodo (DOI 10.5281/zenodo.19821953), so the next compound … at the cost of a few GPU-days."
- **Proposed:** "Code, checkpoints, and 918 mined hard negatives are released on Zenodo, so the next compound to enter the HMX-class band can be discovered, validated, and recommended for synthesis at the cost of a few GPU-days (DOI 10.5281/zenodo.19821953)."
- **Why:** Moves the DOI to the very end so the forward-looking promise lands first; the DOI becomes the citable receipt rather than a mid-sentence stop.

## 9. §1 paragraph 1 harmonisation check

§1 ¶1 reuses the same hook ("no new HMX-class compound disclosed in fifteen years") verbatim. The K-J caveat the abstract carries (S6 parenthetical) is *also* in §1 ¶1 ("empirical formulas (Kamlet–Jacobs) are regime-limited"). This is the redundancy that justifies removing the parenthetical from the abstract: §1 already does that work for the careful reader.

## Verdict

**Minor revision.** The three-paragraph structure now holds, the hook lands, the close lands, and length is acceptable (~217 w). The two residual issues are S6 (E1 caveat deflates a co-headline that should stand on its own) and S8 (five numerals in one sentence overloads retention). Both are surgical fixes; no structural rework needed.

If only one rewrite ships: **rewrite A** (E1 caveat removal) — it returns the second co-headline to full force and the §1 ¶1 already carries the K-J regime-limit note.
