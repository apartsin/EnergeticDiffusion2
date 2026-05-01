# Pass 4 Reviewer 4 (Abstract Reader) Report

Scope: abstract only (lines 75-83) plus §1 paragraph 1 (line 89) for harmonisation.

## Word count

- Para 1: 64 words
- Para 2: 121 words
- Para 3: 73 words
- **Total: ~258 words** (target was ~180; over by ~43%)

## Sentence-by-sentence trace

### Paragraph 1 (problem)

**S1.** "Designing novel high-energy-density carbon-hydrogen-nitrogen-oxygen (CHNO) compounds is a sparse-label, multi-objective generation problem: of roughly 66 k labelled rows only ~3 k derive from experiment or first-principles DFT, the rest from empirical formulas and surrogates of sharply lower reliability."
- Parse difficulty: **HIGH**. ~38 words, two compound modifiers ("high-energy-density", "sparse-label, multi-objective"), spelled-out CHNO, then a colon-attached statistical clause with two numbers and a comparative quality clause. Reader must hold the framing while ingesting the data.
- Ends on "sharply lower reliability", which is a strong evaluative phrase, good.

**S2.** "Naive generative models trained on this mixture either memorise the high-performance tail or extrapolate without calibration; classifier-free guidance (CFG) silently misfires under the short sampling trajectories required for molecular latent diffusion."
- Parse difficulty: **HIGH**. Semicolon-joined compound sentence; second clause requires the reader to know what CFG does and what a "short sampling trajectory" means. The phrase "silently misfires" is memorable but the surrounding context is technical.
- Ends on "molecular latent diffusion", which is descriptive but not punchy.

### Paragraph 2 (method + L1)

**S3.** "We introduce **Domain-Gated Latent Diffusion (DGLD)**."
- Parse difficulty: **LOW**. Crisp 5-word thesis sentence. Acronym defined at first use. Excellent.

**S4.** "At training time, a label-quality gate controls which examples drive the conditional gradient, preventing low-confidence surrogate labels from corrupting the generation signal."
- Parse difficulty: **MEDIUM**. "Conditional gradient" presumes diffusion-training literacy; "label-quality gate" is intuitive. The why-clause ("preventing...") rescues the meaning.
- Ends on "corrupting the generation signal", strong.

**S5.** "At sample time, a multi-task score model provides selectable per-step steering across viability, sensitivity, and hazard axes without retraining the diffusion backbone."
- Parse difficulty: **MEDIUM**. Parallel structure to S4 (good). "Selectable per-step steering" is a coined phrase; "viability, sensitivity, hazard axes" is jargon-light. "Without retraining the diffusion backbone" is the value-prop punchline.

**S6.** "Decoded candidates pass through a four-stage chemistry-validation funnel escalating from rule-based filters to full first-principles DFT audit."
- Parse difficulty: **MEDIUM**. "Four-stage funnel" + "rule-based filters" + "first-principles DFT audit" sets a clear escalation. Reader-friendly.

**S7.** "The result is 12 DFT-confirmed novel leads; the headline lead, trinitro-1,2-isoxazole (L1), reaches \(\rho_{\text{cal}} = 2.09 \pm 0.15\) g/cm³ and \(D_{\text{K-J,cal}} = 8.25\) km/s, placing it within the HMX/CL-20 performance band (by Kamlet-Jacobs relative ranking; absolute D requires a thermochemical equilibrium solver, see §7), structurally dissimilar from all 65 980 training molecules (nearest-neighbour Tanimoto 0.27)."
- Parse difficulty: **VERY HIGH**. ~62-word monster. Six numerical claims, two parenthetical qualifications (one a forward-section reference), three subscripted symbols (rho_cal, D_K-J,cal), three named compounds (L1, HMX, CL-20), one similarity metric (Tanimoto). The mid-sentence Kamlet-Jacobs caveat breaks momentum and reads like a referee-induced patch.
- Sentence-end strength: trails into "(nearest-neighbour Tanimoto 0.27)", a parenthetical, which is a weak ending for the abstract's primary result.

### Paragraph 3 (comparison + Zenodo) - note: E1 is missing despite assignment claim

**S8.** "DGLD is the only method with consistent novel productive-quadrant coverage (simultaneously novel and on-target for performance) across seeds."
- Parse difficulty: **MEDIUM**. "Productive-quadrant" is an internal coinage that requires the parenthetical gloss to land. The parenthetical works, but the inline definition slows the reader.

**S9.** "The compute-matched SELFIES-GA (40k pool) best novel candidate collapses from \(D_{\text{surrogate}} = 9.73\) km/s to \(D_{\text{DFT}} = 6.28\) km/s under DFT audit (a 3.5 km/s surrogate artefact); SMILES-LSTM memorises 18.3% of its outputs exactly; REINVENT 4 generates genuinely novel heterocycles but peaks at \(D = 9.02\) km/s."
- Parse difficulty: **VERY HIGH**. Triple-baseline semicolon chain with seven numerical claims and four method names (SELFIES-GA, SMILES-LSTM, REINVENT 4, plus DFT). The reader cannot retain this.
- Ends on "peaks at D = 9.02 km/s" - factual but flat.

**S10.** "Trained checkpoints, sampling scripts, and 918 self-distillation hard negatives are released on Zenodo (DOI 10.5281/zenodo.19821953)."
- Parse difficulty: **LOW**. Clean release sentence. The "918 self-distillation hard negatives" is unexplained jargon for the abstract reader (they will not know what self-distillation hard negatives are without §4) but it is parsable as "we release artefact X".

## Jargon inventory

| Term | Defined inline? | Notes |
| --- | --- | --- |
| CHNO | Yes (carbon-hydrogen-nitrogen-oxygen) | Good |
| DFT | No | Common enough in NMI but worth a brief gloss |
| Kamlet-Jacobs | No | The reader sees only "K-J" subscript |
| classifier-free guidance (CFG) | Defined as acronym only, not explained | Function is implied, not stated |
| latent diffusion | No | Assumed prior knowledge |
| conditional gradient | No | ML-jargon |
| DGLD | Yes | Good |
| label-quality gate | Self-explanatory | Good |
| multi-task score model | Partially | Reader infers |
| productive-quadrant | Yes (parenthetical) | Works |
| surrogate artefact | No, but "collapse from 9.73 to 6.28" tells the story | Acceptable |
| Tanimoto | No | Standard cheminformatics; an NMI generalist may not know |
| HMX, CL-20 | No | Domain reader knows; generalist does not, but performance "band" framing is enough |
| self-distillation hard negatives | No | Pure ML jargon, used in S10 |
| SELFIES-GA, SMILES-LSTM, REINVENT 4 | No (method names, citations expected later) | Acceptable for an abstract |

## Number-density per sentence

| Sentence | Distinct numerical claims | Verdict |
| --- | --- | --- |
| S1 | 2 (66 k, ~3 k) | OK |
| S2 | 0 | OK |
| S3 | 0 | OK |
| S4 | 0 | OK |
| S5 | 0 | OK |
| S6 | 1 (four-stage) | OK |
| **S7** | **6** (12 leads, 2.09 ± 0.15 g/cm³, 8.25 km/s, 65 980, 0.27) | **OVERLOAD** |
| S8 | 0 | OK |
| **S9** | **7** (40k, 9.73, 6.28, 3.5, 18.3%, REINVENT 4, 9.02) | **OVERLOAD** |
| S10 | 1 (918) plus DOI | OK |

Two sentences (S7, S9) blow past the 2-numbers-per-sentence retention ceiling. They are the load-bearing result sentences and are also the longest.

## Three-paragraph structure

- Para 1: problem - holds.
- Para 2: method + L1 result - holds, but L1 result-sentence (S7) is bloated and contains a defensive forward reference to §7.
- Para 3: comparison + E1 + Zenodo - **E1 IS MISSING**. The assignment specified that para 3 should include E1, and §1 para 1 (line 93) does mention E1 (4-nitro-1,2,3,5-oxatriazole, D = 9.00 km/s), but the abstract does not. Either the assignment description is slightly off, or E1 should be added back. Given that para 3 is currently 73 words, there is room.

## First-sentence hook

S1 leads with "Designing novel high-energy-density CHNO compounds is a sparse-label, multi-objective generation problem". The hook is intellectual ("hard ML problem") not visceral ("HMX-class compound in 15 years"). §1 para 1 has the punchier hook ("the field has not surfaced a new HMX-class compound in the last fifteen years") and that line would make the abstract considerably stronger if echoed. The abstract currently misses the chance to convey *why the reader should care*.

## Harmonisation with §1 para 1

§1 para 1 is dense but ordered: motivation -> field stagnation -> design space -> failure modes of existing families -> tier stratification. The abstract S1 echoes only the last point (tier stratification). A stronger abstract S1 would echo the §1 motivation ("no new HMX-class compound in fifteen years") rather than diving into the data composition.

## Specific phrase-level rewrites

### Rewrite A - first-sentence hook

**Current (S1, 38 words):**
> Designing novel high-energy-density carbon-hydrogen-nitrogen-oxygen (CHNO) compounds is a sparse-label, multi-objective generation problem: of roughly 66 k labelled rows only ~3 k derive from experiment or first-principles DFT, the rest from empirical formulas and surrogates of sharply lower reliability.

**Proposed (28 words):**
> No HMX-class energetic compound has been disclosed in fifteen years. Designing one is a sparse-label problem: of ~66 k labelled CHNO molecules, only ~3 k carry experimental or DFT measurements; the remainder rely on empirical formulas and lower-fidelity surrogates.

Splits a 38-word run-on into two sentences, opens with a visceral hook, defers "multi-objective" until method paragraph (where it is paid off by the multi-task score model).

### Rewrite B - decompose the L1 result sentence

**Current (S7, 62 words):**
> The result is 12 DFT-confirmed novel leads; the headline lead, trinitro-1,2-isoxazole (L1), reaches \(\rho_{\text{cal}} = 2.09 \pm 0.15\) g/cm³ and \(D_{\text{K-J,cal}} = 8.25\) km/s, placing it within the HMX/CL-20 performance band (by Kamlet-Jacobs relative ranking; absolute D requires a thermochemical equilibrium solver, see §7), structurally dissimilar from all 65 980 training molecules (nearest-neighbour Tanimoto 0.27).

**Proposed (split into two sentences, ~45 words):**
> The pipeline yields 12 DFT-confirmed novel leads. The headline compound, trinitro-1,2-isoxazole (L1), reaches \(\rho_{\text{cal}} = 2.09\) g/cm³ and \(D_{\text{K-J,cal}} = 8.25\) km/s (Kamlet-Jacobs relative ranking, §7), HMX/CL-20 class, and is structurally novel against all 65 980 training molecules (max Tanimoto 0.27).

Drops the mid-sentence solver caveat (the §7 reference suffices), drops the ±0.15 (one decimal less for retention), and ends on the punchy novelty number.

### Rewrite C - decompose the baseline comparison

**Current (S9, 49 words, 7 numerical claims):**
> The compute-matched SELFIES-GA (40k pool) best novel candidate collapses from \(D_{\text{surrogate}} = 9.73\) km/s to \(D_{\text{DFT}} = 6.28\) km/s under DFT audit (a 3.5 km/s surrogate artefact); SMILES-LSTM memorises 18.3% of its outputs exactly; REINVENT 4 generates genuinely novel heterocycles but peaks at \(D = 9.02\) km/s.

**Proposed (sequential bullet-style sentences, ~46 words):**
> Three baselines fail differently. SELFIES-GA collapses 3.5 km/s under DFT audit (9.73 -> 6.28 km/s), exposing a surrogate artefact; SMILES-LSTM memorises 18.3% of outputs verbatim; REINVENT 4 produces novel heterocycles but plateaus at \(D = 9.02\) km/s.

Adds a topic sentence ("Three baselines fail differently") so the reader has scaffolding before the numbers, and trims the parenthetical.

### Rewrite D - restore E1

After the L1 sentence (or as part of S8), add:
> A second lead, E1 (4-nitro-1,2,3,5-oxatriazole, \(D_{\text{K-J,cal}} = 9.00\) km/s), comes from a chemically distinct scaffold family.

This realigns the abstract with §1 para 1 and the assignment's stated three-paragraph structure.

## Verdict

**Minor revision.** The structural arc is sound and DGLD/L1 land. But the abstract has drifted to ~258 words (target 180), the two result sentences (S7 at 62 words, S9 at 49 words with seven numbers) are unreadable for the NMI generalist on first pass, and the first-sentence hook is technical rather than motivational. E1 has fallen out of the abstract despite remaining in §1 para 1. Phrase-level rewrites A, B, C above (with optional D) tighten the abstract to ~210 words and restore retention.

## Top 2-3 actionable rewrites for reply

1. Replace S1 with the two-sentence hook (Rewrite A): lead with the fifteen-year HMX-class drought.
2. Split S7 into two sentences and drop the inline thermochemical-solver caveat (Rewrite B).
3. Add a topic sentence before the baseline triplet in S9 (Rewrite C) and consider restoring E1 (Rewrite D) for harmony with §1 para 1.
