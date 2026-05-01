# Pass-4 Reviewer 1 (AI/ML) line-by-line findings
_Generated 2026-04-30_

Sections audited line-by-line: §1 (L86-115), §2.1-2.3 (L120-139), §4 (L202-363),
§5.5 (L627-655), Appendix C (L738-873), Appendix G (L1216-1320).

## Methodological concerns
- L302-304: The sample-time formula
  `ε_hat = ε_θ^cfg − σ_t · Σ s_h ∇ L_h(z_t,σ_t)`
  uses a (−σ_t) prefactor on the head-gradient bus. The conventional
  classifier-guidance form (Dhariwal-Nichol) injects a (−σ_t) term only when
  the gradient is of log p_h (ascending log-probability). Yet two of the three
  active heads are described in L304 as "descend" objectives (sens_z and
  −log(1−P_hazard)), so the global sign is ambiguous in the current statement
  and depends on whether L_h is treated as "loss-to-minimise" or as
  "log-prob-to-ascend". Suggest spelling out the sign convention to remove
  the ambiguity.
- L302: "DDIM steps = 40" is asserted as production but Table C.5 (L859) calls
  it "prior-art default; not separately swept". The number of DDIM steps is
  a first-order knob for guided diffusion (see Karras et al. 2022, Lu et al.
  2022 DPM-Solver); not sweeping it is defensible but the paper should at
  least cite a prior work that establishes 40 DDIM steps as adequate for a
  guided 1000-step VP-cosine model in a 1024-dim latent.
- L243 vs L278: §4.4 trains the denoiser at "batch 128, 20 epochs" but the
  score model at "batch 1024, ~40k steps". The relation between "20 epochs"
  on the 326k corpus and the diffusion update count is not stated; reviewers
  cannot reproduce the compute footprint without it.
- L311: Anneal+clamp ablation rests on "natural gradient magnitudes at low σ
  are 8-30, well above 5" but no measurement is shown in this section; the
  appendix pointer (E.5) is the only evidence. Production setting σ_max=0
  with C_g=50 essentially disables both safeguards. The claim that "per-head
  scales s_h have negligible effect" under the original Dhariwal recipe
  warrants a one-line numerical anchor in §4.9 itself.
- L791 (Table C.1b): Score model has "Dropout(0.1)" in per-block layers, but
  this is never mentioned in §4.7. Minor methodological detail not surfaced
  in body.
- L785 (Table C.1b): "σ ~ U(0, σ_max=2.0) per batch; z_t = z_0 + σ·ε" — this
  is a *variance-exploding* parameterisation for the score model, while the
  denoiser of §4.4 uses the *variance-preserving* cosine schedule
  (z_t = √ᾱ_t z_0 + √(1-ᾱ_t) ε). The score model and denoiser therefore
  see incompatible noise levels at the same nominal "t". Sample-time
  guidance evaluates the score head at σ_t derived from the VP schedule
  (L276 says σ_t = √(1-ᾱ_t)), but the head was trained on a flat U(0,2.0)
  VE distribution. This is a real methodological issue and warrants an
  explicit remark on the σ-domain alignment.
- L1241 (Table G.2): The 40k Gaussian baseline is reported as "not ranked by
  D; N-fraction proxy" while DGLD is ranked by UniMol D. The "+0.45 km/s"
  lift in Table 8 (L635) and §6 then compares DGLD (UniMol-ranked, M7 100k)
  against the Gaussian *3k* run (UniMol-ranked) — not against the
  compute-matched 40k baseline. This compute-mismatch caveat deserves a
  one-line acknowledgement in §5.5 and Table 8.

## Numerical drift / contradictions
- L243 vs L787: §4.4 says score model trains "~40k steps, batch 1024", but
  Table C.1b row "Epochs / batch" lists "12 / 512". 12 epochs × 326k corpus
  / 512 batch ≈ 7.6k steps (or × labelled-corpus only ≈ 1.5k steps),
  neither of which matches "~40k steps". This is an internal contradiction.
- L243 vs L826: §4.4 lists "~40k steps" for score model; Appendix C.1
  step 1 (L826) restates "~40k AdamW steps" — consistent with §4.4 but not
  with Table C.1b.
- L304 vs L862: §4.9 lists `s_viab=1.0, s_sens=0.3, s_hazard=1.0` —
  consistent with Table C.5 row "Active steering heads". OK.
- L311 vs L864: §4.9 says clamp `C_g = 50`; Table C.5 says `C_g = 50` — OK.
- L93/L107 vs L678: §1 reports L1's max-Tanimoto as "0.27". §7 (L678)
  notes a packing-factor scenario shifting D_KJ by "±0.3 km/s" not the
  Tanimoto. No drift here, but check that "max-Tanimoto 0.27 ± 0.03"
  in Table G.4 (L1284, Hz-C2 condition) is the same scalar as L1's
  reported 0.27 — they refer to *different* objects (Hz-C2 condition mean
  vs L1 specific lead) but the paper uses the same number, which may
  confuse readers.
- L257 vs L765-777: §4.5 says "DGLD-H tilts toward HOF tail
  (factor 5×)" and "DGLD-P ... +5× high-tail oversampling on the top decile";
  §3.2 (L200) says "5×–10×". Table C.1b (L777) lists
  "5×–10× above 90th percentile". §4.5 (5×) and §3.2 (5×–10×) are mildly
  inconsistent — pick one factor.
- L1316 vs L320: Tier-gate ablation Table G.6 shows keep rate 53.9%; Table 8
  (L634) lists "53.9 % when off"; G.6 prose (L1320) reproduces this. OK.
- L304 vs L1284: Production scales s_viab=1.0, s_sens=0.3, s_hazard=1.0
  (L304) but the multi-seed Hz-C2 condition in Table G.4 is described
  identically and adopted as production. Consistent.
- L1232: "yields a top-1 composite of 1.10" for 3k Gaussian, but Table G.2
  (L1240) lists same row as composite "1.10". Then "Lift" row (L1243)
  says composite "−0.44 (better)" — DGLD composite is "0.665"
  (L1242), and 1.10 − 0.665 ≈ 0.435, not 0.44. Trivial rounding but worth
  fixing.

## Notation / consistency
- L131, L243, L251, L304: ε is variously written as `\epsilon` and `\varepsilon`
  in the same equations (L243 uses `\varepsilon`; L131, L251 use `\epsilon`).
  Adopt one form throughout.
- L228, L237, L130: CFG scale is `w` in the body; L237 introduces
  `w_tier` (per-property tier weight) and emphasises it is "distinct from the
  CFG scale w". The collision is fine once flagged but the same letter `w`
  also appears in L278 as `w_k` (head loss weight) and again in L313
  (`w` CFG). Three different `w`s within ten lines is hard to follow;
  consider renaming one.
- L228: `m ∈ {0,1}⁴` written without LaTeX delimiters in body text (HTML
  entities `&isin;`); other places use `\(m \in \{0,1\}^4\)`. Minor stylistic
  inconsistency.
- L676: `UHF=1` capitalised here; check whether other occurrences (e.g. xTB
  inputs in Appendix D) use `uhf=1`. Spot check shows §7 uses `UHF` only;
  consistent.
- L304: `P_viab`, `P_hazard` in math are P with subscript text; check the
  same notation in Appendix E.12 (L1167-1170 uses `P_hazard` consistently).
  OK.
- L316: Composite formula uses unitalicised `S_perf^band(x)` etc.; the
  body refers to it later as `S(x)` — fine.
- L789: "alpha-anneal" in plain text (Table C.1b) but §4.9 (L311) writes
  it with LaTeX `\alpha(\sigma_t)`. Standardise.
- L739, L777: "5×–10×" written with `&times;` HTML entity vs `\times`
  LaTeX in math contexts. Minor.

## Missing or weak citations
- L302: "40 DDIM steps" — DDIM should cite Song, Meng, Ermon (2021),
  "Denoising Diffusion Implicit Models", ICLR 2021. The body cites
  `dhariwal2021` for the cosine schedule (L243) but never DDIM itself.
  Suggested cite: Song et al. 2021 DDIM.
- L243: "cosine T=1000 DDPM schedule of Nichol & Dhariwal" — the cosine
  schedule was introduced in Nichol & Dhariwal 2021 ("Improved DDPM",
  ICML 2021), not in the IDDPM/Diffusion-beats-GANs paper. The current
  citation `dhariwal2021diffusion` resolves to the
  "Diffusion Models Beat GANs" paper (Dhariwal-Nichol NeurIPS 2021), which
  is *not* the cosine-schedule paper. Recommend retargeting this citation
  to Nichol & Dhariwal 2021 ("Improved DDPM").
- L243: EMA decay 0.999 — standard practice but worth a one-line citation
  to either Karras et al. 2022 or Song-Ermon 2020 for the convention.
- L276-278: SmoothL1 / Huber loss — uncited; minor.
- L311: Per-row gradient-norm clamp — the technique is loosely "Liu et al.
  2023" or any guided-diffusion review; the appendix-E.5 pointer is the
  only justification. Consider citing a specific reference for
  per-step gradient clamping in classifier guidance.
- L739 (C.0): Random Forest viability classifier — Breiman 2001 not cited
  here (it's standard but the paper does cite obscure earlier ML works
  elsewhere, so symmetry argues for inclusion).

## Suspicious factual claims
- L217: "validation token-accuracy is 64.5%" combined with
  "molecule-level (full-sequence) validity is 100% by construction" and
  "Reconstruction accuracy ... is 31.4%". A 64.5% per-token accuracy on
  a 72-token sequence implies *expected* full-sequence reconstruction far
  below 31.4% under independence (0.645^72 ≈ 10^-13), so the empirical
  31.4% reconstruction implies token errors are heavily *correlated*
  (clustered on a few hard sequences) — which is plausible. The claim is
  not wrong but the reader gets no anchor for why 64.5% token + 31.4%
  reconstruction are compatible. A one-sentence note on the
  per-sequence error distribution would help.
- L217: "‖μ‖ ≈ 8 on average, well below the √1024 ≈ 32 expected of N(0,I)".
  Under N(0,I_1024), E[‖μ‖] ≈ √1023 ≈ 31.98, so the comparison is
  correct. OK.
- L741: Random Forest validation AUC = 0.9986 (Table C.1c, L802).
  Suspiciously high for a 2089-feature ~150k-row classification with
  balanced class weights and an unaugmented test split. Either the test
  split contains scaffold leakage from training (a known failure mode for
  Morgan-FP descriptors on energetic chemistry, where many rows share
  scaffolds) or the labels are nearly linearly separable. The paper's
  own §4.8 narrative ("the Random Forest generalises poorly", L819)
  appears to contradict the 0.9986 AUC — the RF is described as generalising
  poorly *to the latent regions the diffusion sampler inhabits*, not on
  its own held-out test split. A one-line acknowledgement that the AUC
  is in-distribution and the OOD generalisation is what self-distillation
  is fixing would close the apparent contradiction.
- L808: Smoke ensemble "Validation R² 0.84-0.92 (5-fold CV, per output)" —
  reasonable for property prediction in this domain but the 5-fold CV
  scheme (random vs scaffold split) is not specified. For energetic CHNO
  with strong scaffold correlations this matters; recommend stating
  "scaffold-CV" if that is what was done (L804 says "2-fold scaffold-CV
  ensemble" for the model itself, but the validation R² may be from a
  different split).
- L257: "Both are 44.6 M-parameter FiLM-ResNets with the §4.4 architecture;
  only the training-data tilt differs." A 44.6 M-param 8-block ResNet over
  a 1024 latent (per-block ~5.5 M) is consistent with Linear(1024→2048)
  + Linear(2048→1024) ≈ 4.2 M params/block × 8 blocks ≈ 33.6 M, plus
  FiLM, embeddings, time/property embeds. Order-of-magnitude check: 44.6 M
  is in the right ballpark.
- L845: "their trained weights add ~2 × 256k parameters (negligible against
  the 1.4 M parameter score-model trunk)". The trunk is described as a
  4-block FiLM-MLP with hidden 1024 (L779); per block ~2× Linear(1024→1024)
  = 2.1 M params, × 4 blocks = 8.4 M params, plus FiLM and σ-embed and
  six heads. The "1.4 M parameter score-model trunk" claim is *off by ~6×*.
  This warrants a recount.

## Verdict
**Minor revision.** The methodology is sound and the contributions are
well-stated, but two genuine internal contradictions remain (score-model
step count §4.4 vs Table C.1b; score-model trunk parameter count "1.4 M"
in C.4 vs the architecture in Table C.1b which gives ~8 M), one
methodological issue deserves an explicit note (VP-schedule denoiser vs
VE-schedule score-model alignment of σ at sample time), and the
`dhariwal2021` citation is mis-targeted for the cosine schedule. None of
these is a methodological flaw; all are surface-level fixes that would
take a single editing pass.
