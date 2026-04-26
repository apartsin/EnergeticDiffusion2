---
name: Target chemistry is energetic materials only
description: Project goal is generating energetic materials, not general molecules. Avoid training/sampling choices that broaden the prior to inert chemistry.
type: project
originSessionId: bd41692a-abf9-40c5-ab30-18dd4abf4359
---
The diffusion+VAE pipeline is for **novel energetic materials only** (NO2,
N3, ONO2, nitramine, nitroaromatic, polynitro frames, CHNO heterocycles,
furazan/tetrazole/triazine rings, etc.). The target use case is high-D /
high-P / high-HOF candidates.

**Why this matters in practice:**

Of the 382,604 molecules in `latents_expanded.pt`:
- Only ~2,400–19,000 per property are curated energetics (Tier-A experimental
  or Tier-B DFT). The other ~360k are generic ZINC/PubChem-style molecules
  with 3DCNN-predicted properties.
- Smoke-model 3DCNN predictions for non-energetics push the conditioning
  distribution toward inert chemistry (low HOF, low D, low P).

**How to apply:**

- When training the denoiser: bias the prior toward real energetic frames.
  Don't oversample "extremes" symmetrically (the low tail is dominated by
  non-energetics). Restrict oversampling to Tier-A/B *and* the high-property
  end (mode=high, min_cond_weight=0.9). v3 already does this.
- For sampling/eval: prefer guidance scales that stay on-manifold for energetic
  chemistry. Long alkane/aromatic generations (e.g.
  `CCCCCCCCCCC...`) are warning signs that the model has drifted off-target.
- Future v4+ retrains should consider an "energetic-frame" filter on the
  training set itself: require ≥1 energetic motif (NO2/N3/ONO2/azo/furazan)
  OR Tanimoto ≥ 0.3 to a known-energetic seed list, rather than letting the
  model freely interpolate generic chemistry.
- Validation should always condition on *high* targets (q90), since hitting
  q10 of D/P just means generating an inert molecule.
