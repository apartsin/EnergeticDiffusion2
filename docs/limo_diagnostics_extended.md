# LIMO root-cause diagnostics — extended

Existing D2 (roundtrip Tanimoto) and D3 (property predictability) tell us
LIMO is the bottleneck, but not **why**. This catalog adds focused
diagnostics that pinpoint which part of the encode → decode chain breaks
on energetic chemistry, and what to fix.

Each is cheap (CPU + small GPU) and produces one binary verdict. Run after
LIMO v2.1b and on every future LIMO retrain.

---

## Category A — Tokenization / vocabulary

### L1 — Per-motif tokenization survival

**Question**: Does SELFIES even *represent* the energetic motifs we care
about, or do tokens drop on encode?

**How**: Take 200 SMILES known to contain {NO₂, N₃, ONO₂, furazan,
tetrazole, triazole, tetrazine, nitramine}. SELFIES-encode each, then
decode the SELFIES back to SMILES (no model). Check whether the motif
survives the SELFIES roundtrip alone.

**Pass**: ≥95 % motif survival on all eight families.
**Fail**: motif survival < 80 % → SELFIES alphabet itself loses the motif;
**LIMO can't possibly learn what its tokens drop**. Fix: extend the
SELFIES alphabet (e.g. add explicit `[N+1H0]`-like grouped tokens).

### L2 — OOV rate per motif family

**Question**: Do we have vocab tokens for all the SELFIES tokens that
energetic chemistry uses?

**How**: Tokenise 5,000 random SMILES, count tokens that aren't in LIMO's
vocabulary. Bin by motif family of the parent SMILES.

**Pass**: OOV ≤ 0.5 % overall, ≤ 2 % per family.
**Fail**: high OOV in {furazan, triazole, tetrazole} → vocab gap. Fix:
rebuild vocab from a motif-rich corpus.

---

## Category B — Encoder geometry

### L3 — Tanimoto-vs-latent distance correlation

**Question**: Does the encoder map structurally similar molecules to
nearby latents?

**How**: Pick 500 random pairs of training SMILES (across the energetic
distribution). For each pair (a, b): compute Tanimoto(a, b) and
‖z_a − z_b‖₂. Spearman ρ(Tanimoto, −distance) should be strongly positive.

**Pass**: ρ ≥ 0.6.
**Fail**: ρ < 0.3 → encoder doesn't preserve chemistry similarity;
property head training works "by accident". Indicates encoder collapse
or overfit. Fix: KL regularisation tuning, more capacity, or
architectural change.

### L4 — Per-dim KL distribution (posterior-collapse probe)

**Question**: Are most latent dims being used, or has the model
collapsed?

**How**: On 5,000 val SMILES, compute per-row per-dim
`0.5·(μ² + σ² − log σ² − 1)` (the KL term). Average per dim. Plot the
sorted KL contributions.

**Pass**: ≤ 5 % of dims have per-dim KL < 0.05; entropy
(over normalized KL) ≥ 0.7 of max.
**Fail**: heavy collapse → bottleneck unused → no useful representation
in collapsed dims. Fix: lower β, free-bits, or larger encoder.

### L5 — Latent-norm distribution match to N(0, I)

**Question**: Do training latents look like the prior diffusion expects?

**How**: Compute `‖z_mu‖₂` distribution over 5,000 training rows.
Compare to χ(d=1024) expected for unit-Gaussian (mean ≈ √1024 ≈ 32).

**Pass**: distribution mean within ±25 % of √d, std ≈ √2·√d/√d = 1.
**Fail**: norm too small (e.g. 8 — current LIMO!) → latents live near the
origin → DDIM trajectories explode at high noise (matches our schedule
bug). Fix: rescale latents post-encode, or train with proper KL anchoring.

This one **explains the diffusion-sampler instability we hit earlier**.
Run it first.

---

## Category C — Decoder behaviour

### L6 — Per-token decode error map

**Question**: Where in the SELFIES sequence does the decoder make
mistakes for energetic chemistry?

**How**: For 200 high-D Tier-A/B SMILES, encode → decode, compare token
by token. Build a heatmap: per-position cross-entropy. Distinguish:
- positions with ring-opening/closing tokens
- positions with charged tokens (`[N+]`, `[O-]`)
- positions with branch-open/close

**Pass**: error spread ≤ 2× across positions.
**Fail**: errors concentrated at ring tokens or `[N+]` → decoder doesn't
learn polynitro patterns. Fix: token-class loss weighting in retrain.

### L7 — Top-K decode retrieval

**Question**: When the decoder is wrong, is the right answer in its
top-3 candidates?

**How**: For each position where argmax decode disagrees with truth,
record whether the truth token is in top-3 of the softmax. Bin by motif
family.

**Pass**: ≥ 60 % top-3 retrieval on rare motifs.
**Fail**: < 30 % top-3 retrieval → the decoder doesn't even *know* the
right token is plausible. Fix: more training on motif-rich data, or
multinomial-decode + chemistry-rule filtering at sample time.

### L8 — Roundtrip failure taxonomy

**Question**: Of failed roundtrips, what kind of mistake is it?

**How**: For 200 random Tier-A/B SMILES with non-exact roundtrip, classify
the decoded vs original difference:
- lost ring (ring count decreased)
- lost charge (formal charge changed)
- changed scaffold (Murcko scaffold mismatch)
- atom-count mismatch
- functional-group swap (e.g. `[N+](=O)[O-]` → `C=O`)

**Pass**: < 30 % each of {lost ring, lost charge, scaffold change}.
**Fail**: dominant failure mode → tells us *which* training-data emphasis
fixes most cases. E.g. if "lost charge" dominates → upweight charged-token
loss. If "lost ring" → upweight ring-bond tokens.

---

## Category D — Latent-space coverage of energetic chemistry

### L9 — Per-motif clustering in latent space

**Question**: Do molecules with the same motif cluster, or are they
scattered?

**How**: Take all training latents with each motif (furazan, tetrazole,
…). Compute centroid + intra-class distance. Compare to inter-class
distance. Davies-Bouldin index per motif family.

**Pass**: DB index ≤ 1.0 on rare motif families.
**Fail**: motif latents scattered (high DB) → diffusion can't reliably
sample motif chemistry by conditioning. Fix: motif-aware encoder
(auxiliary classification head during VAE training).

### L10 — Linear-probe motif presence

**Question**: Can a linear classifier on z predict motif presence?

**How**: For each motif family, train a logistic-regression classifier
on (z_mu, motif_label) where motif_label is binary RDKit-SMARTS hit.
Report ROC-AUC per family.

**Pass**: AUC ≥ 0.85 for furazan, tetrazole, triazole, azide.
**Fail**: AUC < 0.7 → motif info **missing from latent**. The encoder
discarded it. Fix: only an architecture/training change can add it back.

This pairs with L6 — if L6 says decoder is fine but L10 says encoder is
bad, the bottleneck is encoder. If L10 OK but L6 bad, decoder is bad.
Together they tell us *which side* to fix.

### L11 — Sample-from-prior coverage

**Question**: When we sample z ∼ N(0, I) and decode, what chemistry do we
get? Does it cover the energetic distribution?

**How**: Sample 1,000 z ∼ N(0, I), decode, count valid SMILES, count
each motif family in the output, count overlap with training set
(Tanimoto > 0.8 = "memorised").

**Pass**: ≥ 50 % valid; motif distribution matches training within 50 %;
< 5 % memorised.
**Fail**: validity < 30 % or motif distribution very off → the prior
mismatch makes diffusion sampling unreliable. Fix: KL retuning, latent
rescaling, or fine-tune on energetic data.

---

## Category E — Cross-version diagnostics (when retraining)

### L12 — Per-layer weight delta v_old → v_new

**Question**: Did the retrain meaningfully update the model, or just
the head/early layers?

**How**: For each parameter tensor, compute Frobenius norm of
(v_new − v_old) / ‖v_old‖. Histogram.

**Pass**: encoder + decoder both have ≥ 1 % delta on average.
**Fail**: delta concentrated in one component → other component still
v1, training imbalanced. Fix: unfreeze schedule, or higher LR.

### L13 — Self-consistency delta on energetic seeds

**Question**: Did self-consistency improve on the seeds we *care* about?

**How**: Run `decode(encode(s))` exact match + Tanimoto on the 10 seeds
in `data/c2c/seeds.csv`, compare to v1.

**Pass**: Tanimoto improvement ≥ +0.10 on average across seeds.
**Fail**: no improvement (or regression) → retrain didn't help the
target distribution. Fix: data augmentation strategy needs rethinking.

This is the **single most important post-retrain check**.

---

## Suggested execution order after each LIMO retrain

1. **L13** — self-consistency on seeds (5 min). If fail, abort and revisit.
2. **L5** — latent norm distribution (1 min). Detects scale issues
   that explain diffusion-sampler instability.
3. **L4** — per-dim KL (5 min). Detects collapse early.
4. **L13's** Tanimoto delta + **L8** roundtrip taxonomy (15 min).
5. **L10** linear probes (10 min). Tells us if motif info is present.
6. **L11** sample-from-prior (10 min). Tells us if generation works.
7. **L6** per-token error map (15 min). Diagnoses decoder weaknesses.
8. **L1, L2, L3, L7, L9, L12** — only if any of the above fail and we
   need finer-grained signal.

Total budget: ~1 hour for the basic suite.

---

## Mapping to actionable fixes

| Diagnostic fails | Most likely fix |
|---|---|
| L1 SELFIES drops motif | extend SELFIES alphabet; switch to SMILES-BPE base VAE |
| L2 OOV high | rebuild vocab on motif-rich corpus |
| L3 Tanimoto/distance disconnected | tune KL β, more capacity |
| L4 KL collapse | lower β, free-bits ε, or KL warmup |
| L5 wrong norm | rescale latents post-encode; document the scale |
| L6 errors at ring tokens | per-token loss weighting |
| L7 not in top-3 | more training, motif data |
| L8 dominated by lost-ring | ring-token loss boost |
| L9 motifs scattered | auxiliary motif classification head |
| L10 motif AUC low | encoder retrain with motif aux loss |
| L11 prior mismatch | KL retuning |
| L12 unbalanced delta | LR / unfreeze schedule |
| L13 no Tanimoto delta | data augmentation didn't reach the target distribution |

---

## What to implement first

**L5** (latent-norm) takes one minute and would have caught the diffusion
sampler instability days earlier. It's the cheapest highest-information
test we could run.

**L13** (self-consistency on seeds) is the single most informative post-
retrain gate.

**L10** (motif linear probes) is the deepest "is the encoder OK" test.

If those three pass, almost everything else will. Build a single
`scripts/diagnostics/limo_full.py` that runs L5 + L13 + L10 in ~10 min.
