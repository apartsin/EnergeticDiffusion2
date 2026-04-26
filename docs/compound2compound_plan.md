# Compound-to-compound (c2c) pipeline — implementation + validation plan

Generate **analogue** molecules of a known seed compound: take a real
energetic material, perturb it in latent space, and produce structurally-related
candidates with potentially better properties. The biological-imaging
analogue is "img2img" / SDEdit; the molecule analogue is what `evaluate.py`
already calls **compound2compound**.

This is fundamentally different from the unconditional / property-conditional
pipeline (v6, v6-multi, v6-feasibility) because the target is anchored to a
**reference molecule**, not to a property quantile.

---

## 1. Why c2c matters

| Use-case | What c2c gives |
|---|---|
| Lead optimisation around a known compound (e.g. RDX, HMX, CL-20) | candidates with similar scaffolds but pushed toward higher D / P / HOF |
| Neighbourhood exploration | "what's around this molecule that might be a better explosive" |
| Patent-busting | candidates structurally distinct from a target, but matched in properties |
| Synthesisability anchoring | start from a molecule we know how to make; perturb gently → keep synthesisable |
| Fast-iteration on a hit | DFT-validate a compound, use it as seed for next round |

vs unconditional / property-conditional (current production):
- Property-conditional explores the latent space at large; results often have **no obvious synthesis precedent**.
- c2c is anchored to a real molecule, so results inherit some of its synthesis path / motif.

---

## 2. Algorithm — SDEdit on latent diffusion

For a seed SMILES `s_seed`:

```
z_seed = LIMO.encode(s_seed)                            # (1, 1024)
t_edit  ∈ [0, T] = "edit strength"                       # higher = more aggressive
z_T     = q_sample(z_seed, t_edit)                       # noisy version of seed
for i in [t_edit, t_edit-1, ..., 0]:
    eps   = denoiser(z, t, c, mask)                      # property-conditioned ε
    z     = ddim_step(z, eps, t)                         # standard reverse step
out_smi   = LIMO.decode(z)
```

Two knobs:
- **strength** = `t_edit / T` ∈ [0, 1]. 0 = identity (returns seed). 1 = full denoise (≈ unconditional). Sweet spot typically 0.3–0.7 — perturbs scaffold, keeps frame.
- **conditioning c** = optional property targets if you want the analogue to be *better* than the seed (e.g. higher D, higher HOF).

### Variant — c2c with classifier guidance

Stack the property-head and SA/SC gradients on the partial-denoise trajectory:

```
z_t = q_sample(z_seed, t_edit)
for i in [t_edit, ..., 0]:
    eps_cond  = denoiser(z, t, c, mask)             # CFG'd
    g_prop    = ∇_z f_prop(z_t, t)                  # optional
    g_feas    = ∇_z f_SA(z_t, t) + ∇_z f_SC(z_t, t)
    eps       = eps_cond + σ_t (λ_p g_prop + λ_f g_feas)
    z         = ddim_step(z, eps, t)
```

This is exactly the **feasibility_sampler** path but seeded from `z_seed`
instead of pure noise. Implementation cost: trivial.

---

## 3. What's already built (existing scaffolding)

`scripts/diffusion/evaluate.py` already exposes a simple SDEdit at `--compound2compound`:

```python
sample_from_seed(denoiser, schedule, limo, tok, device,
                 seed_smiles, n_variants_per_seed,
                 strength=0.3 | 0.6 | 0.9, ...)
```

- Encodes seed via LIMO
- `q_sample` to noise level proportional to `strength`
- DDIM-denoises using current denoiser
- Decodes variants

**Limitations of the current implementation**:

1. No property conditioning during the c2c trajectory (mask is left as random / unconditional).
2. No SA/SC feasibility guidance.
3. No 3DCNN validation of outputs (currently just reports "5/5 unique").
4. No similarity / novelty metrics vs seed.
5. No comparison against the seed's own properties.
6. Single point-estimate per seed; no proper sweep over strength.

A first-class c2c production pipeline rebuilds this with:
- Multi-property conditioning at sample time.
- Optional feasibility guidance (sampler we just built).
- 3DCNN validation of outputs.
- Tanimoto + MCS overlap to seed.
- Property *delta* tracking (Δρ, ΔD, ΔP, ΔHOF) vs seed.
- Strength sweep + n_variants sweep, output ranked.

---

## 4. Implementation plan

### 4.1 New file: `scripts/diffusion/c2c_pipeline.py`

```bash
python scripts/diffusion/c2c_pipeline.py \
    --exp <denoiser exp dir> \
    --seeds data/c2c/seeds.smi \
    --strengths 0.3 0.5 0.7 \
    --n_variants 50 \
    --target_density 1.85 --target_d 8.5 --target_p 30.0 --target_hof 200 \
    --lambda_sa 5 --lambda_sc 2 \
    --require_neutral
```

Pipeline per seed × strength:

1. **Encode** — `z_seed, mu, log_var = LIMO.encode(seed)`.
2. **Noise** — `t_edit = int(strength × T)`; `z_t = q_sample(z_seed, t_edit)`.
3. **Conditional denoise** — call `ddim_sample_feasibility(...)` *with the
   z_t initial state passed in* (not re-sampled from noise). This is a
   small modification to the sampler signature: accept `init_z` argument.
4. **Decode** — `smi_var = LIMO.decode(z)` → canonical SMILES.
5. **Filter** — drop invalids; require neutral if requested.
6. **Validate** — 3DCNN ensemble predicts (ρ, D, P, HOF) per variant.
7. **Score**:
   - Tanimoto similarity to seed (Morgan fp, 2048 bits, radius 2)
   - MCS atom count (max common substructure)
   - Per-property Δ vs seed
   - Composite rank: weighted (property error to target) + similarity bonus
8. **Output** — markdown report per seed with top-N variants.

### 4.2 Code change to `feasibility_sampler.py`

Add `init_z: Optional[torch.Tensor]` and `start_step: Optional[int]` arguments:

```python
def ddim_sample_feasibility(..., init_z=None, start_step=None, ...):
    if init_z is not None:
        z = init_z.clone().to(device)
        ts = torch.linspace(start_step, 0, n_steps + 1, device=device).long()
    else:
        # existing path
        ...
```

This unifies the unconditional and c2c paths through one sampler.

### 4.3 Seeds list

Curate `data/c2c/seeds.smi` with ~10–20 well-known energetic materials:

| Tier | Examples |
|---|---|
| **Classic** | TNT, RDX, HMX, PETN, NTO, TATB |
| **Modern high-performance** | CL-20, ICM-101, FOX-7, DAAF |
| **Polynitrogen** | TKX-50, hydrazinium azotetrazolate |
| **Furazans** | LLM-105, BTF, DNAF |
| **Strained / cage** | DDF, ONC (octanitrocubane) |

Pick from `labeled_master.csv`; their canonical SMILES are already in our
training set. Encoding will give clean `z_seed` (no decode error).

---

## 5. Metrics to measure + report

Per (seed × strength × λ) configuration:

### 5.1 Generation quality

| Metric | Description | Target |
|---|---|---|
| **valid %** | RDKit-parsable + (optionally) charge-neutral | ≥ 90 % |
| **unique %** | unique canonical SMILES out of generated | ≥ 80 % |
| **non-trivial %** | not identical to seed (Tanimoto < 0.99) | ≥ 95 % at strength ≥ 0.3 |

### 5.2 Similarity to seed

| Metric | Description | What it tells us |
|---|---|---|
| **Tanimoto-mean** | Morgan-fp similarity to seed | how perturbed the chemistry is |
| **Tanimoto-p25 / p75** | spread | are variants in a tight neighbourhood or wandering? |
| **MCS-mean (atom count)** | maximum common substructure size | scaffold preservation |
| **scaffold-match %** | Murcko scaffold identical to seed | strict scaffold preservation |

Expected: Tanimoto declines with strength; for 0.3 → ~0.7, 0.5 → ~0.5, 0.7 → ~0.3.

### 5.3 Property shift

| Metric | Description |
|---|---|
| **Δρ_mean** | mean(predicted ρ_var − ρ_seed) across variants |
| **ΔD_mean**, **ΔP_mean**, **ΔHOF_mean** | same for D, P, HOF |
| **% variants with all 4 Δ ≥ 0** | fraction strictly improving |
| **% variants with composite improvement** | weighted across 4 properties |

This is the headline metric: the c2c pipeline is **useful** if it
generates variants with ΔD > 0 or ΔP > 0 vs seed without breaking
synthesisability.

### 5.4 Feasibility

| Metric | Description |
|---|---|
| **SA-mean (variants) vs SA(seed)** | does perturbation push toward harder synthesis? |
| **SC-mean delta** | same for SC |
| **% variants with SA ≤ seed_SA + 0.5** | "no worse" feasibility |

### 5.5 Novelty

| Metric | Description |
|---|---|
| **% novel vs 382k internal** | exact-canon match against training data |
| **% novel vs PubChem (top-N)** | external-DB match using `match_candidates.py` |

### 5.6 Strength × λ sweep

For each seed, run a 3 × 4 grid:
- strengths ∈ {0.3, 0.5, 0.7}
- (λ_SA, λ_SC) ∈ {(0,0), (3,1.5), (10,5), (30,15)}

Plot Pareto frontier: `Tanimoto_to_seed` vs `Δ_property`.

---

## 6. Validation protocol

Three test conditions:

### 6.1 Self-consistency

Strength = 0, λ = 0 → output should be identical (or very close) to seed.
This validates the encoder-decoder roundtrip on each seed.

Pass: ≥ 90 % exact recovery; mean Tanimoto ≥ 0.95.

### 6.2 Strength curve

Vary strength = {0, 0.1, 0.2, ..., 1.0} for one seed. Plot:
- Tanimoto (seed → variant) — should decrease smoothly with strength
- Validity — should stay high throughout
- Property prediction — should drift smoothly

A monotone Tanimoto curve confirms the noise→denoise pipeline behaves
correctly.

### 6.3 Comparator: c2c vs unconditional

For the same property targets (e.g. q90 of D), run:
- **A**. unconditional generation (current production v6) — pool 400, top-40
- **B**. c2c from a high-performance seed (e.g. CL-20) at strength 0.5,
       same property targets, pool 400, top-40

Compare:
- mean Tanimoto to known energetic chemistry (B should be higher)
- novelty rate (B will be lower — that's the point)
- property fidelity (rel_MAE — should be similar or better with c2c)
- SA mean (B should be lower since seeded from a known synthesisable)

If c2c beats unconditional on `SA + similarity + property fidelity` jointly,
it's the production answer for "give me analogues of CL-20 with higher D".

### 6.4 Per-seed acceptance gate

A seed s is "well-supported" by c2c if:
- self-consistency ≥ 90 % (R6.1)
- non-trivial variants at strength=0.5 ≥ 80 %
- Tanimoto_mean(variants) ∈ [0.3, 0.7] at strength=0.5
- ≥ 5 variants with predicted ΔD ≥ 0 *and* SA ≤ seed_SA + 0.5

Seeds that fail self-consistency (R6.1) are dropped from the seed list —
LIMO can't roundtrip them, so c2c will be unreliable.

---

## 7. Expected results

| Seed type | Predicted self-consistency | Predicted Tanimoto at strength=0.5 |
|---|---|---|
| Classic (TNT, RDX, HMX) | high (≥ 0.9 exact) | ~0.5 |
| Modern (CL-20, ICM-101) | medium (~0.5 exact) | ~0.4 |
| Furazans (LLM-105, BTF) | low (per D2 finding) | ~0.3, drift to non-furazan |
| Polynitrogen (TKX-50) | low | hard to predict |
| Strained cage (ONC) | low | likely scaffold-broken |

**Honest caveat**: the same D2 / D15 LIMO weakness on N-rich heterocycles
that breaks the unconditional pipeline will also break c2c on furazan /
tetrazole / triazole seeds. Self-consistency will fail, so c2c can only be
trusted on classic and a subset of modern seeds until LIMO v2 lands.

---

## 8. Implementation cost

| Step | Time |
|---|---|
| New `c2c_pipeline.py` (~400 LOC) | half day |
| Patch `feasibility_sampler.py` for `init_z`, `start_step` | 30 min |
| Curate seeds.smi (10–20 SMILES + verify they're in training) | 1 hour |
| Build similarity / MCS / property-delta metrics | 2 hours |
| Run validation suite (R6.1–R6.4) | 1 hour compute |
| Markdown reporting + Pareto plots | 2 hours |
| **Total** | **~1 day** |

---

## 9. Output report structure

`<exp>/c2c_results/<seed_name>/<strength>_<lambda>.md`:

```
# c2c results: seed = CL-20

## Seed
SMILES: O=[N+](...)
Tier-A/B properties: ρ=2.04, D=9.4, P=42, HOF=+90

## Configuration
strength = 0.5
λ_SA = 5, λ_SC = 2
n_variants = 200

## Aggregate metrics
Validity: 92 %    Unique: 89 %    Non-trivial: 96 %
Tanimoto-mean: 0.43    p25: 0.32    p75: 0.55
SA mean (variants): 4.2 (vs seed 4.4)
ΔD-mean: +0.6 km/s    Δρ-mean: -0.05    ΔP-mean: +1.4
% variants with ΔD ≥ 0: 38 %

## Top 10 variants (composite-ranked)

| rank | Tanimoto | ρ | D | P | HOF | SA | SMILES |
| ...

## Pareto front (Tanimoto vs ΔD)
[scatter plot saved as png]
```

A summary index `<exp>/c2c_index.md` lists all (seed, strength, λ) runs and
their headline metrics.

---

## 10. What gets reported back

After running R6.1 (self-consistency on all seeds) → R6.2 (strength curve
on best 3 seeds) → R6.3 (vs unconditional) → R6.4 (per-seed gates):

1. Self-consistency scoreboard — which seeds LIMO can roundtrip.
2. Strength-curve plot — confirms sane c2c behaviour.
3. Pareto plots (Tanimoto vs ΔD, vs SA, vs all 4 props) per seed.
4. Top-K variant tables for each seed.
5. c2c-vs-unconditional comparison on identical property targets.
6. Acceptance gate verdict per seed.
7. Recommendation: "c2c is production-ready for {classic seeds}; not yet
   reliable for {N-rich seeds}; gate-fail seeds {list}."

---

## 11. What's deliberately NOT in this plan

- New training. We use the existing v4-B denoiser + LIMO v1 + property heads.
- Multi-seed conditioning (interpolation between two seeds) — leave as v2.
- Reaction-prediction overlay (would require ChemFormer or similar) — out of scope.
- Property *increase* via gradient guidance from property heads — already
  available via `lambda_property` in `feasibility_sampler`; just hooked in here.

---

## 12. Sketch of the production wrapper

```bash
# Standard c2c run for one seed, three strengths, with feasibility:
python scripts/diffusion/c2c_pipeline.py \
    --exp experiments/diffusion_subset_cond_expanded_v4b_<...> \
    --seeds data/c2c/seeds.smi \
    --strengths 0.3 0.5 0.7 \
    --n_variants 200 \
    --target_density 1.90 --target_d 8.5 --target_p 30.0 --target_hof 200 \
    --lambda_sa 5 --lambda_sc 2 \
    --require_neutral \
    --use_t_aware
```

Output: `<exp>/c2c_results/<seed>_<strength>_<lambda>.md` plus
`<exp>/c2c_index.md`.
