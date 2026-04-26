# C2c diagnose-and-patch rounds — final report

Goal of the rounds: improve c2c Tanimoto-to-seed without retraining LIMO.

## Findings

| Seed | LIMO self-consistency Tanimoto | Theoretical c2c ceiling |
|---|---|---|
| TNT | 0.13 | 0.13 |
| RDX | 0.50 | 0.50 |
| HMX | 0.12 | 0.12 |
| PETN | 0.46 | 0.46 |
| NTO | 0.31 | 0.31 |
| TATB | 0.13 | 0.13 |
| CL-20 | 0.12 | 0.12 |
| FOX-7 | 0.04 | 0.04 |
| DAAF | 0.44 | 0.44 |
| LLM-105 | failed encode | – |

**The decode(encode(seed)) Tanimoto is the ceiling**: c2c cannot produce
variants more similar to the seed than the seed's *own* roundtrip already
gives. For most energetic seeds (especially those with N-rich heterocycles
and polynitro frames) this ceiling is ≤ 0.15, far below the [0.30, 0.60]
"useful analog" window.

## Patches tested

| Round | Patch | Variants/seed (mean) | Tan-to-seed (mean) | Verdict |
|---|---|---|---|---|
| 1 | anchor_alpha=0.3, cfg=2 | 8–25 | 0.10–0.20 | matches self-consistency ceiling |
| 2 | strength=0.1 + anchor=0.5 | 1–2 | 0.04–0.30 (= ceiling) | too few variants |
| 3 | cfg=1.0, strength=0.2 | 4–18 | 0.12–0.25 | best volume but no Tanimoto gain |
| 4 | strength=0.1 + anchor=0.6 + cfg=1 | 1 | 0.04–0.30 (= ceiling) | overly constrained |

None of the patches lifted Tanimoto above the seed's self-consistency
ceiling. The variants ARE different molecules (low Tanimoto to each other)
but they're not analogs of the seed.

## Why patches can't fix this

The c2c trajectory:

```
seed_smi  -- LIMO.encode -->  z_seed
                              ↓
                         q_sample (noise level controlled by strength)
                              ↓
                          z_t (noisy latent)
                              ↓
                         DDIM denoise + property/feasibility guidance
                              ↓
                           z_final
                              ↓
                         LIMO.decode (argmax)
                              ↓
                          variant_smi
```

The bottleneck is the **last step**: argmax decode of `z_final` is
unstable for energetic chemistry — even when z_final ≈ z_seed (which we
arrange via the anchor patch), `LIMO.decode(z_seed) ≠ seed_smi` for most
energetic seeds.

So the c2c output is bounded by **LIMO's decoder on energetic chemistry**,
not by the diffusion sampler. No anchor / strength / CFG knob alters the
decoder.

## What this implies for the breakthrough plan

Path A (de novo joint pool) is unaffected — it doesn't rely on
self-consistency. Path B (c2c) is **gated on LIMO v2**: motif-rich
fine-tune is expected to lift self-consistency from ≤ 0.15 to ≥ 0.5 on
energetic seeds, which makes the [0.30, 0.60] analog window reachable.

After LIMO v2 trains, re-run c2c with anchor_alpha=0.2 + cfg=2 +
strength=0.3. If self-consistency improves even modestly, c2c becomes the
fastest path to Class-A and Class-C breakthroughs.

## Saved artifacts

- `experiments/<v4b>/c2c_round1/` — anchor_alpha=0.3, cfg=2
- `experiments/<v4b>/c2c_round2/` — strength=0.1 + anchor_alpha=0.5
- `experiments/<v4b>/c2c_round3/` — cfg=1.0, strength=0.2
- `experiments/<v4b>/c2c_round4/` — combo

Each contains a `c2c_index.md` plus per-seed variant tables.
