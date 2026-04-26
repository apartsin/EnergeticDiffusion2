# Full diagnostic summary (8 diagnostics on v4-B / v6 pipeline)

All run at 2026-04-26 against `diffusion_subset_cond_expanded_v4b_20260426T000541Z`.

| # | Diagnostic | Verdict | Headline |
|---|---|---|---|
| **D1** | Validator self-consistency | density / D / P **strong** (r ≥ 0.95); HOF **ok** (r=0.72) | metric is reliable for D / P / ρ; HOF metric partly clouded by validator noise |
| **D2** | LIMO encode-decode roundtrip on top-D molecules | **weak** | 2 / 50 exact recovery; mean Tanimoto 0.18; 84 % retain NO₂. **LIMO is a bottleneck** |
| **D3** | Property predictability from latents | density **strong** (r=0.94), P **strong** (r=0.89), D **ok** (r=0.85), HOF **ok** (r=0.70) | latent geometry encodes properties → conditioning is *recoverable* |
| **D5** | Out-of-range conditioning (z=+3) | **saturated** for all 4 properties | pred mean barely changes z=+1.28 → z=+3; max approaches target only for ρ / D / P |
| **D8** | Tier-D label noise (3DCNN smoke vs Tier-A/B truth) | density / D / P **OK** (MAE/std 16–23 %); HOF **noisy** (44 %) | smoke labels are decent for ρ/D/P; HOF labels are partly poisoning |
| **D10** | Conditioning-signal correlation | **broken** for all 4 properties | denoiser uses *which* property (mask), not *what value*; cos = +0.21 (ρ) to +0.91 (HOF) |
| **D14** | Property correlations on Tier-A/B | HOF decoupled from ρ/D/P (r=0.15) | high-HOF rows still have higher ρ/D/P (+0.7 to +1.1 σ); joint optimisation is possible |
| **D15** | Motif distribution: training vs top candidates | top candidates miss N-rich rings | furazan/tetrazole/triazole/azide all 0 % in top vs 8–17 % in real high-HOF; nitramine over-represented (45 % vs 13 %) |

---

## Causal chain identified

```
[D2] LIMO can't decode N-rich heterocycles cleanly
        │
        ▼
[D15] Top candidates lack furazan/tetrazole/triazole/azide
        │
        ▼
HOF q90 ceiling (max +257 kcal/mol vs target +268)
        │
        └─── unrelated: [D10] denoiser ignores cond value
                          (sidestepped successfully by rerank)
```

Other findings sit on the side:
- [D1] / [D3] / [D8] / [D14] establish that the **validator and the latent geometry
  and the data correlations** are all healthy enough to support a fix.
- [D5] confirms that more aggressive conditioning targets won't help — the
  ceiling is in the support of LIMO's decoder, not in the conditioning input.

---

## What's solved already

- ρ / D / P at q90: **rerank pool = 1500 → rel_MAE 0.2–0.9 %, 100 % within-10 %**.
- HOF q90 mean: **−156 → +166 kcal/mol** with bigger pool (single-prop rerank).
- HOF q90 max: **+185 → +257 kcal/mol** (1500 pool).

## What's left

| Gap | Fixable by | Cost |
|---|---|---|
| HOF q90 within-10 still only 2 % | Active-learning loop (DFT-validate top → augment Tier-B → retrain) | ~1.5 days/cycle |
| LIMO can't produce furazan / tetrazole / triazole / azide | Motif-aware LIMO retrain, or switch base VAE | 6–12 h GPU |
| D10 cond signal broken (architecturally) | Cross-attention / DiT denoiser or contrastive auxiliary loss | 1 day code + 90 min train |
| Top candidates have unbalanced formal charges | Add `--require_neutral` filter to rerank | 30 min code |

Recommended next move: **motif-aware LIMO retrain** — biggest expected lift,
addresses the root cause identified by D2 + D15 simultaneously, and it makes
the subsequent active-learning loop more efficient because the new LIMO will
be able to decode the chemistry the loop is trying to discover.
