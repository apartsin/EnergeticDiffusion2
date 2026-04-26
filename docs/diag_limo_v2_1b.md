# LIMO root-cause bundle — L5 + L10 + L13

## L5 — latent-norm distribution

| version | n | mean | p10 | p90 | expected (√d) | ratio | verdict |
|---|---|---|---|---|---|---|---|
| current | 2000 | 8.03 | 5.80 | 10.11 | 32.00 | 0.25 | scale-mismatch |
| baseline | 2000 | 7.92 | 5.98 | 9.70 | 32.00 | 0.25 | scale-mismatch |

## L10 — motif linear-probe AUC

| version | nitro | nitramine | furazan | tetrazole | triazole | azide | polynitro3 |
|---|---|---|---|---|---|---|---|
| current | 0.96 | 0.78 | 0.61 | 0.61 | 0.84 | n/a | 0.97 |
| baseline | 0.96 | 0.79 | 0.62 | 0.60 | 0.82 | n/a | 0.98 |

Verdict: ≥ 0.85 strong, 0.7–0.85 ok, < 0.7 weak.

## L13 — self-consistency on energetic seeds

| seed | current | baseline | Δ |
|---|---|---|---|
| TNT | 0.13 | 0.13 | +0.00 |
| RDX | 0.39 | 0.50 | -0.11 |
| HMX | 0.12 | 0.12 | -0.01 |
| PETN | 0.37 | 0.46 | -0.09 |
| NTO | 0.13 | 0.31 | -0.19 |
| TATB | 0.23 | 0.13 | +0.10 |
| CL-20 | 0.09 | 0.12 | -0.03 |
| FOX-7 | 0.04 | 0.04 | +0.00 |
| LLM-105 | 0.00 | 0.00 | +0.00 |
| DAAF | 0.24 | 0.44 | -0.21 |

## Summary verdicts

- **L5**: scale-mismatch  (mean ‖z‖ = 8.03, expected 32.00)
- **L10**: 2 weak motif(s): ['furazan', 'tetrazole']
- **L13**: 0/10 exact roundtrip; mean Tanimoto 0.17
- **L13 vs baseline**: mean ΔTanimoto = -0.055; improved on 1/10 seeds.