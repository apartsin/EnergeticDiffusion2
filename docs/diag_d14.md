# D14: property correlations (Tier-A/B only)

## Pearson r between properties

| | density | heat_of_formation | detonation_velocity | detonation_pressure |
|---|---|---|---|---|
| **density** | +1.00 | +0.15 | +0.89 | +0.93 |
| **heat_of_formation** | +0.15 | +1.00 | +0.16 | +0.16 |
| **detonation_velocity** | +0.89 | +0.16 | +1.00 | +0.99 |
| **detonation_pressure** | +0.93 | +0.16 | +0.99 | +1.00 |

## High-HOF rows: do they have high D / P / density?

Pick rows in top-10 % of HOF, report mean of other props vs whole-set mean.

| metric | top-10 % HOF | all Tier-A/B | Δ |
|---|---|---|---|
| HOF cutoff (top-10 %) | +175.4 | – | – |
| density (mean) | 1.695 | 1.517 | +0.178 (+0.74 σ) |
| detonation_velocity (mean) | 7.442 | 5.918 | +1.524 (+1.00 σ) |
| detonation_pressure (mean) | 24.763 | 15.039 | +9.724 (+1.08 σ) |