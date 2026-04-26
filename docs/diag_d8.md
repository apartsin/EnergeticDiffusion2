# D8: Tier-D label noise

Compares 3DCNN-smoke predictions against Tier-A/B ground truth on the rows where both exist.

| Property | n_rows | std(A/B) | MAE(smoke) | rel_MAE % | MAE/std % | verdict |
|---|---|---|---|---|---|---|
| density | 19,176 | 0.241 | 0.055 | 3.6 % | 23 % | **OK** |
| heat_of_formation | 2,625 | 284.785 | 126.519 | 130.3 % | 44 % | **noisy** |
| detonation_velocity | 2,539 | 1.520 | 0.346 | 5.8 % | 23 % | **OK** |
| detonation_pressure | 2,366 | 9.000 | 1.454 | 9.7 % | 16 % | **OK** |

Verdict thresholds (MAE / std):
- < 30 % = OK (smoke labels usable as is)
- 30-60 % = noisy (use, but with weight < 0.7)
- > 60 % = poison (drop or use only as auxiliary)