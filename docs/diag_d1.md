# D1: Validator self-consistency (3DCNN smoke ensemble)

Validator runs on Tier-A/B SMILES whose values are *known*. Compare predicted to ground truth.

| Property | n | MAE | rel_MAE % | r | verdict |
|---|---|---|---|---|---|
| density | 1500 | 0.055 | 3.6 % | 0.959 | **strong** (MAE/std=23%) |
| heat_of_formation | 1500 | 123.269 | 131.8 % | 0.723 | **ok** (MAE/std=44%) |
| detonation_velocity | 1500 | 0.341 | 5.7 % | 0.948 | **strong** (MAE/std=22%) |
| detonation_pressure | 1500 | 1.468 | 9.7 % | 0.977 | **strong** (MAE/std=16%) |

verdict: strong = validator < 25 % of std error; ok = 25–50 %; weak = > 50 % (unreliable as ground truth)