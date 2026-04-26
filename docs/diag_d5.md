# D5: out-of-range conditioning
Model: `diffusion_subset_cond_expanded_v4b_20260426T000541Z`

Tests whether the model can extrapolate beyond q90 (z=+1.281).
If predicted property at z=+3 ≈ z=+1.281, the model has saturated.

| Property | target z | target raw | pred mean | pred max | rel_MAE % |
|---|---|---|---|---|---|
| density | q90 | +1.83 | +1.63 | +1.83 | 10.7 % |
| density | z=+2 | +2.00 | +1.64 | +1.88 | 18.0 % |
| density | z=+3 | +2.24 | +1.54 | +1.88 | 31.4 % |
| heat_of_formation | q90 | +267.82 | -190.91 | +86.76 | 171.3 % |
| heat_of_formation | z=+2 | +472.62 | -163.18 | +50.23 | 134.5 % |
| heat_of_formation | z=+3 | +757.46 | -202.96 | +86.34 | 126.8 % |
| detonation_velocity | q90 | +7.86 | +7.34 | +9.32 | 11.3 % |
| detonation_velocity | z=+2 | +8.96 | +7.21 | +8.76 | 19.5 % |
| detonation_velocity | z=+3 | +10.48 | +5.66 | +9.01 | 46.0 % |
| detonation_pressure | q90 | +26.57 | +22.15 | +35.89 | 27.5 % |
| detonation_pressure | z=+2 | +33.04 | +23.11 | +35.09 | 30.5 % |
| detonation_pressure | z=+3 | +42.04 | +21.08 | +36.14 | 49.9 % |

If 'z=+3' pred mean is similar to q90 pred mean → saturated.