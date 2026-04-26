# D10: conditioning signal correlation
Model: `diffusion_subset_cond_expanded_v4b_20260426T000541Z`

Same z_t,t fed three ways: (A) target_z=+1.281 for prop p, (B) target_z=-1.281, (C) all-zero unconditional. Measures cosine(eps_A − eps_C, eps_B − eps_C) — should be **negative** (opposite targets push eps in opposite directions). Cosine ≈ 0 means the model is ignoring conditioning.

| Property | t=100 | t=500 | t=900 | mean | verdict |
|---|---|---|---|---|---|
| density | +0.079 | +0.471 | +0.080 | +0.210 | **broken** |
| heat_of_formation | +0.979 | +0.980 | +0.757 | +0.905 | **broken** |
| detonation_velocity | +0.967 | +0.986 | +0.665 | +0.873 | **broken** |
| detonation_pressure | +0.966 | +0.980 | +0.661 | +0.869 | **broken** |

verdict: **strong** if cosine < -0.3 (good signal); **weak** if -0.3..0; **broken** if positive.