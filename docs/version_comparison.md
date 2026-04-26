# Version comparison (best CFG per cell)

metric: `rel_mae_pct`  (lower better)

- **v1**: `diffusion_subset_cond_expanded_20260425T095335Z` (n=50)
- **v2**: `diffusion_subset_cond_expanded_v2_20260425T121727Z` (n=50)
- **v3**: `diffusion_subset_cond_expanded_v3_20260425T140941Z` (n=50)
- **v4**: `diffusion_subset_cond_expanded_v4_20260425T160108Z` (n=50)
- **v4-nf**: `diffusion_subset_cond_expanded_v4_nofilter_20260425T175119Z` (n=50)
- **v4-B**: `diffusion_subset_cond_expanded_v4b_20260426T000541Z` (n=50)
- **v5**: `diffusion_subset_cond_expanded_v5_20260425T224932Z` (n=50)

| Property | q | target | v1 | v2 | v3 | v4 | v4-nf | v4-B | v5 |
|---|---|---|---|---|---|---|---|---|---|
| density | q10 | +1.21 | **3@g7.0** | 3@g7.0 | 3@g2.0 | 3@g5.0 | 3@g7.0 | 6@g7.0 | 6@g7.0 |
| density | q50 | +1.52 | 12@g7.0 | 13@g2.0 | 12@g2.0 | 12@g2.0 | **11@g2.0** | 11@g2.0 | 13@g2.0 |
| density | q90 | +1.83 | 16@g7.0 | 13@g7.0 | 16@g5.0 | 17@g7.0 | **11@g7.0** | 12@g7.0 | 18@g7.0 |
| | | | | | | | | | | |
| heat_of_formation | q10 | -461.94 | 29@g7.0 | 26@g2.0 | **22@g2.0** | 31@g5.0 | 25@g7.0 | 34@g2.0 | 30@g2.0 |
| heat_of_formation | q50 | -97.06 | 169@g7.0 | 272@g2.0 | 199@g7.0 | 233@g5.0 | 213@g7.0 | **140@g7.0** | 194@g7.0 |
| heat_of_formation | q90 | +267.82 | 127@g7.0 | 162@g7.0 | **111@g7.0** | 178@g7.0 | 153@g7.0 | 171@g7.0 | 162@g7.0 |
| | | | | | | | | | | |
| detonation_velocity | q10 | +3.97 | 10@g7.0 | **7@g7.0** | 8@g7.0 | 7@g7.0 | 9@g7.0 | 33@g2.0 | 14@g7.0 |
| detonation_velocity | q50 | +5.92 | 24@g7.0 | 31@g2.0 | 24@g7.0 | 26@g2.0 | **18@g7.0** | 19@g7.0 | 23@g2.0 |
| detonation_velocity | q90 | +7.86 | 30@g7.0 | 22@g7.0 | 27@g5.0 | 31@g7.0 | 19@g7.0 | **12@g7.0** | 30@g2.0 |
| | | | | | | | | | | |
| detonation_pressure | q10 | +3.51 | 33@g7.0 | 40@g5.0 | 21@g5.0 | 46@g5.0 | **21@g5.0** | 183@g2.0 | 93@g7.0 |
| detonation_pressure | q50 | +15.04 | 48@g5.0 | 59@g2.0 | 50@g5.0 | 51@g2.0 | 42@g7.0 | **36@g5.0** | 45@g2.0 |
| detonation_pressure | q90 | +26.57 | 51@g7.0 | 48@g7.0 | 47@g5.0 | 48@g7.0 | 41@g7.0 | **26@g7.0** | 52@g7.0 |
| | | | | | | | | | | |