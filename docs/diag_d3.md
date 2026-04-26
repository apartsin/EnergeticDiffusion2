# D3: Property predictability from LIMO latents

MLP 1024→512→512→1, 80/20 split on Tier-A/B latents.

| Property | n_train | n_test | r | MAE | rel_MAE % | verdict |
|---|---|---|---|---|---|---|
| density | 15,340 | 3,836 | 0.941 | 0.060 | 4.0 % | **strong** |
| heat_of_formation | 2,100 | 525 | 0.702 | 136.959 | 137.4 % | **strong** |
| detonation_velocity | 2,031 | 508 | 0.847 | 0.583 | 9.9 % | **ok** |
| detonation_pressure | 1,892 | 474 | 0.893 | 2.910 | 18.9 % | **strong** |