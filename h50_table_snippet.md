# h50 (impact-sensitivity, drop-height) per-lead predictions

Two independent free routes:

- **Route 1 (model)**: `score_model_v3e_h50` — sensitivity head fine-tuned against Huang & Massa h50 literature data on top of the LIMO 1024-d latent (calibration set ~307 rows; sens proxy `sigmoid((log10(40)-log10(h50))*1.5)`; inverted to cm at inference).

- **Route 2 (BDE)**: Politzer & Murray 2014 linear correlation `h50 (cm) = 1.93 * BDE(X-NO2) - 52.4`, with chemotype-heuristic BDE (Ar-NO2 ≈70, R-CH-NO2 ≈55, R2N-NO2 ≈47, R-O-NO2 ≈40 kcal/mol).

Sensitivity scale: h50 < 20 cm very sensitive; 20-50 cm sensitive; 50-100 cm moderately sensitive; >100 cm insensitive. RDX literature value ≈25-30 cm; TATB ≈140-490 cm.


## Table 1 (compact)

| id   | h50_model_cm | h50_BDE_cm |
|------|--------------|------------|
| L1 | 30.3 | 82.7 |
| L2 | 33.5 | 53.8 |
| L3 | 82.6 | 38.3 |
| L4 | 27.8 | 38.3 |
| L5 | 33.4 | 24.8 |
| L9 | 21.9 | 38.3 |
| L11 | 26.5 | 38.3 |
| L13 | 31.0 | 38.3 |
| L16 | 82.6 | 38.3 |
| L18 | 38.6 | 38.3 |
| L19 | 45.5 | 24.8 |
| L20 | 74.1 | 38.3 |
| RDX | 26.3 | 38.3 |
| TATB | 89.2 | 82.7 |

## Table D.1 (full)

| id   | chemotype          | BDE (kcal/mol) | h50_BDE_cm | h50_model_cm | within 30%? |
|------|--------------------|----------------|------------|--------------|-------------|
| L1 | nitroaromatic | 70.0 | 82.7 | 30.3 | no |
| L2 | unknown | 55.0 | 53.8 | 33.5 | no |
| L3 | nitramine | 47.0 | 38.3 | 82.6 | no |
| L4 | nitramine | 47.0 | 38.3 | 27.8 | yes |
| L5 | nitrate_ester | 40.0 | 24.8 | 33.4 | yes |
| L9 | nitramine | 47.0 | 38.3 | 21.9 | no |
| L11 | nitramine | 47.0 | 38.3 | 26.5 | no |
| L13 | nitramine | 47.0 | 38.3 | 31.0 | yes |
| L16 | nitramine | 47.0 | 38.3 | 82.6 | no |
| L18 | nitramine | 47.0 | 38.3 | 38.6 | yes |
| L19 | nitrate_ester | 40.0 | 24.8 | 45.5 | no |
| L20 | nitramine | 47.0 | 38.3 | 74.1 | no |
| RDX | nitramine | 47.0 | 38.3 | 26.3 | no |
| TATB | nitroaromatic | 70.0 | 82.7 | 89.2 | yes |

*Caption.* Per-lead impact-sensitivity h50 predictions from two independent routes. Route 1 is the literature-grounded `score_model_v3e_h50` head (Huang & Massa 2021 calibration set, ~307 rows). Route 2 is Politzer & Murray's published BDE-h50 linear fit applied to the chemotype-class BDE typical of the weakest X-NO2 bond. Sensitivity classes (cm): <20 very sensitive; 20-50 sensitive; 50-100 moderately sensitive; >100 insensitive. Anchors: RDX experimental h50 ≈25-30 cm; TATB ≈140-490 cm.
