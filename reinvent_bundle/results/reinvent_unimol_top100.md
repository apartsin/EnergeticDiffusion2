# REINVENT seed-42 top-100: UniMol 3D-CNN scores

Input: 100 SMILES (top-100 by N-fraction, seed 42, 40k pool)
Scored: 100 / 100 (rest: UniMol returned NaN)

## Summary (for Table 6a)

| Metric | Value |
|--------|-------|
| top-1 D (km/s) | 9.0199 |
| top-1 rho (g/cm3) | 1.8527 |
| top-1 P (GPa) | 34.5151 |
| top-1 N-fraction | 0.6 |
| top-1 max-Tani | 0.4878 |
| top-1 SMILES | `O=[N+]([O-])c1nnn(Cc2nnnn2O)n1` |
| mean D across top-100 | 7.484 |
| max D across top-100 | 9.02 |

## Top-10 by UniMol D

| Rank | SMILES | D (km/s) | rho | P (GPa) | N-frac | max-Tani |
|------|--------|----------|-----|---------|--------|----------|
| 1 | `O=[N+]([O-])c1nnn(Cc2nnnn2O)n1` | 9.0199 | 1.8527 | 34.5151 | 0.6 | 0.4878 |
| 2 | `Nc1c([N+](=O)[O-])c([N+](=O)[O-])nn2nnnc12` | 8.9266 | 1.9156 | 36.3712 | 0.5 | 0.4571 |
| 3 | `N#CCn1nnc([N+](=O)[O-])n1` | 8.727 | 1.8339 | 31.803 | 0.5455 | 0.5556 |
| 4 | `O=[N+]([O-])c1ncnnn1` | 8.6647 | 1.8389 | 35.1049 | 0.5556 | 0.3636 |
| 5 | `O=[N+]([O-])c1cnnnn1` | 8.6241 | 1.8754 | 35.2611 | 0.5556 | 0.48 |
| 6 | `O=[N+]([O-])Nc1nnc2nncn2n1` | 8.5885 | 1.8696 | 32.9469 | 0.6154 | 0.4839 |
| 7 | `[N-]=[N+]=Nc1nnnn1CCO[N+](=O)[O-]` | 8.4046 | 1.8443 | 32.6537 | 0.5714 | 0.3333 |
| 8 | `O=[N+]([O-])c1nnn(Cc2ncon2)n1` | 8.3349 | 1.7997 | 30.4488 | 0.5 | 0.4651 |
| 9 | `O=[N+]([O-])c1nonc1Nc1nnc2nncn2n1` | 8.2715 | 1.8944 | 32.1706 | 0.5556 | 0.4314 |
| 10 | `O=[N+]([O-])Cn1cnnn1` | 8.252 | 1.7991 | 29.9446 | 0.5556 | 0.3077 |