# D15: motif distribution — Tier-A/B vs high-HOF subset vs top-ranked candidates

- Tier-A/B HOF rows scanned: 2625
- Tier-A/B HOF >+200 kcal/mol: 231
- v6-multi top candidates: 20

| Motif | Tier-A/B % | High-HOF % | Top-cand % | Δ (top − A/B) |
|---|---|---|---|---|
| nitro | 95.2 | 81.4 | 90.0 | -5.2 |
| nitrate_ester | 4.2 | 0.4 | 5.0 | +0.8 |
| nitramine_NNO2 | 10.1 | 13.0 | 45.0 | +34.9 |
| azide | 0.8 | 8.2 | 0.0 | -0.8 |
| furazan | 2.6 | 16.9 | 0.0 | -2.6 |
| tetrazole | 2.0 | 11.3 | 0.0 | -2.0 |
| triazole | 2.7 | 12.1 | 0.0 | -2.7 |
| triazine | 0.4 | 3.0 | 0.0 | -0.4 |
| tetrazine | 0.0 | 0.0 | 0.0 | +0.0 |
| nitroso | 95.7 | 82.3 | 100.0 | +4.3 |
| dinitromethyl | 9.3 | 2.6 | 5.0 | -4.3 |
| polynitro | 23.4 | 42.0 | 10.0 | -13.4 |

**Reading**: Δ < 0 means top candidates *under-produce* this motif vs trusted training data — a likely bottleneck for HOF.