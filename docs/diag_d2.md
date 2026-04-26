# D2: LIMO encoder–decoder roundtrip on high-D molecules

50 highest-D Tier-A/B SMILES → encode → argmax-decode → canonicalise.

- exact recovery: **2/50** (4 %)
- mean Tanimoto when both decode: **0.176** (n=50)
- decoded SMILES still containing NO2: **42/50** (84 %)
- verdict: **weak**

## Failure examples (truth → decoded)
