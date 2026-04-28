# E2 SMARTS audit: gem-dinitro nitramine (`O=[N+]([O-])NC([N+](=O)[O-])[N+](=O)[O-]`)

## Background

E2 carries 3 nitro groups: 2 gem-dinitro on a single sp3 carbon plus 1
nitramine (N-NO2) on the adjacent nitrogen. The §4.5 prose mentions a
hard-filter rule "≥3 nitro groups on a 1- or 2-carbon skeleton".

## (a) The existing rule

There are TWO chem-filter modules in the repository, used at different
stages of the pipeline:

1. `m6_postprocess_bundle/chem_redflags.py::screen()` (the production
   reranker filter, also mirrored in `scripts/diffusion/chem_redflags.py`).
   It contains an explicit hard-reject rule on line 238:
   ```python
   if nC <= 2 and nNO2 >= 3:
       out["reasons"].append(f"polynitro_C{nC}_chain:{nNO2}_nitro"); return out
   ```
   Plus a SMARTS `trinitromethyl_aliphatic`:
   `[CX4]([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]` (severity "strong",
   not reject).

2. `m6_postprocess_bundle/chem_filter.py::chem_filter_batch()` (the
   E-set mining filter, used by `scripts/mine_extension_set.py`). This
   module is intentionally lighter: it catches `N4_chain`, peroxide,
   `nitro_alkyne`, and halogen / P / S, but has no rule on small-skeleton
   polynitro at all.

## (b) Should E2 have matched?

Tested with RDKit (see `_e2_audit_check.py`):

- `screen(E2)` -> `status: reject, reasons: ['polynitro_C1_chain:3_nitro']`.
  E2 is correctly rejected by the production reranker filter via the
  `nC<=2 and nNO2>=3` hard rule (E2 has nC=1, nNO2=3).
- `chem_filter_batch(E2)` (the E-set mining filter) -> `pass`. The
  light filter does not have this rule.
- The narrower SMARTS `trinitromethyl_aliphatic` does NOT match E2:
  the third NO2 is on the nitramine nitrogen, not on the central
  carbon, so the SMARTS pattern (3 NO2 on the same sp3 C) misses it.

## (c) Recommendation

**This is a Stage-1-vs-Stage-2 filter scope mismatch, not a
production-rule miss.** The §4.5 hard rule is correctly implemented in
`chem_redflags.screen()`; the E-set mining script bypasses it because
it uses the lighter `chem_filter` (designed to be permissive at the
500-candidate distributional view). The L-set pipeline runs through
`screen()` and would have rejected E2.

For paper §5.7, the honest framing is: E2 is a candidate that the
distributional-view filter retained but the headline production filter
rejects — exactly the kind of scaffold the lighter Stage-1 mining is
meant to surface for inspection. We do NOT recommend tightening
production code; we recommend documenting that the E-set filter is
intentionally looser than the L-set filter.

## (d) Tightened SMARTS (if useful for future filters)

For completeness, the SMARTS that catches gem-dinitro-WITH-adjacent-
nitramine (E2-class, not currently in `ALERTS`):

```
[CX4]([N+](=O)[O-])([N+](=O)[O-])[NX3][N+](=O)[O-]
```

False-positive scan against L1-L20 + R-rejects + RDX/HMX/PETN/FOX-7/
TATB/NTO/TNT: **0 hits** (the validated leads do not carry this motif,
and HMX-style cyclic nitramines have CH2 between the C and the N-NO2
so the [CX4] gem-dinitro carbon is not present).

This SMARTS could be added to `ALERTS` with severity `"reject"` if the
narrow gem-dinitro nitramine motif is judged universally too sensitive
to keep, but it is not currently necessary because the existing
`nC<=2 and nNO2>=3` hard-reject already covers E2.
