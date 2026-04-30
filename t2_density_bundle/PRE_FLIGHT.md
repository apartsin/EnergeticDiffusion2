# T2 pre-flight: ML density cross-check on L1 / E1

## Question
Did the Chem. Mater. 2024 paper "Machine Learning Models for High Explosive
Crystal Density and Performance" (Stewart et al., LANL,
DOI:10.1021/acs.chemmater.4c01978) release model weights or code?

## Finding: NO public release
A web search (April 2026) of the paper page on `pubs.acs.org`, the PMC mirror,
the OSTI record, the arXiv preprint server, the LANL github org, and a
direct query for "DOI 10.1021/acs.chemmater.4c01978 github" returned no
public repository or weights file. The paper itself does not link to a
software supplement; OSTI lists the work at `osti.gov/biblio/2476700`
without a code/data deposit. Authors (Stewart, Coleman et al.) are LANL
researchers; the dataset (21k experimentally measured high-explosive
density / DFT-computed velocity / pressure) is also not publicly mirrored
as far as a public search can resolve.

A second-best fall-back was the Casey et al. 2020 3D-CNN paper
(DOI:10.1021/acs.jcim.0c00259); the same search found no public repo or
weights. The Force-field-inspired transformer network (Lin et al., 2023,
J. Cheminform.) is on github but trains on a smaller (~5k row) labelled
set and would itself be re-derivable from the public DataWarrior/EMDP
labelled corpus we already use; running it would not provide an
independent third estimator since its training data overlaps with our
DataWarrior subset.

## Decision: fall-back C, Bondi-vdW with bracketed packing factor
Following the EXPERIMENTATION_PLAN.md fall-back guidance, this bundle
runs the same Bondi vdW grid integration we use in m8/m2 DFT pipelines
(BLOCKED at the same B3LYP/6-31G(d) optimised geometry) and brackets the
packing factor at three values:

  pk = 0.65  (loose-packing lower bound)
  pk = 0.69  (production value, used in §5.2.2 anchor calibration)
  pk = 0.72  (tight-packing upper bound)

The bracket gives an honest plus/minus range for the predicted crystal
density without requiring an unpublished proprietary model.  The output
JSON also reports the bracket relative to the in-paper 6-anchor calibrated
densities (rho_cal = 2.09 g/cm3 for L1, 2.04 g/cm3 for E1) so the paper
can quote a delta and a packing-factor span.

For input geometry, since we do not have a Modal-side B3LYP/6-31G* optimised
geometry on hand, we use RDKit ETKDGv3 + MMFF94 single-conformer geometry
(the same starting geometry the m8 pipeline uses, before B3LYP). This
gives a slightly smaller volume than a B3LYP geometry; the production-
value (pk=0.69) row is therefore the right one to compare against the
6-anchor calibration target.

## What this changes in the paper
A new line in Appendix D for L1 and E1:
  "Bondi-vdW packing-factor bracket: rho_pk0.65 - rho_pk0.72 = [a, b]
   g/cm3, centred on the production pk=0.69 estimate of c g/cm3, vs the
   6-anchor calibrated rho_cal of d g/cm3 (delta = e %)."
This is an honest cross-check of the Bondi-vdW component of the
calibration; it cannot detect packing-factor mis-specification, but it
does rule out wide-cell / loose-packing alternative readings.
