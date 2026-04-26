# EXPLO5 Academic License Request

**Recipient:** Prof. Muhamed Sućeska
**Email:** muhamed.suceska@fkit.unizg.hr
**Affiliation:** Faculty of Chemical Engineering and Technology, University of Zagreb, Croatia
**Alt contact (his collaborator):** robert.matyas@upce.cz

---

## Draft email

**Subject:** Request for EXPLO5 academic license — generative ML for CHNO energetic materials

Dear Prof. Sućeska,

I am working on a machine-learning project that develops a generative model
for novel CHNO energetic materials. Our current labeled dataset contains
~65,000 molecules, but only a small fraction have experimental detonation
velocity and pressure (roughly 600 and 400 molecules respectively). The
remaining labels come from the Kamlet-Jacobs formula or from
data-driven ML predictors, neither of which match the accuracy of a proper
CJ thermochemistry calculation.

I would like to request an academic (non-commercial) license for EXPLO5 so
that I can compute detonation velocity and pressure for a curated subset of
our molecules (initially ~5,000 to 10,000 CHNO structures with experimental
or DFT-computed density and HOF already in hand). The outputs would be used
exclusively for training and validation of a generative model, and the
resulting predictions would be tagged as EXPLO5-derived in our public data
release.

Specific questions:

1. Is an academic license available for this non-commercial use?
2. What version of EXPLO5 would you recommend for batch-scripting CHNO
   calculations (ideally driven from Python via command-line or Fortran
   input files)?
3. Are there any constraints on how we cite or release the EXPLO5-derived
   values in a public dataset, beyond the standard citation of your paper
   (Sućeska, *Propellants Explos. Pyrotech.* 1999, 24, 280)?

I would be happy to share the final dataset with you for validation and to
add proper acknowledgement in any publication that uses it.

Thank you very much for your time.

Kind regards,

[Your name]
[Your affiliation]
[Your ORCID]
[Link to the project / GitHub if public]

---

## What to include in the reply

Sućeska typically responds within 1-2 weeks. He may ask for:
- Proof of academic affiliation (university email is usually sufficient)
- Short description of the intended use (the paragraph above covers it)
- Occasionally: a preprint or link to a prior related paper

Once granted, you receive a Windows installer + license file. EXPLO5 runs
on Windows; it is command-line scriptable via input/output text files, so
it is straightforward to batch-drive from Python.
