# T4 pre-flight: searching for a 1,2,3,5-oxatriazole-class anchor

## Purpose
Find a literature 1,2,3,5-oxatriazole-class compound with BOTH (a)
published experimental crystal density rho_exp and (b) published
experimental detonation velocity D_exp, so that we can run our
B3LYP/6-31G(d) opt + wB97X-D3BJ/def2-TZVP single-point pipeline on it as
a true 7th anchor in the same heteroatom class as E1
(4-nitro-1,2,3,5-oxatriazole, SMILES `O=[N+]([O-])c1nnon1`).  The
existing m8 attempt (DNTF, 3,4-bis(4-nitrofurazan-3-yl)furazan) is a
furazan / 1,2,5-oxadiazole ring system, not the 1,2,3,5-oxatriazole
class.

## Search performed (April 2026)

I queried the open literature via web search and Google Scholar style
listing for the following candidate scaffolds and substituents.

### Class definition
1,2,3,5-oxatriazole = a 5-membered ring with the heteroatom sequence
O - N - N - N - C  (O at position 1, N at 2/3/5, C at 4).  Members of
the class would include:
  - parent 1,2,3,5-oxatriazole
  - 4-nitro-1,2,3,5-oxatriazole  (this is E1)
  - 4-amino-1,2,3,5-oxatriazole
  - 4-amino-1,2,3,5-oxatriazol-3(2H)-one
  - 4-nitro-1,2,3,5-oxatriazol-3(2H)-one and its 2-N-oxide
  - energetic salts (3-oxido, 3-oxide derivatives)

Note carefully: the often-cited "5-amino-1,2,3,4-oxatriazole-3-oxide"
(the most-studied energetic oxatriazole in the open literature) is
1,2,3,4-oxatriazole, NOT 1,2,3,5-oxatriazole, so it does not satisfy
condition (a) of "same heteroatom sequence as E1".

### Findings per candidate

1. **5-Amino-1,2,3,4-oxatriazole-3-oxide** (Politzer, Murray, Lane,
   _Propellants Explos. Pyrotech._ 2021, doi:10.1002/prep.202000243)
   - Wrong class (1,2,3,4 vs 1,2,3,5)
   - Properties are computational (B3LYP densities, K-J detonation
     velocities); no experimental rho_exp + D_exp pair.
   - Politzer's paper itself frames the family as a "potential
     framework" that has "not yet been studied".

2. **4-Amino-1,2,3,5-oxatriazole** and its 4-nitro analog
   - No open-literature single-crystal X-ray structures or detonation-
     velocity measurements found.  The closest is the 2025 _Structural
     Chemistry_ paper "Zwitterionic energetic materials containing an
     oxatriazole explosophore" (Springer 2025,
     doi:10.1007/s11224-025-02456-z), which is an "exploration of
     structure and performance" framed paper -- the experimental
     hooks are crystal structures of zwitterionic SALTS, not of the
     parent oxatriazole, and the detonation-velocity numbers are
     K-J-level computed, not experimental.

3. **3-substituted 1,2,3,5-oxatriazole salts** (e.g. ammonium,
   hydroxylammonium, hydrazinium)
   - Some exist on PubChem with computed properties only; no
     experimental D_exp value located.

4. **4-Nitro-1,2,3,5-oxatriazol-3(2H)-one** and its 2-N-oxide
   - Computational entries in EMDP / CAS exist; no experimental
     detonation-velocity measurement located in PubMed / Scopus / DOAJ.

### General observation
The 1,2,3,5-oxatriazole class is computationally well-explored
(Politzer 2021 and follow-ups) but experimentally under-explored.  The
synthesis literature is fragmentary, single-crystal X-ray structures
where they exist are often for non-energetic precursor amines or for
salts with non-energetic counterions, and quantitative detonation-
velocity / Vod measurements (which require gram-scale synthesis,
crystallisation, density measurement, and a small-charge initiation
test) have not been published for any neutral 1,2,3,5-oxatriazole-class
compound that I can find.

## Decision: do NOT launch the Modal job

Per the EXPERIMENTATION_PLAN.md fall-back ("If no viable candidate
exists, write up the negative finding so the paper §7 can cite ..."),
this bundle does NOT launch a DFT job.  We document the literature
search above and recommend the following replacement language for §7
in the paper:

> "We searched the open literature for a 1,2,3,5-oxatriazole-class
> compound with both experimental crystal density and experimental
> detonation velocity that could serve as a 7th DFT calibration
> anchor for E1 (4-nitro-1,2,3,5-oxatriazole). The class is
> computationally well-characterised (Politzer, Murray and Lane 2021;
> Springer 2025) but experimentally under-explored: at the time of
> writing, no neutral 1,2,3,5-oxatriazole-class compound appears in
> the open literature with both rho_exp and D_exp measured. The closest
> open-literature anchor would be the closely-related furazan
> (1,2,5-oxadiazole) scaffold DNTF, which we did add as a 7th anchor
> (Appendix D); however, DNTF differs from E1 in heteroatom sequence,
> which is why we retain E1's 'provisional co-headline' framing.
> Promoting E1 from provisional to confirmed therefore requires either
> (i) a synthesis-then-characterisation campaign that establishes
> rho_exp and D_exp for at least one 1,2,3,5-oxatriazole-class
> compound, or (ii) an indirect cross-validation via covolume EOS
> (T5)."

## Modal launcher present but disabled

The `modal_t4_oxatriazole_anchor.py` script in this directory is a
copy-modified template of `m8_bundle/modal_m8_oxatriazole_anchor.py`
that would run the same DFT pipeline once a viable target is identified.
It is committed in disabled form (the local entrypoint raises with
"PRE_FLIGHT FAILED -- no candidate") so that the bundle is ready to
launch as soon as a viable anchor compound surfaces.  To activate:

  1. Edit the `CANDIDATE_*` constants at the top of
     `modal_t4_oxatriazole_anchor.py` with the chosen SMILES and
     experimental references.
  2. Remove the `raise RuntimeError(...)` line in `main()`.
  3. `modal run modal_t4_oxatriazole_anchor.py`.
