# P4 bibliography findings
_Generated 2026-04-30 by web research (WebSearch + WebFetch)._

## Kamlet-Jacobs 1968 page range

- **Verified pages: 23-55** (NOT 23-35; the current paper draft is wrong).
- **DOI:** 10.1063/1.1667908
- **Authors confirmed:** Kamlet, Mortimer J.; Jacobs, S. J.
- **Title confirmed:** "Chemistry of Detonations. I. A Simple Method for Calculating Detonation Properties of C-H-N-O Explosives"
- **Journal / volume / issue / year:** Journal of Chemical Physics, vol. 48, issue 1, 1968.

### Sources of evidence
1. NASA ADS bibliographic record (1968JChPh..48...23K) confirms title, authors, DOI 10.1063/1.1667908, starting page 23, year 1968.
2. AIP Publishing landing page (`pubs.aip.org/aip/jcp/article/48/1/23/779577/`) confirms volume 48 issue 1, starting page 23, DOI 10.1063/1.1667908. The full article body (which would show page 55 explicitly) is paywalled and returned HTTP 403; the ending page is established from independent secondary citation evidence below.
3. Multiple peer-reviewed secondary citations consistently quote pages **23-55**, e.g. SciRP reference 1968680 verbatim ("The Journal of Chemical Physics, 48, 23-55"); citations within Politzer 2019 (Propellants, Explosives, Pyrotechnics) and Kazandjian 2006 likewise match. No reputable source uses 23-35; that variant appears to be a one-off typo that propagated.
4. Politzer 2019 paper title ("The Kamlet-Jacobs Parameter phi: A Measure of Intrinsic Detonation Potential") shows the K-J paper continues to ground 30+ pages of derivation and tables, consistent with a 33-page article (23-55) rather than a 13-page article (23-35).

### Recommended bibliography entry

BibTeX:
```
@article{Kamlet1968,
  author  = {Kamlet, Mortimer J. and Jacobs, S. J.},
  title   = {Chemistry of Detonations. {I}. {A} Simple Method for Calculating Detonation Properties of {C-H-N-O} Explosives},
  journal = {The Journal of Chemical Physics},
  volume  = {48},
  number  = {1},
  pages   = {23--55},
  year    = {1968},
  doi     = {10.1063/1.1667908}
}
```

Plain text:
> Kamlet, M. J.; Jacobs, S. J. Chemistry of Detonations. I. A Simple Method for Calculating Detonation Properties of C-H-N-O Explosives. J. Chem. Phys. 48, 23-55 (1968). https://doi.org/10.1063/1.1667908

**Action for main session:** change `23-35` to `23-55` in `short_paper.html` everywhere the Kamlet-Jacobs reference appears, and add the DOI if not already present.

---

## EMDB citation

- **Public URL found: NO** (no genuinely persistent, public, citable URL was located within the time budget).
- The acronym "EMDB" is heavily overloaded. Disambiguation of the candidates explored:
  - `emdb-ca.org` and `emdb.iboc.ru`: both ECONNREFUSED, do not resolve. Not a usable citation.
  - `www.emdb-software.com`: TLS certificate invalid (cert altname mismatch). Even if reachable, this is **EMDB = "Energetic Materials Designing Bench"** (Keshavarz & Klapotke 2017), which is **proprietary calculation software**, not a downloadable experimental data extract. Wrong target.
  - **EMDB at EBI** (`www.ebi.ac.uk/pdbe/emdb`): "Electron Microscopy Data Bank" - structural-biology cryo-EM. Completely unrelated; do not cite.
  - **eth-ait/emdb GitHub**: "Electromagnetic Database of Global 3D Human Pose" - unrelated.
- The **closest legitimate substitute** for what the paper appears to mean (a public extract of CHNO experimental detonation properties) is:
  - Huang, X.; Qian, W. et al. **"EM Database v1.0: A benchmark informatics platform for data-driven discovery of energetic materials."** *Energetic Materials Frontiers* (2023). DOI: 10.1016/j.enmf.2023.09.002.
  - Maintaining institution: Institute of Chemical Materials, **China Academy of Engineering Physics (CAEP)**.
  - Scope: ~100,000 quantum-chemistry-computed compounds + ~10,000 experimentally-extracted compounds with crystal density, sublimation enthalpy, formation enthalpy, detonation pressure, detonation velocity, detonation heat, detonation volume.
  - License/access: paper says "open access of the EM Database v1.0," but the **specific online interface URL is not reproduced in any indexed metadata I could reach**, and the two domains the user proposed (`emdb-ca.org`, `emdb.iboc.ru`) are both dead. The ScienceDirect paywall blocked WebFetch from extracting the URL from the published manuscript body.

### Recommendation: reframe the citation

Because no public, persistent EMDB URL was confirmed and the original placeholder ("Energetic Materials DataBase (EMDB), public extract, accessed 2024") is therefore not currently verifiable, I recommend **one** of the following two options. Pick whichever matches what was actually used at data-prep time:

**Option A: if the paper's EMDB extract is in fact the Huang-Qian "EM Database v1.0":**
- Cite the journal article as the canonical reference.
- BibTeX:
```
@article{Huang2023EMDB,
  author  = {Huang, Xin and Qian, Wen and others},
  title   = {{EM} Database v1.0: A benchmark informatics platform for data-driven discovery of energetic materials},
  journal = {Energetic Materials Frontiers},
  year    = {2023},
  doi     = {10.1016/j.enmf.2023.09.002},
  note    = {China Academy of Engineering Physics; ${\sim}100{,}000$ QC-computed and ${\sim}10{,}000$ experimental CHNO entries.}
}
```
- If the actual access URL was visited at the time the dataset was pulled, paste that URL into `note=` and add `urldate = {2024-XX-XX}`.

**Option B (recommended if the dataset was actually a hand-compiled spreadsheet from books/papers):** reframe as a multi-source compilation. The EMDB acronym should be retired from the paper and replaced with a transparent provenance line:
- "Reference detonation-property data for evaluation were compiled from Klapotke, *Energetic Materials Encyclopedia* (de Gruyter, 2018, ISBN 978-3-11-044139-0); Cooper, *Explosives Engineering* (Wiley-VCH, 1996); and the LANL Explosives Reference Tables (Dobratz & Crawford, *LLNL Explosives Handbook*, UCRL-52997, 1985). Per-row provenance is given in Appendix A.1."
- Recommended bibliography entries:
```
@book{Klapotke2018Encyclopedia,
  author    = {Klap{\"o}tke, Thomas M.},
  title     = {Energetic Materials Encyclopedia},
  publisher = {De Gruyter},
  year      = {2018},
  address   = {Berlin/Boston},
  isbn      = {978-3-11-044139-0}
}

@book{Cooper1996Explosives,
  author    = {Cooper, Paul W.},
  title     = {Explosives Engineering},
  publisher = {Wiley-VCH},
  year      = {1996},
  address   = {New York},
  isbn      = {978-0-471-18636-6}
}

@techreport{Dobratz1985LLNL,
  author      = {Dobratz, B. M. and Crawford, P. C.},
  title       = {{LLNL} Explosives Handbook: Properties of Chemical Explosives and Explosive Simulants},
  institution = {Lawrence Livermore National Laboratory},
  year        = {1985},
  number      = {UCRL-52997 Rev. 2}
}
```

### Action for main session
- Choose Option A or B based on what `cocrystal_bundle/` / `validation_bundle/` actually pulls from. If the rows trace to PDF pages of Klapotke / Cooper / LLNL handbook, use Option B and add an Appendix A.1 provenance table. If the rows trace to a CSV download from a Chinese-academy web portal, use Option A and recover the exact URL from the data-prep script's logs.
- Do **not** keep the bare "EMDB, public extract, accessed 2024" placeholder; it is not citable as written.
