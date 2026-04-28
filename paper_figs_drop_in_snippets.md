# Drop-in `<figure>` snippets for new paper visualisations

Each block below is ready to paste into `docs/paper/index.html`. The recommended
figure number and the existing table/figure it replaces are noted above each block.
Captions end with a bold-italic takeaway.

---

## H1 Lead-card grid (recommended Fig. 6, replaces Table 1)

```html
<figure id="fig-lead-cards-grid">
  <img src="figs/fig_lead_cards_grid.svg"
       alt="Twelve DGLD lead cards in a 3-by-4 grid; each cell shows the RDKit 2D depiction, lead ID, chemotype name, formula, and anchor-calibrated rho, D, P, with a coloured border encoding HMX-class pass status." />
  <figcaption>
    <strong>Figure 6.</strong> Twelve chem-pass DGLD leads (L1, L2, L3, L4, L5, L9,
    L11, L13, L16, L18, L19, L20). Each card carries the RDKit 2D depiction, the
    chemotype label, molecular formula, and the anchor-calibrated DFT/Kamlet-Jacobs
    triple (rho, D, P). Border colour indicates HMX-class pass status: green if
    rho >= 1.85 g/cc, D >= 9.0 km/s, and P >= 35 GPa all hold; orange for partial;
    red for fail or NaN-collapsed calibration. <strong><em>The grid replaces the
    text-only Table 1 with a structure-aware view, making at a glance which leads
    actually clear the HMX-class triple after DFT calibration.</em></strong>
  </figcaption>
</figure>
```

---

## H2 Baseline forest plot (recommended Fig. 7, replaces Table 4)

```html
<figure id="fig-baseline-forest">
  <img src="figs/fig_baseline_forest.svg"
       alt="Horizontal forest plot of top-1 Phase-A composite penalty (lower is better) for DGLD C0/C1/C2/C3 hazard-axis conditions, DGLD SA-axis conditions, SMILES-LSTM, and MolMIM 70M. Bars show mean and standard deviation across seeds; a dashed reference line marks the SMILES-LSTM value." />
  <figcaption>
    <strong>Figure 7.</strong> Forest plot of top-1 Phase-A composite penalty (mean
    +/- s.d. across 1-6 seeds, lower is better) for DGLD hazard-axis conditions
    (C0 unguided, C1 viab+sens, C2 viab+sens+hazard, C3 hazard-only), DGLD SA-axis
    conditions (C1v, C2vs, C3vs+SA), SMILES-LSTM (5k validated samples), and
    MolMIM 70M. Vertical dashed line marks the SMILES-LSTM reference; family
    colours separate DGLD-hazard (blue), DGLD-SA (green), SMILES-LSTM (red), and
    MolMIM (orange). <strong><em>SMILES-LSTM holds the best top-1 composite by a
    wide margin, while MolMIM 70M is dominated by every DGLD condition; this
    quantifies the head-to-head gap that the lead bundle must subsequently
    overcome via DFT validation.</em></strong>
  </figcaption>
</figure>
```

---

## H3 Distribution-learning small multiples (recommended Fig. 8, replaces / augments Table 6)

```html
<figure id="fig-moses-small-multiples">
  <img src="figs/fig_moses_small_multiples.svg"
       alt="Six small bar-chart panels comparing SMILES-LSTM and DGLD conditions on validity proxy, scaffold uniqueness, internal diversity, FCD, and two annotated panels for novelty and scaffold-NN-Tanimoto where data are not available." />
  <figcaption>
    <strong>Figure 8.</strong> Distribution-learning small-multiples comparing
    SMILES-LSTM (red) against the seven DGLD conditions (blue). Panels show
    validity proxy (validated SMILES per run), top-100 scaffold uniqueness,
    internal diversity (Tanimoto), and FCD vs the reference set; novelty and
    scaffold-NN-Tanimoto are reserved as annotated NOTE panels because those
    metrics are reported in Table 6 / Fig. 4 and were not recomputed for this
    pass. <strong><em>SMILES-LSTM dominates on FCD (0.52 vs 24-27 for DGLD) and
    scaffold uniqueness, while DGLD wins on internal diversity, exposing the
    classical mode-coverage versus targeted-search trade-off this paper
    foregrounds.</em></strong>
  </figcaption>
</figure>
```

---

## H4 DFT dumbbell + N-fraction residual (recommended Fig. 9, augments Section 5.13)

```html
<figure id="fig-dft-dumbbell">
  <img src="figs/fig_dft_dumbbell.svg"
       alt="Two-panel figure: left, per-lead dumbbell connecting 3D-CNN-predicted detonation velocity to anchor-calibrated DFT Kamlet-Jacobs detonation velocity, ordered by predicted D; right, scatter of N-fraction versus residual (DFT minus CNN) with linear fit and Pearson r." />
  <figcaption>
    <strong>Figure 9.</strong> Left, dumbbell plot connecting the 3D-CNN-predicted
    detonation velocity D (blue) to the anchor-calibrated DFT/Kamlet-Jacobs D
    (orange) for each DFT-converged lead; the dotted green line is the HMX-class
    9.0 km/s threshold. Right, residual (D_DFT_cal minus D_CNN) versus
    N-fraction (N atoms / total atoms) with linear fit and Pearson r overlaid;
    the figure title cites the global Pearson r from the 575-row pool reported
    in d9_kj_nfrac_table.json. <strong><em>The residual trends positive with
    rising N-fraction, reproducing the global N-rich bias of KJ relative to
    CNN and explaining why several leads jump from sub-HMX to HMX-class once
    DFT calibration is applied.</em></strong>
  </figcaption>
</figure>
```

---

## H5 xTB gap-gate strip (recommended Fig. 10, replaces / augments Table 5)

```html
<figure id="fig-xtb-strip">
  <img src="figs/fig_xtb_strip.svg"
       alt="Vertical strip of top xTB candidates; each row pairs an RDKit depiction with the xTB GFN2 HOMO-LUMO gap in eV, a pass or fail verdict against a 1.5 eV gate, and a graph-survives flag. Pass rows are bordered green, fail rows red." />
  <figcaption>
    <strong>Figure 10.</strong> xTB GFN2 HOMO-LUMO gap gate for the top
    candidates from <code>experiments/xtb_topN.json</code>. Each row shows the
    RDKit 2D depiction, optimised xTB gap (eV), the 1.5 eV pass/fail verdict
    used as a sensitivity-proxy gate, and whether the input graph survived xTB
    optimisation (post-SMILES intact vs altered). Green border indicates pass,
    red indicates fail. <strong><em>Most ranked candidates either fall below
    the 1.5 eV gate or fail the graph-survival check, which is exactly why the
    DFT bundle in Fig. 6 was filtered down to the chemist-curated 12 leads
    rather than promoted directly from the xTB ranking.</em></strong>
  </figcaption>
</figure>
```
