# DGLD visual-abstract prompts (4 candidate styles)

## Specification

The visual abstract should communicate, at a single glance:

1. **Domain**: energetic-materials chemistry (CHNO molecules; nitro groups visible)
2. **The bottleneck**: a tiny labelled core (RDX, HMX, CL-20, TATB) inside a much larger pool of weakly-labelled molecules
3. **The method**: latent diffusion in a frozen VAE space, classifier guidance from multi-head property predictors, chemistry-credibility filter
4. **The output**: a top candidate (trinitro-1,2-isoxazole, L1) at HMX-class properties (rho ≈ 2 g/cm³, D ≈ 9.5 km/s)
5. **The validation chain**: DFT confirmation, xTB stability, retrosynthesis check
6. **The "productive quadrant" framing**: novel + on-target

## Candidate prompts

Each candidate emphasises a different visual metaphor.

### Candidate A — Pipeline schematic, editorial style

> Editorial scientific-illustration in a deep navy + gold + cream palette, aspect ratio 16:9, suitable for a Nature Machine Intelligence visual abstract. Three-panel left-to-right layout. Left panel: a stylised pyramid of molecule cards labelled "experimental (3000)", "DFT (9000)", "Kamlet-Jacobs (25000)", "3D-CNN surrogate (30000)" — only the top two layers glow gold ("trusted"). Centre panel: a frozen VAE encoder feeding a latent cloud, then a denoising-diffusion process visualised as a series of concentric circles converging to a single bright point. Multiple property gauges (density, detonation velocity, heat of formation, sensitivity) feed gradient arrows into the diffusion process. Bottom-edge filter labelled "chemistry rules". Right panel: a chemical structure rendered as a 2D skeletal formula of a trinitroisoxazole (5-membered ring with three NO2 groups), with a small inset showing a 3D ball-and-stick model and the label "rho=2.00, D=9.56, P=40.5". A subtle background of TNT-like benzene rings fades into the dataset pyramid. White background; clean serif typography. No people, no hands, no smiling robots. Editorial energy, Nature paper, not commercial AI art.

### Candidate B — Productive-quadrant scatter, framed as the headline

> Scientific data-visualisation graphic, 16:9, deep navy + gold accent on cream. Centred 2D scatter plot with axis labels "novelty (1 - max Tanimoto)" on x and "predicted detonation velocity D (km/s)" on y. The plot space is divided into four quadrants by faint gridlines. Most points (small grey dots) cluster in the bottom-left and bottom-right; one cluster of bright gold points sits ALONE in the top-right quadrant labelled "DGLD (productive quadrant: novel AND HMX-class)". Two single red points labelled "SMILES-LSTM" and "MolMIM 70M" sit far from the gold cluster - one at novelty=0 and one at low-D. Several small chemical-structure thumbnails (skeletal formulas, nitro groups visible) annotate the gold cluster with arrows. A single large gold thumbnail is highlighted at top-right with the label "L1 trinitro-1,2-isoxazole, rho=2.00, D=9.56 km/s". Top of image: small all-caps title "DGLD: Domain-Gated Latent Diffusion for Energetic Materials". Editorial style, no AI-art kitsch.

### Candidate C — The chemotype rediscovery story

> A visual narrative in three vertically-stacked frames, vertical 4:5 aspect ratio. Cream background, navy line-art with gold highlights. Frame 1 (top): a labyrinthine cloud of small molecule shapes labelled "65,980 known energetic molecules"; one specific molecule (3,4,5-trinitro-1H-pyrazole, the published Hervé compound) glows. Frame 2 (middle): a geometric depiction of a latent space with a diffusion trajectory threading from a Gaussian distribution toward a bright target region; small grey arrows representing classifier-guidance gradients pull the trajectory toward "high D, high rho, low sensitivity". Frame 3 (bottom): a single highlighted skeletal structure of trinitro-1,2-isoxazole (the L1 lead) with three NO2 groups on a five-membered O-N-C-C-C ring; below it the label "predicted: rho=2.00 g/cm^3, D=9.56 km/s; DFT-corroborated: D=9.52 km/s; chemotype-rediscovered (Sabatini 2018, Tang 2017)". Editorial scientific-illustration, no human figures, no commercial slick.

### Candidate D — The full pipeline as one page

> A clean methodology figure spanning the full page horizontally, 16:9, deep navy + cream + gold accent. Left third: a stylised "trust pyramid" of training data layers (A experimental, B DFT, C K-J, D surrogate) feeding a frozen VAE encoder. Middle third: a latent-space diffusion model with classifier-guidance heads (small icons for "viability", "sensitivity", "performance", "hazard"); a chemistry-rules filter as a sieve symbol below. Right third: outputs - a 3-by-4 grid of skeletal chemical structures in muted grey, with one structure highlighted in gold (the L1 lead) and labels "12 chem-pass leads, DFT-validated, retrosynthesis-checked". Connecting arrows: data-flow in navy, classifier-guidance gradients in gold dashed, chemistry-filter rejection in red dashed. Bottom band: small icons for "DFT", "xTB", "AiZynth" with checkmarks. Caption-style serif typography. No hands, no robots, no decorative background. Editorial register, suitable for a single-page graphical abstract on a Nature Machine Intelligence preprint.

## Implementation note

Generate all four via Imagen 4.0 standard quality (1K resolution, 16:9 except candidate C which is 4:5).
Output directory: `visual_abstract_candidates/`.
