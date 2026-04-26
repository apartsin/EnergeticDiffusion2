# Energetic Materials Generation Project  
## Full Experimental Design and Training Protocol

**Goal:** Generate novel candidate energetic materials with desired properties using SELFIES representation, VAE latent space, property prediction, and latent diffusion.

**Core pipeline:**

```text
Molecules → SELFIES → VAE latent space → property predictors → latent diffusion → generated SELFIES/SMILES → filtering/evaluation
```

Target properties:

```text
density
heat of formation / explosion heat
detonation velocity
detonation pressure
```

Supported generation modes:

| Mode | Property target | Starting compound |
|---|---:|---:|
| De novo property-conditioned generation | yes | no |
| De novo regressor-guided generation | yes | no |
| Combined conditional + regressor-guided generation | yes | no |
| Compound editing / optimization | yes/optional | yes |

---

# 1. Data Preparation

## 1.1 Data sources

Use two molecular datasets:

```text
A. Energetic materials dataset
   - labeled
   - contains target properties

B. Generic molecular dataset
   - large
   - unlabeled
   - chemically diverse
```

Energetic data is used for supervised property learning and target-domain fine-tuning.

Generic data is used to learn broad chemical structure and improve the VAE/diffusion molecular prior.

---

## 1.2 Molecular standardization

For every molecule:

1. Parse SMILES with RDKit.
2. Sanitize molecule.
3. Remove salts/mixtures.
4. Keep largest valid fragment.
5. Optionally neutralize charges.
6. Canonicalize SMILES.
7. Convert canonical SMILES to SELFIES.

Store both representations:

```text
SELFIES          → model input/output
canonical SMILES → identity, deduplication, lookup, similarity, evaluation
```

Required stored fields:

```text
molecule_id
source_dataset: generic / energetic
canonical_smiles
selfies
properties if available
scaffold_id or cluster_id
split: train / validation / test
```

---

## 1.3 SELFIES vocabulary

Build the SELFIES vocabulary from the training data only.

Include special tokens:

```text
<bos>
<eos>
<pad>
<unk>
```

Recommended:

```text
max_sequence_length = percentile_99_length or fixed cutoff after inspection
```

Remove molecules that are too long, malformed, or outside allowed atom set.

---

## 1.4 Data splits

For energetic materials, use scaffold-aware splitting:

```text
train: 80%
validation: 10%
test: 10%
```

Primary recommendation for energetic molecules:

```text
cluster by fingerprint similarity and split by cluster ID (group split)
```

This is specifically to prevent close analog leakage across train/test.

If RDX is in train and a methyl-RDX analog is in test, model error can look artificially low because the model is effectively doing nearest-neighbor interpolation.
Group splitting prevents this by keeping entire similarity groups on one side of the boundary.

### 1.4.1 Similarity representation

For Butina clustering, compute Morgan fingerprints from canonical SMILES:

```text
ECFP4 (radius = 2, nBits = 2048)
similarity = Tanimoto(fp_i, fp_j)
distance = 1 - similarity
```

### 1.4.2 Butina clustering protocol

Use a distance threshold corresponding to the desired similarity cutoff.
Example:

```text
similarity cutoff = 0.4
distance threshold = 0.6
```

Procedure:

1. Compute neighbor count for each molecule under the threshold.
2. Sort molecules by neighbor count (descending).
3. Select highest-count molecule as centroid; assign all neighbors within threshold to that cluster.
4. Remove assigned molecules and repeat.
5. Continue until all molecules are assigned (singletons are valid clusters).

Output:

```text
one cluster_id per molecule
partition of the dataset
```

Threshold guidance:

```text
0.6 similarity -> tighter, small clusters
0.4 similarity -> practical "same family" grouping
0.3 similarity -> broader, larger clusters
```

### 1.4.3 Split assignment rule

Split cluster IDs, not molecules.

Procedure:

1. Sort clusters (by size; optionally shuffle with fixed seed).
2. Assign whole clusters to train/validation/test until target ratios are reached.
3. Every molecule inherits its cluster split.

This creates a real chemical gap between train and test and turns evaluation into a generalization test, not interpolation.

### 1.4.4 Split verification checks

After split assignment, always compute:

1. Maximum nearest-neighbor Tanimoto from each test molecule to the train set.
2. Property distribution comparison across train/validation/test.

Acceptance guidance:

```text
max test->train NN similarity should be below the intended clustering cutoff (allow small slack)
property histograms should be reasonably comparable across splits
```

If a split is skewed, reshuffle cluster-to-split assignment with a different seed and regenerate.

### 1.4.5 Relation to Bemis–Murcko grouping

Bemis–Murcko is also a group split, but with a different similarity definition:

```text
Bemis–Murcko: same group only if ring skeleton is identical
Butina/ECFP: same group if full fingerprint similarity is above cutoff
```

For energetic materials, fingerprint-based grouping is usually more faithful because substituent patterns often drive energetic behavior while scaffold-only grouping can hide that information.

### 1.4.6 Reuse cluster IDs downstream

Keep the grouping ID (scaffold or fingerprint cluster) in the dataset manifest as a first-class field.
This ID is reused for:

1. Novelty analysis:
   - candidate in seen vs unseen training cluster
2. Training stability:
   - cluster-aware batch sampling to reduce dominance from one chemical family
3. Property model validation:
   - group-aware k-fold cross-validation where each fold holds out whole clusters

Group-aware CV preserves the same train/test contamination guarantee used in the main split.

For generic molecules:

```text
train: 90%
validation: 5%
test: 5%
```

Implementation note:

```text
RDKit Butina (rdkit.ML.Cluster.Butina.ClusterData) uses a condensed distance matrix and is O(N^2) in memory/time.
```

This is usually practical for the energetic subset, but not for ~10^6-scale generic corpora.
For very large generic sets, use scaffold grouping or approximate alternatives (for example MinHash-based clustering, or sphere exclusion with nearest-neighbor indexing).

Do not use random splits only for energetic materials. Random splits may overestimate performance because close analogs can appear in train and test.

---

## 1.5 Property normalization

Normalize each property using energetic training split statistics only:

```text
y_norm = (y - mean_train) / std_train
```

Store:

```text
property_mean
property_std
normalization_version
```

All property conditioning and regression training should use normalized values.

---

# 2. SELFIES VAE

## 2.1 Purpose

Train a molecular autoencoder:

```text
SELFIES → latent z → SELFIES
```

The VAE defines the latent space used later for:

```text
property regression
latent diffusion
compound editing
optimization
```

---

## 2.2 Training data

Use mixed training data:

```text
generic molecules + energetic molecules
```

Do not fix the energetic/generic ratio blindly. Treat it as an experiment.

Recommended energetic oversampling ratios:

```text
0.10
0.25
0.50
```

Example for batch size 128 and energetic ratio 0.25:

```text
32 energetic molecules
96 generic molecules
```

Training may resample generic molecules each epoch, but validation and test sets must remain fixed.

---

## 2.3 VAE architecture

Recommended baseline:

```text
Input: SELFIES token sequence
Encoder: token embedding + BiLSTM or Transformer encoder
Latent: μ and logσ²
Decoder: autoregressive LSTM or Transformer decoder
Output: SELFIES token logits
```

Baseline hyperparameters:

```text
embedding_dim = 256
encoder_hidden_dim = 512
decoder_hidden_dim = 512
latent_dim = 128
dropout = 0.1
```

Tune:

```text
latent_dim ∈ {64, 128, 256}
```

---

## 2.4 VAE loss

Use:

```text
L = L_reconstruction + β * L_KL + λ * L_property_aux
```

Where:

```text
L_reconstruction = token cross-entropy
L_KL = KL divergence to standard normal
L_property_aux = optional weak supervised property loss on energetic samples
```

Recommended default:

```text
λ = 0 for first baseline
λ = small value for property-aligned VAE ablation
```

KL annealing:

```text
β starts at 0
β increases to β_max during warmup
β_max ∈ {0.5, 1.0}
warmup = 50k steps
```

---

## 2.5 Training settings

```text
batch_size = 128
optimizer = AdamW
learning_rate = 3e-4
weight_decay = 1e-2
gradient_clipping = 1.0
training_steps = 200k–500k
precision = mixed precision
```

Teacher forcing:

```text
start = 1.0
decay to = 0.5
```

Checkpoint every:

```text
5k steps
```

Validate every:

```text
2k steps
```

---

## 2.6 VAE validation metrics

Evaluate:

```text
SELFIES validity
RDKit SMILES validity after decoding
exact reconstruction rate
canonical reconstruction rate
uniqueness
latent smoothness
nearest-neighbor similarity
property preservation on energetic validation molecules
```

Property preservation checks:

```text
property(x) vs property(reconstruct(x))
latent distance vs property distance
nearest latent neighbors vs property similarity
```

This does not mean the VAE must perfectly predict properties. It means the latent geometry should not destroy property-relevant structure.

---

## 2.7 VAE checkpoint selection

Primary selection criterion:

```text
high validity × high reconstruction
```

Secondary criteria:

```text
good latent smoothness
good property preservation
stable KL
no posterior collapse
```

---

## 2.8 Latent export

After selecting the VAE checkpoint, encode all molecules:

```text
z = μ_encoder(SELFIES)
```

Use encoder mean, not sampled latent, for downstream training.

Store:

```text
molecule_id
canonical_smiles
selfies
z
source_dataset
split
properties
scaffold_id
```

---

# 3. Property Predictors

## 3.1 Purpose

Train predictors for target properties.

They are used for:

```text
property evaluation
regressor-guided generation
uncertainty estimation
candidate ranking
```

Do not rely on one predictor only.

Use at least two predictors:

```text
A. latent predictor
B. independent molecular predictor
```

---

## 3.2 Dataset

Use only labeled energetic materials.

Inputs:

```text
latent predictor: z
molecular predictor: molecular graph, fingerprint, or RDKit descriptors
```

Targets:

```text
density
heat of formation / explosion heat
detonation velocity
detonation pressure
```

Use normalized properties.

---

## 3.3 Latent property predictor

Architecture:

```text
Input: latent z
Backbone: MLP
Heads: 4 property heads
Each head outputs μ and logσ
```

Baseline:

```text
input_dim = latent_dim
hidden_layers = [256, 256, 128]
activation = GELU
dropout = 0.1
output_per_property = μ, logσ
```

---

## 3.4 Independent molecular predictor

Use one of:

```text
Morgan fingerprint + MLP
RDKit descriptors + MLP
Graph neural network
```

Recommended first baseline:

```text
Morgan fingerprint radius = 2
nBits = 2048
MLP hidden layers = [512, 256, 128]
```

This model is used as an independent evaluator and sanity check against the latent predictor.

---

## 3.5 Uncertainty

Preferred method:

```text
ensemble of 5 predictors trained with different seeds
```

For each generated candidate:

```text
μ = ensemble mean
σ = ensemble standard deviation
```

Alternative:

```text
MC dropout
heteroscedastic regression
Bayesian head
```

---

## 3.6 Loss

Use heteroscedastic Gaussian negative log likelihood:

```text
L_i = ((y_i - μ_i)^2 / exp(logσ_i)) + logσ_i
```

Total:

```text
L = sum over all available properties
```

If some properties are missing, use property masks.

---

## 3.7 Training settings

```text
batch_size = 256
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-4
epochs = 200
early_stopping = validation loss patience 20
```

Train 5 independent seeds for ensemble.

---

## 3.8 Evaluation metrics

For each property:

```text
MAE
RMSE
R²
Spearman rank correlation
NLL
calibration error
```

Report separately:

```text
validation split
scaffold test split
high-novelty test subset
```

---

# 4. Latent Diffusion Model

## 4.1 Purpose

Train one flexible denoiser that supports:

```text
unconditional generation
property-conditioned generation
partial-property conditioning
regressor-guided generation
source-compound editing
```

Do not train four separate denoisers.

---

## 4.2 Denoiser input/output

Model:

```text
εθ(z_t, t, property_values, property_mask, source_z, source_mask)
```

Inputs:

```text
z_t              noisy latent
t                diffusion timestep
property_values  normalized target property vector
property_mask    which properties are provided
source_z         optional starting compound latent
source_mask      whether source compound is provided
```

Output:

```text
predicted noise ε̂
```

---

## 4.3 Denoiser architecture

Recommended baseline:

```text
time embedding: sinusoidal, dim=128
condition embedding: MLP over property_values + property_mask + source_z + source_mask
main network: residual MLP
conditioning mechanism: concatenation or FiLM
```

Baseline:

```text
latent_dim = 128
hidden_dim = 512
num_layers = 6
activation = SiLU or GELU
dropout = 0.1
```

---

## 4.4 Diffusion setup

```text
T = 1000 training timesteps
noise_schedule = cosine
training_objective = predict ε
loss = MSE(ε, ε̂)
```

Sampler:

```text
DDIM
sampling_steps = 50–100
```

---

# 5. Diffusion Training Stages

## 5.1 Stage A — unconditional pretraining

Data:

```text
generic latents + energetic latents
```

Conditions:

```text
property_mask = 0
source_mask = 0
```

Purpose:

```text
learn broad valid molecular latent distribution
```

Training:

```text
steps = 200k
batch_size = 512
optimizer = AdamW
learning_rate = 2e-4
EMA_decay = 0.999
gradient_clipping = 1.0
```

---

## 5.2 Stage B — property-conditioned fine-tuning

Data:

```text
labeled energetic latents
```

Conditions:

```text
property_values = normalized property vector
property_mask = binary property availability vector
source_mask = 0
```

Condition dropout:

```text
p_drop_all_properties = 0.15
p_drop_each_property = 0.20
```

This enables:

```text
classifier-free guidance
partial property conditioning
unconditional fallback
```

Use masks, not zero values, to represent missing properties.

Training:

```text
steps = 200k
batch_size = 512
learning_rate = 2e-4 or 1e-4
```

---

## 5.3 Stage C — source-conditioned editing training

Purpose:

```text
teach the model to generate variants around a starting compound
```

Training pairs:

### Self-pairs

```text
source_z = z_i
target_z = z_i
```

### Similar-molecule pseudo-pairs

Find similar molecules by Tanimoto similarity or latent nearest neighbors:

```text
source_z = z_i
target_z = z_j
similarity range: 0.4–0.8
```

Recommended pair mixture:

```text
50% self-pairs
50% similar-molecule pairs
```

Conditions:

```text
source_z provided
source_mask = 1
property_values optionally provided
```

Source dropout:

```text
p_drop_source = 0.30
```

Training:

```text
steps = 100k–200k
batch_size = 512
learning_rate = 1e-4
```

---

# 6. Generation Protocol

## 6.1 Common decoding pipeline

For every generated latent:

```text
z_generated
→ VAE decoder
→ SELFIES
→ SMILES
→ canonical SMILES
→ RDKit validation
```

---

## 6.2 Mode A — de novo property-conditioned generation

Inputs:

```text
property_values = desired normalized targets
property_mask = 1 for provided properties
source_mask = 0
```

Sampling:

```text
DDIM steps = 100
CFG scale w ∈ {1.0, 2.0, 3.0, 5.0}
```

Classifier-free guidance:

```text
ε̂ = (1 + w) ε̂_cond - w ε̂_uncond
```

---

## 6.3 Mode B — regressor-guided generation

Use property predictor gradient during sampling.

For target matching:

```text
score(z) = -Σ_i |μ_i(z) - target_i| - α Σ_i σ_i(z)
```

For maximization:

```text
score(z) = Σ_i μ_i(z) - α Σ_i σ_i(z)
```

Gradient step:

```text
z ← z + η ∇z score(z)
```

Recommended:

```text
η ∈ {1e-4, 5e-4, 1e-3}
α ∈ {0.1, 0.5, 1.0}
```

Use weak guidance. Strong guidance can produce adversarial latents.

---

## 6.4 Mode C — combined conditioning + regressor guidance

Use both:

```text
property-conditioned diffusion
weak regressor guidance
uncertainty penalty
```

Recommended final candidate generation mode.

Scoring during sampling:

```text
score(z) =
target_match_score
- α uncertainty
```

Final ranking adds novelty and diversity.

---

## 6.5 Mode D — starting-compound editing

Inputs:

```text
source compound
target properties
source_z = VAE.encode(source)
source_mask = 1
```

Initialize sampling near source:

```text
z_start = source_z + τ noise
```

Noise strength:

| τ | Expected result |
|---:|---|
| low | close analogs |
| medium | meaningful modifications |
| high | de novo-like candidates |

Recommended τ sweep:

```text
τ ∈ {0.1, 0.3, 0.5, 0.7}
```

---

# 7. Candidate Filtering

For each generated molecule:

## 7.1 Validity

Check:

```text
SELFIES decodes
SMILES parses in RDKit
RDKit sanitization succeeds
```

## 7.2 Deduplication

Remove:

```text
duplicates within generated set
duplicates from training set
duplicates from validation/test set
```

Use canonical SMILES and InChIKey.

## 7.3 Basic chemical constraints

Filter by:

```text
allowed atom types
heavy atom count range
molecular weight range
charge policy
valence sanity
fragment count
```

## 7.4 Energetic-domain classifier

Optional but recommended.

Train classifier:

```text
generic compound vs energetic material
```

Use it as soft filter:

```text
P(energetic-like) > threshold
```

## 7.5 Synthetic accessibility and stability proxies

Compute:

```text
synthetic accessibility score
ring strain / unstable groups proxy
sensitivity proxy if available
```

Do not rank candidates only by energetic strength.

---

# 8. Novelty and Diversity

Compute for each candidate:

```text
exact canonical SMILES match
InChIKey match
Tanimoto similarity to nearest training molecule
Bemis–Murcko scaffold novelty
cluster novelty (is candidate assigned to unseen cluster_id vs train)
latent distance to nearest training molecule
pairwise diversity among generated molecules
```

Candidate classes:

| Class | Meaning |
|---|---|
| duplicate | already exists in train/known data |
| close analog | high similarity to known compound |
| scaffold novel | new scaffold |
| OOD risky | far from training distribution |

Best candidates are usually:

```text
novel but not extremely out-of-distribution
```

---

# 9. Candidate Scoring

For each candidate, compute:

```text
property prediction from latent ensemble
property prediction from independent molecular model
uncertainty
novelty
diversity
domain score
synthetic accessibility
```

Example score:

```text
Score =
- Σ_i |μ_i - target_i|
- α Σ_i σ_i
+ β novelty
+ δ domain_score
- γ synthetic_difficulty
```

Recommended starting weights:

```text
α = 0.5
β = 0.2
δ = 0.2
γ = 0.1
```

Select top-K with diversity constraint:

```text
maximum pairwise Tanimoto similarity < threshold
```

Suggested threshold:

```text
0.7–0.85
```

---

# 10. Evaluation Protocol

## 10.1 Generation metrics

Report:

```text
validity %
uniqueness %
novelty %
scaffold novelty %
mean nearest-neighbor similarity
diversity
property target error
uncertainty
domain classifier score
```

## 10.2 Property evaluation

Use three levels:

### Level 1 — model-based screening

```text
latent regressor ensemble
independent molecular predictor
uncertainty
```

Candidates are stronger if both predictors agree.

### Level 2 — database/literature lookup

Search by:

```text
canonical SMILES
InChIKey
molecular formula
known aliases if available
```

Sources:

```text
PubChem
domain-specific energetic material databases
published literature
internal datasets
```

Labels:

```text
known compound
known analog
unknown/novel
properties known
hazard information known
```

### Level 3 — physics/chemistry validation

For shortlisted candidates only:

```text
DFT or semi-empirical heat of formation estimate
density estimate
oxygen balance
Kamlet–Jacobs or related detonation estimates
stability/sensitivity proxies
```

Use DFT/physics validation for top candidates, not for all generated molecules.

---

# 11. Ablation Studies

## 11.1 VAE ablations

```text
SELFIES vs SMILES
generic-only vs energetic-only vs mixed
energetic oversampling ratio: 0.10 / 0.25 / 0.50
latent_dim: 64 / 128 / 256
with vs without auxiliary property loss
```

## 11.2 Property predictor ablations

```text
latent MLP only
fingerprint MLP only
GNN
single model vs ensemble
with vs without uncertainty
```

## 11.3 Diffusion ablations

```text
with vs without unconditional pretraining
with vs without property conditioning
with vs without property masks
with vs without condition dropout
with vs without source conditioning
self-pairs only vs self + neighbor pairs
```

## 11.4 Guidance ablations

```text
condition-only
regressor-only
condition + regressor
condition + regressor + uncertainty penalty
condition + regressor + uncertainty + novelty filtering
```

## 11.5 Editing ablations

```text
source conditioning off
source conditioning on
noise strength τ sweep
self-pair training only
neighbor-pair training included
```

---

# 12. Default Baseline Configuration

Use this as the first full experiment.

## Representation

```text
SELFIES for model input/output
canonical SMILES for identity/evaluation
```

## VAE

```text
latent_dim = 128
embedding_dim = 256
hidden_dim = 512
batch_size = 128
lr = 3e-4
β_max = 1.0
KL warmup = 50k steps
training_steps = 300k
energetic_ratio = 0.25
```

## Property predictors

```text
latent MLP ensemble: 5 seeds
independent Morgan fingerprint MLP ensemble: 5 seeds
batch_size = 256
lr = 1e-3
epochs = 200
```

## Diffusion

```text
single denoiser
latent_dim = 128
hidden_dim = 512
layers = 6
T = 1000
noise_schedule = cosine
batch_size = 512
lr = 2e-4
EMA = 0.999
```

Training stages:

```text
Stage A unconditional pretraining: 200k steps
Stage B property-conditioned fine-tuning: 200k steps
Stage C source-conditioned editing: 100k steps
```

Condition dropout:

```text
p_drop_all_properties = 0.15
p_drop_each_property = 0.20
p_drop_source = 0.30
```

## Sampling

```text
DDIM steps = 100
CFG scale = 2.5
regressor guidance α = 0.5
guidance step η = 5e-4
editing noise τ ∈ {0.1, 0.3, 0.5, 0.7}
```

---

# 13. Output Artifacts

Each experiment run should save:

```text
config.yaml
data_split_manifest.csv
vae_checkpoint.pt
vae_metrics.csv
latent_dataset.parquet
property_predictor_checkpoints/
property_predictor_metrics.csv
diffusion_checkpoint.pt
generation_config.yaml
generated_candidates.csv
filtered_candidates.csv
ranked_candidates.csv
evaluation_report.md
```

Candidate table fields:

```text
candidate_id
generated_selfies
generated_smiles
canonical_smiles
inchi_key
generation_mode
target_properties
predicted_properties_latent
predicted_properties_independent
uncertainty
novelty_score
nearest_train_smiles
nearest_train_similarity
scaffold
domain_score
synthetic_accessibility
final_score
filter_status
database_lookup_status
```

---

# 14. Success Criteria

A successful model should show:

```text
high validity
high uniqueness
high novelty
reasonable similarity to energetic-material domain
property predictions close to requested targets
low uncertainty for selected candidates
diverse generated candidates
independent predictor agreement
shortlisted candidates surviving physics-based validation
```

Minimum acceptable early baseline:

```text
validity > 80%
uniqueness > 70%
novelty > 50%
property predictor agreement on top candidates
clear improvement over random latent sampling
```

---

# 15. Main Risks and Controls

## Risk 1 — predictor exploitation

The generator may produce molecules that fool the property predictor.

Controls:

```text
uncertainty penalty
independent property predictor
database lookup
physics validation
diversity and OOD filtering
```

## Risk 2 — poor latent geometry

The VAE may reconstruct molecules but produce a latent space unsuitable for optimization.

Controls:

```text
latent smoothness checks
property preservation checks
nearest-neighbor property consistency
optional weak property auxiliary loss
```

## Risk 3 — generic compounds dominate the distribution

The generator may produce generic molecules rather than energetic-like molecules.

Controls:

```text
conditional fine-tuning on energetic materials
energetic-domain classifier
balanced sampling
target-domain filtering
```

## Risk 4 — invalid or unstable molecules

SELFIES improves syntactic validity but does not guarantee useful chemistry.

Controls:

```text
RDKit validation
allowed atom filters
stability proxies
synthetic accessibility
expert/physics-based review
```

---

# 16. Final Recommended System

The recommended final system is:

```text
SELFIES VAE trained on generic + energetic compounds
+
fixed latent export using encoder mean
+
latent property regressor ensemble
+
independent molecular property regressor ensemble
+
single latent diffusion denoiser with:
    property values
    property masks
    classifier-free condition dropout
    optional source compound latent
    source mask
+
unconditional diffusion pretraining on generic + energetic latents
+
conditional fine-tuning on labeled energetic materials
+
editing fine-tuning using self-pairs and similar-molecule pseudo-pairs
+
generation using:
    property conditioning
    weak regressor guidance
    uncertainty penalty
    novelty/diversity filtering
+
evaluation using:
    validity
    uniqueness
    novelty
    independent property prediction
    uncertainty
    database lookup
    DFT/physics validation for shortlisted candidates
```

Use one denoiser. Use masks for missing conditions. Use SELFIES for modeling and canonical SMILES for evaluation. Use uncertainty and independent validation to avoid trusting the generator blindly.
