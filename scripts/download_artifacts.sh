#!/usr/bin/env bash
# Download production-essential artefacts from GitHub Releases and restore
# into the paths the codebase expects. Run once after cloning the repo.

set -e
cd "$(dirname "$0")/.."

REPO="apartsin/EnergeticDiffusion2"
TAG="v0.1.0"

declare -A MAP=(
    ["limo_v1_best.pt"]="experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt"
    ["denoiser_v3_best.pt"]="experiments/diffusion_subset_cond_expanded_v3_20260425T140941Z/checkpoints/best.pt"
    ["denoiser_v4b_best.pt"]="experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/checkpoints/best.pt"
    ["denoiser_v4nf_best.pt"]="experiments/diffusion_subset_cond_expanded_v4_nofilter_20260425T175119Z/checkpoints/best.pt"
    ["sa_surrogate_clean.pt"]="experiments/guidance_sa_sc_20260424T172820Z/checkpoints/sa_best.pt"
    ["sc_surrogate_clean.pt"]="experiments/guidance_sa_sc_20260424T172820Z/checkpoints/sc_best.pt"
    ["property_heads_clean.pt"]="data/training/guidance/property_heads.pt"
    ["property_heads_t_aware.pt"]="data/training/guidance/property_heads_t.pt"
    ["preds_3dcnn_smoke.pt"]="data/training/diffusion/preds_3dcnn.pt"
)

for asset in "${!MAP[@]}"; do
    target="${MAP[$asset]}"
    mkdir -p "$(dirname "$target")"
    if [ -f "$target" ]; then
        echo "  [skip] $target already exists"
        continue
    fi
    echo "  downloading $asset → $target"
    gh release download "$TAG" --repo "$REPO" --pattern "$asset" --output "$target"
done

echo
echo "Heavy data still missing (must regenerate locally):"
echo "  data/training/diffusion/latents.pt        (≈1.5 GB; rebuild with scripts/diffusion/encode_latents.py)"
echo "  data/training/diffusion/latents_expanded.pt   (≈1.6 GB; rebuild with scripts/diffusion/expand_conditioning.py)"
echo "  data/training/diffusion/latents_trustcond.pt  (rebuild with scripts/diffusion/build_latents_trustcond.py)"
echo "  data/training/master/labeled_master.csv   (curated input — see DATA_DESCRIPTION.md)"
echo "  external/LIMO/                             (clone from upstream LIMO repo)"
echo "  external/scscore/                          (clone from upstream SCScore repo)"
