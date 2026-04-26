#!/usr/bin/env bash
# Pack production-essential artefacts as GitHub Release assets.
# Each file is < 2 GB so gh release upload accepts them individually.
#
# Usage:
#     bash scripts/archive_artifacts.sh           # dry-run, just lists sizes
#     bash scripts/archive_artifacts.sh --upload  # creates a v0.1.0 release and uploads
#
# After upload, scripts/download_artifacts.sh restores into the right paths.

set -e
cd "$(dirname "$0")/.."

REPO="apartsin/EnergeticDiffusion2"
TAG="v0.1.0"

declare -a ARTEFACTS=(
    "experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt"
    "experiments/diffusion_subset_cond_expanded_v3_20260425T140941Z/checkpoints/best.pt"
    "experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/checkpoints/best.pt"
    "experiments/diffusion_subset_cond_expanded_v4_nofilter_20260425T175119Z/checkpoints/best.pt"
    "experiments/guidance_sa_sc_20260424T172820Z/checkpoints/sa_best.pt"
    "experiments/guidance_sa_sc_20260424T172820Z/checkpoints/sc_best.pt"
    "data/training/guidance/property_heads.pt"
    "data/training/guidance/property_heads_t.pt"
    "data/training/diffusion/preds_3dcnn.pt"
)

declare -a ASSET_NAMES=(
    "limo_v1_best.pt"
    "denoiser_v3_best.pt"
    "denoiser_v4b_best.pt"
    "denoiser_v4nf_best.pt"
    "sa_surrogate_clean.pt"
    "sc_surrogate_clean.pt"
    "property_heads_clean.pt"
    "property_heads_t_aware.pt"
    "preds_3dcnn_smoke.pt"
)

echo "Production-essential artefacts:"
total=0
for i in "${!ARTEFACTS[@]}"; do
    src="${ARTEFACTS[$i]}"
    name="${ASSET_NAMES[$i]}"
    if [ -f "$src" ]; then
        sz=$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src")
        printf "  %-30s %10d bytes  -> %s\n" "$name" "$sz" "$src"
        total=$((total + sz))
    else
        echo "  MISSING: $src"
    fi
done
echo
echo "Total: $((total / 1024 / 1024)) MiB"

if [ "$1" != "--upload" ]; then
    echo "(dry run; pass --upload to actually create release)"
    exit 0
fi

echo
echo "Creating release $TAG on $REPO …"
gh release create "$TAG" \
    --repo "$REPO" \
    --title "EnergeticDiffusion2 v0.1.0 — production checkpoints" \
    --notes "Production checkpoints for the LIMO + denoiser pipeline.

Includes:
- LIMO v1 fine-tuned VAE
- Denoiser v3, v4-nf, v4-B (current production set)
- SA + SC latent surrogates (clean and time-conditional)
- Property heads (per-property predictors)
- 3DCNN smoke-ensemble predictions on all 382 k SMILES

After download, run scripts/download_artifacts.sh to restore into the
correct repo paths." \
    --draft

# upload each
for i in "${!ARTEFACTS[@]}"; do
    src="${ARTEFACTS[$i]}"
    name="${ASSET_NAMES[$i]}"
    [ -f "$src" ] || continue
    echo "  uploading $name …"
    gh release upload "$TAG" "$src#$name" --repo "$REPO" --clobber
done

echo
echo "Done. Edit the draft on GitHub before publishing."
