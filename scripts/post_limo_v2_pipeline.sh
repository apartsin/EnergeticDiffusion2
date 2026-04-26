#!/usr/bin/env bash
# Post-LIMO-v2.1 pipeline: re-encode + retrain denoisers, preserving v1-v6.
#
# Idempotent + resumable: each step skips if its output exists.
# Nothing from the v1 lineage is overwritten — all v2-suffixed.
#
# Usage:
#   bash scripts/post_limo_v2_pipeline.sh                    # auto-detect newest v2.1 ckpt
#   bash scripts/post_limo_v2_pipeline.sh /path/to/best.pt   # explicit ckpt

set -euo pipefail
cd "$(dirname "$0")/.."

PY="/c/Python314/python"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# ── 1. resolve LIMO v2.1 checkpoint ──────────────────────────────────────
if [ "${1:-}" != "" ]; then
    LIMO_V2_CKPT="$1"
else
    LIMO_V2_DIR=$(ls -1dt experiments/limo_ft_motif_rich_v2_1_* 2>/dev/null | head -1)
    if [ -z "$LIMO_V2_DIR" ]; then
        echo "ERROR: no experiments/limo_ft_motif_rich_v2_1_* found"
        exit 1
    fi
    LIMO_V2_CKPT="$LIMO_V2_DIR/checkpoints/best.pt"
fi
[ -f "$LIMO_V2_CKPT" ] || { echo "ERROR: $LIMO_V2_CKPT not found"; exit 1; }
echo "LIMO v2.1 ckpt: $LIMO_V2_CKPT"

# ── 2. self-consistency / R1-R3 check (cheap diagnostic) ─────────────────
echo
echo "============================================================"
echo "Stage 1: VAE self-consistency (R1) on energetic seeds"
echo "============================================================"
SC_LOG="experiments/limo_ft_motif_rich_v2_1_*/r_diagnostics.md"
if ! ls $SC_LOG >/dev/null 2>&1; then
    "$PY" scripts/vae/limo_self_consistency.py --ckpt "$LIMO_V2_CKPT" --seeds_csv data/c2c/seeds.csv 2>&1 | tail -30 || \
        echo "(self-consistency script not built yet; continuing)"
fi

# ── 3. re-encode all 382 k SMILES with v2.1 weights → latents_v2.pt ──────
echo
echo "============================================================"
echo "Stage 2: re-encode latents with LIMO v2.1 → latents_v2.pt"
echo "============================================================"
LATENTS_V2="data/training/diffusion/latents_v2.pt"
if [ ! -f "$LATENTS_V2" ]; then
    "$PY" scripts/diffusion/encode_latents.py \
        --limo_ckpt "$LIMO_V2_CKPT" \
        --out "$LATENTS_V2" 2>&1 | tail -20 || \
        { echo "encode_latents may need --limo_ckpt support; check script"; }
else
    echo "  [skip] $LATENTS_V2 already exists"
fi

# ── 4. expand conditioning + trustcond on the new latents ────────────────
echo
echo "============================================================"
echo "Stage 3: expand_conditioning + build_latents_trustcond on v2"
echo "============================================================"
EXP_V2="data/training/diffusion/latents_expanded_v2.pt"
TRUST_V2="data/training/diffusion/latents_trustcond_v2.pt"
if [ ! -f "$EXP_V2" ]; then
    "$PY" scripts/diffusion/expand_conditioning.py \
        --latents_in "$LATENTS_V2" \
        --preds_3dcnn "data/training/diffusion/preds_3dcnn.pt" \
        --out "$EXP_V2" --tier_d_weight 0.7 2>&1 | tail -10
else
    echo "  [skip] $EXP_V2"
fi
if [ ! -f "$TRUST_V2" ]; then
    "$PY" scripts/diffusion/build_latents_trustcond.py \
        --inp "$EXP_V2" --out "$TRUST_V2" 2>&1 | tail -10
else
    echo "  [skip] $TRUST_V2"
fi

# ── 5. train denoiser v9 (v4-B recipe on new latents) ────────────────────
echo
echo "============================================================"
echo "Stage 4: train denoiser v9 (v4-B-style) on latents_trustcond_v2"
echo "============================================================"
V9_DIR=$(ls -1dt experiments/diffusion_subset_cond_expanded_v9_* 2>/dev/null | head -1)
if [ -z "$V9_DIR" ] || [ ! -f "$V9_DIR/checkpoints/best.pt" ]; then
    # build a v9 config from v4b but with new latents path
    cat > /tmp/diffusion_expanded_v9.yaml <<EOF
$(sed "s|latents_trustcond.pt|latents_trustcond_v2.pt|; s|diffusion_subset_cond_expanded_v4b|diffusion_subset_cond_expanded_v9|" \
       configs/diffusion_expanded_v4b.yaml)
EOF
    cp /tmp/diffusion_expanded_v9.yaml configs/diffusion_expanded_v9.yaml
    "$PY" scripts/diffusion/train.py --config configs/diffusion_expanded_v9.yaml 2>&1 | tail -20
    V9_DIR=$(ls -1dt experiments/diffusion_subset_cond_expanded_v9_* | head -1)
else
    echo "  [skip] $V9_DIR exists"
fi

# ── 6. train denoiser v10 (v3 recipe on new latents) for HOF ─────────────
echo
echo "============================================================"
echo "Stage 5: train denoiser v10 (v3-style, HOF tail) on latents_expanded_v2"
echo "============================================================"
V10_DIR=$(ls -1dt experiments/diffusion_subset_cond_expanded_v10_* 2>/dev/null | head -1)
if [ -z "$V10_DIR" ] || [ ! -f "$V10_DIR/checkpoints/best.pt" ]; then
    cat > /tmp/diffusion_expanded_v10.yaml <<EOF
$(sed "s|latents_expanded.pt|latents_expanded_v2.pt|; s|diffusion_subset_cond_expanded_v3|diffusion_subset_cond_expanded_v10|" \
       configs/diffusion_expanded_v3.yaml)
EOF
    cp /tmp/diffusion_expanded_v10.yaml configs/diffusion_expanded_v10.yaml
    "$PY" scripts/diffusion/train.py --config configs/diffusion_expanded_v10.yaml 2>&1 | tail -20
    V10_DIR=$(ls -1dt experiments/diffusion_subset_cond_expanded_v10_* | head -1)
else
    echo "  [skip] $V10_DIR exists"
fi

# ── 7. sweep both ─────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "Stage 6: cfg_sweep + rerank on v9, v10"
echo "============================================================"
"$PY" scripts/diffusion/cfg_sweep.py --exp "$V9_DIR" --scales 5 7 --n_per_target 50 2>&1 | tail -10
"$PY" scripts/diffusion/cfg_sweep.py --exp "$V10_DIR" --scales 5 7 --n_per_target 50 2>&1 | tail -10
"$PY" scripts/diffusion/rerank_sweep.py --exp "$V9_DIR" --cfg 7 --n_pool 1500 --n_keep 40 --require_neutral 2>&1 | tail -10
"$PY" scripts/diffusion/rerank_sweep.py --exp "$V10_DIR" --cfg 7 --n_pool 1500 --n_keep 40 --require_neutral 2>&1 | tail -10

# ── 8. joint rerank breakthrough run ─────────────────────────────────────
echo
echo "============================================================"
echo "Stage 7: joint v9 + v10 breakthrough rerank"
echo "============================================================"
"$PY" scripts/diffusion/joint_rerank.py \
    --exp_v4b "$V9_DIR" --exp_v3 "$V10_DIR" \
    --cfg 7 --n_pool_each 1500 --n_keep 80 \
    --target_density 1.95 --target_d 9.5 --target_p 40 --target_hof 220 \
    --hard_sa 5.0 --hard_sc 3.5 \
    --tanimoto_min 0.20 --tanimoto_max 0.55 \
    --require_neutral --with_chem_filter --with_feasibility \
    --out experiments/joint_v9_v10_breakthrough.md 2>&1 | tail -20

# ── 9. c2c re-eval with v2.1 self-consistency ────────────────────────────
echo
echo "============================================================"
echo "Stage 8: c2c re-eval on v9 + LIMO v2.1"
echo "============================================================"
"$PY" scripts/diffusion/c2c_pipeline.py \
    --exp "$V9_DIR" --seeds_csv data/c2c/seeds.csv \
    --strengths 0.3 0.5 --n_variants 60 --anchor_alpha 0.3 --cfg 2 \
    --require_neutral --with_chem_filter \
    --out_dir "$V9_DIR/c2c_results_v2_1" 2>&1 | tail -10

echo
echo "============================================================"
echo "DONE. Outputs:"
echo "  LIMO v2.1 ckpt:    $LIMO_V2_CKPT"
echo "  latents v2:        $LATENTS_V2"
echo "  expanded v2:       $EXP_V2"
echo "  trustcond v2:      $TRUST_V2"
echo "  denoiser v9 dir:   $V9_DIR"
echo "  denoiser v10 dir:  $V10_DIR"
echo "  joint rerank:      experiments/joint_v9_v10_breakthrough.md"
echo "============================================================"
echo
echo "Original v1-v6 lineage preserved untouched:"
echo "  experiments/limo_ft_energetic_20260424T150825Z/"
echo "  data/training/diffusion/latents.pt"
echo "  data/training/diffusion/latents_expanded.pt"
echo "  data/training/diffusion/latents_trustcond.pt"
echo "  experiments/diffusion_subset_cond_expanded_*"
