#!/usr/bin/env bash
# Autonomous Phase B+ controller. Each invocation reads state, advances one
# step. Each step prepends a smoke-test that fails fast on env / data issues
# before doing the heavy work.
set -e
PY=/c/Python314/python
ROOT=E:/Projects/EnergeticDiffusion2
cd "$ROOT"
mkdir -p logs

state_file=logs/cycle_state_phaseB.txt
state=$(cat "$state_file" 2>/dev/null || echo "wait_relabel")

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $1"; }
adv() { echo "$1" > "$state_file"; log "-> $1"; }
gpu_busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q -E "[0-9]+,"; }

log "state=$state"

# ── Smoke ──────────────────────────────────────────────────────────────
run_smoke() {
  log "[smoke] python + rdkit + torch"
  $PY -c "import torch, rdkit, sklearn; print('  py+torch+rdkit+sklearn OK', torch.__version__)" || { log "[smoke] FAILED env"; return 1; }
  log "[smoke] OK"
}

case "$state" in
  wait_relabel)
    if [ -f experiments/latent_labels_v3_scaffoldaware.pt ]; then
      adv fix_sc
    else
      log "[wait] relabel still running"
    fi
    ;;
  fix_sc)
    run_smoke || exit 1
    if [ ! -d external/scscore ]; then
      log "[start] extracting scscore tarball"
      mkdir -p external
      tar -xzf vast_jobs/scscore_pkg.tar.gz -C external/ 2>/dev/null || \
          tar -xzf vast_jobs/scscore_pkg.tar.gz -C ./ 2>/dev/null
    fi
    if $PY -c "import sys; sys.path.insert(0, 'external/scscore/scscore'); from standalone_model_numpy import SCScorer" 2>/dev/null; then
      log "[ok] SCScorer importable"
      adv relabel_with_sc
    else
      log "[fail] SCScorer still not importable; skipping SC head, advancing anyway"
      adv add_redflag_head
    fi
    ;;
  relabel_with_sc)
    run_smoke || exit 1
    log "[start] relabel with working SCScore"
    $PY scripts/viability/prepare_latent_labels.py \
        --latents data/training/diffusion/latents_trustcond.pt \
        --rf_v2 experiments/viability_rf_v2_hardneg/model.joblib \
        --out experiments/latent_labels_v3b_scsc.pt \
        > logs/relabel_v3b.log 2>&1 &
    echo $! > logs/relabel_v3b.pid
    adv wait_relabel_v3b
    ;;
  wait_relabel_v3b)
    pid=$(cat logs/relabel_v3b.pid 2>/dev/null || echo 0)
    if ! kill -0 "$pid" 2>/dev/null && [ -f experiments/latent_labels_v3b_scsc.pt ]; then
      log "[done] v3b labels ready"
      adv add_redflag_head
    else
      log "[wait] v3b relabel running (pid=$pid)"
    fi
    ;;
  add_redflag_head)
    log "[start] add redflag head + energetic z-scoring patch"
    # Patch handled inside the trainer script — read existing, branch on flag
    adv train_v3
    ;;
  train_v3)
    if gpu_busy; then log "[wait] GPU busy"; exit 0; fi
    run_smoke || exit 1
    log "[start] train multihead v3"
    LABELS=experiments/latent_labels_v3b_scsc.pt
    [ -f "$LABELS" ] || LABELS=experiments/latent_labels_v3_scaffoldaware.pt
    $PY scripts/viability/train_multihead_latent.py \
        --labels "$LABELS" \
        --out experiments/score_model_v3 \
        --epochs 12 --batch 512 \
        > logs/multihead_train_v3.log 2>&1 &
    echo $! > logs/cycle_train_v3.pid
    adv wait_train_v3
    ;;
  wait_train_v3)
    pid=$(cat logs/cycle_train_v3.pid 2>/dev/null || echo 0)
    if ! kill -0 "$pid" 2>/dev/null && [ -f experiments/score_model_v3/model.pt ]; then
      log "[done] score model v3 trained"
      adv diagnose_v3
    else
      log "[wait] training in progress"
    fi
    ;;
  diagnose_v3)
    run_smoke || exit 1
    log "[start] diagnose v3"
    $PY scripts/viability/diagnose_score_model.py \
        --model experiments/score_model_v3/model.pt \
        --out experiments/score_model_v3/diagnostics.json \
        > logs/diagnose_v3.log 2>&1
    log "[done] diagnostics -> experiments/score_model_v3/"
    adv sweep_5heads_pending
    ;;
  sweep_5heads_pending)
    if gpu_busy; then log "[wait] GPU busy"; exit 0; fi
    run_smoke || exit 1
    log "[start] 5-head sweep"
    bash scripts/diffusion/run_head_sweep_v3.sh > logs/sweep_5heads.log 2>&1 &
    echo $! > logs/sweep_5heads.pid
    adv wait_sweep
    ;;
  wait_sweep)
    pid=$(cat logs/sweep_5heads.pid 2>/dev/null || echo 0)
    if ! kill -0 "$pid" 2>/dev/null; then
      log "[done] 5-head sweep complete"
      adv cycle7_pending
    else
      log "[wait] sweep running"
    fi
    ;;
  cycle7_pending)
    if gpu_busy; then log "[wait] GPU busy"; exit 0; fi
    run_smoke || exit 1
    log "[start] cycle 7: pool=20k guided with best v3 scales"
    $PY scripts/diffusion/joint_rerank.py \
        --exp_v3 experiments/diffusion_subset_cond_expanded_v3_20260425T140941Z \
        --exp_v4b experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z \
        --score_model experiments/score_model_v3/model.pt \
        --guide_viab 1.0 --guide_sens 0.3 --guide_sa 0.0 \
        --cfg 7 --n_pool_each 10000 --n_keep 800 \
        --target_density 1.95 --target_d 9.5 --target_p 40 --target_hof 220 \
        --hard_sa 5.0 --hard_sc 3.5 --tanimoto_min 0.20 --tanimoto_max 0.55 \
        --require_neutral --with_chem_filter --with_feasibility \
        --out experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_cycle7_pool20k.md \
        > logs/cycle7.log 2>&1 &
    echo $! > logs/cycle7.pid
    adv wait_cycle7
    ;;
  wait_cycle7)
    pid=$(cat logs/cycle7.pid 2>/dev/null || echo 0)
    if ! kill -0 "$pid" 2>/dev/null && [ -f experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_cycle7_pool20k.md ]; then
      log "[done] cycle7 generated"
      adv cycle7_strict
    else
      log "[wait] cycle7 generating"
    fi
    ;;
  cycle7_strict)
    run_smoke || exit 1
    log "[start] cycle7 v2 strict"
    $PY scripts/diffusion/rerank_v2.py \
        --in experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_cycle7_pool20k.md \
        --out experiments/diffusion_subset_cond_expanded_v4b_20260426T000541Z/joint_rerank_cycle7_pool20k_v2.md \
        --viability_model experiments/viability_rf_v2_hardneg/model.joblib \
        --limit 800 \
        > logs/cycle7_strict.log 2>&1
    log "[done] cycle7 strict complete"
    adv done
    ;;
  done)
    log "[done] all phases complete"
    ;;
  *) log "[unknown state $state]"; exit 1 ;;
esac
