# T3 launch notes

Same `PYTHONIOENCODING=utf-8` env-var workaround as T1 (see T1 LAUNCH_NOTES.md).

## Successful launch

    App URL:  https://modal.com/apps/llmcourse/main/ap-JUyk3qA0ZkI61RjisNK2PU
    App ID:   ap-JUyk3qA0ZkI61RjisNK2PU
    Image:    pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel + numpy + pyyaml
    GPU:      A100 per seed (each seed runs serially)
    Timeout:  8 hr hard cap per function call
    Seeds:    1, 2  (default)
    Per-seed wall clock: 360 min (6 hr) configured into the v4b YAML
    Modal Volumes:
        dgld-t3-data:    1.6 GB latents file uploaded once on first run
        dgld-t3-results: per-seed best.pt + train.jsonl

## Expected runtime

The latents upload (1.6 GB at ~50-100 MB/s out of this Windows box) is
~30-300 s.  The two A100 seeds run serially: 6 hr each = ~12 hr total
GPU time, plus ~10 min image build + container startup overhead.

## Output paths after job completion

Run after job finishes:
    modal volume get dgld-t3-results / ./t3_denoiser_seeds_bundle/results/

That populates:
    t3_denoiser_seeds_bundle/results/denoiser_v4b_seed1.pt
    t3_denoiser_seeds_bundle/results/denoiser_v4b_seed2.pt
    t3_denoiser_seeds_bundle/results/train_v4b_seed1.jsonl
    t3_denoiser_seeds_bundle/results/train_v4b_seed2.jsonl

The local entrypoint also writes:
    t3_denoiser_seeds_bundle/results/t3_seed{1,2}_summary.json
    t3_denoiser_seeds_bundle/results/t3_summary.json

## Note on training-time budget

The plan's "~6 hr per seed" assumes the trainer is wall-clock-bound.
The v4b config's `total_time_minutes` is 90 by default; the launcher
overrides it to 360 min (6 hr) per seed via the local entrypoint flag
`--total-time-minutes 360`, matching the EXPERIMENTATION_PLAN budget.
The trainer also has `early_stop.enabled = true` in the YAML, so it
will stop earlier if val_loss plateaus for 12 evals.
