# T1 launch notes

## First-attempt failure: charmap codec encode of U+2713

The first `modal run` from Git Bash on this Windows box failed with:

    'charmap' codec can't encode character '✓' in position 0:
    character maps to <undefined>

Modal's CLI client streams progress with the U+2713 (✓) check-mark, but
Python's default stdio encoding under Git Bash is `cp1252` (charmap),
which does not contain that codepoint.

## Fix

Set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` before invoking
`python -m modal run`.  This is now baked into the launch command:

    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m modal run --detach \
        t1_bde_bundle/modal_t1_bde.py > t1_bde_bundle/launch.log 2>&1

The `--detach` flag is also required so the job survives the local
client being torn down by the harness.

## Successful launch

    App URL:  https://modal.com/apps/llmcourse/main/ap-nCIuBZpnK4DyJDara4CcnP
    App ID:   ap-nCIuBZpnK4DyJDara4CcnP
    Image:    im-5mWnOC5LR3ZSM2RNdPWdEu  (custom: ubuntu:22.04 + xtb 6.6.1 + rdkit)
    GPU:      none (CPU 4 cores, 8 GB RAM)
    Timeout:  60 min hard cap

## Output paths

  - per-compound results:    `t1_bde_bundle/results/t1_bde_<id>.json`
  - merged summary:          `t1_bde_bundle/results/t1_bde.json`
  - text log of this launch: `t1_bde_bundle/launch.log`
