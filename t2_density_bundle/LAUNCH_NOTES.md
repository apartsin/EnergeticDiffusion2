# T2 launch notes

Same `PYTHONIOENCODING=utf-8` env-var workaround as T1 (see T1 LAUNCH_NOTES.md).

## Successful launch

    App URL:  https://modal.com/apps/llmcourse/main/ap-W6e8ZL0osRz4TvfMWEXWEW
    App ID:   ap-W6e8ZL0osRz4TvfMWEXWEW
    Image:    im-LsNbiihvYXXhKLgaDisdS7  (debian-slim + python 3.11 + rdkit + numpy)
    GPU:      none (CPU 2 cores, 4 GB RAM)
    Timeout:  30 min hard cap

## Output paths

  - merged JSON:  `t2_density_bundle/results/t2_density_crosscheck.json`
  - text log:     `t2_density_bundle/launch.log`
