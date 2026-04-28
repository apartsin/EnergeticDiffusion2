"""Patch AiZynth config to widen expansion-policy cutoffs.
Default cutoff_cumulative=0.995, cutoff_number=50; we relax to 1.0 / 500
so weak-probability templates still fire on energetic-domain target nodes."""
import sys, yaml
config_path = sys.argv[1]
with open(config_path) as f:
    cfg = yaml.safe_load(f)
exp = cfg.get('expansion') or cfg.get('policy', {}).get('expansion', {})
# Both old and new aizynth schemas; patch any stochastic-policy block
def patch_section(d):
    for k, v in d.items() if isinstance(d, dict) else []:
        if isinstance(v, dict):
            patch_section(v)
            if 'cutoff_cumulative' in v: v['cutoff_cumulative'] = 1.0
            if 'cutoff_number' in v: v['cutoff_number'] = 500
patch_section(cfg)
with open(config_path, 'w') as f:
    yaml.safe_dump(cfg, f)
print('[relax] patched cutoff_cumulative=1.0 cutoff_number=500 in', config_path)
