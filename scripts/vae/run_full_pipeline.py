"""
One-shot runner: smoke test + fine-tune + evaluate + report.

Useful for fully-automated end-to-end runs. Each stage is idempotent — if an
artefact already exists, the corresponding stage is skipped unless --force.

Usage:
    python scripts/vae/run_full_pipeline.py --config configs/vae_limo.yaml
    python scripts/vae/run_full_pipeline.py --config configs/vae_limo.yaml --skip-smoke
    python scripts/vae/run_full_pipeline.py --config configs/vae_limo.yaml --resume <exp_dir>
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], desc: str) -> int:
    print("\n" + "=" * 72)
    print(f"STAGE: {desc}")
    print(f"  cmd: {' '.join(cmd)}")
    print("=" * 72)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, env=env)
    print(f"[{desc}] exit={proc.returncode}  elapsed={time.time()-t0:.1f}s")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval",  action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--base",   default="E:/Projects/EnergeticDiffusion2")
    args = ap.parse_args()

    base = Path(args.base)
    py = sys.executable
    os.chdir(base)

    # 1. smoke test
    if not args.skip_smoke:
        smoke_report = base / "external/LIMO/smoke_report.json"
        if not smoke_report.exists():
            rc = run([py, "scripts/vae/limo_smoke.py"], "smoke test")
            if rc != 0:
                print("Smoke test FAILED — aborting.")
                return rc
        else:
            print(f"[skip] smoke report exists: {smoke_report}")

    # 2. fine-tune
    exp_dir = None
    if not args.skip_train:
        cmd = [py, "scripts/vae/limo_finetune.py", "--config", args.config]
        if args.resume:
            cmd += ["--resume", args.resume]
        rc = run(cmd, "fine-tune")
        if rc != 0:
            print("Training FAILED — aborting.")
            return rc

    # find most recent experiment matching the run name
    import yaml
    cfg = yaml.safe_load(open(args.config))
    run_name = cfg["run"]["name"]
    exps = sorted((base / cfg["paths"]["experiments_dir"]).glob(f"{run_name}_*"),
                   key=lambda p: p.stat().st_mtime)
    if not exps:
        print("No experiment directory found — did training create one?")
        return 1
    exp_dir = exps[-1]
    print(f"\nExperiment: {exp_dir}")

    # 3. evaluation
    if not args.skip_eval:
        rc = run([py, "scripts/vae/limo_evaluate.py", "--exp", str(exp_dir)],
                  "evaluation")
        if rc != 0:
            print("Evaluation failed; will still try to render report.")

    # 4. HTML report
    if not args.skip_report:
        rc = run([py, "scripts/vae/limo_report.py", "--exp", str(exp_dir)],
                  "html report")
        if rc != 0:
            return rc

    print(f"\n✓ Pipeline complete.")
    print(f"  Experiment: {exp_dir}")
    print(f"  Report:     {exp_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
