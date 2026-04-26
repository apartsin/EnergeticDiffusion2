"""Full diffusion pipeline runner: encode (if needed) + train + evaluate + report.

Usage:
    python scripts/diffusion/run_full_pipeline.py --config configs/diffusion.yaml
    python scripts/diffusion/run_full_pipeline.py --config configs/diffusion.yaml --resume <exp>
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def run(cmd, desc):
    print("\n" + "=" * 72)
    print(f"STAGE: {desc}")
    print(f"  cmd: {' '.join(cmd)}")
    print("=" * 72)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    rc = subprocess.call(cmd, env=env)
    print(f"[{desc}] exit={rc}  elapsed={time.time()-t0:.0f}s")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limo_ckpt", default=None,
                    help="LIMO fine-tuned best.pt path (only needed if latents.pt missing)")
    ap.add_argument("--skip_encode", action="store_true")
    ap.add_argument("--skip_train",  action="store_true")
    ap.add_argument("--skip_eval",   action="store_true")
    ap.add_argument("--skip_report", action="store_true")
    ap.add_argument("--resume",  default=None)
    ap.add_argument("--smoke",   action="store_true")
    ap.add_argument("--base",    default="E:/Projects/EnergeticDiffusion2")
    args = ap.parse_args()

    base = Path(args.base)
    py = sys.executable
    os.chdir(base)
    cfg = yaml.safe_load(open(args.config))

    # 1. encode latents if needed
    latents_path = base / cfg["paths"]["latents_pt"]
    if not args.skip_encode and not latents_path.exists():
        if not args.limo_ckpt:
            # auto-discover most recent LIMO best.pt
            candidates = sorted(
                (base / "experiments").glob("limo_ft_energetic_*/checkpoints/best.pt"),
                key=lambda p: p.stat().st_mtime)
            if not candidates:
                print("ERROR: no LIMO best.pt found and --limo_ckpt not supplied"); return 1
            args.limo_ckpt = str(candidates[-1])
            print(f"Auto-detected LIMO ckpt: {args.limo_ckpt}")
        rc = run([py, "scripts/diffusion/encode_latents.py",
                   "--ckpt", args.limo_ckpt, "--out", str(latents_path)],
                  "encode latents")
        if rc != 0: return rc
    else:
        print(f"[skip] latents exist: {latents_path} ({latents_path.stat().st_size/1e6:.1f} MB)")

    # 2. train
    if not args.skip_train:
        cmd = [py, "scripts/diffusion/train.py", "--config", args.config]
        if args.resume: cmd += ["--resume", args.resume]
        if args.smoke:  cmd += ["--smoke"]
        rc = run(cmd, "train diffusion")
        if rc != 0: return rc

    # 3. find experiment dir
    exps = sorted((base / cfg["paths"]["experiments_dir"]).glob(f"{cfg['run']['name']}_*"),
                   key=lambda p: p.stat().st_mtime)
    if not exps:
        print("No experiment directory found"); return 1
    exp_dir = exps[-1]
    print(f"\nExperiment: {exp_dir}")

    # 4. evaluate
    if not args.skip_eval:
        rc = run([py, "scripts/diffusion/evaluate.py", "--exp", str(exp_dir)],
                  "evaluate")
        if rc != 0: print("Eval failed; continuing to report.")

    # 5. report
    if not args.skip_report:
        rc = run([py, "scripts/diffusion/report.py", "--exp", str(exp_dir)],
                  "html report")
        if rc != 0: return rc

    print(f"\n✓ Done.\n  Experiment: {exp_dir}\n  Report:     {exp_dir/'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
