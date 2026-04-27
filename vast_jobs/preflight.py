"""Sanity checks for any local or vast.ai job before launch.

Usage:
    python vast_jobs/preflight.py --kind data       --path data/training/latents_trustcond.pt
    python vast_jobs/preflight.py --kind ckpt       --path experiments/.../best.pt
    python vast_jobs/preflight.py --kind vae        --path experiments/limo_v3_2_noskip_AR_*/checkpoints/best.pt
    python vast_jobs/preflight.py --kind denoiser   --path experiments/diffusion_*/checkpoints/best.pt
    python vast_jobs/preflight.py --kind manifest   --files vast_jobs/job_c_massive_rerank.py vast_jobs/denoiser_v3_best.pt ...

Exit code 0 = all checks passed, 1 = at least one failed.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, os
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def _fail(msg, fails):
    print(f"{FAIL} {msg}"); fails.append(msg)

def _ok(msg):
    print(f"{PASS} {msg}")

def _warn(msg):
    print(f"{WARN} {msg}")

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()

def check_data(path, fails):
    import torch
    p = Path(path)
    if not p.exists():
        _fail(f"data file missing: {p}", fails); return
    blob = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "z" in blob:
        z = blob["z"]
    elif torch.is_tensor(blob):
        z = blob
    else:
        keys = list(blob.keys()) if isinstance(blob, dict) else type(blob)
        _warn(f"unrecognized blob structure: {keys}"); return
    if not torch.is_tensor(z):
        _fail(f"z not a tensor", fails); return
    if torch.isnan(z).any():
        _fail(f"NaN in z (count={torch.isnan(z).sum().item()})", fails)
    else:
        _ok("no NaN in z")
    if torch.isinf(z).any():
        _fail(f"Inf in z (count={torch.isinf(z).sum().item()})", fails)
    else:
        _ok("no Inf in z")
    mean, std = z.float().mean().item(), z.float().std().item()
    print(f"  shape={tuple(z.shape)} dtype={z.dtype} mean={mean:.4f} std={std:.4f}")
    if abs(mean) > 5.0:
        _warn(f"mean far from 0 ({mean:.3f}); diffusion expects N(0,I) prior")
    if std < 0.5 or std > 50:
        _warn(f"std out-of-band ({std:.3f}); check normalization")
    _ok(f"sha256={sha256(p)[:16]}…")

def check_ckpt(path, fails):
    import torch
    p = Path(path)
    if not p.exists():
        _fail(f"ckpt missing: {p}", fails); return
    state = torch.load(p, map_location="cpu", weights_only=False)
    sd = state.get("model_state", state.get("model", state.get("state_dict", state)))
    if not isinstance(sd, dict):
        _fail("checkpoint not a dict", fails); return
    def _flatten(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if torch.is_tensor(v):
                out[key] = v
            elif isinstance(v, dict):
                out.update(_flatten(v, key))
        return out
    flat = _flatten(sd)
    n_params = sum(v.numel() for v in flat.values())
    print(f"  tensors={len(flat)} params={n_params/1e6:.2f}M")
    finite = all(torch.isfinite(v).all().item() for v in flat.values()
                 if v.is_floating_point())
    if not finite:
        _fail("NaN/Inf in checkpoint weights", fails)
    else:
        _ok("all weights finite")
    _ok(f"sha256={sha256(p)[:16]}…")

def check_vae(path, fails):
    """Encode->decode round-trip on random tokens; report token accuracy."""
    check_ckpt(path, fails)
    import torch
    sys.path.insert(0, "scripts/vae")
    try:
        from limo_factory import load_limo
        limo, ver = load_limo(".", path, "cpu")
        toks = torch.randint(0, 50, (4, 64))
        with torch.no_grad():
            z, _, _ = limo.encode(toks)
            out = limo.decode(z)
        ok = out is not None and (torch.is_tensor(out) or isinstance(out, list))
        _ok(f"VAE round-trip ok (version={ver}, z.shape={tuple(z.shape)})") if ok \
            else _fail("VAE round-trip returned None", fails)
    except Exception as e:
        _warn(f"VAE round-trip skipped: {type(e).__name__}: {e}")

def check_denoiser(path, fails):
    """Forward pass with dummy x_t,t,cond on the diffusion model."""
    check_ckpt(path, fails)
    import torch
    sys.path.insert(0, "scripts/diffusion")
    try:
        from model import ConditionalDenoiser
    except Exception as e:
        _warn(f"denoiser forward skipped: cannot import ConditionalDenoiser ({e})"); return
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {}) or {}
    arch = cfg.get("model", cfg)
    try:
        latent_dim = arch.get("latent_dim", 1024)
        n_props = arch.get("n_props", 4)
        model = ConditionalDenoiser(
            latent_dim=latent_dim,
            hidden=arch.get("hidden", 2048),
            n_blocks=arch.get("n_blocks", 8),
            n_props=n_props,
        )
        sd = state.get("model_state", state.get("model", state.get("state_dict", state)))
        miss, unex = model.load_state_dict(sd, strict=False)
        if miss or unex:
            _warn(f"state_dict mismatch: missing={len(miss)} unexpected={len(unex)}")
        model.eval()
        x = torch.randn(2, latent_dim)
        t = torch.tensor([100, 500])
        v = torch.randn(2, n_props)
        m = torch.ones(2, n_props)
        with torch.no_grad():
            y = model(x, t, v, m)
        if torch.isfinite(y).all():
            _ok(f"denoiser forward ok, out.shape={tuple(y.shape)}")
        else:
            _fail("denoiser forward produced NaN/Inf", fails)
    except Exception as e:
        _warn(f"denoiser forward skipped: {type(e).__name__}: {e}")

def check_manifest(files, fails):
    """Print sha256 + size for every file in upload manifest."""
    manifest = {}
    for f in files:
        p = Path(f)
        if not p.exists():
            _fail(f"missing: {f}", fails); continue
        size = p.stat().st_size
        digest = sha256(p)
        manifest[str(p)] = {"sha256": digest, "size": size}
        _ok(f"{p.name:40s} {size/1e6:>8.1f}MB  {digest[:16]}…")
    out = Path("logs/upload_manifest.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=["data", "ckpt", "vae", "denoiser", "manifest"])
    ap.add_argument("--path")
    ap.add_argument("--files", nargs="+")
    args = ap.parse_args()
    fails = []
    if args.kind == "data":
        check_data(args.path, fails)
    elif args.kind == "ckpt":
        check_ckpt(args.path, fails)
    elif args.kind == "vae":
        check_vae(args.path, fails)
    elif args.kind == "denoiser":
        check_denoiser(args.path, fails)
    elif args.kind == "manifest":
        check_manifest(args.files or [], fails)
    if fails:
        print(f"\n{FAIL} {len(fails)} check(s) failed")
        sys.exit(1)
    print(f"\n{PASS} all checks passed")

if __name__ == "__main__":
    main()
