"""Upload DGLD checkpoints to Zenodo as a draft deposition.

Leaves the deposition in DRAFT state so the user can review metadata + files
before clicking 'Publish' (publishing mints the DOI and is irreversible).

Files uploaded:
    v3_best.pt          (denoiser ckpt, expanded-v3 config)
    v4b_best.pt         (denoiser ckpt, expanded-v4b config)
    limo_best.pt        (LIMO VAE encoder/decoder)
    score_model_v3e.pt  (multi-head latent score model, 5-head)
    score_model_v3f.pt  (multi-head latent score model, 6-head with hazard)
    vocab.json          (SELFIES alphabet)
    meta.json           (latent_dim, n_props, stats, config blobs)

Run: python zenodo_upload.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import requests

HERE = Path(__file__).parent
TOKEN = (HERE / "zenodo.key").read_text().strip()

# Public Zenodo (sandbox: https://sandbox.zenodo.org/api)
BASE = "https://zenodo.org/api"

FILES = [
    HERE / "m1_bundle" / "v3_best.pt",
    HERE / "m1_bundle" / "v4b_best.pt",
    HERE / "m1_bundle" / "limo_best.pt",
    HERE / "m1_bundle" / "score_model_v3e.pt",
    HERE / "m1_bundle" / "score_model_v3f.pt",
    HERE / "m1_bundle" / "vocab.json",
    HERE / "m1_bundle" / "meta.json",
]

METADATA = {
    "metadata": {
        "title": "DGLD: Domain-Gated Latent Diffusion checkpoints for energetic-materials discovery",
        "upload_type": "dataset",
        "description": (
            "<p>Trained model checkpoints accompanying the manuscript "
            "<em>DGLD: Domain-Gated Latent Diffusion for the Discovery of Novel "
            "Energetic Materials</em>. Includes two conditional latent denoisers "
            "(<code>v3_best.pt</code>, <code>v4b_best.pt</code>), the LIMO VAE "
            "encoder/decoder (<code>limo_best.pt</code>), two multi-head "
            "classifier-guidance score models (<code>score_model_v3e.pt</code> "
            "5-head, <code>score_model_v3f.pt</code> 6-head with hazard), the "
            "SELFIES alphabet (<code>vocab.json</code>), and run metadata "
            "(<code>meta.json</code>: latent dim 1024, four conditional "
            "properties, normalisation stats, denoiser config). All weights are "
            "fp32 PyTorch state-dicts. See the manuscript §3-§4 for architecture "
            "and §5 for evaluation. Sampling reproduction: "
            "<code>m1_bundle/m1_sweep.py</code> in the companion code release.</p>"
        ),
        "creators": [
            {"name": "Anonymous", "affiliation": "Anonymous"}
        ],
        "keywords": [
            "diffusion models", "molecular generation",
            "energetic materials", "classifier-free guidance",
            "latent diffusion", "VAE", "SELFIES",
        ],
        "license": "cc-by-4.0",
        "access_right": "open",
    }
}


def main():
    s = requests.Session()
    s.params = {"access_token": TOKEN}

    print(f"[zenodo] base={BASE}")
    print("[zenodo] Creating draft deposition ..."); sys.stdout.flush()
    r = s.post(f"{BASE}/deposit/depositions", json={})
    r.raise_for_status()
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print(f"[zenodo] deposition id = {dep_id}")
    print(f"[zenodo] bucket        = {bucket}")
    print(f"[zenodo] html          = {dep['links'].get('html')}")
    sys.stdout.flush()

    print("[zenodo] Setting metadata ..."); sys.stdout.flush()
    r = s.put(f"{BASE}/deposit/depositions/{dep_id}", json=METADATA)
    r.raise_for_status()
    print("[zenodo] metadata OK")

    for f in FILES:
        if not f.exists():
            print(f"[zenodo] SKIP missing: {f}"); continue
        sz_mb = f.stat().st_size / 1e6
        print(f"[zenodo] uploading {f.name} ({sz_mb:.1f} MB) ..."); sys.stdout.flush()
        t0 = time.time()
        with f.open("rb") as fh:
            r = s.put(f"{bucket}/{f.name}", data=fh)
        r.raise_for_status()
        print(f"[zenodo]   -> done in {time.time()-t0:.0f}s, MD5={r.json().get('checksum')}")
        sys.stdout.flush()

    # Save deposition link for later
    out = {"deposition_id": dep_id, "html": dep["links"].get("html"),
           "doi_when_published": dep.get("metadata", {}).get("prereserve_doi", {}).get("doi")}
    (HERE / "zenodo_deposition.json").write_text(json.dumps(out, indent=2))
    print(f"\n[zenodo] DRAFT created. Review at: {dep['links'].get('html')}")
    print(f"[zenodo] Reserved DOI (mints on publish): {out['doi_when_published']}")
    print(f"[zenodo] Deposition info -> zenodo_deposition.json")
    print("[zenodo] === DONE ===")


if __name__ == "__main__":
    main()
