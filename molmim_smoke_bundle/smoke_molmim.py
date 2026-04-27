"""5-minute MolMIM smoke test on vast.ai (bionemo image).

Validates the entire MolMIM path before running the full 382k encode + denoiser
retrain. Loads .nemo, encodes 100 SMILES, prints shapes + a sanity stat.

Required uploads:
  - molmim_70m_24_3.nemo
  - smiles_cond_bundle.pt   (we'll grab the first 100 SMILES from this)
"""
import os, sys, time, json, traceback
os.makedirs("results", exist_ok=True)


def log(msg):
    print(f"[smoke] {msg}", flush=True)


t0 = time.time()
log(f"python={sys.version.split()[0]}")

import torch
log(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA required"
log(f"GPU: {torch.cuda.get_device_name(0)}")

result = {"phase": "starting", "ok": False}

try:
    log("importing nemo …")
    import nemo
    log(f"nemo {nemo.__version__}")

    log("importing MolMIMModel …")
    from nemo.collections.nlp.models.language_modeling.megatron.molmim_model import MolMIMModel
    log("MolMIMModel imported")

    log("loading .nemo …")
    t1 = time.time()
    model = MolMIMModel.restore_from("molmim_70m_24_3.nemo", map_location="cuda")
    model.eval()
    log(f"  loaded in {time.time()-t1:.1f}s")
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  params={n_params/1e6:.1f}M")

    log("loading 100 test SMILES from smiles_cond_bundle.pt …")
    bundle = torch.load("smiles_cond_bundle.pt", weights_only=False)
    test_smiles = bundle["smiles"][:100]
    log(f"  {len(test_smiles)} SMILES (first: {test_smiles[0]!r})")

    log("encoding …")
    t1 = time.time()
    with torch.no_grad():
        z = model.encode(test_smiles)
    log(f"  encoded in {time.time()-t1:.1f}s, shape={tuple(z.shape)}, dtype={z.dtype}")
    log(f"  z stats: mean={z.float().mean().item():.4f}  std={z.float().std().item():.4f}")

    log("decode round-trip on first 8 …")
    try:
        with torch.no_grad():
            roundtrip = model.decode(z[:8])
        log(f"  decoded {len(roundtrip)} sequences (first: {roundtrip[0]!r})")
    except Exception as e:
        log(f"  decode failed (non-blocking for encode-only path): {e}")

    result = {"phase": "ok", "ok": True, "n_params_M": n_params / 1e6,
              "z_shape": list(z.shape), "z_mean": float(z.float().mean()),
              "z_std": float(z.float().std()), "first_smi": test_smiles[0]}
    log("=== SMOKE TEST PASSED ===")
except Exception as e:
    result["error"] = str(e)
    result["traceback"] = traceback.format_exc()
    log(f"=== SMOKE TEST FAILED: {type(e).__name__}: {e} ===")
    traceback.print_exc()

result["elapsed_s"] = time.time() - t0
with open("results/smoke_result.json", "w") as f:
    json.dump(result, f, indent=2)
log(f"total elapsed: {result['elapsed_s']:.1f}s")
log(f"result saved to results/smoke_result.json")
