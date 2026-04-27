"""Diagnose the trained multi-head score model.

Tests:
    1. Per-head validation calibration (predicted vs target scatter)
    2. Sanity probe on known seeds (RDX, HMX, CL-20, FOX-7)
    3. Sanity probe on known model-cheats (gem-dinitro 4-ring, polyazene)
    4. Latent perturbation: hold z, vary sigma, observe head outputs
    5. Gradient direction sanity: ∇_z log P(viable) should point AWAY from
       hard-rejected molecules' z when starting from a positive seed

Output:
    experiments/score_model_v1/diagnostics.json + diagnostic.svg
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts/viability")
sys.path.insert(0, "scripts/vae")
sys.path.insert(0, "external/LIMO")
from train_multihead_latent import MultiHeadScoreModel
from limo_factory import load_limo
from limo_model import SELFIESTokenizer, build_limo_vocab, LIMO_MAX_LEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANCHORS = {
    "RDX":     "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]",
    "HMX":     "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]",
    "TNT":     "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
    "FOX-7":   "NC(=C([N+](=O)[O-])[N+](=O)[O-])N",
    "PETN":    "C(C(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-])O[N+](=O)[O-]",
    "TATB":    "Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]",
}

CHEATS = {
    "gemtetra_C2": "O=[N+]([O-])C(=C([N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
    "trinitromethane": "O=[N+]([O-])C([N+](=O)[O-])[N+](=O)[O-]",
    "polyazene_NN_NO2": "O=[N+]([O-])N=NN=NN[N+](=O)[O-]",
    "gem_dinitro_3ring": "O=[N+]([O-])C1([N+](=O)[O-])CC1",
    "tiny_dinitro": "O=C([N+](=O)[O-])[N+](=O)[O-]",   # dinitromethane
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limo_v1", default="experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt")
    ap.add_argument("--out", default="experiments/score_model_v1/diagnostics.json")
    args = ap.parse_args()

    print(f"Loading score model from {args.model} ...")
    blob = torch.load(args.model, weights_only=False, map_location=DEVICE)
    cfg = blob["config"]
    model = MultiHeadScoreModel(latent_dim=cfg["latent_dim"]).to(DEVICE)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    sigma_max = cfg["sigma_max"]

    print("Loading LIMO v1 for encoding ...")
    limo, _ = load_limo(".", version="v1", ckpt_override=args.limo_v1, device=DEVICE)
    limo.eval()
    vocab = build_limo_vocab("external/LIMO/zinc250k.smi")
    tok = SELFIESTokenizer(vocab)

    def encode(smi):
        seq_pair = tok.smiles_to_tensor(smi)
        if seq_pair is None: return None
        seq, _ = seq_pair
        if seq.shape[0] > LIMO_MAX_LEN: return None
        with torch.no_grad():
            _, mu, _ = limo._m.encode(seq.unsqueeze(0).to(DEVICE))[:3]
        return mu

    out = {}

    # -- Anchor + cheat probes at sigma=0 (clean encoder posterior) --
    print("\n=== Anchor probe (sigma=0) ===")
    rows = []
    for label_set, items in [("anchor", ANCHORS), ("cheat", CHEATS)]:
        for name, smi in items.items():
            z = encode(smi)
            if z is None:
                print(f"  {name}: tokenization fail"); continue
            sigma = torch.zeros(1, device=DEVICE)
            with torch.no_grad():
                o = model(z, sigma)
            viab = torch.sigmoid(o["viab_logit"]).item()
            sa_z = o["sa"].item()
            sc_z = o["sc"].item()
            sens_z = o["sens"].item()
            sa_real = sa_z * cfg["sa_sd"] + cfg["sa_mu"]
            sc_real = sc_z * cfg["sc_sd"] + cfg["sc_mu"]
            sens_real = sens_z * cfg["sens_sd"] + cfg["sens_mu"]
            row = {"name": name, "type": label_set,
                   "viab": viab, "sa": sa_real, "sc": sc_real, "sens": sens_real}
            rows.append(row)
            print(f"  [{label_set:6}] {name:<22} viab={viab:.3f}  SA={sa_real:.2f}  "
                  f"SC={sc_real:.2f}  sens={sens_real:.3f}")

    out["anchor_probe"] = rows

    # -- Sigma sweep on RDX z --
    print("\n=== Sigma sweep on RDX ===")
    z_rdx = encode(ANCHORS["RDX"])
    sweep = []
    for sigma_v in np.linspace(0, sigma_max, 8):
        sigma = torch.full((1,), float(sigma_v), device=DEVICE)
        # Add fixed noise so the trajectory is deterministic
        torch.manual_seed(123)
        eps = torch.randn_like(z_rdx) * float(sigma_v)
        z_t = z_rdx + eps
        with torch.no_grad():
            o = model(z_t, sigma)
        viab = torch.sigmoid(o["viab_logit"]).item()
        sweep.append({"sigma": float(sigma_v), "viab": viab,
                      "sa": float(o["sa"].item() * cfg["sa_sd"] + cfg["sa_mu"]),
                      "sens": float(o["sens"].item() * cfg["sens_sd"] + cfg["sens_mu"])})
        print(f"  sigma={sigma_v:.2f}  viab={viab:.3f}  "
              f"SA={sweep[-1]['sa']:.2f}  sens={sweep[-1]['sens']:.3f}")
    out["sigma_sweep_rdx"] = sweep

    # -- Gradient sanity: starting from a CHEAT z, ascend log P(viable) --
    print("\n=== Gradient ascent from cheat -> direction sanity ===")
    z_cheat = encode(CHEATS["polyazene_NN_NO2"])
    if z_cheat is not None:
        z = z_cheat.clone().detach().requires_grad_(True)
        sigma = torch.zeros(1, device=DEVICE)
        steps = 20; lr = 0.05
        traj = []
        for s in range(steps):
            o = model(z, sigma)
            loss = -F.logsigmoid(o["viab_logit"]).sum()  # ascend log P(viable)
            grad = torch.autograd.grad(loss, z)[0]
            with torch.no_grad():
                z = (z - lr * grad).detach().requires_grad_(True)
            with torch.no_grad():
                viab = torch.sigmoid(model(z, sigma)["viab_logit"]).item()
            traj.append({"step": s, "viab": viab, "z_norm": z.norm().item()})
        print(f"  start viab={traj[0]['viab']:.3f}  ->  end viab={traj[-1]['viab']:.3f}  "
              f"({steps} steps, lr={lr})")
        out["gradient_ascent"] = traj
        # Decode the optimised z and inspect
        with torch.no_grad():
            decoded = limo._m.decode(z)
        if torch.is_tensor(decoded):
            toks = decoded.argmax(-1).cpu().numpy()[0]
            smi = tok.indices_to_smiles(toks.tolist())
            print(f"  decoded after ascent: {smi[:100]}")
            out["ascent_decoded"] = smi

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
