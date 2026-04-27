"""Mine hard negatives + synthetic perturbations and encode them through LIMO
v1 to produce labelled latents (viab=0) for retraining the multi-head score
model.

Hard negatives come from:
    1. SMILES from past rerank markdowns that fail chem_redflags.screen()
    2. Generated SMILES that *pass* redflags but contain manual chemistry
       red flags (open-chain N-chain >= 3, gem-tetra, etc.) — we synthesise
       these by template rewrites of real positives.

Output augments the existing `latent_labels_v1.pt` into v2.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, "scripts/diffusion")
sys.path.insert(0, "scripts/vae")
sys.path.insert(0, "external/LIMO")
from chem_redflags import screen
from limo_factory import load_limo
from limo_model import SELFIESTokenizer, build_limo_vocab, LIMO_MAX_LEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


SYNTHETIC_HARD_TEMPLATES = [
    # (description, SMARTS-pattern, replacement) - kept simple for now;
    # we instead generate from-scratch hard negatives.
]

CURATED_HARD_NEGATIVES = [
    # known model-cheat patterns; many short variations
    "O=[N+]([O-])N=NN=NN[N+](=O)[O-]",
    "O=[N+]([O-])N=NNN=N[N+](=O)[O-]",
    "O=[N+]([O-])N=NN=NN=NN[N+](=O)[O-]",
    "O=[N+]([O-])C(=C([N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
    "O=[N+]([O-])CC([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",
    "O=C([N+](=O)[O-])[N+](=O)[O-]",
    "O=[N+]([O-])C1([N+](=O)[O-])CC1",
    "O=[N+]([O-])C1([N+](=O)[O-])CCC1",
    "C(=NN=NN[N+](=O)[O-])N[N+](=O)[O-]",
    "O=[N+]([O-])N([N+](=O)[O-])N([N+](=O)[O-])[N+](=O)[O-]",
    "O=[N+]([O-])C=C=C=C[N+](=O)[O-]",
    "O=[N+]([O-])C=C(N=N[N+](=O)[O-])[N+](=O)[O-]",
    "C(=C[N+](=O)[O-])C=C[N+](=O)[O-]",
    "[O-][N+](=O)CN=NCN=NC[N+](=O)[O-]",
    "O=[N+]([O-])C1=C([N+](=O)[O-])C1([N+](=O)[O-])[N+](=O)[O-]",
    "O=[N+]([O-])C=N[N+](=O)[O-]",   # too small
    "N#C[N+](=O)[O-]",                # nitryl cyanide
    "C#C[N+](=O)[O-]",                # very small
    "[O-][N+](=O)C(=O)C[N+](=O)[O-]",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_in",  required=True)
    ap.add_argument("--pool_mds",   nargs="+", default=[])
    ap.add_argument("--limo_v1",    default="experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt")
    ap.add_argument("--out",        required=True)
    args = ap.parse_args()

    # Mine hard negatives from past rerank MDs
    pool_smis = set()
    for md in args.pool_mds:
        if not Path(md).exists(): continue
        text = Path(md).read_text(encoding="utf-8")
        for line in text.split("\n"):
            if line.startswith("| ") and "rank" not in line and "---" not in line:
                cells = [c.strip(" `") for c in line.split("|")]
                if len(cells) >= 12: pool_smis.add(cells[-2])
    print(f"Scanned {len(pool_smis)} unique SMILES from {len(args.pool_mds)} pool MDs")

    hard_neg = []
    for smi in pool_smis:
        scr = screen(smi)
        if scr["status"] != "ok":
            hard_neg.append(smi)
        elif scr["red_flag_score"] <= -1.5:
            # passed hard filters but heavy soft penalties
            hard_neg.append(smi)
    # Add curated cheats
    hard_neg += CURATED_HARD_NEGATIVES
    # Dedup by canonical
    seen = set(); deduped = []
    for s in hard_neg:
        try:
            m = Chem.MolFromSmiles(s)
            if m is None: continue
            c = Chem.MolToSmiles(m)
            if c not in seen and len(s) <= 200:
                seen.add(c); deduped.append(c)
        except Exception:
            pass
    print(f"Total hard negatives: {len(deduped)}")

    print("Loading LIMO v1 for encoding ...")
    limo, _ = load_limo(".", version="v1", ckpt_override=args.limo_v1, device=DEVICE)
    limo.eval()
    vocab = build_limo_vocab("external/LIMO/zinc250k.smi")
    tok = SELFIESTokenizer(vocab)

    encoded_z = []; encoded_smi = []
    skipped = 0
    for smi in deduped:
        seq_pair = tok.smiles_to_tensor(smi)
        if seq_pair is None: skipped += 1; continue
        seq, _ = seq_pair
        if seq.shape[0] > LIMO_MAX_LEN: skipped += 1; continue
        with torch.no_grad():
            _, mu, _ = limo._m.encode(seq.unsqueeze(0).to(DEVICE))[:3]
        encoded_z.append(mu.squeeze(0).cpu())
        encoded_smi.append(smi)
    print(f"Encoded: {len(encoded_z)} / {len(deduped)} (skipped {skipped})")

    if not encoded_z:
        print("Nothing to add. Exiting."); return

    # Load existing labels
    print(f"Loading existing labels from {args.labels_in} ...")
    blob = torch.load(args.labels_in, weights_only=False, map_location="cpu")

    # Stack new entries
    new_z = torch.stack(encoded_z)
    n_new = new_z.shape[0]
    new_y_viab = torch.zeros(n_new)             # hard negatives = viab 0
    new_y_sa   = torch.full((n_new,), float("nan"))
    new_y_sc   = torch.full((n_new,), float("nan"))
    new_y_sens = torch.ones(n_new)              # treat as max sensitivity
    new_pass   = torch.zeros(n_new, dtype=torch.bool)

    # Concatenate
    out_blob = {
        "z_mu":         torch.cat([blob["z_mu"], new_z]),
        "smiles":       list(blob["smiles"]) + encoded_smi,
        "y_viab":       torch.cat([blob["y_viab"], new_y_viab]),
        "y_sa":         torch.cat([blob["y_sa"], new_y_sa]),
        "y_sc":         torch.cat([blob["y_sc"], new_y_sc]),
        "y_sens":       torch.cat([blob["y_sens"], new_y_sens]),
        "redflag_pass": torch.cat([blob["redflag_pass"], new_pass]),
    }
    if "perf_target" in blob and isinstance(blob["perf_target"], torch.Tensor):
        n_perf = blob["perf_target"].shape[1]
        out_blob["perf_target"] = torch.cat([
            blob["perf_target"], torch.zeros(n_new, n_perf)])
    if "perf_mask" in blob and isinstance(blob["perf_mask"], torch.Tensor):
        out_blob["perf_mask"] = torch.cat([
            blob["perf_mask"], torch.zeros(n_new, blob["perf_mask"].shape[1] if blob["perf_mask"].dim()==2 else 1, dtype=torch.bool)])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_blob, out)
    print(f"-> {out}")
    print(f"   total rows: {out_blob['z_mu'].shape[0]}  (added {n_new} hard negatives)")


if __name__ == "__main__":
    main()
