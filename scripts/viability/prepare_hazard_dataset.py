"""S1 hazard dataset prep.

Builds a binary hazard label dataset for the new HAZARD head:
    target=1 (hazardous): h50 < 10 cm + 23 chemist-audit rejects from
        merged top-100 + ~15 literature primary explosives
    target=0 (safe-ish):   h50 > 50 cm secondary explosives

All encoded through LIMO -> latents.

Output: experiments/hazard_dataset.pt with z_mu, hazard_target, smiles, source.

Run:
    /c/Python314/python scripts/viability/prepare_hazard_dataset.py \
        --ckpt experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt \
        --out experiments/hazard_dataset.pt
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


# Known primary explosives + extreme-sensitivity reference set.
# Curated from Klapotke "Chemistry of High-Energy Materials" + IM database.
PRIMARY_EXPLOSIVES = [
    ("lead_azide_anion",      "[N-]=[N+]=[N-]"),     # lead azide active anion
    ("hydrazoic_acid",        "N#N=N"),
    ("mercury_fulminate_anion", "[O-][N+]#C"),       # fulminate ion
    ("nitrogen_triiodide_HN3", "N(N=N)N=N"),         # azide-like
    ("triacetone_triperoxide", "O1OC(C)(C)OOC(C)(C)OOC1(C)C"),
    ("hexamethylenetriperoxidediamine", "C1OOC2OOC(N1)N2"),  # HMTD - skeleton only
    ("acetone_peroxide",      "CC1(C)OOC(C)(C)OO1"),
    ("nitroglycerin",         "[O-][N+](=O)OCC(O[N+]([O-])=O)CO[N+](=O)[O-]"),
    ("PETN",                  "O=[N+]([O-])OCC(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]"),
    ("tetrazene_explosive",   "NN=NNC(=N)NN"),       # tetrazene 1
    ("mercury_styphnate",     "Oc1c([N+](=O)[O-])cc([N+](=O)[O-])c([N+](=O)[O-])c1O"),  # styphnic acid (sensitised)
    ("methylene_glycol_dinitrate", "[O-][N+](=O)OCO[N+]([O-])=O"),
    ("ethyleneglycol_dinitrate", "[O-][N+](=O)OCCO[N+]([O-])=O"),
    ("glycidyl_azide",        "C(C1CO1)N=[N+]=[N-]"),
    ("diazodinitrophenol",    "[N+](=N)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1O"),
    ("nitromethane",          "C[N+]([O-])=O"),
    ("benzo_furoxan",         "c1ccc2c(c1)on[n+]2[O-]"),
    # Also add chemist-audit reject classes as canonical examples:
    ("nnitroimine_canonical",  "C(=NN(=O)=O)N"),
    ("polyazene_canonical",    "N=NNC=NNN(=O)=O"),
    ("vinyl_azonitrate",       "[O-][N+](=O)N=NC=C"),
]


def load_chemist_audit_rejects() -> list:
    """The 23 SMILES rejected by the strengthened chem_redflags filter."""
    sys.path.insert(0, "scripts/diffusion")
    from chem_redflags import screen
    text = Path("experiments/final_merged_topN.md").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or "rank" in line or "---" in line:
            continue
        cells = [c.strip(" `") for c in line.split("|")]
        if len(cells) < 14:
            continue
        try:
            rank = int(cells[1])
            smi = cells[14]
            rows.append((rank, smi))
        except ValueError:
            continue
    rejects = []
    for rank, smi in rows:
        s = screen(smi)
        if s["status"] == "reject":
            rejects.append((f"chemist_reject_rank{rank}", smi,
                            s["reasons"][0] if s["reasons"] else "unknown"))
    return rejects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h50_dataset", default="experiments/sens_h50_dataset.pt")
    ap.add_argument("--h50_threshold_pos", type=float, default=10.0,
                    help="h50 < this is hazardous (cm)")
    ap.add_argument("--h50_threshold_neg", type=float, default=50.0,
                    help="h50 >= this is safe-ish (cm)")
    ap.add_argument("--out", default="experiments/hazard_dataset.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    base = Path.cwd()
    sys.path.insert(0, str(base / "scripts" / "vae"))
    from limo_model import (LIMOVAE, SELFIESTokenizer, load_vocab,
                              build_limo_vocab, save_vocab, LIMO_MAX_LEN, find_limo_repo)

    # ── 1. Load h50 dataset and split by threshold ───────────────────────
    print(f"Loading h50 dataset: {args.h50_dataset}"); sys.stdout.flush()
    h_blob = torch.load(args.h50_dataset, weights_only=False, map_location="cpu")
    z_h = h_blob["z_mu"].float()
    h50 = h_blob["h50_obs"].numpy()
    smis_h = h_blob["smiles"]
    pos_idx = np.where(h50 < args.h50_threshold_pos)[0]
    neg_idx = np.where(h50 >= args.h50_threshold_neg)[0]
    print(f"  h50 < {args.h50_threshold_pos}: {len(pos_idx)} (hazardous)"); sys.stdout.flush()
    print(f"  h50 >= {args.h50_threshold_neg}: {len(neg_idx)} (safe)"); sys.stdout.flush()

    # ── 2. Load chemist-audit rejects from merged top-100 ───────────────
    print("\nLoading chemist-audit rejects from merged top-100 ..."); sys.stdout.flush()
    rejects = load_chemist_audit_rejects()
    print(f"  found {len(rejects)} reject SMILES"); sys.stdout.flush()
    for name, smi, reason in rejects[:5]:
        print(f"    {name} -> {reason}"); sys.stdout.flush()

    # ── 3. Build full hazard SMILES list (positives) ────────────────────
    pos_set = []
    # 3a: from h50 < threshold
    for i in pos_idx:
        pos_set.append(("h50_lt_" + str(int(h50[i])) + "cm", smis_h[i], "h50_low"))
    # 3b: chemist-audit rejects
    for name, smi, reason in rejects:
        pos_set.append((name, smi, reason))
    # 3c: literature primary explosives
    for name, smi in PRIMARY_EXPLOSIVES:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print(f"  WARN: invalid SMILES for {name}: {smi}"); continue
        canon = Chem.MolToSmiles(m)
        pos_set.append((name, canon, "primary_explosive"))
    print(f"\nPositives total: {len(pos_set)}"); sys.stdout.flush()

    # 3d: deduplicate by canonical SMILES
    seen = {}
    for name, smi, reason in pos_set:
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        c = Chem.MolToSmiles(m)
        if c not in seen:
            seen[c] = (name, reason)
    pos_unique = list(seen.items())
    print(f"  unique positives after canonical dedup: {len(pos_unique)}"); sys.stdout.flush()

    # ── 4. Build negatives (safe-ish) ───────────────────────────────────
    neg_set = []
    for i in neg_idx:
        neg_set.append((f"h50_gt_{int(h50[i])}cm", smis_h[i], "h50_high"))
    print(f"Negatives total: {len(neg_set)}"); sys.stdout.flush()

    # ── 5. Encode positives that don't already have a latent ────────────
    print("\nLoading LIMO encoder for new SMILES encoding ..."); sys.stdout.flush()
    limo_dir = find_limo_repo(base)
    vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)
    blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = LIMOVAE().to(args.device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    def encode_one(smi):
        t = tok.smiles_to_tensor(smi)
        if t is None: return None
        x = t[0].unsqueeze(0).to(args.device)
        if x.shape[0] == 1: x = torch.cat([x, x])
        with torch.no_grad():
            _, mu, _ = model.encode(x)
        return mu[:1].cpu().float().squeeze(0)

    # Build latent map for h50 positives (already have)
    h50_latent_map = {smis_h[i]: z_h[i] for i in range(len(smis_h))}

    pos_z = []
    pos_smiles = []
    pos_names = []
    pos_reasons = []
    for canon_smi, (name, reason) in pos_unique:
        if canon_smi in h50_latent_map:
            z = h50_latent_map[canon_smi]
        else:
            z = encode_one(canon_smi)
            if z is None:
                print(f"  SKIP (tokenize fail): {name}: {canon_smi}"); continue
        pos_z.append(z); pos_smiles.append(canon_smi); pos_names.append(name); pos_reasons.append(reason)
    print(f"Positives encoded: {len(pos_z)}"); sys.stdout.flush()

    # Negatives: all already have latents in h50_latent_map
    neg_z = []
    neg_smiles = []
    neg_names = []
    for name, smi, reason in neg_set:
        if smi in h50_latent_map:
            neg_z.append(h50_latent_map[smi])
            neg_smiles.append(smi)
            neg_names.append(name)
    print(f"Negatives kept: {len(neg_z)}"); sys.stdout.flush()

    # ── 6. Build dataset tensor + save ──────────────────────────────────
    z_pos = torch.stack(pos_z)
    z_neg = torch.stack(neg_z)
    z_all = torch.cat([z_pos, z_neg], dim=0)
    y = torch.cat([torch.ones(len(z_pos)), torch.zeros(len(z_neg))]).float()
    smis_all = pos_smiles + neg_smiles
    names_all = pos_names + neg_names
    reasons_all = pos_reasons + ["h50_high"] * len(neg_z)

    out_blob = {
        "z_mu": z_all,
        "hazard_target": y,
        "smiles": smis_all,
        "names": names_all,
        "reasons": reasons_all,
        "meta": {
            "n_positive": int(len(z_pos)),
            "n_negative": int(len(z_neg)),
            "h50_threshold_pos_cm": args.h50_threshold_pos,
            "h50_threshold_neg_cm": args.h50_threshold_neg,
            "ckpt": args.ckpt,
        },
    }
    out_path = Path(args.out); out_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(out_blob, out_path)
    print(f"\nSaved -> {out_path}"); sys.stdout.flush()
    print(f"  positive: {len(z_pos)}, negative: {len(z_neg)}"); sys.stdout.flush()
    print(f"  imbalance: pos/total = {len(z_pos) / len(y):.2f}"); sys.stdout.flush()


if __name__ == "__main__":
    main()
