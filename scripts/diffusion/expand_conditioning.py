"""
Merge 3DCNN predictions (from run_3dcnn_all.py output) into a new latents
file with expanded conditioning coverage.

Rules for building the conditioning pool:
  - Tier A+B values: always used, cond_valid = True (weight 1.0)
  - 3DCNN predictions for rows with no Tier A/B: used at tier D* (weighted)
  - Properties that 3DCNN doesn't cover (D, P, Q) for rows with only 3DCNN-ρ
    and 3DCNN-HOF: can optionally K-J-impute and tag as Tier C*

Outputs a new file `latents_expanded.pt` with fields:
    z_mu         (N, 1024)   — unchanged
    smiles       list        — unchanged
    values_raw   (N, 4)      — original Tier A+B, backfilled with 3DCNN where missing
    values_norm  (N, 4)      — standardized (re-uses original stats)
    cond_valid   (N, 4) bool — expanded: True for Tier A/B OR 3DCNN prediction
    cond_weight  (N, 4) float — 1.0 for Tier A/B, --tier_d_weight for 3DCNN, 0 elsewhere
    tiers        (N, 4) int8 — 0=A 1=B 2=C 3=D 4=missing (unchanged except D for 3DCNN backfill)
    stats        dict        — unchanged
    source_mask  (N, 4) int8 — 0=Tier A/B original, 1=3DCNN backfill, 2=K-J imputed, 4=missing

The denoiser trainer (train.py) can optionally use cond_weight to down-weight
Tier D examples when sampling subsets (currently uses only cond_valid; future
enhancement).

Usage:
    python scripts/diffusion/expand_conditioning.py \\
        --latents_in  data/training/diffusion/latents_with_3dcnn.pt \\
        --out         data/training/diffusion/latents_expanded.pt \\
        --tier_d_weight 0.7
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


BASE = Path("E:/Projects/EnergeticDiffusion2")


# Map 3DCNN 8-output indices to our 4 target property indices
# 3DCNN order:          density, DetoD, DetoP, DetoQ, DetoT, DetoV, HOF_S, BDE
# Our property order:   density, heat_of_formation, detonation_velocity, detonation_pressure
THREEDCNN_TO_OUR = {
    0: 0,    # density → density
    6: 1,    # HOF_S → heat_of_formation
    1: 2,    # DetoD → detonation_velocity
    2: 3,    # DetoP → detonation_pressure
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_in", default="data/training/diffusion/latents.pt",
                    help="Original latents file (has z_mu, smiles, values_raw)")
    ap.add_argument("--preds_3dcnn", default="data/training/diffusion/preds_3dcnn.pt",
                    help="Small 3DCNN predictions file from run_3dcnn_all.py")
    ap.add_argument("--out",        default="data/training/diffusion/latents_expanded.pt")
    ap.add_argument("--tier_d_weight", type=float, default=0.7,
                    help="Down-weight applied to 3DCNN-backfilled conditioning examples")
    ap.add_argument("--kj_impute",  action="store_true",
                    help="Also K-J-impute D/P from (ρ, HOF) where both known")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    in_path  = BASE / args.latents_in
    out_path = BASE / args.out

    preds_path = BASE / args.preds_3dcnn

    print(f"Loading base latents {in_path} …")
    blob = torch.load(in_path, weights_only=False)
    print(f"Loading 3DCNN predictions {preds_path} …")
    if not preds_path.exists():
        print(f"ERROR: {preds_path} missing. Run scripts/diffusion/run_3dcnn_all.py first.")
        return 1
    preds_blob = torch.load(preds_path, weights_only=False)
    # Splice predictions into the blob so the rest of the code works unchanged
    blob["predictions_3dcnn"]       = preds_blob["predictions"]
    blob["predictions_3dcnn_valid"] = preds_blob["valid"]

    N = blob["z_mu"].shape[0]
    prop_names = blob["property_names"]        # 4 names
    stats      = blob["stats"]                  # standardization stats from original Tier A+B

    vals_raw  = blob["values_raw"].numpy().copy()      # (N, 4)  NaN where missing
    vals_norm = blob["values_norm"].numpy().copy()     # (N, 4)
    tiers     = blob["tiers"].numpy().copy()           # (N, 4)  4=missing
    cond_vld  = blob["cond_valid"].numpy().copy()      # (N, 4)  original Tier A+B

    preds = blob["predictions_3dcnn"].numpy()          # (N, 8)
    preds_valid = blob["predictions_3dcnn_valid"].numpy()  # (N,) bool

    cond_weight = cond_vld.astype(np.float32).copy()   # 1.0 for Tier A/B
    source_mask = np.where(cond_vld, 0, 4).astype(np.int8)   # 0 = tier A/B, 4 = missing

    # ── Backfill 3DCNN smoke-model predictions ONLY where the property
    #    value is genuinely missing. Critical fix: the original "3DCNN" rows
    #    in master are DFT-computed targets (Tier-D-labelled but Tier-B-quality);
    #    do NOT overwrite them with our weaker smoke-model predictions.
    backfilled = 0
    for cnn_idx, our_idx in THREEDCNN_TO_OUR.items():
        prop = prop_names[our_idx]
        mu_p = stats[prop]["mean"]; sd_p = max(stats[prop]["std"], 1e-6)
        cnn_vals = preds[:, cnn_idx]

        # Backfill only where:
        #   (1) 3DCNN smoke-model prediction is valid
        #   (2) the existing master value is NaN (no labelled data of any tier)
        existing_is_nan = np.isnan(vals_raw[:, our_idx])
        backfill_mask = preds_valid & existing_is_nan & ~np.isnan(cnn_vals)
        # Promote any existing Tier-D rows (with non-NaN values) to be
        # "valid for conditioning" too — they're DFT-quality data we shouldn't
        # ignore. Mark them with their original (lower) weight.
        existing_tier_d = (~existing_is_nan) & ~cond_vld[:, our_idx]
        cond_vld[existing_tier_d, our_idx] = True
        cond_weight[existing_tier_d, our_idx] = args.tier_d_weight
        # Standardize the existing values too if they weren't standardized before
        for i in np.where(existing_tier_d)[0]:
            if np.isnan(vals_norm[i, our_idx]):
                vals_norm[i, our_idx] = (vals_raw[i, our_idx] - mu_p) / sd_p
        n_bf = int(backfill_mask.sum())

        vals_raw[backfill_mask,  our_idx] = cnn_vals[backfill_mask]
        vals_norm[backfill_mask, our_idx] = (cnn_vals[backfill_mask] - mu_p) / sd_p
        cond_vld[backfill_mask,  our_idx] = True
        cond_weight[backfill_mask, our_idx] = args.tier_d_weight
        tiers[backfill_mask, our_idx] = 3            # tier D
        source_mask[backfill_mask, our_idx] = 1      # 3DCNN backfill

        backfilled += n_bf
        print(f"  {prop:28s} backfilled {n_bf:>7,} rows "
              f"(Tier A+B was {int((cond_weight[:, our_idx] == 1.0).sum()):,})")

    # ── Optional K-J imputation of D, P where ρ and HOF are both present ───
    if args.kj_impute:
        print("\nK-J imputation of D/P from (ρ, HOF) …")
        from rdkit import Chem
        HOF_H2O, HOF_CO2, HOF_CO = -57798.0, -94051.0, -26416.0
        KJ_TO_CAL = 239.006

        def kj_phi(smi: str, hof_kj_mol: float):
            mol = Chem.MolFromSmiles(smi)
            if mol is None: return None, None
            mol_h = Chem.AddHs(mol)
            c = {"C": 0, "H": 0, "N": 0, "O": 0, "OTHER": 0}
            for atom in mol_h.GetAtoms():
                s = atom.GetSymbol()
                c[s if s in c else "OTHER"] += 1
            if c["OTHER"] > 0: return None, None
            a, b, g, d = c["C"], c["H"], c["N"], c["O"]
            MW = 12.011*a + 1.008*b + 14.007*g + 15.999*d
            if MW <= 0: return None, None
            n2 = g / 2.0
            h_rem, o_rem, c_rem = b, d, a
            h2o = min(h_rem/2, o_rem); o_rem -= h2o; h_rem -= 2*h2o
            co2 = min(c_rem, o_rem/2); o_rem -= 2*co2; c_rem -= co2
            co = min(c_rem, o_rem); o_rem -= co; c_rem -= co
            o2 = max(o_rem/2, 0.0)
            n_gas = n2 + h2o + co2 + co + o2
            if n_gas <= 0: return None, None
            N_ = n_gas / MW
            M = (n2*28 + h2o*18 + co2*44 + co*28 + o2*32) / n_gas
            hof_cal = hof_kj_mol * KJ_TO_CAL
            Q = (hof_cal - (h2o*HOF_H2O + co2*HOF_CO2 + co*HOF_CO)) / MW
            if Q <= 0: return None, None
            return N_ * np.sqrt(M * Q), M

        smiles = blob["smiles"]
        d_idx = prop_names.index("detonation_velocity")
        p_idx = prop_names.index("detonation_pressure")
        rho_idx = prop_names.index("density")
        hof_idx = prop_names.index("heat_of_formation")

        n_kj = 0
        for i in range(N):
            if cond_vld[i, d_idx] or cond_vld[i, p_idx]:
                continue
            if not (cond_vld[i, rho_idx] and cond_vld[i, hof_idx]):
                continue
            rho = vals_raw[i, rho_idx]
            hof = vals_raw[i, hof_idx]
            phi, _ = kj_phi(smiles[i], hof)
            if phi is None: continue
            D = 1.01 * np.sqrt(phi) * (1 + 1.3 * rho)
            P = 1.558 * rho**2 * phi
            mu_D = stats["detonation_velocity"]["mean"]; sd_D = stats["detonation_velocity"]["std"]
            mu_P = stats["detonation_pressure"]["mean"]; sd_P = stats["detonation_pressure"]["std"]
            vals_raw[i, d_idx] = float(D)
            vals_raw[i, p_idx] = float(P)
            vals_norm[i, d_idx] = (D - mu_D) / max(sd_D, 1e-6)
            vals_norm[i, p_idx] = (P - mu_P) / max(sd_P, 1e-6)
            cond_vld[i, d_idx] = True; cond_vld[i, p_idx] = True
            cond_weight[i, d_idx] = 0.5; cond_weight[i, p_idx] = 0.5
            tiers[i, d_idx] = 2; tiers[i, p_idx] = 2    # Tier C
            source_mask[i, d_idx] = 2; source_mask[i, p_idx] = 2
            n_kj += 1
            if n_kj % 10000 == 0:
                print(f"    K-J imputed {n_kj:,} rows …")
        print(f"  K-J imputed {n_kj:,} rows for D + P")

    # ── Final stats ─────────────────────────────────────────────────────────
    print("\nFinal conditioning inventory:")
    for j, p in enumerate(prop_names):
        n_ab = int((source_mask[:, j] == 0).sum())
        n_3d = int((source_mask[:, j] == 1).sum())
        n_kj = int((source_mask[:, j] == 2).sum())
        n_any = int(cond_vld[:, j].sum())
        print(f"  {p:28s}  Tier A+B: {n_ab:>7,}  3DCNN: {n_3d:>7,}  K-J: {n_kj:>7,}  total_valid: {n_any:>7,}  ({100*n_any/N:.1f}%)")

    # ── Write expanded blob ─────────────────────────────────────────────────
    out = dict(blob)
    out["values_raw"]   = torch.from_numpy(vals_raw)
    out["values_norm"]  = torch.from_numpy(vals_norm)
    out["tiers"]        = torch.from_numpy(tiers)
    out["cond_valid"]   = torch.from_numpy(cond_vld)
    out["cond_weight"]  = torch.from_numpy(cond_weight)
    out["source_mask"]  = torch.from_numpy(source_mask)
    out["expansion_meta"] = {
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier_d_weight":  args.tier_d_weight,
        "kj_impute":      args.kj_impute,
        "source_legend":  {0: "Tier A+B (original)", 1: "3DCNN backfill",
                            2: "K-J imputed", 4: "missing"},
    }
    tmp = str(out_path) + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
