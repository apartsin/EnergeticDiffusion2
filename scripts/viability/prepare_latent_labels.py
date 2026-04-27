"""Build training labels for the multi-headed latent score model.

For each row in latents_trustcond.pt:
    SMILES (cached, no decode needed)
    z_mu     (cached, encoder posterior mean)

Compute four targets:
    y_viab  =  1[chem_redflags.passes(smi)] * RF_v2(smi)        - hard floor
    y_SA    =  RDKit SA score (Ertl-Schuffenhauer 2009)         - lower is better
    y_SC    =  SCScore (Coley 2018)                              - lower is better
    y_sens  =  sensitivity_proxy(redflags.screen(smi))           - higher = more sensitive
    + perf  =  the existing values_raw / cond_valid from latents

Output:
    experiments/latent_labels_v1.pt
        z_mu, smiles, y_viab, y_sa, y_sc, y_sens,
        perf_target (4-d), perf_mask (4-d bool)
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "scripts/viability")
sys.path.insert(0, "scripts/diffusion")
from chem_redflags import screen
from train_viability import featurize


def sa_score_safe(mol):
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
        import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--rf_v2",   required=True)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--batch_rf", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"Loading latents from {args.latents} ...")
    blob = torch.load(args.latents, weights_only=False, map_location="cpu")
    z = blob["z_mu"].float()
    smiles = blob.get("smiles") or blob.get("smiles_list")
    print(f"  z.shape={tuple(z.shape)}, smiles={len(smiles)}")
    if args.max_rows:
        z = z[:args.max_rows]; smiles = smiles[:args.max_rows]
        print(f"  truncated to {args.max_rows}")

    # SMILES-RF v2 (hard-neg variant)
    print(f"Loading RF_v2 from {args.rf_v2} ...")
    import joblib
    rf = joblib.load(args.rf_v2)

    n = len(smiles)
    y_viab = np.zeros(n, dtype=np.float32)
    y_sa   = np.zeros(n, dtype=np.float32)
    y_sc   = np.zeros(n, dtype=np.float32)
    y_sens = np.zeros(n, dtype=np.float32)
    pass_mask = np.zeros(n, dtype=np.bool_)
    skipped = 0

    print("Computing labels (SA + SC + sens + viab) ...")
    # Optional SCScorer (Coley)
    sc_score_fn = None
    try:
        sys.path.insert(0, "external/scscore/scscore")
        from standalone_model_numpy import SCScorer
        sc_model = SCScorer()
        sc_model.restore(weight_path="external/scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz")
        def _sc(smi):
            try:
                _, s = sc_model.get_score_from_smi(smi)
                return float(s)
            except Exception: return float("nan")
        sc_score_fn = _sc
        print("  SCScorer loaded")
    except Exception as e:
        print(f"  SCScorer unavailable, will use NaN: {e}")

    # Sensitivity proxy
    def sens_score(scr):
        if scr["status"] != "ok": return 1.0
        nitro_dens = scr["nitro_per_heavy"]
        base = max(0.0, min(1.0, (nitro_dens - 0.20) / 0.20))
        chain_n = 1.0 - scr["frac_n_ring"]
        chain_term = chain_n * max(0.0, min(1.0, (scr["nitro"] - 2) / 3))
        small_polynitro = 0.5 if (scr["mw"] < 180 and scr["nitro"] >= 3) else 0.0
        floor = 0.0
        names = {n for n,_ in scr.get("alerts", [])}
        if scr.get("longest_n_chain", 0) >= 4: floor = max(floor, 0.65)
        if scr.get("explosophore_density", 0) > 0.4: floor = max(floor, 0.45)
        if {"nitrohydrazone", "open_chain_NNN", "diazo_chain_with_nitro"} & names:
            floor = max(floor, 0.50)
        return min(1.0, max(base + 0.5*chain_term + small_polynitro, floor))

    # Stream through molecules
    rf_X_buf = []
    rf_idx_buf = []

    def flush_rf():
        if not rf_X_buf: return
        X_arr = np.stack(rf_X_buf)
        probs = rf.predict_proba(X_arr)[:, 1]
        for i, p in zip(rf_idx_buf, probs):
            if pass_mask[i]:
                y_viab[i] = float(p)
        rf_X_buf.clear(); rf_idx_buf.clear()

    t1 = time.time()
    for i, smi in enumerate(smiles):
        if i % 20000 == 0 and i > 0:
            flush_rf()
            print(f"  {i}/{n}  elapsed={time.time()-t1:.0f}s  pass_rate={pass_mask[:i].mean():.3f}")
        scr = screen(smi)
        if scr["status"] != "ok":
            pass_mask[i] = False
            y_viab[i] = 0.0   # hard floor
            y_sens[i] = 1.0
            # Still compute SA/SC for completeness
            try:
                from rdkit import Chem
                m = Chem.MolFromSmiles(smi)
                if m:
                    y_sa[i] = sa_score_safe(m)
                    if sc_score_fn: y_sc[i] = sc_score_fn(smi)
            except Exception: skipped += 1
            continue
        pass_mask[i] = True
        m = scr["mol"]
        y_sa[i]   = sa_score_safe(m)
        y_sens[i] = sens_score(scr)
        if sc_score_fn: y_sc[i] = sc_score_fn(smi)
        # RF: featurize + queue
        f = featurize(smi)
        if f is not None:
            rf_X_buf.append(f); rf_idx_buf.append(i)
            if len(rf_X_buf) >= args.batch_rf:
                flush_rf()
        else:
            skipped += 1

    flush_rf()
    print(f"  done labels in {time.time()-t1:.0f}s; skipped_featurize={skipped}")

    # Performance targets (already in latents blob)
    perf = blob.get("values_raw")
    if perf is None:
        perf = blob.get("values_norm")
    cond_valid = blob.get("cond_valid")
    if isinstance(perf, torch.Tensor):
        perf = perf[:n].float()
    if isinstance(cond_valid, torch.Tensor):
        cond_valid = cond_valid[:n]

    print("\nLabel statistics:")
    print(f"  pass_rate (redflags): {pass_mask.mean():.3f}")
    print(f"  y_viab mean (passers only): {y_viab[pass_mask].mean():.3f}")
    print(f"  y_SA   mean: {np.nanmean(y_sa):.3f}")
    print(f"  y_SC   mean: {np.nanmean(y_sc):.3f}")
    print(f"  y_sens mean: {y_sens.mean():.3f}")

    # Phase B: scaffold-aware viability target boost
    # y_viab' = y_viab * (1 + 0.2 * 1[has_aromatic_heterocycle])
    print("Computing aromatic-heterocycle boost ...")
    y_arom = np.zeros(n, dtype=np.float32)
    from rdkit import Chem
    for i, smi in enumerate(smiles):
        if not pass_mask[i]: continue
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        has_arom_het = any(
            (a.GetIsAromatic() and a.GetSymbol() in ("N", "O"))
            for a in m.GetAtoms())
        y_arom[i] = 1.0 if has_arom_het else 0.0
    y_viab_boost = y_viab * (1.0 + 0.20 * y_arom)
    print(f"  aromatic_het rate among passers: {y_arom[pass_mask].mean():.3f}")

    save = {
        "z_mu":      z[:n],
        "smiles":    smiles[:n],
        "y_viab":    torch.from_numpy(np.minimum(1.0, y_viab_boost)),  # clamp
        "y_arom":    torch.from_numpy(y_arom),
        "y_sa":      torch.from_numpy(y_sa),
        "y_sc":      torch.from_numpy(y_sc),
        "y_sens":    torch.from_numpy(y_sens),
        "redflag_pass": torch.from_numpy(pass_mask),
        "perf_target":  perf,
        "perf_mask":    cond_valid,
    }
    torch.save(save, out)
    print(f"\n-> {out}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
