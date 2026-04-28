"""M2 calibration extension: add HMX, PETN, FOX-7, NTO to the 2-anchor
(RDX, TATB) DFT calibration.

Reuses opt + Hessian + SP + atomization-HOF + Bondi-vdW rho machinery from
m2_dft_pipeline.py. Refits (a, b) for rho and c for HOF over the union of
six anchors (RDX, TATB, HMX, PETN, FOX-7, NTO) and reports leave-one-out
residuals.

Outputs (in --results dir):
    m2_anchor_HMX.json
    m2_anchor_PETN.json
    m2_anchor_FOX7.json
    m2_anchor_NTO.json
    m2_calibration_6anchor.json
    m2_summary_6anchor.json   (12 chem-pass leads recomputed under new cal)

Usage on RunPod (after gpu4pyscf + pyscf + rdkit-pypi installed and atom refs
copied to results/m2_atom_refs.json):
    python3 m2_anchors_extension.py --results results
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np

import m2_dft_pipeline as dft  # local import; same dir on the pod


# Experimental anchor values used for the 6-anchor refit. These are the
# numbers the chemist-reviewer expects to see and that the user listed in
# the brief. Cross-checked against Talawar (HMX/PETN), Bemm-Ostmark (FOX-7),
# Lee-Coburn (NTO) reviews.
ANCHOR_LIT = {
    "RDX":   {"rho_lit": 1.82, "HOF_lit_kJmol":  +70.0,
              "smiles": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1"},
    "TATB":  {"rho_lit": 1.94, "HOF_lit_kJmol": -141.0,
              "smiles": "Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]"},
    "HMX":   {"rho_lit": 1.91, "HOF_lit_kJmol":  +75.0,
              "smiles": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]"},
    "PETN":  {"rho_lit": 1.77, "HOF_lit_kJmol": -538.0,
              "smiles": "O=[N+]([O-])OCC(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]"},
    "FOX7":  {"rho_lit": 1.89, "HOF_lit_kJmol": -134.0,
              "smiles": "NC(=C([N+](=O)[O-])[N+](=O)[O-])N"},
    "NTO":   {"rho_lit": 1.93, "HOF_lit_kJmol": -129.0,
              "smiles": "O=C1NN=C(N1)[N+](=O)[O-]"},
}

# Only these need to be computed fresh on the pod; RDX and TATB are cached
# from the prior run (m2_lead_RDX.json, m2_lead_TATB.json).
NEW_ANCHORS = ["HMX", "PETN", "FOX7", "NTO"]


def load_anchor_dft(anchor_id, results_dir):
    """Return {'rho_dft', 'HOF_kJmol_wb97xd'} from a per-lead JSON."""
    # RDX/TATB cached in m2_lead_<id>.json; new anchors saved as m2_anchor_<id>.json
    candidates = [
        results_dir / f"m2_anchor_{anchor_id}.json",
        results_dir / f"m2_lead_{anchor_id}.json",
    ]
    for p in candidates:
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("rho_dft") is not None and d.get("HOF_kJmol_wb97xd") is not None:
                return {"rho_dft": float(d["rho_dft"]),
                        "HOF_kJmol_wb97xd": float(d["HOF_kJmol_wb97xd"]),
                        "formula": d.get("formula")}
    return None


def compute_anchor(anchor_id, atomic_refs, results_dir, use_gpu=True):
    """Run the full B3LYP opt + Hessian + wB97X-D SP pipeline for one anchor."""
    info = ANCHOR_LIT[anchor_id]
    out_path = results_dir / f"m2_anchor_{anchor_id}.json"
    if out_path.exists():
        cached = json.loads(out_path.read_text())
        if cached.get("HOF_kJmol_wb97xd") is not None and not cached.get("errors"):
            print(f"[ext] {anchor_id} cached, skipping"); sys.stdout.flush()
            return cached
    # Reuse the lead-runner; just hand it a synthetic lead dict.
    lead = {"id": anchor_id, "smiles": info["smiles"], "name": anchor_id}
    out = dft.run_lead(lead, atomic_refs["B3LYP_631Gss"],
                       atomic_refs["wB97XD_def2TZVP"], results_dir, use_gpu=use_gpu)
    # rename to anchor file (run_lead wrote m2_lead_<id>.json; we want a
    # separate namespace so the chem-pass lead summary is not polluted).
    src = results_dir / f"m2_lead_{anchor_id}.json"
    if src.exists() and src != out_path:
        out_path.write_text(src.read_text())
        try:
            src.unlink()
        except Exception:
            pass
    return out


def fit_calibration(anchor_dft, anchor_lit):
    """Least-squares fit rho_cal = a*rho_DFT + b on N anchors;
    constant offset c on HOF. Returns (a, b, c, residuals dict)."""
    ids = list(anchor_dft.keys())
    rho_x = np.array([anchor_dft[i]["rho_dft"] for i in ids])
    rho_y = np.array([anchor_lit[i]["rho_lit"] for i in ids])
    hof_x = np.array([anchor_dft[i]["HOF_kJmol_wb97xd"] for i in ids])
    hof_y = np.array([anchor_lit[i]["HOF_lit_kJmol"] for i in ids])
    A = np.vstack([rho_x, np.ones_like(rho_x)]).T
    a, b = np.linalg.lstsq(A, rho_y, rcond=None)[0]
    c = float(np.mean(hof_y - hof_x))
    rho_cal = a * rho_x + b
    hof_cal = hof_x + c
    rho_res = rho_cal - rho_y
    hof_res = hof_cal - hof_y
    return float(a), float(b), float(c), {
        "rho_rms_g_cm3": float(np.sqrt(np.mean(rho_res ** 2))),
        "hof_rms_kJmol": float(np.sqrt(np.mean(hof_res ** 2))),
        "per_anchor": {i: {
            "rho_DFT": float(anchor_dft[i]["rho_dft"]),
            "HOF_DFT_kJmol": float(anchor_dft[i]["HOF_kJmol_wb97xd"]),
            "rho_exp": float(anchor_lit[i]["rho_lit"]),
            "HOF_exp_kJmol": float(anchor_lit[i]["HOF_lit_kJmol"]),
            "rho_cal": float(a * anchor_dft[i]["rho_dft"] + b),
            "HOF_cal_kJmol": float(anchor_dft[i]["HOF_kJmol_wb97xd"] + c),
            "rho_residual": float(a * anchor_dft[i]["rho_dft"] + b - anchor_lit[i]["rho_lit"]),
            "HOF_residual_kJmol": float(anchor_dft[i]["HOF_kJmol_wb97xd"] + c - anchor_lit[i]["HOF_lit_kJmol"]),
        } for i in ids},
    }


def loo_calibration(anchor_dft, anchor_lit):
    """Leave-one-out: refit on N-1 anchors, predict the held-out one."""
    ids = list(anchor_dft.keys())
    rho_pred_err = []
    hof_pred_err = []
    per = {}
    for held in ids:
        fit_ids = [i for i in ids if i != held]
        sub_dft = {i: anchor_dft[i] for i in fit_ids}
        a, b, c, _ = fit_calibration(sub_dft, anchor_lit)
        rho_p = a * anchor_dft[held]["rho_dft"] + b
        hof_p = anchor_dft[held]["HOF_kJmol_wb97xd"] + c
        rho_e = rho_p - anchor_lit[held]["rho_lit"]
        hof_e = hof_p - anchor_lit[held]["HOF_lit_kJmol"]
        rho_pred_err.append(rho_e)
        hof_pred_err.append(hof_e)
        per[held] = {"rho_pred": float(rho_p), "rho_err": float(rho_e),
                     "HOF_pred_kJmol": float(hof_p), "HOF_err_kJmol": float(hof_e),
                     "fit_a": float(a), "fit_b": float(b), "fit_c": float(c)}
    return {
        "loo_rho_rms_g_cm3": float(np.sqrt(np.mean(np.array(rho_pred_err) ** 2))),
        "loo_hof_rms_kJmol": float(np.sqrt(np.mean(np.array(hof_pred_err) ** 2))),
        "per_anchor": per,
    }


def recompute_leads_under_new_cal(results_dir, a, b, c):
    """Apply the new (a, b, c) to the 12 chem-pass leads in m2_summary.json
    and write m2_summary_6anchor.json mirroring the old schema with K-J
    recomputed under the calibrated rho and HOF."""
    summary_path = results_dir / "m2_summary.json"
    if not summary_path.exists():
        print("[ext] no m2_summary.json found; skipping lead recompute")
        return None
    summary = json.loads(summary_path.read_text())
    new_summary = []
    for s in summary:
        s2 = dict(s)
        if s.get("rho_dft") is None or s.get("HOF_kJmol_wb97xd") is None:
            new_summary.append(s2); continue
        s2["rho_cal"] = float(a * s["rho_dft"] + b)
        s2["HOF_kJmol_wb97xd_cal"] = float(s["HOF_kJmol_wb97xd"] + c)
        # Need formula for K-J — pull from per-lead JSON.
        lead_path = results_dir / f"m2_lead_{s['id']}.json"
        formula = None
        if lead_path.exists():
            try:
                formula = json.loads(lead_path.read_text()).get("formula")
            except Exception:
                pass
        if formula:
            s2["kj_dft_cal"] = dft.kamlet_jacobs(s2["rho_cal"],
                                                  s2["HOF_kJmol_wb97xd_cal"], formula)
        new_summary.append(s2)
    out = results_dir / "m2_summary_6anchor.json"
    out.write_text(json.dumps(new_summary, indent=2, default=str))
    print(f"[ext] wrote {out}")
    return new_summary


def main():
    dft._start_heartbeat()
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-compute", action="store_true",
                    help="Skip DFT and just refit from cached anchor JSONs")
    args = ap.parse_args()
    use_gpu = not args.cpu

    if use_gpu and not args.skip_compute:
        try:
            from gpu4pyscf import dft as _gpu_dft  # noqa
            import torch
            assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
            print(f"[ext] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()
        except (ImportError, AssertionError) as e:
            print(f"[ext] FATAL: gpu4pyscf or CUDA unavailable ({e})", flush=True)
            sys.exit(1)

    results_dir = Path(args.results); results_dir.mkdir(parents=True, exist_ok=True)

    # Atomic references — must already be on disk (uploaded with job) or
    # computed from scratch. Reuse the cached file.
    atom_refs_path = results_dir / "m2_atom_refs.json"
    if atom_refs_path.exists():
        atom_refs = json.loads(atom_refs_path.read_text())
        print(f"[ext] loaded cached atom refs from {atom_refs_path}"); sys.stdout.flush()
    else:
        if args.skip_compute:
            print("[ext] FATAL: --skip-compute but no atom refs"); sys.exit(1)
        print(f"[ext] computing atomic reference energies"); sys.stdout.flush()
        atom_refs = {"B3LYP_631Gss": {}, "wB97XD_def2TZVP": {}}
        for sym in ["H", "C", "N", "O"]:
            atom_refs["B3LYP_631Gss"][sym] = dft.atomic_reference_energy(
                sym, "6-31g*", "b3lyp", use_gpu=use_gpu)
            atom_refs["wB97XD_def2TZVP"][sym] = dft.atomic_reference_energy(
                sym, "def2-tzvp", "wb97x-d3bj", use_gpu=use_gpu)
        atom_refs_path.write_text(json.dumps(atom_refs, indent=2))

    # Compute the 4 new anchors (RDX, TATB are cached from prior run).
    if not args.skip_compute:
        for aid in NEW_ANCHORS:
            t0 = time.time()
            print(f"\n[ext] === anchor {aid} ===", flush=True)
            try:
                compute_anchor(aid, atom_refs, results_dir, use_gpu=use_gpu)
                print(f"[ext] {aid} done in {time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                print(f"[ext] {aid} FAILED: {e}", flush=True)

    # Collect all 6 anchor DFT outputs.
    anchor_dft = {}
    for aid in ANCHOR_LIT:
        d = load_anchor_dft(aid, results_dir)
        if d is None:
            print(f"[ext] WARNING: missing DFT for {aid}, skipping from fit")
            continue
        anchor_dft[aid] = d
    print(f"[ext] anchors with DFT data: {list(anchor_dft.keys())}")
    if len(anchor_dft) < 3:
        print("[ext] FATAL: fewer than 3 anchors converged; refit not meaningful")
        sys.exit(2)

    # Refit on the union of available anchors (target: 6).
    a, b, c, fit_info = fit_calibration(anchor_dft, ANCHOR_LIT)
    loo = loo_calibration(anchor_dft, ANCHOR_LIT)

    cal = {
        "n_anchors": len(anchor_dft),
        "anchors_used": list(anchor_dft.keys()),
        "a_rho": a,
        "b_rho": b,
        "c_hof_kJmol": c,
        "fit_residual": fit_info,
        "loo_residual": loo,
        "note": (f"6-anchor calibration extending RDX/TATB with "
                 f"HMX/PETN/FOX-7/NTO. ZPE unit-bug-fixed pipeline (commit 4abe43d)."),
    }
    out_path = results_dir / "m2_calibration_6anchor.json"
    out_path.write_text(json.dumps(cal, indent=2))
    print(f"\n[ext] === FIT ===")
    print(f"[ext] rho_cal = {a:.4f} * rho_DFT + {b:+.4f}  "
          f"(fit RMS {fit_info['rho_rms_g_cm3']:.3f} g/cm3, "
          f"LOO RMS {loo['loo_rho_rms_g_cm3']:.3f} g/cm3)")
    print(f"[ext] HOF_cal = HOF_DFT + ({c:+.1f}) kJ/mol  "
          f"(fit RMS {fit_info['hof_rms_kJmol']:.1f} kJ/mol, "
          f"LOO RMS {loo['loo_hof_rms_kJmol']:.1f} kJ/mol)")
    print(f"[ext] -> {out_path}")

    # Apply to the 12 chem-pass leads + write summary.
    recompute_leads_under_new_cal(results_dir, a, b, c)

    print(f"\n[ext] === DONE ===")


if __name__ == "__main__":
    main()
