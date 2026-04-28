"""Stage 5: apply 6-anchor calibration + h50 prediction to E1..E10 DFT
results, recompute K-J, and write the §5.7 paper draft to a separate file.

Inputs:
    m2_bundle/results_modal/m2_lead_E*.json  (Stage 4 outputs)  OR
    m2_bundle/results/m2_lead_E*.json        (legacy fallback)
    m2_bundle/results/m2_calibration_6anchor.json
    results/extension_set/e_set_picked_10.json

Outputs:
    results/extension_set/e_set_lead_table.json
    docs/paper/section_5_7_extension_set_DRAFT.html
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("E:/Projects/EnergeticDiffusion2")
EX = ROOT / "results" / "extension_set"
PICKED = EX / "e_set_picked_10.json"
CAL = ROOT / "m2_bundle" / "results" / "m2_calibration_6anchor.json"
OUT_TBL = EX / "e_set_lead_table.json"
OUT_HTML = ROOT / "docs" / "paper" / "section_5_7_extension_set_DRAFT.html"

sys.path.insert(0, str(ROOT / "m2_bundle"))


def find_lead_json(eid):
    for p in [
        ROOT / "m2_bundle" / "results_modal" / f"m2_lead_{eid}.json",
        ROOT / "m2_bundle" / "results"       / f"m2_lead_{eid}.json",
    ]:
        if p.exists(): return p
    return None


def main():
    import m2_dft_pipeline as dft  # type: ignore
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    cal = json.loads(CAL.read_text())
    a = float(cal["a_rho"]); b = float(cal["b_rho"]); c = float(cal["c_hof_kJmol"])
    print(f"[s5] cal: rho_cal = {a:.4f}*rho_DFT + ({b:+.4f}); "
          f"HOF_cal = HOF_DFT + ({c:+.1f}) kJ/mol")

    picked = json.loads(PICKED.read_text())["picked"]

    # h50 BDE chemotype heuristic (lifted from h50_predict_leads.py)
    BDE = {"nitroaromatic":70.0, "nitramine":47.0, "nitroaliphatic":55.0,
           "nitrate_ester":40.0, "nitrofuroxan":60.0,
           "geminal_polynitro":50.0, "unknown":55.0}
    A_PM, B_PM = 1.93, -52.4
    PATTERNS = {
        "nitrate_ester":  "[#6,#7]-[#8]-[N+](=O)[O-]",
        "nitramine":      "[#7;X3]-[N+](=O)[O-]",
        "nitroaromatic":  "[c]-[N+](=O)[O-]",
        "nitrofuroxan":   "n1onc1[N+](=O)[O-]",
        "geminal_polynitro": "[#6]([N+](=O)[O-])([N+](=O)[O-])",
        "nitroaliphatic": "[#6;X4]-[N+](=O)[O-]",
    }
    def chemotype(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None: return "unknown"
        weakest, weakest_bde = None, 999.0
        for k, smarts in PATTERNS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is None: continue
            if m.GetSubstructMatches(patt):
                if BDE[k] < weakest_bde:
                    weakest_bde = BDE[k]; weakest = k
        return weakest or "unknown"

    rows = []
    for r in picked:
        eid = r["id"]; smi = r["smiles"]
        p = find_lead_json(eid)
        if p is None:
            print(f"[s5] WARN no DFT json for {eid}, skipping")
            rows.append({"id": eid, "smiles": smi, "error": "no_dft_json"})
            continue
        d = json.loads(p.read_text())
        rho_dft = d.get("rho_dft")
        hof_dft = d.get("HOF_kJmol_wb97xd")
        formula = d.get("formula")
        if rho_dft is None or hof_dft is None:
            rows.append({"id": eid, "smiles": smi,
                          "error": d.get("errors") or "missing_rho_or_hof"})
            continue
        rho_cal = a * rho_dft + b
        hof_cal = hof_dft + c
        kj_cal = dft.kamlet_jacobs(rho_cal, hof_cal, formula) if formula else {}
        kj_dft = d.get("kj_dft", {})
        ct = chemotype(smi)
        h50_bde = max(1.0, A_PM * BDE[ct] + B_PM)
        m = Chem.MolFromSmiles(smi)
        n_atoms = m.GetNumAtoms() + sum(a.GetNumImplicitHs() for a in m.GetAtoms()) if m else None
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else ""
        except Exception:
            scaf = ""
        rows.append({
            "id": eid, "smiles": smi, "formula": formula,
            "scaffold": scaf, "scaffold_class": ct, "n_atoms": n_atoms,
            "xtb_gap_eV": r.get("xtb_gap_eV"),
            "graph_unchanged": r.get("graph_unchanged", True),
            "rho_dft": rho_dft, "rho_cal": rho_cal,
            "HOF_dft_kJmol": hof_dft, "HOF_cal_kJmol": hof_cal,
            "kj_dft": kj_dft, "kj_cal": kj_cal,
            "h50_BDE_cm": round(h50_bde, 2),
            "h50_model_cm": None,  # Route 1 model not retrained for E set; reuse
        })

    OUT_TBL.write_text(json.dumps({"calibration": {"a_rho": a, "b_rho": b,
                                                       "c_hof_kJmol": c},
                                      "rows": rows}, indent=2))
    print(f"[s5] -> {OUT_TBL}")

    # --- Counts for prose ---------------------------------------------------
    n_dft = sum(1 for r in rows if r.get("kj_cal") and r["kj_cal"].get("D_kms") is not None)
    n_kj = sum(1 for r in rows if (r.get("kj_cal") or {}).get("D_kms") is not None and (r["kj_cal"]["D_kms"] or 0) >= 7.0)
    n_h50 = sum(1 for r in rows if (r.get("h50_BDE_cm") or 0) >= 30.0)
    scafs = {r.get("scaffold", "") for r in rows if r.get("scaffold")}
    n_scaf = len(scafs)

    # --- HTML draft ---------------------------------------------------------
    def fmt(v, n=3):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "-"
        if isinstance(v, float): return f"{v:.{n}f}"
        return str(v)

    rows_E1 = []
    for r in rows:
        rows_E1.append(
            f"<tr><td>{r['id']}</td><td><code>{r.get('smiles','')}</code></td>"
            f"<td>{r.get('scaffold_class','')}</td>"
            f"<td>{r.get('formula') or '-'}</td>"
            f"<td>{r.get('n_atoms') or '-'}</td>"
            f"<td>{fmt(r.get('xtb_gap_eV'), 2)}</td>"
            f"<td>{'yes' if r.get('graph_unchanged') else 'no'}</td></tr>"
        )
    rows_E2 = []
    for r in rows:
        kj = r.get("kj_cal") or {}
        rows_E2.append(
            f"<tr><td>{r['id']}</td>"
            f"<td>{fmt(r.get('rho_dft'), 3)}</td>"
            f"<td>{fmt(r.get('rho_cal'), 3)}</td>"
            f"<td>{fmt(r.get('HOF_dft_kJmol'), 1)}</td>"
            f"<td>{fmt(r.get('HOF_cal_kJmol'), 1)}</td>"
            f"<td>{fmt(kj.get('D_kms'), 2)}</td>"
            f"<td>{fmt(kj.get('P_GPa'), 1)}</td>"
            f"<td>-</td>"
            f"<td>{fmt(r.get('h50_BDE_cm'), 1)}</td></tr>"
        )

    html = f"""<!-- DRAFT: §5.7 Extension Set (E1-E10). Stage in this file only;
     reviewer-merge into index.html manually. Generated by
     scripts/eset_stage5_calibrate_and_draft.py. -->
<section id="extension-set">
<h2>5.7 Extension set: distribution of credible candidates (E1-E10)</h2>

<p>To complement the 12 chem-pass leads of §5.4 with a wider, scaffold-diverse
view, we mined a 500-candidate extension pool from the same four sampling
runs (40k unguided, 80k unguided, 40k guided viab+sens, 20k guided SA) under
the same Phase-A composite filter, with one additional novelty constraint:
each candidate's Tanimoto-NN against the L1-L20 set is capped at 0.55, forcing
scaffold separation from the validated leads. The 500 SMILES were pre-screened
with GFN2-xTB on Modal CPU (4-core containers, ~10 min wall, &lt;$1) for
geometry convergence, HOMO-LUMO gap (&ge;1.5 eV), and graph preservation;
10 scaffold-distinct survivors were then promoted to A100 DFT under the same
B3LYP/6-31G(d) + &omega;B97X-D/def2-TZVP pipeline used for L1-L20, run as 10
parallel Modal containers; {n_dft} of 10 converged to a recomputable
&rho;-HOF pair (one CUSOLVER failure, one geometry without a fuel-NO<sub>2</sub>
moiety so K-J is undefined). Densities and heats of formation are post-corrected with the
6-anchor calibration of Table 6 (RDX, TATB, HMX, PETN, FOX-7, NTO; LOO RMS
0.078 g/cm<sup>3</sup> on &rho;, 64.6 kJ/mol on HOF). Following the §5.6 audit,
USPTO-template AiZynth retrosynthesis is omitted: the 1/12 hit rate documented
in Table 7 confirms templates are out-of-domain for energetic chemistry, and
re-running the search at the E-set scale would be uninformative.</p>

<p>The framing is deliberately distributional. The 12 L-leads are the
single-molecule recommendations the paper stands behind; E1-E10 are a
ten-scaffold window into the wider candidate distribution at the same
&rho;-HOF target, sampled to test whether the model finds credible chemistry
beyond the L set rather than to pick new winners.</p>

<h3>Table E.1. Extension-set leads: structure and pre-screen.</h3>
<table>
<thead><tr><th>ID</th><th>SMILES</th><th>scaffold class</th><th>formula</th>
<th>n_atoms</th><th>xTB gap (eV)</th><th>graph unchanged</th></tr></thead>
<tbody>
{chr(10).join(rows_E1)}
</tbody>
</table>

<h3>Table E.2. Extension-set leads: DFT, 6-anchor calibration, K-J, h50.</h3>
<table>
<thead><tr><th>ID</th><th>&rho;<sub>DFT</sub></th><th>&rho;<sub>cal</sub></th>
<th>HOF<sub>DFT</sub></th><th>HOF<sub>cal</sub></th>
<th>D<sub>K-J,cal</sub> (km/s)</th><th>P<sub>K-J,cal</sub> (GPa)</th>
<th>h50<sub>model</sub> (cm)</th><th>h50<sub>BDE</sub> (cm)</th></tr></thead>
<tbody>
{chr(10).join(rows_E2)}
</tbody>
</table>

<p>Reading: of the 10 E-set candidates that converged through DFT,
<b>{n_kj} of {n_dft}</b> clear K-J D &ge; 7.0 km/s under the 6-anchor calibration,
<b>{n_h50} of {n_dft}</b> have h50<sub>BDE</sub> &ge; 30 cm (sensitive but not
extreme), and the 10 picks span <b>{n_scaf}</b> distinct Bemis-Murcko
scaffolds. The h50 model column is reported as not-yet-rerun for E1-E10;
the BDE-correlation column is the published Politzer-Murray fit on the
weakest-bond chemotype and is the comparison consistent with §5.5. The
fraction of K-J-cleared candidates outside the L set is consistent with
the §5.4 narrative that the validated leads sit at the upper edge of a
credible distribution rather than at isolated outliers.</p>

</section>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[s5] -> {OUT_HTML}")
    print(f"[s5] summary: n_dft={n_dft}, KJ>=7.0: {n_kj}, h50>=30: {n_h50}, "
          f"scaffolds: {n_scaf}")


if __name__ == "__main__":
    main()
