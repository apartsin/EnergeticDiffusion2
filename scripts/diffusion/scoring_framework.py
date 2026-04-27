"""Scoring framework: data-driven SMARTS + descriptor + composite rules.

Loads YAML rule files (see rules/alerts.yaml) and provides a single
`screen(smi) -> dict` entry-point with the same shape as the hand-coded
`chem_redflags.screen()` for backward compatibility.

Three layers per the migration design:
  1. Hard rejects: descriptor gates + reject_smarts (RDKit FilterCatalog)
  2. Soft demerits: smarts_alerts + feature_alerts → summed → reject_threshold
  3. Sensitivity floors: high-risk-motif tags for downstream consumers

The rest of the pipeline (rerank_v2 composite, multi-head guidance) keeps
its existing logic; only the alert/screen layer migrates.
"""
from __future__ import annotations
import math
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import FilterCatalog


@lru_cache(maxsize=2)
def _load_rules(yaml_path: str) -> dict:
    return yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _build_filter_catalog(yaml_path: str) -> FilterCatalog.FilterCatalog:
    """Build a FilterCatalog from the smarts_alerts + reject_smarts entries.
    The catalog enables fast simultaneous screening; we still record the
    per-rule demerit for soft scoring."""
    rules = _load_rules(yaml_path)
    params = FilterCatalog.FilterCatalogParams()
    catalog = FilterCatalog.FilterCatalog(params)
    for r in rules.get("reject_smarts", []) + rules.get("smarts_alerts", []):
        m = Chem.MolFromSmarts(r["smarts"])
        if m is None: continue
        entry = FilterCatalog.FilterCatalogEntry(r["name"], FilterCatalog.SmartsMatcher(m))
        catalog.AddEntry(entry)
    return catalog


def _features(mol) -> Dict:
    """Compute the descriptor + feature dictionary used by feature alerts."""
    n_C = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    n_N = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    n_O = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    n_H = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    n_heavy = mol.GetNumHeavyAtoms()
    mw = Descriptors.MolWt(mol)
    nNO2 = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N+](=O)[O-]")))
    n_nitramine = len(mol.GetSubstructMatches(
        Chem.MolFromSmarts("[N;!$([N+](=O)[O-])][N+](=O)[O-]")))
    n_nitrate = len(mol.GetSubstructMatches(
        Chem.MolFromSmarts("[#6][O][N+](=O)[O-]")))
    n_azide = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N-]=[N+]=[N]")))
    n_expl = nNO2 + n_nitramine + n_nitrate + n_azide
    n_nitrohydrazone = len(mol.GetSubstructMatches(
        Chem.MolFromSmarts("[#6]=[#7;!R][#7;!R][N+](=O)[O-]")))
    OB = -1600.0 * (2*n_C + n_H/2.0 - n_O) / max(mw, 1e-3)
    n_atoms_ring_N = sum(1 for a in mol.GetAtoms()
                          if a.GetSymbol() == "N" and a.IsInRing())
    frac_n_in_ring = n_atoms_ring_N / max(n_N, 1)
    has_arom = any(a.GetIsAromatic() for a in mol.GetAtoms())
    n_arom_het_rings = sum(
        1 for r in mol.GetRingInfo().AtomRings()
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() and
               mol.GetAtomWithIdx(i).GetSymbol() in ("N", "O") for i in r))
    rings = mol.GetRingInfo().AtomRings()
    min_ring_size = min((len(r) for r in rings), default=99)
    longest_n_chain = _longest_open_n_chain(mol)
    return {
        "MolWt": mw, "n_C": n_C, "n_N": n_N, "n_O": n_O, "n_H": n_H,
        "n_heavy": n_heavy, "n_nitro": nNO2, "n_nitramine": n_nitramine,
        "n_nitrate": n_nitrate, "n_azide": n_azide,
        "n_explosophore": n_expl,
        "explosophore_density": n_expl / max(n_heavy, 1),
        "n_nitrohydrazone": n_nitrohydrazone,
        "OB": OB,
        "frac_n_in_ring": frac_n_in_ring,
        "has_aromatic": has_arom,
        "n_aromatic_het_rings": n_arom_het_rings,
        "min_ring_size": min_ring_size,
        "longest_n_chain": longest_n_chain,
        "h_per_heavy": n_H / max(n_heavy, 1),
        "h_per_C": n_H / max(n_C, 1),
        "n_rings": mol.GetRingInfo().NumRings(),
        "formal_charge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        "atoms": {a.GetSymbol() for a in mol.GetAtoms()},
    }


def _longest_open_n_chain(mol) -> int:
    n_atoms = [a for a in mol.GetAtoms()
               if a.GetSymbol() == "N" and not a.IsInRing()]
    if not n_atoms: return 0
    nidx = {a.GetIdx() for a in n_atoms}
    adj = {i: set() for i in nidx}
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetIdx() in nidx and a2.GetIdx() in nidx and not b.IsInRing():
            adj[a1.GetIdx()].add(a2.GetIdx())
            adj[a2.GetIdx()].add(a1.GetIdx())
    best = 0
    def dfs(node, visited):
        nonlocal best
        visited.add(node)
        if len(visited) > best: best = len(visited)
        for nb in adj[node]:
            if nb not in visited: dfs(nb, visited)
        visited.remove(node)
    for s in nidx:
        dfs(s, set())
    return best


def _eval_when(expr: str, feats: Dict, allowed_atoms: set) -> bool:
    """Safely evaluate a 'when' predicate expression against the feature dict.
    Limited eval: only feature names + arithmetic + comparisons + booleans."""
    safe = {k: v for k, v in feats.items()}
    safe["allowed_atoms"] = allowed_atoms
    safe["atoms"] = feats.get("atoms", set())
    safe["mol"] = "OK"
    # Convert "atoms - allowed_atoms != {}" style: Python set diff is fine
    try:
        return bool(eval(expr, {"__builtins__": {}}, safe))
    except Exception:
        return False


def screen(smi: str, rules_yaml: str = "rules/alerts.yaml") -> Dict:
    """Drop-in replacement for chem_redflags.screen with same return shape."""
    out = {"smiles": smi, "mol": None, "status": "reject",
           "alerts": [], "reasons": [],
           "mw": None, "ob": None, "nitro": None, "n_heavy": None,
           "nitro_per_heavy": None, "frac_n_ring": None,
           "longest_n_chain": 0, "explosophore_density": 0.0,
           "red_flag_score": 0.0, "demerits": 0, "triage": "ok"}
    rules = _load_rules(rules_yaml)
    allowed = set(rules.get("allowed_atoms", ["C", "H", "N", "O", "F"]))

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        out["reasons"].append("rdkit_invalid"); return out
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        out["reasons"].append("sanitize_failed"); return out
    out["mol"] = mol

    feats = _features(mol)
    out.update(mw=feats["MolWt"], ob=feats["OB"], nitro=feats["n_nitro"],
               n_heavy=feats["n_heavy"],
               nitro_per_heavy=feats["n_nitro"] / max(feats["n_heavy"], 1),
               frac_n_ring=feats["frac_n_in_ring"],
               longest_n_chain=feats["longest_n_chain"],
               explosophore_density=feats["explosophore_density"],
               n_C=feats["n_C"], n_N=feats["n_N"], n_O=feats["n_O"],
               n_H=feats["n_H"],
               has_aromatic=feats["has_aromatic"],
               n_arom_het_rings=feats["n_aromatic_het_rings"],
               n_nitrohydrazone=feats["n_nitrohydrazone"],
               h_per_heavy=feats["h_per_heavy"])

    # 1. Descriptor-based hard rejects
    if feats["formal_charge"] != 0:
        out["reasons"].append(f"net_charge:{feats['formal_charge']}"); return out
    bad = feats["atoms"] - allowed
    if bad:
        out["reasons"].append(f"disallowed_atoms:{','.join(sorted(bad))}"); return out
    for r in rules.get("descriptor_rejects", []):
        when = r["when"]
        # Skip the rdkit-invalid / charge / atoms cases (already handled)
        if "mol is None" in when or "formal_charge" in when or "allowed_atoms" in when:
            continue
        if _eval_when(when, feats, allowed):
            out["reasons"].append(f"{r['name']}"); return out

    # 2. SMARTS rejects + alerts (via FilterCatalog for speed)
    catalog = _build_filter_catalog(rules_yaml)
    matches = catalog.GetMatches(mol)
    name_to_demerit = {}
    for r in rules.get("smarts_alerts", []):
        name_to_demerit[r["name"]] = r["demerit"]
    reject_names = {r["name"] for r in rules.get("reject_smarts", [])}

    demerits = 0
    for m in matches:
        n = m.GetDescription()
        if n in reject_names:
            out["reasons"].append(f"alert:{n}"); return out
        d = name_to_demerit.get(n, 0)
        sev = "strong" if d >= 60 else ("weak" if d >= 30 else "mild")
        out["alerts"].append((n, sev))
        demerits += d

    # 3. Feature alerts
    for r in rules.get("feature_alerts", []):
        if _eval_when(r["when"], feats, allowed):
            sev = "strong" if r["demerit"] >= 60 else (
                  "weak" if r["demerit"] >= 30 else "mild")
            out["alerts"].append((r["name"], sev))
            demerits += r["demerit"]

    # 4. Reject if total demerits exceed threshold
    threshold = rules.get("reject_threshold", 150)
    out["demerits"] = demerits
    if demerits >= threshold:
        out["reasons"].append(f"demerits>={threshold}({demerits})")
        return out

    # 5. Map demerits → red_flag_score (negative; back-compat with old code)
    norm = rules.get("soft_normalisation", 100)
    out["red_flag_score"] = -demerits / norm

    # 6. Triage tag
    if demerits >= threshold * 0.7:
        out["triage"] = "high_risk"
    elif demerits >= 60:
        out["triage"] = "medium_risk"

    out["status"] = "ok"
    return out


def sensitivity_floor(scr: dict, rules_yaml: str = "rules/alerts.yaml") -> float:
    """Return the floor on the sensitivity proxy implied by the active alerts."""
    rules = _load_rules(rules_yaml)
    floors = rules.get("sensitivity_floors", {"high": 0.5, "medium": 0.3, "very_high": 0.65})
    very_high = {"n_chain_len_4plus", "explosophore_density_very_high"}
    high = {"nitrohydrazone", "open_chain_NNN", "diazo_chain_with_nitro",
            "nitroalkene_hydrazone", "multi_nitrohydrazone", "nitrohydrazone_1",
            "small_unsaturated_polynitro_no_arom"}
    medium = {"open_chain_NN_NO2", "nitroalkene_open", "n_chain_len_3",
              "explosophore_density_high"}
    names = {n for n, _ in scr.get("alerts", [])}
    if names & very_high: return floors.get("very_high", 0.65)
    if names & high:      return floors.get("high",      0.50)
    if names & medium:    return floors.get("medium",    0.30)
    return 0.0


if __name__ == "__main__":
    test = [
        ("RDX",         "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]"),
        ("HMX",         "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]"),
        ("FOX-7",       "NC(=C([N+](=O)[O-])[N+](=O)[O-])N"),
        ("our_lead_2",  "O=[N+]([O-])NC(=C([N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"),
        ("dinitromethane", "O=C([N+](=O)[O-])[N+](=O)[O-]"),
        ("polyazene",   "O=[N+]([O-])N=NN=NN[N+](=O)[O-]"),
    ]
    for n, s in test:
        r = screen(s)
        flag = "OK    " if r["status"] == "ok" else "REJECT"
        print(f"{flag}  {n:<22} mw={(r['mw'] or 0):.0f}  demerits={r['demerits']}  "
              f"alerts={len(r['alerts'])}  reasons={r['reasons'] or '-'}")
