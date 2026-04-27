"""SMARTS red-flag filters and formula/atom-ratio gates for the v2 reranker.

Each function takes an RDKit Mol and returns either a list of (name, severity)
tuples or a numeric metric. Severity convention:
  "reject"  - drop the candidate entirely
  "strong"  - large negative score component (-1.0)
  "weak"    - small negative score component (-0.3)

Patterns are based on energetic-materials chemistry red flags and the model-
cheat patterns we observed in our pool=40k output.
"""
from __future__ import annotations
from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors


# ── Hard structural alerts (SMARTS) ──────────────────────────────────────
ALERTS = {
    "trinitromethyl_aliphatic":
        ("[CX4]([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", "strong"),
    "gemtetranitro":
        ("[CX4]([N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", "reject"),
    "polynitro_cyclopropene":
        ("[C]1[C]=[C]1", "weak"),
    "polynitro_strained_ring":
        ("[C]1([N+](=O)[O-])[C]([N+](=O)[O-])[C]1[N+](=O)[O-]", "strong"),
    "nitro_on_carbene_like":
        ("[C]=[C]=[N][N+](=O)[O-]", "strong"),
    "n_nitro_adjacent_imine":
        ("[#7]([N+](=O)[O-])[N]=[#6]", "weak"),
    "n_nitro_adjacent_hydrazone":
        ("[#7]=[#7][N+](=O)[O-]", "weak"),
    "azido_carbene":
        ("[N-]=[N+]=[N][C]=[C]", "weak"),
    "oxetane_polynitro":
        ("[O]1[C]([N+](=O)[O-])[C]([N+](=O)[O-])[C]1", "weak"),
    "peroxide":
        ("[O][O]", "reject"),
    "hydroxylamine":
        ("[N][O][H]", "weak"),
    "cyanate":
        ("[N]=[C]=[O]", "weak"),
    "diazo":
        ("[C]=[N+]=[N-]", "weak"),
    "open_chain_polynitroamide":
        ("[NX3]([N+](=O)[O-])([N+](=O)[O-])[NX3]([N+](=O)[O-])[N+](=O)[O-]", "strong"),
    # added 2026-04-26 from chemistry-expert review of v2 lead #1
    "gemdinitro_in_3ring":
        ("[C;r3]([N+](=O)[O-])([N+](=O)[O-])", "strong"),
    "gemdinitro_in_4ring":
        ("[C;r4]([N+](=O)[O-])([N+](=O)[O-])", "strong"),
    "nn_in_4ring":
        ("[#7;r4][#7;r4]", "weak"),
    "nn_in_3ring":
        ("[#7;r3][#7;r3]", "strong"),
    "n_oxide_adjacent_n_oxide":
        ("[#7]([O-])[#7;+]=O", "weak"),
    "geminal_diazo":
        ("[#6]=[N+]=[N-][#6]", "strong"),
    "small_ring_polynitro":
        ("[r3,r4][N+](=O)[O-]", "weak"),
    # added 2026-04-26 (round 2) from review of v2 top-5 (open-chain loophole)
    "nitrohydrazone":
        ("[#6]=[#7;!R][#7;!R][N+](=O)[O-]", "strong"),
    "nitroalkene_open":
        ("[#6;!R]=[#6;!R][N+](=O)[O-]", "weak"),
    "nitroalkene_hydrazone":
        ("[N+](=O)[O-][#6]=[#6][#6]=[#7][#7][N+](=O)[O-]", "strong"),
    "open_chain_NNN":
        ("[#7;!R]=[#7;!R][#7;!R]", "strong"),
    "open_chain_NN_NO2":
        ("[#7;!R][#7;!R][N+](=O)[O-]", "weak"),
    "diazo_chain_with_nitro":
        ("[#7]=[#7][#7]=[#7][N+](=O)[O-]", "strong"),
    # added 2026-04-26 (round 3): cycle-8 cumulated-cyclopropene artifact
    "cumulated_cyclopropene":
        ("[#6]1=[#6]=[#6]1", "reject"),
    "allene_in_3ring":
        ("[#6;r3]=[#6;r3]=[#6;r3]", "reject"),
}

# Compiled SMARTS (lazy)
_COMPILED = None
def _compile():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = {n: (Chem.MolFromSmarts(s), sev) for n, (s, sev) in ALERTS.items()}
    return _COMPILED


# ── Allowed-atom check ──────────────────────────────────────────────────
ALLOWED_ATOMS = {"C", "H", "N", "O", "F"}


def disallowed_atoms(mol) -> List[str]:
    """Return atom symbols present in mol but not in the allowed set."""
    seen = {a.GetSymbol() for a in mol.GetAtoms()}
    return sorted(seen - ALLOWED_ATOMS)


# ── Formula / atom-ratio gates ───────────────────────────────────────────
def oxygen_balance(mol) -> float:
    """Oxygen balance for CHNO molecules, percent (Mathieu/Klapotke convention).
    OB = 100 * (2*C + H/2 - O) / MW   ->  inverted sign
    Actually use the standard formula:
        OB% = -1600 * (2*nC + nH/2 - nO) / MW
    Negative for under-oxidized, positive for over-oxidized.
    """
    nC = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    nH = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    nO = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    mw = Descriptors.MolWt(mol)
    if mw <= 0: return 0.0
    return -1600.0 * (2*nC + nH/2.0 - nO) / mw


def nitro_count(mol) -> int:
    p = Chem.MolFromSmarts("[N+](=O)[O-]")
    return len(mol.GetSubstructMatches(p))


def n_heavy(mol) -> int:
    return mol.GetNumHeavyAtoms()


def _longest_open_n_chain(mol) -> int:
    """Length of the longest open-chain (non-ring) N-N or N=N path of nitrogens.
    Returns the number of consecutive non-ring N atoms connected by any
    N-N or N=N bond (single, double, or aromatic non-ring)."""
    # Consider only non-ring N atoms; ring nitrogens are stabilised by ring.
    n_atoms = [a for a in mol.GetAtoms()
               if a.GetSymbol() == "N" and not a.IsInRing()]
    if not n_atoms: return 0
    # Build adjacency among non-ring N atoms, only counting N-N edges
    nidx = {a.GetIdx() for a in n_atoms}
    adj = {i: set() for i in nidx}
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetIdx() in nidx and a2.GetIdx() in nidx and not b.IsInRing():
            adj[a1.GetIdx()].add(a2.GetIdx())
            adj[a2.GetIdx()].add(a1.GetIdx())
    # DFS from each N, return longest simple path length (# nodes)
    best = 0
    def dfs(node, visited):
        nonlocal best
        visited.add(node)
        if len(visited) > best: best = len(visited)
        for nb in adj[node]:
            if nb not in visited:
                dfs(nb, visited)
        visited.remove(node)
    for s in nidx:
        dfs(s, set())
    return best


def fraction_n_in_ring(mol) -> float:
    """Fraction of nitrogen atoms inside a ring vs. total N. Open-chain N-NO2
    motifs are more sensitive than ring-stabilised analogues."""
    n_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    if not n_atoms: return 0.0
    in_ring = sum(1 for a in n_atoms if a.IsInRing())
    return in_ring / len(n_atoms)


# ── Top-level filter result ─────────────────────────────────────────────
def screen(smi: str) -> dict:
    """Apply hard filters + descriptor extraction. Returns a dict with keys:
        smiles, mol, status, alerts (list of (name, sev)), reasons (list of str),
        mw, ob, nitro, n_heavy, nitro_per_heavy, frac_n_ring,
        red_flag_score (sum of penalty severities)
    Status is one of {ok, reject}.
    """
    out = {"smiles": smi, "mol": None, "status": "reject",
           "alerts": [], "reasons": [],
           "mw": None, "ob": None, "nitro": None,
           "n_heavy": None, "nitro_per_heavy": None,
           "frac_n_ring": None, "red_flag_score": 0.0}
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        out["reasons"].append("rdkit_invalid"); return out
    Chem.SanitizeMol(mol, catchErrors=True)
    out["mol"] = mol

    # Atom whitelist
    bad = disallowed_atoms(mol)
    if bad:
        out["reasons"].append(f"disallowed_atoms:{','.join(bad)}"); return out

    # Charge: only neutral
    fc = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if fc != 0:
        out["reasons"].append(f"net_charge:{fc}"); return out

    # Descriptors
    mw   = Descriptors.MolWt(mol)
    nC   = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    nN   = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    nO   = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    nH   = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    nHv  = n_heavy(mol)
    nNO2 = nitro_count(mol)
    fring = fraction_n_in_ring(mol)
    ob   = oxygen_balance(mol)

    out.update(mw=mw, ob=ob, nitro=nNO2, n_heavy=nHv,
               nitro_per_heavy=nNO2 / max(nHv, 1),
               frac_n_ring=fring, n_C=nC, n_N=nN, n_O=nO, n_H=nH)

    # --- Hard rejects ---
    if mw < 130:
        out["reasons"].append(f"mw_too_low:{mw:.0f}"); return out
    if mw > 600:
        out["reasons"].append(f"mw_too_high:{mw:.0f}"); return out
    if nNO2 / max(nHv, 1) > 0.42:
        out["reasons"].append(f"nitro_density_too_high:{nNO2}/{nHv}"); return out
    if nC <= 2 and nNO2 >= 3:
        out["reasons"].append(f"polynitro_C{nC}_chain:{nNO2}_nitro"); return out
    if ob > 25:
        out["reasons"].append(f"oxygen_balance_too_high:{ob:.0f}"); return out
    if ob < -120:
        out["reasons"].append(f"oxygen_balance_too_low:{ob:.0f}"); return out

    # SMARTS alerts
    score = 0.0
    for name, (patt, sev) in _compile().items():
        if patt is None: continue
        if mol.HasSubstructMatch(patt):
            out["alerts"].append((name, sev))
            if sev == "reject":
                out["reasons"].append(f"alert:{name}"); return out
            score -= 1.0 if sev == "strong" else 0.3

    # Soft penalties
    if ob > 15:
        out["alerts"].append(("ob_high_soft", "weak")); score -= 0.3
    if nC <= 2 and nNO2 >= 2:
        out["alerts"].append(("small_carbon_polynitro", "weak")); score -= 0.5
    if fring == 0 and nNO2 >= 3:
        out["alerts"].append(("open_chain_polynitro", "weak")); score -= 0.5

    # Low H/C ratio -> highly oxidised, sensitive (chemistry-expert flag)
    hc = nH / max(nC, 1)
    if hc < 1.0 and nNO2 >= 2:
        out["alerts"].append(("low_HC_ratio_polynitro", "weak")); score -= 0.4

    # MW < 200 with >= 2 nitro groups: tiny + multi-nitro = high sensitivity risk
    if mw < 200 and nNO2 >= 2:
        out["alerts"].append(("small_mw_multi_nitro", "weak")); score -= 0.3

    # Highly-strained ring + ANY nitro: strain + explosophore = sensitivity spike
    has_3ring = mol.GetRingInfo().AtomRingSizes() if False else None
    rsize = [len(r) for r in mol.GetRingInfo().AtomRings()]
    if rsize and min(rsize) <= 4 and nNO2 >= 2:
        out["alerts"].append(("strained_ring_multi_nitro", "weak")); score -= 0.4

    # No aromatic stabilization + >=3 nitro groups
    if Chem.GetSSSR(mol) and not any(a.GetIsAromatic() for a in mol.GetAtoms()) and nNO2 >= 3:
        out["alerts"].append(("nonaromatic_polynitro", "weak")); score -= 0.3

    # Longest contiguous open-chain N-N or N=N path (no rings, any bond order)
    longest_n_chain = _longest_open_n_chain(mol)
    out["longest_n_chain"] = longest_n_chain
    if longest_n_chain >= 4:
        out["alerts"].append((f"n_chain_len_{longest_n_chain}", "strong")); score -= 1.0
    elif longest_n_chain == 3:
        out["alerts"].append(("n_chain_len_3", "weak")); score -= 0.5

    # Explosophore density: nitro + nitramine-N + nitrate-ester + azide
    # (no double-count of nitro [O-]; N-oxide overlap with nitro skipped)
    expl = (
        nNO2 +
        len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N;!$([N+](=O)[O-])][N+](=O)[O-]"))) +
        len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6][O][N+](=O)[O-]"))) +
        len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N-]=[N+]=[N]")))
    )
    expl_density = expl / max(nHv, 1)
    out["explosophore_density"] = expl_density
    if expl_density > 0.45:
        out["alerts"].append(("explosophore_density_very_high", "strong")); score -= 1.0
    elif expl_density > 0.35:
        out["alerts"].append(("explosophore_density_high", "weak")); score -= 0.4

    # Very low hydrogen count with small MW
    h_per_heavy = nH / max(nHv, 1)
    out["h_per_heavy"] = h_per_heavy
    if mw < 220 and h_per_heavy < 0.25:
        out["alerts"].append(("low_H_density", "weak")); score -= 0.4

    # Suspiciously high D-prediction context: small + multi-explosophore -> uncertain
    if mw < 200 and expl >= 3:
        out["alerts"].append(("small_multi_explosophore_OOD", "weak")); score -= 0.3

    # New (Phase A2 expert-driven): aromatic / scaffold features
    has_arom = any(a.GetIsAromatic() for a in mol.GetAtoms())
    arom_n_o_ring = mol.HasSubstructMatch(Chem.MolFromSmarts("[a;n,o]")) if has_arom else False
    n_arom_het_rings = sum(
        1 for r in mol.GetRingInfo().AtomRings()
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() and
               mol.GetAtomWithIdx(i).GetSymbol() in ("N", "O") for i in r)
    )
    out["has_aromatic"] = has_arom
    out["n_arom_het_rings"] = n_arom_het_rings

    # nitrohydrazone count (C=N-N-NO2)
    nitrohydrazone_pat = Chem.MolFromSmarts("[#6]=[#7][#7][N+](=O)[O-]")
    n_hydra = len(mol.GetSubstructMatches(nitrohydrazone_pat))
    out["n_nitrohydrazone"] = n_hydra
    if n_hydra >= 2:
        out["alerts"].append(("multi_nitrohydrazone", "strong")); score -= 1.0
    elif n_hydra == 1:
        out["alerts"].append(("nitrohydrazone_1", "weak")); score -= 0.4

    # Suspiciously high D in MW < 220 with no aromatic stabilisation
    # We don't have D here, but the rerank caller can also enforce.
    # As a structural proxy: very oxidised + no aromatic = high-risk bracket
    if mw < 220 and not has_arom and nNO2 >= 3:
        out["alerts"].append(("small_unsaturated_polynitro_no_arom", "weak")); score -= 0.4

    # Very low H per heavy atom + multi-explosophore (electron-poor)
    if h_per_heavy < 0.20 and expl >= 3:
        out["alerts"].append(("very_low_H_with_multi_expl", "weak")); score -= 0.4

    # Triage bucket: high-risk but not auto-rejected
    triage = "ok"
    if score <= -1.5:
        triage = "high_risk"
    elif score <= -0.6:
        triage = "medium_risk"

    out["status"] = "ok"
    out["red_flag_score"] = score
    out["triage"] = triage
    return out


if __name__ == "__main__":
    import sys
    test = [
        ("RDX", "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]"),
        ("HMX", "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]"),
        ("TNT", "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]"),
        ("FOX-7", "NC(=C([N+](=O)[O-])[N+](=O)[O-])N"),
        ("our_lead_2", "O=[N+]([O-])NC(=C([N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"),
        ("our_lead_5_pool40k", "O=C([N+](=O)[O-])[N+](=O)[O-]"),
        ("our_lead_1_pool40k", "O=[N+]([O-])C=N[N+](=O)[O-]"),
        ("dinitromethane", "O=C([N+](=O)[O-])[N+](=O)[O-]"),
    ]
    for name, smi in test:
        r = screen(smi)
        flag = "OK   " if r["status"] == "ok" else "REJECT"
        print(f"{flag}  {name:<25} mw={r['mw'] or 0:.0f}  OB={r['ob'] or 0:+.0f}  "
              f"nNO2={r['nitro']}  alerts={len(r['alerts'])}  "
              f"reasons={r['reasons'] or '-'}")
