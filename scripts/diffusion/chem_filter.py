"""Physics + chemistry filters that drop obvious false-positive candidates.

Applied AFTER 3DCNN validation but BEFORE composite ranking. Rejects
molecules whose predicted properties are physically implausible OR whose
chemistry contains known unstable / unrealistic patterns.

Designed to be conservative — only kill candidates that are clearly bad.
Borderline cases stay in the pool and get sorted by the composite score.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ── property bounds ───────────────────────────────────────────────────────
# Outside these ranges, the 3DCNN smoke prediction is almost certainly an
# extrapolation error. Bounds are deliberately wide to keep real candidates.
PROPERTY_BOUNDS = {
    "density":              (0.5,  2.5),    # g/cm³
    "heat_of_formation":    (-1200, 1500),  # kcal/mol
    "detonation_velocity":  (1.0,  12.0),   # km/s
    "detonation_pressure":  (0.5,  60.0),   # GPa
}

# ── chemistry: unstable / unrealistic substructures ──────────────────────
UNSTABLE_SMARTS = [
    # ≥ 4 consecutive N atoms (chain) — most are pathologically unstable
    ("N4_chain",   "[#7]-[#7]-[#7]-[#7]-[#7]"),
    # peroxide-like O-O bond not in known oxocarbon contexts (very unstable)
    ("peroxide",   "[OX2]-[OX2]"),
    # vicinal nitro on triple-bonded C (alkynyl nitro — typically explosive
    # to handle / SMILES decoder artefact)
    ("nitro_alkyne", "[N+](=O)([O-])-C#C"),
    # halogens — rare in real energetics; usually a smoke-model OOD prediction
    ("halogen",    "[F,Cl,Br,I]"),
    # phosphorus / sulfur — rare in CHNO energetics
    ("P_or_S",     "[P,S]"),
    # sulfur disable separate to phosphorus to make filter selective if needed
    # (kept here for clarity; same effect via "P_or_S")
]

# ── composition rules ─────────────────────────────────────────────────────
def composition_ok(mol: Chem.Mol) -> tuple[bool, str]:
    """Real CHNO energetics have N + O > 0 and reasonable atom counts."""
    n = mol.GetNumHeavyAtoms()
    if n < 4:   return False, f"too small ({n} heavy atoms)"
    if n > 60:  return False, f"too large ({n} heavy atoms)"
    elem = {a.GetSymbol() for a in mol.GetAtoms()}
    if "N" not in elem and "O" not in elem:
        return False, "no N or O — not CHNO"
    # require at least one nitrogen for energetic relevance
    n_N = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    if n_N == 0:
        return False, "no N atoms"
    return True, "ok"


def has_unstable_motif(mol: Chem.Mol) -> Optional[str]:
    for label, smarts in UNSTABLE_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None and mol.HasSubstructMatch(pat):
            return label
    return None


def oxygen_balance(mol: Chem.Mol) -> float:
    """Oxygen balance % (Kistiakowsky-Wilson convention).
    Definition for CₐHᵦNᵧOₔ: OB% = -1600 (2a + b/2 - d) / MW.
    Returns 0.0 if undefined."""
    counts = {"C": 0, "H": 0, "N": 0, "O": 0}
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        if s in counts: counts[s] += 1
        counts["H"] += a.GetTotalNumHs()
    mw = Descriptors.MolWt(mol)
    if mw <= 0: return 0.0
    return -1600 * (2*counts["C"] + counts["H"]/2 - counts["O"]) / mw


# ── main filter ──────────────────────────────────────────────────────────
def chem_filter(smi: str, props: dict | None = None,
                  obal_min: float = -250.0, obal_max: float = 60.0
                  ) -> tuple[bool, str]:
    """Returns (keep, reason).
    `props` may include any of the keys in PROPERTY_BOUNDS; their values
    will be range-checked. Pass None to skip property bounds (chemistry only).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return False, "rdkit parse failed"

    # composition
    ok, why = composition_ok(mol)
    if not ok: return False, f"comp:{why}"

    # unstable motifs
    bad = has_unstable_motif(mol)
    if bad: return False, f"motif:{bad}"

    # oxygen balance sanity
    obal = oxygen_balance(mol)
    if not (obal_min <= obal <= obal_max):
        return False, f"obal:{obal:.0f}"

    # property bounds
    if props is not None:
        for p, bounds in PROPERTY_BOUNDS.items():
            v = props.get(p)
            if v is None or (isinstance(v, float) and v != v):
                continue
            if not (bounds[0] <= v <= bounds[1]):
                return False, f"prop:{p}={v:.2f}"
    return True, "ok"


def chem_filter_batch(smiles_list: list[str], cols: dict,
                        prop_names: list[str]
                        ) -> tuple[list[int], list[str]]:
    """Returns (keep_indices, reasons_for_drop[i] for all i)."""
    keep = []
    reasons = []
    for i, smi in enumerate(smiles_list):
        props = {p: float(cols[p][i]) for p in prop_names}
        ok, why = chem_filter(smi, props)
        reasons.append(why if not ok else "")
        if ok: keep.append(i)
    return keep, reasons
