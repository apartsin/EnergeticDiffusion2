"""
Independent property predictor for validation: wraps the EMDP/Uni-Mol v1
multi-task regressor (the "3DCNN" in our pipeline nomenclature).

Predicts 8 targets per SMILES:
    density, DetoD (velocity), DetoP (pressure), DetoQ (heat of explosion),
    DetoT (temperature), DetoV (gas volume), HOF_S (HOF solid), BDE

Of these, the 4 that match our training targets are:
    density ↔ density
    DetoD   ↔ detonation_velocity
    DetoP   ↔ detonation_pressure
    HOF_S   ↔ heat_of_formation

We use this as an *independent* predictor to validate conditional diffusion
sampling: for a molecule generated with target (ρ=1.85, D=9.2), we ask the
3DCNN "what do YOU think its ρ and D are?" and measure the agreement.

Usage:
    from unimol_validator import UniMolValidator
    v = UniMolValidator('data/raw/energetic_external/EMDP/Data/smoke_model')
    preds = v.predict(['CCO', 'O=[N+]([O-])N1CN(C(=O)C=C1)[N+](=O)[O-]'])
    # preds: dict {property: np.array(N,)}
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# The 8 columns the EMDP 3DCNN predicts, in order
PROP_ORDER = ["density", "DetoD", "DetoP", "DetoQ",
              "DetoT", "DetoV", "HOF_S", "BDE"]

# Mapping to our training target names
PROP_MAP = {
    "density":   "density",
    "DetoD":     "detonation_velocity",
    "DetoP":     "detonation_pressure",
    "HOF_S":     "heat_of_formation",
}


class UniMolValidator:
    """Wrapper around unimol_tools.MolPredict that takes SMILES in and returns
    predicted properties. Handles caching, error masking, and name mapping.
    """
    def __init__(self, model_dir: str | Path,
                 remove_hs: bool = False,
                 cache_dir: str | Path | None = None):
        self.model_dir = Path(model_dir).resolve()
        if not (self.model_dir / "model_0.pth").exists():
            raise FileNotFoundError(f"No model_0.pth in {self.model_dir}")
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Import here so unimol_tools is optional
        try:
            from unimol_tools import MolPredict
        except ImportError as e:
            raise ImportError(
                f"unimol_tools not available: {e}. "
                "Install with: pip install unimol_tools"
            )
        self._predictor_class = MolPredict
        self._predictor = None   # lazy init to avoid loading model unless used

    def _lazy_init(self):
        if self._predictor is None:
            self._predictor = self._predictor_class(load_model=str(self.model_dir))

    def predict(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        """SMILES list -> dict with keys in PROP_ORDER and our training-target
        names. Entries where prediction fails are np.nan.
        """
        self._lazy_init()
        # unimol_tools takes a csv or list; easiest is pass list directly
        # via a DataFrame with "smiles" column.
        df_in = pd.DataFrame({"smiles": list(smiles)})
        # Run prediction; returns a numpy array (N, n_targets) or similar
        try:
            out = self._predictor.predict(df_in)
        except Exception as e:
            print(f"[unimol] prediction failed: {e}")
            out = None
        results = {}
        if out is None:
            for p in PROP_ORDER:
                results[p] = np.full(len(smiles), np.nan)
        else:
            arr = np.asarray(out)
            # arr shape may be (N, 8) or (N,)
            if arr.ndim == 1:
                arr = arr[:, None]
            for i, p in enumerate(PROP_ORDER):
                if i < arr.shape[1]:
                    results[p] = arr[:, i]
                else:
                    results[p] = np.full(len(smiles), np.nan)
        # also under our training-target names
        for emdp_name, our_name in PROP_MAP.items():
            results[our_name] = results[emdp_name]
        return results


def validate_generation(
    generated_smiles: list[str],
    target_values_raw: dict[str, float],      # {property: target_value_raw}
    validator: UniMolValidator,
) -> dict:
    """Given N generated SMILES and target values, run the validator and
    return per-property stats.
    """
    preds = validator.predict(generated_smiles)
    out = {}
    for our_name, target in target_values_raw.items():
        if our_name not in preds:
            continue
        p = preds[our_name]
        mask = ~np.isnan(p)
        if mask.sum() == 0:
            out[our_name] = {"n_valid": 0}
            continue
        diffs = p[mask] - target
        out[our_name] = {
            "target":       float(target),
            "n_valid":      int(mask.sum()),
            "n_total":      int(len(p)),
            "mean_pred":    float(p[mask].mean()),
            "median_pred":  float(np.median(p[mask])),
            "mae":          float(np.abs(diffs).mean()),
            "rmse":         float(np.sqrt((diffs**2).mean())),
            "bias":         float(diffs.mean()),
            "rel_mae_pct":  100 * float(np.abs(diffs).mean() / max(abs(target), 1e-6)),
            "within_10_pct": 100 * float(
                (np.abs(diffs) / max(abs(target), 1e-6) < 0.10).mean()),
        }
    return out


if __name__ == "__main__":
    # Quick smoke test
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",
                    default="data/raw/energetic_external/EMDP/Data/smoke_model")
    ap.add_argument("--smi", nargs="+",
                    default=["O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1",  # RDX
                              "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",   # TNT
                              "CCO"])   # ethanol (negative control)
    ap.add_argument("--base", default="E:/Projects/EnergeticDiffusion2")
    args = ap.parse_args()

    v = UniMolValidator(Path(args.base) / args.model_dir)
    preds = v.predict(args.smi)
    print("\nSMILES → predicted properties:\n")
    for i, smi in enumerate(args.smi):
        print(f"  {smi[:60]}")
        for p in PROP_ORDER:
            vals = preds[p]
            print(f"    {p:12s}  {vals[i]:.3f}")
        print()
