# M8 Anchor Run: DNTF 7th Anchor (Furazan Class)

## Purpose

Extends the 6-anchor K-J calibration (RDX, TATB, HMX, PETN, FOX-7, NTO) to 7
anchors by running B3LYP/6-31G(d) DFT on DNTF
(3,4-bis(4-nitrofurazan-3-yl)furazan), a high-density furazan-class energetic
material.  This closes the paper gap that demoted E1 (4-nitro-1,2,3,5-
oxatriazole) to "provisional co-headline pending an oxatriazole-class DFT anchor."

## Prerequisites

1. Modal account configured (`modal token new` if needed).
2. `modal` Python package installed in your local environment.
3. Internet access to pull the CUDA Docker base image (first run only).

## Run

From the `m8_bundle` directory:

```bash
cd E:/Projects/EnergeticDiffusion2/m8_bundle
modal run modal_m8_oxatriazole_anchor.py
```

Expected wall-clock time on an A100: 3-5 hours (geometry opt dominates).

## Skip flag

If the DNTF SMILES fails local RDKit validation and you want to abort cleanly
without dispatching to Modal:

```bash
modal run modal_m8_oxatriazole_anchor.py --skip-dntf
```

The SMILES is validated locally before the A100 job is dispatched, so no GPU
cost is incurred on a bad structure.

## Output

`m8_bundle/results/m8_anchor_result.json` is written on success with fields:

| Field | Description |
|---|---|
| `dntf_rho_dft` | Raw DFT density (Bondi vdW, packing 0.69) g/cm3 |
| `dntf_hof_dft` | Raw DFT HOF (wB97X-D3BJ + ZPE) kJ/mol |
| `dntf_rho_cal_6anchor` | Density after 6-anchor calibration g/cm3 |
| `dntf_hof_cal_6anchor` | HOF after 6-anchor calibration kJ/mol |
| `residual_rho` | rho_cal_6anchor minus rho_exp g/cm3 |
| `residual_hof` | HOF_cal_6anchor minus HOF_exp kJ/mol |
| `new_7anchor_rho_slope` | Slope of the new 7-anchor rho calibration |
| `new_7anchor_rho_intercept` | Intercept of the new 7-anchor rho calibration |
| `new_7anchor_hof_intercept` | Additive HOF offset for the 7-anchor calibration kJ/mol |
| `new_7anchor_loo_rms_rho` | Leave-one-out RMS for rho (7 anchors) g/cm3 |
| `new_7anchor_loo_rms_hof` | Leave-one-out RMS for HOF (7 anchors) kJ/mol |

## Experimental reference values (hardcoded)

| Property | Value | Source |
|---|---|---|
| rho_exp | 1.937 g/cm3 | Crystal density |
| HOF_exp | +193.0 kJ/mol | Sinditskii et al., condensed-phase 298 K |
| D_exp | 9.25 km/s | Klapotke 2019 |

## Theory level

Same ladder as `m2_bundle/modal_dft_extension.py`:

- Geometry optimisation: B3LYP/6-31G(d), geomopt via geomeTRIC
- Hessian: analytical at B3LYP/6-31G(d)
- Single point: wB97X-D3BJ/def2-TZVP
- HOF method: atomization energy (NIST CCCBDB atomic HOF at 298 K)
- Density: Bondi vdW grid integration, packing factor 0.69
- GPU backend: gpu4pyscf on A100, CPU fallback if unavailable

## Failed-run recovery

If the job fails mid-run, a `m8_anchor_result_FAILED.json` is written locally
with the error traceback.  The most common failure modes are:

- ETKDGv3 embedding failure: add `--skip-dntf` and report.
- SCF convergence failure: check `m8_anchor_result_FAILED.json` for the
  pyscf traceback; the geometry opt may need a different initial conformer.
- Imaginary frequencies: the result JSON reports `freq_n_imag`; one small
  imaginary mode (< 50 cm-1) is usually acceptable for flexible molecules.
