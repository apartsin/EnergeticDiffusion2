"""Modal script: multi-seed REINVENT 4 baseline (seeds 1 and 2).

Seed 42 was already run (reinvent_bundle/results/). This script runs
seeds 1 and 2 to give 3-seed variance on top-1 N-fraction and novelty,
completing the seed-variance row for Table 5 (§5.5.4).

Outputs:
    m9_bundle/results/reinvent_seed1.json
    m9_bundle/results/reinvent_seed2.json
    m9_bundle/results/reinvent_multiseed_summary.json
"""
from __future__ import annotations

import json
import time
import textwrap
from pathlib import Path

import modal

HERE          = Path(__file__).parent.resolve()
PROJECT_ROOT  = HERE.parent
RESULTS_LOCAL = HERE / "results"
RESULTS_LOCAL.mkdir(exist_ok=True)

CORPUS_LOCAL      = PROJECT_ROOT / "baseline_bundle" / "corpus.csv"
SCRIPTS_DIFF      = PROJECT_ROOT / "scripts" / "diffusion"
LABELLED_MASTER   = PROJECT_ROOT / "m4_bundle" / "labelled_master.csv"
REINVENT_PRIOR    = (PROJECT_ROOT / "data" / "raw" / "energetic_external" /
                     "EMDP" / "De novo" / "reinvent.prior")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "libxrender1", "libxext6")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "rdkit==2024.3.5", "networkx", "tomli", "numpy", "pandas",
    )
    .run_commands(
        "pip install git+https://github.com/MolecularAI/REINVENT4.git",
    )
    .add_local_file(str(SCRIPTS_DIFF / "chem_filter.py"),
                    remote_path="/scripts_diff/chem_filter.py")
    .add_local_file(str(CORPUS_LOCAL),
                    remote_path="/data/corpus.csv")
    .add_local_file(str(LABELLED_MASTER),
                    remote_path="/data/labelled_master.csv")
    .add_local_file(str(REINVENT_PRIOR),
                    remote_path="/data/reinvent.prior")
)

app = modal.App("dgld-reinvent-multiseed", image=image)

results_vol = modal.Volume.from_name("dgld-reinvent-multiseed-results",
                                      create_if_missing=True)

def _make_toml(prior: str, seed: int, workdir: str = "/workspace/reinvent_out") -> str:
    """Build a valid REINVENT 4 TOML matching the current GitHub HEAD schema.

    Schema change vs. earlier releases: model/agent/checkpoint keys and
    run-control flags (use_checkpoint, random_seed) now live under [parameters],
    not at the top level.  Reference: EMDP denovo_staged_learning.toml.
    """
    return textwrap.dedent(f"""
        run_type = "staged_learning"
        device = "cuda:0"
        tb_logdir = "{workdir}/tb"
        json_out_config = "{workdir}/out_config.json"

        [parameters]
        prior_file = "{prior}"
        agent_file = "{prior}"
        summary_csv_prefix = "{workdir}/summary"
        use_checkpoint = false
        batch_size = 128
        unique_sequences = true
        randomize_smiles = true

        [learning_strategy]
        type = "dap"
        sigma = 128
        rate = 0.0001

        [diversity_filter]
        type = "IdenticalMurckoScaffold"
        bucket_size = 25
        minscore = 0.2
        minsimilarity = 0.4
        penalty_multiplier = 0.5

        [[stage]]
        chkpt_file = "{workdir}/stage.chkpt"
        termination = "simple"
        max_steps = 2000

        [stage.scoring]
        type = "geometric_mean"

        [[stage.scoring.component]]
        [stage.scoring.component.GroupCount]
        [[stage.scoring.component.GroupCount.endpoint]]
        name = "nitro_count"
        weight = 2.0
        params.smarts = "[N+](=O)[O-]"
        transform.type = "sigmoid"
        transform.high = 3.0
        transform.low = 0.0
        transform.k = 0.25

        [[stage.scoring.component]]
        [stage.scoring.component.GroupCount]
        [[stage.scoring.component.GroupCount.endpoint]]
        name = "n_heterocycle"
        weight = 1.5
        params.smarts = "[nR]"
        transform.type = "sigmoid"
        transform.high = 4.0
        transform.low = 0.0
        transform.k = 0.25

        [[stage.scoring.component]]
        [stage.scoring.component.NumHeteroAtoms]
        [[stage.scoring.component.NumHeteroAtoms.endpoint]]
        name = "heteroatom_count"
        weight = 0.6
        transform.type = "sigmoid"
        transform.high = 8.0
        transform.low = 2.0
        transform.k = 0.25

        [[stage.scoring.component]]
        [stage.scoring.component.SAScore]
        [[stage.scoring.component.SAScore.endpoint]]
        name = "sa_score"
        weight = 1.5
        transform.type = "reverse_sigmoid"
        transform.high = 4.5
        transform.low  = 1.0
        transform.k    = 0.4
    """)


@app.function(
    gpu="A10G",
    timeout=3 * 60 * 60,
    memory=20_480,
    volumes={"/results": results_vol},
)
def run_reinvent_seed(seed: int) -> dict:
    """Run REINVENT 4 RL with given seed, return summary."""
    import os, sys, subprocess, textwrap
    import time as _time
    from pathlib import Path

    sys.path.insert(0, "/scripts_diff")
    import torch
    print(f"[reinvent s={seed}] GPU: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
          flush=True)

    workdir = Path("/workspace/reinvent_out")
    workdir.mkdir(parents=True, exist_ok=True)

    # Use the bundled REINVENT prior uploaded with the image
    prior_path = Path("/data/reinvent.prior")
    if not prior_path.exists():
        # Fallback: search the installed package (should not be needed)
        reinvent_pkg = Path(subprocess.check_output(
            ["python", "-c", "import reinvent; print(reinvent.__file__)"], text=True
        ).strip()).parent
        for p in sorted(reinvent_pkg.rglob("*.prior")):
            prior_path = p; break
    print(f"[reinvent s={seed}] prior: {prior_path} (exists={prior_path.exists()})", flush=True)
    if not prior_path.exists():
        return {"seed": seed, "error": "prior not found", "rc": -1}

    toml_content = _make_toml(str(prior_path), seed, str(workdir))
    toml_path = workdir / f"config_s{seed}.toml"
    toml_path.write_text(toml_content)
    print(f"[reinvent s={seed}] TOML written to {toml_path}", flush=True)
    print(f"[reinvent s={seed}] TOML content:\n{toml_content}", flush=True)

    log_path = workdir / f"run_s{seed}.log"
    print(f"[reinvent s={seed}] Running RL ...", flush=True)
    t0 = _time.time()
    cmd = ["reinvent", "-l", str(log_path), str(toml_path)]
    # Propagate seed via PYTHONHASHSEED so REINVENT's internal RNG is deterministic
    import os as _os
    env = {**_os.environ, "PYTHONHASHSEED": str(seed)}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
    elapsed = _time.time() - t0
    print(f"[reinvent s={seed}] RL done in {elapsed:.0f}s rc={result.returncode}",
          flush=True)
    if result.returncode != 0:
        print(result.stderr[-2000:], flush=True)

    # Collect SMILES from summary CSV
    import pandas as pd
    csv_files = sorted(workdir.glob("summary*.csv"))
    all_smiles = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=["SMILES"])
            all_smiles.extend(df["SMILES"].dropna().tolist())
        except Exception:
            pass
    print(f"[reinvent s={seed}] collected {len(all_smiles)} SMILES from CSVs",
          flush=True)

    if not all_smiles:
        return {"seed": seed, "error": "no SMILES", "rc": result.returncode,
                "stderr": result.stderr[-1000:]}

    # Postprocess
    from rdkit import Chem, DataStructs, RDConfig
    from rdkit.Chem import AllChem
    import os as _os
    sys.path.append(_os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer as _sa
    from chem_filter import chem_filter

    master_df = pd.read_csv("/data/labelled_master.csv", usecols=["smiles"])
    master_set = set(master_df["smiles"].dropna().tolist())
    master_fps = []
    for s in master_df["smiles"].dropna():
        m = Chem.MolFromSmiles(s)
        if m:
            master_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))

    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None

    seen: set = set()
    valid = []
    for s in all_smiles:
        c = canon(s)
        if c and c not in seen:
            seen.add(c); valid.append(c)

    n_memorised = sum(1 for s in valid if s in master_set)
    filtered = [s for s in valid
                if Chem.MolFromSmiles(s).GetNumHeavyAtoms() >= 10
                and chem_filter(s, props=None)[0]]

    def n_frac(s):
        m = Chem.MolFromSmiles(s)
        if not m: return 0.0
        h = m.GetNumHeavyAtoms()
        return sum(1 for a in m.GetAtoms() if a.GetSymbol()=="N") / max(h, 1)

    if filtered:
        top1 = max(filtered, key=n_frac)
        top1_fp = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(top1), 2, 2048)
        sims = DataStructs.BulkTanimotoSimilarity(top1_fp, master_fps)
        top1_tani = max(sims) if sims else 0.0
        try:
            top1_sa = float(_sa.calculateScore(Chem.MolFromSmiles(top1)))
        except Exception:
            top1_sa = None
    else:
        top1 = None; top1_tani = None; top1_sa = None

    r = {
        "seed": seed,
        "n_raw": len(all_smiles),
        "n_valid": len(valid),
        "n_chem_pass": len(filtered),
        "n_memorised": n_memorised,
        "memorisation_rate": round(n_memorised / max(len(valid), 1), 4),
        "top1_smiles": top1,
        "top1_n_frac": round(n_frac(top1), 4) if top1 else None,
        "top1_max_tanimoto": round(top1_tani, 4) if top1_tani else None,
        "top1_sa": round(top1_sa, 3) if top1_sa else None,
        "elapsed_rl_s": round(elapsed, 1),
    }
    print(f"[reinvent s={seed}] n_valid={len(valid)} memorisation={r['memorisation_rate']:.3f}",
          flush=True)
    print("=== DONE ===", flush=True)
    return r


@app.local_entrypoint()
def main():
    seeds = [1, 2]
    print(f"[local] Launching REINVENT 4 seeds {seeds} in parallel ...", flush=True)
    t0 = time.time()

    results = list(run_reinvent_seed.map(seeds))

    elapsed = time.time() - t0
    print(f"[local] All seeds done in {elapsed:.0f}s", flush=True)

    summary = {"seeds": seeds, "seed42_from_prev_run": True, "elapsed_s": round(elapsed, 1),
               "runs": results}

    for r in results:
        s = r["seed"]
        out = RESULTS_LOCAL / f"reinvent_seed{s}.json"
        out.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"  seed {s}: memorisation={r.get('memorisation_rate','n/a')} "
              f"top1={r.get('top1_smiles','n/a')}")

    (RESULTS_LOCAL / "reinvent_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[local] Done. -> {RESULTS_LOCAL/'reinvent_multiseed_summary.json'}")
