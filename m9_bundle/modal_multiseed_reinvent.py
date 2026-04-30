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
)

app = modal.App("dgld-reinvent-multiseed", image=image)

results_vol = modal.Volume.from_name("dgld-reinvent-multiseed-results",
                                      create_if_missing=True)

def _make_toml(prior: str, seed: int, workdir: str = "/workspace/reinvent_out") -> str:
    """Build a valid REINVENT 4 TOML — mirrors the working modal_reinvent_40k.py config."""
    import textwrap
    return textwrap.dedent(f"""
        run_type = "reinforcement_learning"
        model_file = "{prior}"
        output_dir = "{workdir}"
        use_checkpoint = false
        tb_logdir = ""
        random_seed = {seed}

        [parameters]
        summary_csv_prefix = "{workdir}/summary"
        json_out_config = "{workdir}/out_config.json"

        [parameters.diversity_filter]
        type = "IdenticalMurckoScaffold"
        bucket_size = 25

        [parameters.inception]
        memory_size = 20
        sample_size = 5
        smiles = []

        [[stage]]
        chkpt_file = "{workdir}/agent.chkpt"
        termination = "simple"
        max_steps = 2000

        [stage.scoring]
        type = "custom"

        [[stage.scoring.component]]
        [stage.scoring.component.CustomAlerts]
        [[stage.scoring.component.CustomAlerts.endpoint]]
        name = "chno_only"
        weight = 1.0
        [stage.scoring.component.CustomAlerts.endpoint.params]
        smarts = ["[!#6;!#1;!#7;!#8]",
                  "[C]([N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])"]

        [[stage.scoring.component]]
        [stage.scoring.component.SAScore]
        [[stage.scoring.component.SAScore.endpoint]]
        name = "sa_score"
        weight = 1.5
        [stage.scoring.component.SAScore.endpoint.transform]
        type = "reverse_sigmoid"
        high = 4.5
        low  = 1.0
        k    = 0.4

        [[stage.scoring.component]]
        [stage.scoring.component.MatchingSubstructure]
        [[stage.scoring.component.MatchingSubstructure.endpoint]]
        name = "n_fraction_proxy"
        weight = 2.0
        [stage.scoring.component.MatchingSubstructure.endpoint.params]
        smarts = ["[nH0]", "[n]", "[N+](=O)[O-]", "[N-]=[N+]=[N-]", "[N]=[N]"]
        use_chirality = false
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

    # Find REINVENT prior
    reinvent_pkg = Path(subprocess.check_output(
        ["python", "-c", "import reinvent; print(reinvent.__file__)"], text=True
    ).strip()).parent
    prior_path = None
    for p in sorted(reinvent_pkg.rglob("*.prior")):
        prior_path = p; break
    if prior_path is None:
        for p in sorted(reinvent_pkg.rglob("*.pt")):
            if "reinvent" in p.name.lower():
                prior_path = p; break
    print(f"[reinvent s={seed}] prior: {prior_path}", flush=True)

    toml_content = _make_toml(str(prior_path), seed, str(workdir))
    toml_path = workdir / f"config_s{seed}.toml"
    toml_path.write_text(toml_content)

    log_path = workdir / f"run_s{seed}.log"
    print(f"[reinvent s={seed}] Running RL ...", flush=True)
    t0 = _time.time()
    cmd = ["reinvent", "-l", str(log_path), str(toml_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
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
    filtered = [s for s in valid if chem_filter(s, props=None)[0]]

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
