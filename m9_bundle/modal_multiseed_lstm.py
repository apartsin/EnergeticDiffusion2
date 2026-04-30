"""Modal script: multi-seed SMILES-LSTM baseline (seeds 1, 2, 3).

Trains a character-level LSTM on the 326k energetic corpus with three different
random seeds and reports top-1 composite and memorisation rate per seed.
Seeds run in parallel on separate A10G instances.

Addresses the reviewer question: "is SMILES-LSTM memorisation a seed-stable
finding or a fluke of the single seed reported?"

Outputs:
    m9_bundle/results/lstm_seed1.json
    m9_bundle/results/lstm_seed2.json
    m9_bundle/results/lstm_seed3.json
    m9_bundle/results/lstm_multiseed_summary.json
"""
from __future__ import annotations

import json
import time
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
    .pip_install("numpy<2", "pandas", "selfies==2.1.1", "rdkit-pypi")
    .add_local_file(str(CORPUS_LOCAL),
                    remote_path="/data/corpus.csv")
    .add_local_file(str(SCRIPTS_DIFF / "chem_filter.py"),
                    remote_path="/workspace/chem_filter.py")
    .add_local_file(str(LABELLED_MASTER),
                    remote_path="/data/labelled_master.csv")
)

app = modal.App("dgld-lstm-multiseed", image=image)


@app.function(
    gpu="A10G",
    timeout=90 * 60,
    memory=16_384,
)
def run_lstm_seed(seed: int, n_samples: int = 10_000) -> dict:
    """Train SMILES-LSTM with given seed + sample + postprocess."""
    import os, sys, time as _time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path

    sys.path.insert(0, "/workspace")

    assert torch.cuda.is_available(), "CUDA not found"
    device = torch.device("cuda")
    print(f"[lstm s={seed}] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Load corpus ----
    import pandas as pd
    df = pd.read_csv("/data/corpus.csv", usecols=["smiles"])
    smiles = df["smiles"].dropna().astype(str).tolist()
    max_len = 120
    smiles = [s for s in smiles if 5 <= len(s) <= max_len - 2]
    print(f"[lstm s={seed}] {len(smiles)} SMILES", flush=True)

    chars = sorted(set("".join(smiles)))
    SOS, EOS, PAD = "^", "$", " "
    chars = [PAD, SOS, EOS] + [c for c in chars if c not in (PAD, SOS, EOS)]
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)

    def encode(s):
        s = SOS + s + EOS
        return [c2i.get(c, c2i[PAD]) for c in s][:max_len]
    def pad_seq(seq):
        return seq + [c2i[PAD]] * (max_len - len(seq))

    encoded = [pad_seq(encode(s)) for s in smiles]
    data = torch.tensor(encoded, dtype=torch.long)

    class CharLSTM(nn.Module):
        def __init__(self, vocab, hidden=512, layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True,
                                dropout=0.2 if layers > 1 else 0)
            self.out = nn.Linear(hidden, vocab)
        def forward(self, x, h=None):
            e = self.embed(x)
            o, h = self.lstm(e, h)
            return self.out(o), h

    model = CharLSTM(vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)

    print(f"[lstm s={seed}] Training 5 epochs ...", flush=True)
    t0 = _time.time()
    bs, n = 128, len(data)
    for ep in range(5):
        perm = torch.randperm(n)
        model.train()
        loss_sum = loss_cnt = 0
        for i in range(0, n, bs):
            ids = perm[i:i+bs]
            x = data[ids].to(device)
            logits, _ = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), x[:, 1:].reshape(-1),
                                   ignore_index=c2i[PAD])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item(); loss_cnt += 1
        sched.step()
        print(f"  ep {ep+1}/5  loss={loss_sum/loss_cnt:.4f}", flush=True)
    train_elapsed = _time.time() - t0
    print(f"[lstm s={seed}] trained in {train_elapsed:.0f}s", flush=True)

    # ---- Sampling ----
    model.eval()
    samples = []
    sos_id = c2i[SOS]; eos_id = c2i[EOS]; pad_id = c2i[PAD]
    batch_sz = 256
    with torch.no_grad():
        for _ in range(0, n_samples, batch_sz):
            cur = min(batch_sz, n_samples - len(samples))
            x = torch.full((cur, 1), sos_id, dtype=torch.long, device=device)
            done = torch.zeros(cur, dtype=torch.bool, device=device)
            tokens = [[] for _ in range(cur)]
            h = None
            for _ in range(max_len - 1):
                logits, h = model(x, h)
                logits = logits[:, -1, :] / 1.0
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                for j in range(cur):
                    if not done[j]:
                        t = nxt[j].item()
                        if t == eos_id:
                            done[j] = True
                        else:
                            tokens[j].append(i2c.get(t, ""))
                x = nxt
                if done.all(): break
            samples.extend(["".join(t) for t in tokens])
    print(f"[lstm s={seed}] sampled {len(samples)} raw SMILES", flush=True)

    # ---- Postprocess ----
    from rdkit import Chem, DataStructs, RDConfig
    from rdkit.Chem import AllChem
    import os as _os
    sys.path.append(_os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer as _sa

    from chem_filter import chem_filter

    master_df = pd.read_csv("/data/labelled_master.csv", usecols=["smiles"])
    master_fps = []
    for s in master_df["smiles"].dropna():
        m = Chem.MolFromSmiles(s)
        if m:
            master_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    print(f"[lstm s={seed}] master fps: {len(master_fps)}", flush=True)

    master_set = set(master_df["smiles"].dropna().tolist())

    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None

    seen: set = set()
    valid = []
    for s in samples:
        c = canon(s)
        if c and c not in seen:
            seen.add(c); valid.append(c)

    n_memorised = sum(1 for s in valid if s in master_set)
    n_exact_rediscovery = n_memorised
    memorisation_rate = n_exact_rediscovery / max(len(valid), 1)

    # Top-1 by N-fraction (same metric as baselines)
    def n_frac(s):
        m = Chem.MolFromSmiles(s)
        if not m: return 0.0
        h = m.GetNumHeavyAtoms()
        return sum(1 for a in m.GetAtoms() if a.GetSymbol()=="N") / max(h, 1)

    filtered = [s for s in valid if chem_filter(s, props=None)[0]]
    if filtered:
        top1 = max(filtered, key=n_frac)
        top1_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(top1), 2, 2048)
        sims = DataStructs.BulkTanimotoSimilarity(top1_fp, master_fps)
        top1_max_tani = max(sims) if sims else 0.0
    else:
        top1 = None; top1_max_tani = None

    result = {
        "seed": seed,
        "n_raw": len(samples),
        "n_valid": len(valid),
        "n_chem_pass": len(filtered),
        "n_exact_rediscovery": n_exact_rediscovery,
        "memorisation_rate": round(memorisation_rate, 4),
        "top1_smiles": top1,
        "top1_n_frac": round(n_frac(top1), 4) if top1 else None,
        "top1_max_tanimoto": round(top1_max_tani, 4) if top1_max_tani else None,
        "train_elapsed_s": round(train_elapsed, 1),
    }
    print(f"[lstm s={seed}] memorisation_rate={memorisation_rate:.3f} top1_max_tani={top1_max_tani}", flush=True)
    print("=== DONE ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    seeds = [1, 2, 3]
    print(f"[local] Launching SMILES-LSTM seeds {seeds} in parallel ...", flush=True)
    t0 = time.time()

    results = list(run_lstm_seed.map(seeds))

    elapsed = time.time() - t0
    print(f"[local] All seeds done in {elapsed:.0f}s", flush=True)

    summary = {"seeds": seeds, "elapsed_s": round(elapsed, 1), "runs": results}

    for r in results:
        s = r["seed"]
        out = RESULTS_LOCAL / f"lstm_seed{s}.json"
        out.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"  seed {s}: memorisation={r['memorisation_rate']:.3f} "
              f"top1_max_tani={r.get('top1_max_tanimoto','n/a')}")

    (RESULTS_LOCAL / "lstm_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[local] Summary -> {RESULTS_LOCAL / 'lstm_multiseed_summary.json'}")
