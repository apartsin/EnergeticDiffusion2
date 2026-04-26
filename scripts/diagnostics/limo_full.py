"""LIMO root-cause bundle — L5 + L10 + L13.

Runs after every LIMO retrain. Gives a fast pass/fail on:

- L5  latent-norm distribution match to χ(d) prior expected
- L10 linear-probe motif AUCs (encoder retains rare-ring info?)
- L13 self-consistency on the energetic seeds we care about

Compares ckpt against optional --baseline_ckpt for delta.

Usage:
    python scripts/diagnostics/limo_full.py \
        --ckpt experiments/limo_ft_motif_rich_v2_1_<ts>/checkpoints/best.pt \
        --baseline_ckpt experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt \
        --out docs/diag_limo_v2_1b.md
"""
from __future__ import annotations
import argparse, csv, math, sys, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

BASE = Path("E:/Projects/EnergeticDiffusion2")
sys.path.insert(0, str(BASE / "scripts/vae"))
from limo_model import (LIMOVAE, SELFIESTokenizer, load_vocab, build_limo_vocab,
                          save_vocab, LIMO_MAX_LEN, find_limo_repo)


def load_model(ckpt_path: Path, device: str) -> LIMOVAE:
    cb = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = cb.get("model_state") or cb.get("state_dict") or cb
    m = LIMOVAE()
    m.load_state_dict(state)
    m.to(device).eval()
    return m


def smiles_set(blob_path: Path, k: int):
    blob = torch.load(blob_path, weights_only=False)
    smis = blob["smiles"]
    if len(smis) > k:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(smis), k, replace=False)
        return [smis[i] for i in idx], blob.get("z_mu", None), idx
    return list(smis), blob.get("z_mu", None), np.arange(len(smis))


@torch.no_grad()
def encode_many(m: LIMOVAE, tok: SELFIESTokenizer, smiles, device, batch=256):
    out = []
    failed = 0
    for i in range(0, len(smiles), batch):
        chunk = smiles[i:i+batch]
        xs = []
        keep_idx = []
        for k, s in enumerate(chunk):
            try:
                t, _ = tok.smiles_to_tensor(s)
            except Exception:
                failed += 1; continue
            if t is None: failed += 1; continue
            xs.append(t); keep_idx.append(k)
        if not xs:
            for _ in chunk: out.append(None)
            continue
        x = torch.stack(xs).to(device)
        _, mu, _ = m.encode(x)
        for j, kept in enumerate(keep_idx):
            out.append(mu[j].cpu())
        # fill failures
        for k in range(len(chunk)):
            if k not in keep_idx:
                out.append(None)
    return out, failed


# ── L5 latent norm ────────────────────────────────────────────────────────
def L5(model, tok, smiles, device):
    zs, _ = encode_many(model, tok, smiles, device)
    norms = [z.norm().item() for z in zs if z is not None]
    arr = np.array(norms)
    expected = math.sqrt(model.latent_dim) if hasattr(model, "latent_dim") else math.sqrt(1024)
    return {
        "n":         len(arr),
        "mean":      float(arr.mean()),
        "std":       float(arr.std()),
        "p10":       float(np.percentile(arr, 10)),
        "p90":       float(np.percentile(arr, 90)),
        "expected":  expected,
        "ratio":     float(arr.mean() / expected),
        "verdict":   ("on-prior" if 0.75 <= arr.mean()/expected <= 1.25
                       else "scale-mismatch"),
    }


# ── L10 motif linear probes ───────────────────────────────────────────────
MOTIFS_FOR_PROBE = [
    ("nitro",     "[N+](=O)[O-]"),
    ("nitramine", "[NX3][N+](=O)[O-]"),
    ("furazan",   "c1nonc1"),
    ("tetrazole", "c1nnnn1"),
    ("triazole",  "c1nncn1"),
    ("azide",     "[N-]=[N+]=N"),
    ("polynitro3", "$composite"),
]

def has_motif(mol, smarts):
    if smarts == "$composite":
        pat = Chem.MolFromSmarts("[N+](=O)[O-]")
        return len(mol.GetSubstructMatches(pat)) >= 3
    p = Chem.MolFromSmarts(smarts)
    return p is not None and mol.HasSubstructMatch(p)


def L10(model, tok, smiles, device):
    """Train logistic regression z → motif on 80% of rows; report AUC on 20% test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    zs, _ = encode_many(model, tok, smiles, device)
    keep = [(s, z) for s, z in zip(smiles, zs) if z is not None]
    X = torch.stack([z for _, z in keep]).numpy()
    smis = [s for s, _ in keep]
    mols = [Chem.MolFromSmiles(s) for s in smis]
    rng = np.random.default_rng(42)
    n = len(X)
    perm = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = perm[:n_tr], perm[n_tr:]
    out = []
    for name, smarts in MOTIFS_FOR_PROBE:
        y = np.array([1.0 if (m and has_motif(m, smarts)) else 0.0 for m in mols])
        if y.sum() < 30 or y.sum() > n - 30:
            out.append({"motif": name, "auc": None, "n_pos": int(y.sum()),
                          "verdict": "too few/many positives"})
            continue
        try:
            clf = LogisticRegression(max_iter=2000, C=0.5)
            clf.fit(X[tr], y[tr])
            auc = float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        except Exception as e:
            out.append({"motif": name, "auc": None, "verdict": f"err:{e}"})
            continue
        verdict = ("strong" if auc >= 0.85 else "ok" if auc >= 0.7 else "weak")
        out.append({"motif": name, "auc": auc, "n_pos": int(y.sum()),
                      "verdict": verdict})
    return out


# ── L13 self-consistency on seeds ─────────────────────────────────────────
def tanimoto(a, b):
    ma = Chem.MolFromSmiles(a); mb = Chem.MolFromSmiles(b)
    if not ma or not mb: return 0.0
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, 2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, 2048)
    return DataStructs.TanimotoSimilarity(fa, fb)


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def L13(model, tok, seeds_csv: Path, device):
    rows = []
    with open(seeds_csv) as f:
        for r in csv.DictReader(f):
            if r.get("smiles"):
                rows.append((r.get("name", "?"), r["smiles"].strip()))
    out = []
    for name, smi in rows:
        c = canon(smi)
        if not c:
            out.append({"name": name, "tanimoto": 0.0, "exact": False,
                          "verdict": "parse-fail"})
            continue
        try:
            t, _ = tok.smiles_to_tensor(c)
        except Exception:
            out.append({"name": name, "tanimoto": 0.0, "exact": False,
                          "verdict": "encode-fail"})
            continue
        x = t.unsqueeze(0).to(device)
        with torch.no_grad():
            _, mu, _ = model.encode(x)
            logits = model.decode(mu)
        toks = logits.argmax(-1)[0].cpu().tolist()
        dec = canon(tok.indices_to_smiles(toks))
        if dec is None:
            out.append({"name": name, "tanimoto": 0.0, "exact": False,
                          "verdict": "decode-fail"})
            continue
        tan = tanimoto(c, dec)
        out.append({"name": name, "tanimoto": float(tan),
                      "exact": (c == dec),
                      "decoded": dec})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--baseline_ckpt", default=None)
    ap.add_argument("--latents_pt", default="data/training/diffusion/latents.pt",
                    help="Source for L5 + L10 sample SMILES")
    ap.add_argument("--n_smiles", type=int, default=2000)
    ap.add_argument("--seeds_csv", default="data/c2c/seeds.csv")
    ap.add_argument("--out", default="docs/diag_limo_full.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = BASE
    limo_dir = find_limo_repo(base); vc = limo_dir / "vocab_cache.json"
    alphabet = load_vocab(vc) if vc.exists() else build_limo_vocab(limo_dir / "zinc250k.smi")
    if not vc.exists(): save_vocab(alphabet, vc)
    tok = SELFIESTokenizer(alphabet, max_len=LIMO_MAX_LEN)

    print("Loading SMILES sample for L5/L10 …")
    smis, _, _ = smiles_set(base / args.latents_pt, args.n_smiles)
    print(f"  {len(smis)} SMILES sampled")

    def run(ck_path: Path, label: str):
        print(f"\n=== {label}: {ck_path.name} ===")
        m = load_model(ck_path, args.device)
        t0 = time.time()
        l5 = L5(m, tok, smis, args.device)
        print(f"  L5 done in {time.time()-t0:.0f}s — mean norm {l5['mean']:.2f} "
              f"(expected {l5['expected']:.2f}) → {l5['verdict']}")
        t0 = time.time()
        l10 = L10(m, tok, smis, args.device)
        print(f"  L10 done in {time.time()-t0:.0f}s — "
              f"motif AUCs: {[(r['motif'], r['auc']) for r in l10]}")
        t0 = time.time()
        l13 = L13(m, tok, base / args.seeds_csv, args.device)
        mean_tan = float(np.mean([r["tanimoto"] for r in l13]))
        n_exact = sum(1 for r in l13 if r.get("exact"))
        print(f"  L13 done in {time.time()-t0:.0f}s — mean Tanimoto {mean_tan:.2f}, "
              f"exact {n_exact}/{len(l13)}")
        return {"label": label, "ckpt": str(ck_path),
                  "L5": l5, "L10": l10, "L13": l13}

    results = [run(Path(args.ckpt), "current")]
    if args.baseline_ckpt:
        results.append(run(Path(args.baseline_ckpt), "baseline"))

    # ── Markdown report ──────────────────────────────────────────────────
    md = ["# LIMO root-cause bundle — L5 + L10 + L13", ""]

    # L5
    md.append("## L5 — latent-norm distribution")
    md.append("")
    md.append("| version | n | mean | p10 | p90 | expected (√d) | ratio | verdict |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        l5 = r["L5"]
        md.append(f"| {r['label']} | {l5['n']} | {l5['mean']:.2f} | "
                  f"{l5['p10']:.2f} | {l5['p90']:.2f} | {l5['expected']:.2f} | "
                  f"{l5['ratio']:.2f} | {l5['verdict']} |")
    md.append("")

    # L10
    md.append("## L10 — motif linear-probe AUC")
    md.append("")
    motif_names = [m["motif"] for m in results[0]["L10"]]
    header = "| version | " + " | ".join(motif_names) + " |"
    md.append(header)
    md.append("|" + "|".join(["---"]*(1+len(motif_names))) + "|")
    for r in results:
        cells = []
        for entry in r["L10"]:
            a = entry.get("auc")
            cells.append(f"{a:.2f}" if a is not None else "n/a")
        md.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("Verdict: ≥ 0.85 strong, 0.7–0.85 ok, < 0.7 weak.")
    md.append("")

    # L13
    md.append("## L13 — self-consistency on energetic seeds")
    md.append("")
    seeds = [r["name"] for r in results[0]["L13"]]
    header = "| seed | " + " | ".join(r["label"] for r in results) + " | Δ |"
    md.append(header)
    md.append("|" + "|".join(["---"]*(2+len(results))) + "|")
    for k, name in enumerate(seeds):
        cells = []
        vals = []
        for r in results:
            v = r["L13"][k]["tanimoto"]
            vals.append(v)
            mark = " ✓" if r["L13"][k].get("exact") else ""
            cells.append(f"{v:.2f}{mark}")
        delta = vals[0] - vals[-1] if len(vals) > 1 else 0.0
        md.append(f"| {name} | " + " | ".join(cells) + f" | {delta:+.2f} |")
    md.append("")

    # ── Summary verdicts ─────────────────────────────────────────────────
    cur = results[0]
    md.append("## Summary verdicts")
    md.append("")
    md.append(f"- **L5**: {cur['L5']['verdict']}  (mean ‖z‖ = {cur['L5']['mean']:.2f}, expected {cur['L5']['expected']:.2f})")
    aucs = [(e["motif"], e["auc"]) for e in cur["L10"] if e.get("auc") is not None]
    weak_motifs = [n for n, a in aucs if a < 0.7]
    md.append(f"- **L10**: {len(weak_motifs)} weak motif(s): {weak_motifs or '—'}")
    sc = cur["L13"]
    n_exact = sum(1 for r in sc if r.get("exact"))
    md.append(f"- **L13**: {n_exact}/{len(sc)} exact roundtrip; "
              f"mean Tanimoto {np.mean([r['tanimoto'] for r in sc]):.2f}")
    if len(results) > 1:
        base = results[1]["L13"]
        deltas = [c["tanimoto"] - b["tanimoto"]
                    for c, b in zip(cur["L13"], base)]
        md.append(f"- **L13 vs baseline**: mean ΔTanimoto = {np.mean(deltas):+.3f}; "
                  f"improved on {sum(d > 0.05 for d in deltas)}/{len(deltas)} seeds.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    sys.exit(main())
