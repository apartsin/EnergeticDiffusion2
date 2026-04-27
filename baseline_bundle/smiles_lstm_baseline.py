"""SMILES-LSTM no-diffusion baseline for M4.

Train a character-level LSTM on the 326k energetic corpus, sample 10k SMILES,
save for postprocessing alongside our DGLD generations.

Architecture: 2-layer LSTM, 128 hidden, char-level. Train ~5 min on RTX_4090.

This is the "no-diffusion baseline" reviewer R3.3 / R2.6 asked for: same training
data, similar parameter count, but no diffusion + no classifier guidance. Purely
distributional.

Output:
    results/smiles_lstm_train.log
    results/smiles_lstm_samples.txt    (10k SMILES from trained LSTM)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="Training SMILES (one per line, csv with smiles col, or txt)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.results_dir); out_dir.mkdir(exist_ok=True, parents=True)

    # Load corpus
    print(f"[train] Loading corpus from {args.corpus}"); sys.stdout.flush()
    corpus_path = Path(args.corpus)
    smiles = []
    if corpus_path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(args.corpus, usecols=["smiles"])
        smiles = df["smiles"].dropna().astype(str).tolist()
    else:
        smiles = [s.strip() for s in corpus_path.read_text().splitlines() if s.strip()]
    smiles = [s for s in smiles if 5 <= len(s) <= args.max_len - 2]
    print(f"[train] {len(smiles)} valid-length SMILES")

    # Build character vocabulary
    chars = sorted(set("".join(smiles)))
    SOS, EOS, PAD = "^", "$", " "
    chars = [PAD, SOS, EOS] + [c for c in chars if c not in (PAD, SOS, EOS)]
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    print(f"[train] vocab size: {vocab_size}"); sys.stdout.flush()

    def encode(s):
        s = SOS + s + EOS
        return [c2i.get(c, c2i[PAD]) for c in s][:args.max_len]

    def pad(seq):
        return seq + [c2i[PAD]] * (args.max_len - len(seq))

    # Encode all
    print("[train] Encoding ..."); sys.stdout.flush()
    encoded = [pad(encode(s)) for s in smiles]
    data = torch.tensor(encoded, dtype=torch.long)  # (N, L)
    print(f"[train] data shape: {data.shape}"); sys.stdout.flush()

    # Model
    class CharLSTM(nn.Module):
        def __init__(self, vocab, hidden, layers):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=0.2 if layers > 1 else 0)
            self.out = nn.Linear(hidden, vocab)

        def forward(self, x, h=None):
            e = self.embed(x)
            o, h = self.lstm(e, h)
            return self.out(o), h

    model = CharLSTM(vocab_size, args.hidden, args.layers).to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M"); sys.stdout.flush()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Training
    print(f"[train] Training {args.epochs} epochs ..."); sys.stdout.flush()
    n = len(data)
    total_steps = args.epochs * (n // args.batch)
    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        model.train()
        loss_sum = 0; loss_n = 0
        t_start = time.time()
        for i in range(0, n, args.batch):
            ids = perm[i:i + args.batch]
            x = data[ids].to(device)
            tgt = x[:, 1:]
            inp = x[:, :-1]
            logits, _ = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1),
                                    ignore_index=c2i[PAD])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item(); loss_n += 1
            step += 1
            if step % 200 == 0:
                print(f"  {step}/{total_steps} loss={loss.item():.4f} epoch={ep+1}")
                sys.stdout.flush()
        sched.step()
        avg = loss_sum / max(loss_n, 1)
        print(f"[train] epoch {ep+1}/{args.epochs}: avg_loss={avg:.4f} elapsed={time.time()-t_start:.0f}s")
        sys.stdout.flush()

    # Sampling
    print(f"\n[train] Sampling {args.n_samples} SMILES at temperature={args.temperature}")
    sys.stdout.flush()
    model.eval()
    samples = []
    sos_id = c2i[SOS]
    eos_id = c2i[EOS]
    pad_id = c2i[PAD]
    bs = 256
    with torch.no_grad():
        for batch_idx in range(0, args.n_samples, bs):
            cur_bs = min(bs, args.n_samples - batch_idx)
            x = torch.full((cur_bs, 1), sos_id, dtype=torch.long, device=device)
            done = torch.zeros(cur_bs, dtype=torch.bool, device=device)
            tokens = [[] for _ in range(cur_bs)]
            h = None
            for _ in range(args.max_len - 1):
                logits, h = model(x, h)
                logits = logits[:, -1, :] / max(args.temperature, 1e-3)
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                for j in range(cur_bs):
                    if not done[j]:
                        tok = nxt[j].item()
                        if tok == eos_id:
                            done[j] = True
                        else:
                            tokens[j].append(tok)
                x = nxt
                if done.all(): break
            for j in range(cur_bs):
                samples.append("".join(i2c[t] for t in tokens[j] if t != pad_id))

    print(f"[train] Sampled {len(samples)}; {len([s for s in samples if s])} non-empty")
    out_path = out_dir / "smiles_lstm_samples.txt"
    out_path.write_text("\n".join(samples), encoding="utf-8")
    print(f"[train] -> {out_path}")
    print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
