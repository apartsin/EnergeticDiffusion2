"""S1 + S4 retraining: add a 6th HAZARD head to the MultiHeadScoreModel and
fine-tune on the existing v3e checkpoint.

Strategy mirrors retrain_sens_head_h50.py:
  - Freeze the v3e trunk + 5 existing heads (viab, SA, SC, sens, perf)
  - Replace head_hazard with a fresh small MLP (sigmoid logit)
  - Fit only this head on the 196-row hazard dataset
  - Mix 1.0 hazard BCE loss + 0.25 teacher regulariser on existing heads (so
    the trunk doesn't drift)
  - Save as score_model_v3f/model.pt

Run:
    /c/Python314/python scripts/viability/train_v3f_hazard_head.py \
        --base_ckpt experiments/score_model_v3e_h50/model.pt \
        --hazard experiments/hazard_dataset.pt \
        --aux experiments/latent_labels_v3e_hardneg.pt \
        --out experiments/score_model_v3f
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True,
                    help="v3e or v3e_h50 MultiHeadScoreModel checkpoint")
    ap.add_argument("--hazard", default="experiments/hazard_dataset.pt")
    ap.add_argument("--aux", default="experiments/latent_labels_v3e_hardneg.pt",
                    help="big v3e training latent set; teacher regulariser only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_haz", type=int, default=64)
    ap.add_argument("--batch_aux", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma_max", type=float, default=2.0)
    ap.add_argument("--w_haz", type=float, default=1.0)
    ap.add_argument("--w_teacher", type=float, default=0.25)
    args = ap.parse_args()

    sys.path.insert(0, "scripts/viability")
    from train_multihead_latent import MultiHeadScoreModel

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {args.device}"); sys.stdout.flush()

    # ── 1. Load v3e/v3e_h50 base, build a 6-head student ─────────────────
    print(f"Loading base checkpoint: {args.base_ckpt}"); sys.stdout.flush()
    base_blob = torch.load(args.base_ckpt, weights_only=False, map_location="cpu")
    cfg = base_blob["config"]
    teacher = MultiHeadScoreModel(latent_dim=cfg["latent_dim"]).to(args.device)
    teacher.load_state_dict(base_blob["state_dict"])
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    student = MultiHeadScoreModel(latent_dim=cfg["latent_dim"]).to(args.device)
    student.load_state_dict(base_blob["state_dict"])

    # Freeze everything in student
    for p in student.parameters(): p.requires_grad_(False)

    # Add NEW hazard head as a sibling module
    hidden = student.input_proj.out_features    # 1024
    student.head_hazard = nn.Sequential(
        nn.Linear(hidden, 256), nn.SiLU(),
        nn.Linear(256, 1),
    ).to(args.device)
    for p in student.head_hazard.parameters(): p.requires_grad_(True)

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train/1e3:.1f}k"); sys.stdout.flush()

    # ── 2. Patch forward to also output hazard ───────────────────────────
    orig_forward = student.forward

    def forward_with_hazard(z, sigma):
        out = orig_forward(z, sigma)
        # Recompute the trunk hidden state once more for the new head
        s = student.sig_emb(sigma.view(-1, 1).float())
        h = student.input_proj(z)
        for b in student.blocks:
            h = b(h, s)
        out["hazard_logit"] = student.head_hazard(h).squeeze(-1)
        return out

    student.forward = forward_with_hazard

    # ── 3. Load hazard dataset ───────────────────────────────────────────
    print(f"Loading hazard dataset: {args.hazard}"); sys.stdout.flush()
    h_blob = torch.load(args.hazard, weights_only=False, map_location="cpu")
    z_h = h_blob["z_mu"].float()
    y_h = h_blob["hazard_target"].float()
    print(f"  rows: {len(z_h)} (pos {int(y_h.sum())} / neg {int((y_h == 0).sum())})")
    sys.stdout.flush()

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(z_h), generator=g)
    n_val = max(20, len(z_h) // 10)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    z_h = z_h.to(args.device); y_h = y_h.to(args.device)

    # Class imbalance: positive weight for BCE
    pos_weight = torch.tensor([(y_h[tr_idx] == 0).sum().float() /
                               (y_h[tr_idx] == 1).sum().float().clamp(min=1)],
                              device=args.device)
    print(f"  pos_weight (for BCE balance): {pos_weight.item():.2f}"); sys.stdout.flush()

    # ── 4. Load aux dataset for teacher regulariser ──────────────────────
    print(f"Loading aux dataset: {args.aux}"); sys.stdout.flush()
    a_blob = torch.load(args.aux, weights_only=False, map_location="cpu")
    z_a = a_blob["z_mu"].float().to(args.device)
    print(f"  rows: {len(z_a)}"); sys.stdout.flush()

    # ── 5. Train ─────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def sample_sigma(n):
        scalar = float(torch.rand(1).item() * args.sigma_max)
        return scalar, torch.full((n,), scalar, device=args.device)

    best_val = float("inf")
    history = []
    for ep in range(args.epochs):
        student.train()
        h_perm = tr_idx[torch.randperm(len(tr_idx))]
        steps = max(1, len(h_perm) // args.batch_haz)
        agg = {"haz": 0.0, "teacher": 0.0, "total": 0.0}
        for s in range(steps):
            h_ids = h_perm[s * args.batch_haz:(s + 1) * args.batch_haz]
            a_ids = torch.randint(0, len(z_a), (args.batch_aux,), device=args.device)

            sig_h_s, sig_h = sample_sigma(len(h_ids))
            sig_a_s, sig_a = sample_sigma(len(a_ids))

            z_h_in = z_h[h_ids] + torch.randn_like(z_h[h_ids]) * sig_h_s
            z_a_in = z_a[a_ids] + torch.randn_like(z_a[a_ids]) * sig_a_s

            out_h = student(z_h_in, sig_h)
            out_a = student(z_a_in, sig_a)
            with torch.no_grad():
                out_a_t = teacher(z_a_in, sig_a)

            l_h = F.binary_cross_entropy_with_logits(
                out_h["hazard_logit"], y_h[h_ids], pos_weight=pos_weight)
            # Teacher reg: keep viab/sens close to v3e on aux (catches drift)
            l_t = F.smooth_l1_loss(out_a["viab_logit"], out_a_t["viab_logit"]) \
                  + F.smooth_l1_loss(out_a["sens"], out_a_t["sens"])
            total = args.w_haz * l_h + args.w_teacher * l_t

            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0)
            opt.step()
            agg["haz"] += l_h.item(); agg["teacher"] += l_t.item()
            agg["total"] += total.item()
        for k in agg: agg[k] /= steps

        # Validation
        student.eval()
        with torch.no_grad():
            sig_v_s, sig_v = sample_sigma(len(val_idx))
            z_v_in = z_h[val_idx] + torch.randn_like(z_h[val_idx]) * sig_v_s
            out_v = student(z_v_in, sig_v)
            l_val = F.binary_cross_entropy_with_logits(
                out_v["hazard_logit"], y_h[val_idx],
                pos_weight=pos_weight).item()
            preds = (out_v["hazard_logit"] > 0).float()
            acc = (preds == y_h[val_idx]).float().mean().item()
            # AUROC
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_h[val_idx].cpu().numpy(),
                                    out_v["hazard_logit"].cpu().numpy())
            except Exception:
                auc = float("nan")

        sched.step()
        marker = ""
        if l_val < best_val:
            best_val = l_val
            ckpt_cfg = dict(cfg); ckpt_cfg["has_hazard_head"] = True
            torch.save({"state_dict": student.state_dict(),
                         "config": ckpt_cfg,
                         "base_ckpt": args.base_ckpt,
                         "hazard_dataset": args.hazard},
                        out_dir / "model.pt")
            marker = " *"
        history.append({"ep": ep + 1, **agg, "val_haz": l_val, "val_acc": acc,
                        "val_auc": auc})
        print(f"  ep {ep+1:3d}/{args.epochs}  haz_loss={agg['haz']:.4f}  "
              f"teacher={agg['teacher']:.4f}  val_haz={l_val:.4f}  "
              f"val_acc={acc:.3f}  val_auc={auc:.3f}{marker}")
        sys.stdout.flush()

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest val (BCE): {best_val:.4f}"); sys.stdout.flush()
    print(f"-> {out_dir / 'model.pt'}"); sys.stdout.flush()


if __name__ == "__main__":
    main()
