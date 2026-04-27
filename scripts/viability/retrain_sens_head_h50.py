"""2.5: Fine-tune the sensitivity head of an existing v3e MultiHeadScoreModel
on literature h50 (impact-sensitivity drop-height) data.

Strategy:
    - Load the frozen v3e checkpoint (5 heads, all trained against the heuristic
      sensitivity proxy).
    - Freeze the trunk, FiLM, and the four other heads (viab/SA/SC/perf).
    - Replace head_sens with a fresh small MLP (hidden 256) and fit only this
      head against z-scored sens_target derived from h50.
    - Mix two loss terms each minibatch:
          (a) on the h50 dataset (~307 rows): smooth-L1 on z-scored sens_target.
          (b) on the existing v3e training set (~382k+918): smooth-L1 against
              the *frozen old head's prediction* on the same noisy z (a teacher
              regularisation that prevents the new head from drifting too far
              from the heuristic where h50 has no opinion).
    - Save as experiments/score_model_v3e_h50/model.pt with the same config
      shape as v3e + extra "sens_h50_pivot/slope" metadata so the sampler can
      recover the un-z-scored sensitivity prediction at inference.

Usage:
    /c/Python314/python scripts/viability/retrain_sens_head_h50.py \
        --base_ckpt experiments/score_model_v3e/model.pt \
        --h50 experiments/sens_h50_dataset.pt \
        --aux experiments/latent_labels_v3e_hardneg.pt \
        --out experiments/score_model_v3e_h50

After training, sample with --score_model experiments/score_model_v3e_h50/model.pt
to use the h50-grounded sens head; everything else stays identical.
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True,
                    help="frozen v3e MultiHeadScoreModel checkpoint")
    ap.add_argument("--h50", default="experiments/sens_h50_dataset.pt",
                    help="output of prepare_h50_dataset.py")
    ap.add_argument("--aux", default="experiments/latent_labels_v3e_hardneg.pt",
                    help="big v3e training label set; used for teacher regularisation only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_h50", type=int, default=64)
    ap.add_argument("--batch_aux", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma_max", type=float, default=2.0)
    ap.add_argument("--w_h50", type=float, default=1.0,
                    help="weight on the literature h50 loss")
    ap.add_argument("--w_teacher", type=float, default=0.25,
                    help="weight on the v3e-teacher regulariser loss")
    args = ap.parse_args()

    sys.path.insert(0, "scripts/viability")
    from train_multihead_latent import MultiHeadScoreModel

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {args.device}")

    # ── 1. Load frozen base model (v3e) ─────────────────────────────────────
    print(f"Loading base v3e checkpoint: {args.base_ckpt}")
    base_blob = torch.load(args.base_ckpt, weights_only=False, map_location="cpu")
    cfg = base_blob["config"]
    teacher = MultiHeadScoreModel(latent_dim=cfg["latent_dim"]).to(args.device)
    teacher.load_state_dict(base_blob["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = MultiHeadScoreModel(latent_dim=cfg["latent_dim"]).to(args.device)
    student.load_state_dict(base_blob["state_dict"])
    # Freeze everything except head_sens
    for name, p in student.named_parameters():
        p.requires_grad_(name.startswith("head_sens"))
    # Re-init head_sens for a fresh fit
    for m in student.head_sens.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    print(f"  trainable params: {sum(p.numel() for p in student.parameters() if p.requires_grad)/1e3:.1f}k")

    # ── 2. Load h50 dataset ────────────────────────────────────────────────
    print(f"Loading h50 dataset: {args.h50}")
    h_blob = torch.load(args.h50, weights_only=False, map_location="cpu")
    z_h = h_blob["z_mu"].float()
    y_sens_h = h_blob["sens_target"].float()           # in [0, 1]
    print(f"  rows: {len(z_h)}, sens_target mu={y_sens_h.mean():.3f} sd={y_sens_h.std():.3f}")

    # h50 train/val split (90/10, seeded)
    g = torch.Generator().manual_seed(0)
    perm_h = torch.randperm(len(z_h), generator=g)
    n_val = max(20, len(z_h) // 10)
    h_val_idx = perm_h[:n_val]; h_tr_idx = perm_h[n_val:]
    z_h = z_h.to(args.device); y_sens_h = y_sens_h.to(args.device)

    # Normalisation for the new head: z-score the h50 sens_target
    sens_h_mu = float(y_sens_h[h_tr_idx].mean())
    sens_h_sd = float(y_sens_h[h_tr_idx].std() + 1e-6)
    print(f"  h50-sens normalisation: mu={sens_h_mu:.3f} sd={sens_h_sd:.3f}")

    # ── 3. Load aux dataset (v3e training set) for teacher regularisation ──
    print(f"Loading aux dataset: {args.aux}")
    a_blob = torch.load(args.aux, weights_only=False, map_location="cpu")
    z_a = a_blob["z_mu"].float()
    print(f"  rows: {len(z_a)}")
    z_a = z_a.to(args.device)

    # ── 4. Train ───────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def sample_sigma(n):
        sigma_scalar = float(torch.rand(1).item() * args.sigma_max)
        return sigma_scalar, torch.full((n,), sigma_scalar, device=args.device)

    best_val = float("inf")
    history = []
    for ep in range(args.epochs):
        student.train()
        # one epoch over h50 (small) + matched aux mini-batches
        h_perm = h_tr_idx[torch.randperm(len(h_tr_idx))]
        steps = max(1, len(h_perm) // args.batch_h50)
        agg = {"h50": 0.0, "teacher": 0.0, "total": 0.0}
        for s in range(steps):
            h_ids = h_perm[s * args.batch_h50:(s + 1) * args.batch_h50]
            a_ids = torch.randint(0, len(z_a), (args.batch_aux,), device=args.device)

            sigma_h_scalar, sigma_h = sample_sigma(len(h_ids))
            sigma_a_scalar, sigma_a = sample_sigma(len(a_ids))

            z_h_in = z_h[h_ids] + torch.randn_like(z_h[h_ids]) * sigma_h_scalar
            z_a_in = z_a[a_ids] + torch.randn_like(z_a[a_ids]) * sigma_a_scalar

            out_h = student(z_h_in, sigma_h)
            out_a = student(z_a_in, sigma_a)
            with torch.no_grad():
                out_a_t = teacher(z_a_in, sigma_a)

            tgt_h = (y_sens_h[h_ids] - sens_h_mu) / sens_h_sd
            l_h = F.smooth_l1_loss(out_h["sens"], tgt_h)
            # Teacher regulariser: keep aux predictions close to the old z-scored output
            l_t = F.smooth_l1_loss(out_a["sens"], out_a_t["sens"])

            total = args.w_h50 * l_h + args.w_teacher * l_t
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0)
            opt.step()
            agg["h50"] += l_h.item(); agg["teacher"] += l_t.item()
            agg["total"] += total.item()
        for k in agg: agg[k] /= steps

        # Validation: just on h50 val
        student.eval()
        with torch.no_grad():
            sigma_v_scalar, sigma_v = sample_sigma(len(h_val_idx))
            z_v_in = z_h[h_val_idx] + torch.randn_like(z_h[h_val_idx]) * sigma_v_scalar
            out_v = student(z_v_in, sigma_v)
            tgt_v = (y_sens_h[h_val_idx] - sens_h_mu) / sens_h_sd
            l_val = F.smooth_l1_loss(out_v["sens"], tgt_v).item()
            # Spearman / R² as additional diagnostics
            pred_un = out_v["sens"].cpu().numpy() * sens_h_sd + sens_h_mu
            true_un = y_sens_h[h_val_idx].cpu().numpy()
            r = float(np.corrcoef(pred_un, true_un)[0, 1])

        sched.step()
        marker = ""
        if l_val < best_val:
            best_val = l_val
            ckpt_cfg = dict(cfg)
            ckpt_cfg.update({"sens_h50_mu": sens_h_mu,
                              "sens_h50_sd": sens_h_sd,
                              "sens_h50_pivot": h_blob["meta"]["h50_pivot_cm"],
                              "sens_h50_slope": h_blob["meta"]["h50_log_slope"],
                              "sens_source": "h50_literature"})
            torch.save({"state_dict": student.state_dict(),
                        "config": ckpt_cfg,
                        "base_ckpt": args.base_ckpt,
                        "h50_dataset": args.h50},
                       out_dir / "model.pt")
            marker = " *"
        history.append({"ep": ep + 1, **agg, "val_h50": l_val, "val_r": r})
        print(f"  ep {ep+1:3d}/{args.epochs}  h50_loss={agg['h50']:.4f}  "
              f"teacher={agg['teacher']:.4f}  val_h50={l_val:.4f}  "
              f"val_r={r:+.3f}{marker}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest val (smooth-L1): {best_val:.4f}")
    print(f"-> {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
