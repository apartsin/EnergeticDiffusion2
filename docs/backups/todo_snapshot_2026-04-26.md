# Todo snapshot — 2026-04-26

Snapshot of the live todo list at session-audit time. Use as a recovery
point if memory/state is lost.

## Active

- **LIMO v2.1 training** — task `b8nks15e0`, motif-augmented dataset
  (1.22 M rows, rare 5× / polynitro 2×), lr=3e-5, 8 h budget, init from
  `experiments/limo_ft_energetic_20260424T150825Z/checkpoints/best.pt`,
  output `experiments/limo_ft_motif_rich_v2_1_<ts>/`.

## Completed milestones

- LIMO v1 fine-tune (v1 production VAE).
- 7 denoiser experiments: v1 → v6, plus v4-nf (no-filter ablation),
  v5 (Min-SNR), v4-B (Tier-A/B-only conditioning, current ρ/D/P production),
  v3 (current HOF production).
- Diagnostics D1, D2, D3, D5, D7, D8, D10, D14, D15, D16.
- Feasibility integration (rerank-weight composite — production-ready;
  sampling-time gradient — working at small λ but modest gain).
- C2c diagnostic rounds 1–4 (bounded by LIMO self-consistency ceiling).
- `chem_filter.py` + `joint_rerank.py` + `breakthrough_experiment.md`.
- **Joint v3+v4-B breakthrough rerank**: 5 viable leads found, top is
  ρ=1.91, D=9.20, P=36.7, HOF=+163, SA=4.55, MaxTan=0.38.
- Repo created + pushed: `https://github.com/apartsin/EnergeticDiffusion2`.

## Pending (orphan list)

1. **GitHub release v0.1.0 upload** (~2.4 GB ckpts).
   `scripts/archive_artifacts.sh --upload` ready, never executed.
2. **README.md** for the repo (never created).
3. **LIMO v2.1 diagnostics R1–R9** (per `docs/limo_v2_plan.md`).
4. **Re-encode latents → `latents_v2.pt`** (decision pending after v2.1).
5. **Denoiser retrain on `latents_v2.pt`** (decision pending).
6. **C2c re-eval after LIMO v2.1** to test new self-consistency.
7. **HOF-prioritised joint rerank** (weight composite toward HOF).
8. **Synthesis-route / Murcko-scaffold annotation** for top breakthrough leads.
9. **production_overview.md polish** (add joint_rerank + leads section).

## Explicitly dropped (no action)

- MolMIM / ChemFormer / CDDD VAE swaps.
- Active-learning DFT loop.

## Background tasks running at snapshot time

| task_id | command | started | ETA |
|---|---|---|---|
| `b8nks15e0` | LIMO v2.1 training | 2026-04-26 ~08:48Z | ~8 h |
