"""Vast.ai job B — DiT-style transformer denoiser.

Replaces FiLM-MLP denoiser with a transformer that treats the 1024-d latent
as 64×16 tokens, applies AdaLN-Zero conditioning + self-attention, and
predicts noise. Direct fix for the D10 broken-cond-signal finding.

Required input files:
  - latents_trustcond.pt  (1.6 GB — yes, this is heavy on R2)
  - dit_denoiser.py
  - dit_train.py
"""
import sys, os, time, math, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA not available."
device = torch.device("cuda")
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs")
writer.add_text("phase", "model_download: building DiT denoiser", 0)
writer.flush()


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden))

    def sinusoid(self, t):
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, t):
        return self.mlp(self.sinusoid(t))


class DiTBlock(nn.Module):
    """Standard DiT block: AdaLN-Zero modulation + self-attention + MLP."""
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp   = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(),
                                       nn.Linear(dim*mlp_ratio, dim))
        self.adaLN = nn.Sequential(nn.SiLU(),
                                       nn.Linear(dim, 6*dim, bias=True))
        nn.init.zeros_(self.adaLN[1].weight); nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        sca1, scb1, gate1, sca2, scb2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sca1.unsqueeze(1)) + scb1.unsqueeze(1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * a
        h = self.norm2(x) * (1 + sca2.unsqueeze(1)) + scb2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class DiTDenoiser(nn.Module):
    def __init__(self, latent_dim=1024, n_tokens=64, dim=512, n_blocks=12, n_heads=8,
                  n_props=4, prop_emb_dim=64):
        super().__init__()
        self.n_tokens = n_tokens
        self.tok_dim = latent_dim // n_tokens
        self.in_proj = nn.Linear(self.tok_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        self.t_emb = TimestepEmbedder(dim)
        self.prop_emb = nn.Linear(2 * n_props, dim)        # values + mask
        self.cond_combine = nn.Linear(2 * dim, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, n_heads) for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.out_proj = nn.Linear(dim, self.tok_dim)
        self.adaLN_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2*dim, bias=True))
        nn.init.zeros_(self.adaLN_out[1].weight); nn.init.zeros_(self.adaLN_out[1].bias)

    def forward(self, z_t, t, values, mask):
        B = z_t.shape[0]
        x = z_t.view(B, self.n_tokens, self.tok_dim)
        x = self.in_proj(x) + self.pos_embed.unsqueeze(0)
        c = torch.cat([values, mask], dim=-1)              # (B, 2*n_props)
        c_emb = self.prop_emb(c)
        t_emb = self.t_emb(t)
        c_full = self.cond_combine(torch.cat([c_emb, t_emb], dim=-1))
        for blk in self.blocks:
            x = blk(x, c_full)
        sca, scb = self.adaLN_out(c_full).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + sca.unsqueeze(1)) + scb.unsqueeze(1)
        out = self.out_proj(x).view(B, -1)
        return out


# Build cosine alpha-bar schedule
def cosine_alphabar(T, s=0.008):
    t = torch.arange(T+1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return f / f[0]

T_diff = 1000
alpha_bar = cosine_alphabar(T_diff).to(device)[1:]            # (T,)


def q_sample(z0, t):
    ab = alpha_bar[t].unsqueeze(1).clamp(min=1e-4)
    om = (1 - ab).sqrt()
    eps = torch.randn_like(z0)
    return ab.sqrt() * z0 + om * eps, eps


# ── Training loop ────────────────────────────────────────────────────────
print("[train] Loading latents_trustcond.pt …"); sys.stdout.flush()
blob = torch.load("latents_trustcond.pt", weights_only=False)
z = blob["z_mu"].float().to(device)
v = blob["values_norm"].float().to(device)
cv = blob["cond_valid"].bool().to(device)
N = z.shape[0]
print(f"[train] Loaded {N:,} latents (1024-d)"); sys.stdout.flush()
writer.add_text("phase", f"data_loaded: {N} latents", 0)

model = DiTDenoiser(latent_dim=1024, n_tokens=64, dim=512, n_blocks=12, n_heads=8).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[train] Model loaded: DiT denoiser ({n_params/1e6:.1f}M params) on {device}"); sys.stdout.flush()
writer.add_text("phase", f"model_loaded: DiT denoiser {n_params/1e6:.1f}M", 0)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda')

bs = 128
total_steps = 50000
print(f"[train] Training ({total_steps} steps)…"); sys.stdout.flush()
writer.add_text("phase", f"training_start: {total_steps} steps", 0)
writer.flush()

t_start = time.time()
val_idx = torch.randperm(N)[:5000]
tr_idx_pool = torch.randperm(N)
ptr = 0
for step in range(1, total_steps + 1):
    if ptr + bs > N:
        tr_idx_pool = torch.randperm(N); ptr = 0
    idx = tr_idx_pool[ptr:ptr+bs]; ptr += bs
    z0 = z[idx]
    vals = v[idx]
    mask = cv[idx].float()
    t = torch.randint(0, T_diff, (bs,), device=device)
    with torch.amp.autocast('cuda'):
        zt, eps = q_sample(z0, t)
        eps_pred = model(zt, t, vals, mask)
        loss = F.mse_loss(eps_pred, eps)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    writer.add_scalar("train/loss", loss.item(), step)
    if step % 200 == 0:
        print(f"  {step}/{total_steps} loss={loss.item():.4f} epoch=1"); sys.stdout.flush()
    if step % 5000 == 0:
        torch.save({"model_state": model.state_dict(), "step": step},
                   f"results/dit_step{step}.pt")

elapsed = time.time() - t_start
torch.save({"model_state": model.state_dict(), "step": total_steps},
           "results/dit_denoiser_best.pt")
print(f"[train] Loss: {loss.item():.4f}"); sys.stdout.flush()
writer.add_text("phase", f"training_complete: {elapsed:.0f}s", 0)
writer.flush()
shutil.copytree("runs", "results/tb_runs", dirs_exist_ok=True)
print("[train] === DONE ==="); sys.stdout.flush()
