import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonicalize_smiles(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def fp_from_smiles(smiles: str, n_bits: int = 2048):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)


def fp_array_from_smiles(smiles: str, n_bits: int = 2048) -> np.ndarray:
    fp = fp_from_smiles(smiles, n_bits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    if fp is None:
        return arr
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def maybe_sample(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_selfies_string(s: str) -> List[str]:
    if not isinstance(s, str) or not s:
        return []
    try:
        return list(sf.split_selfies(s))
    except Exception:
        return []


def build_vocab(selfies_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    token_set = set()
    lengths = []
    for s in selfies_list:
        toks = split_selfies_string(s)
        if toks:
            token_set.update(toks)
            lengths.append(len(toks))
    stoi = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
    for tok in sorted(token_set):
        if tok not in stoi:
            stoi[tok] = len(stoi)
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def percentile_max_len(selfies_list: List[str], cap: int) -> int:
    lengths = []
    for s in selfies_list:
        toks = split_selfies_string(s)
        if toks:
            lengths.append(len(toks) + 2)
    if not lengths:
        return min(64, cap)
    return int(min(cap, max(8, np.percentile(lengths, 99))))


def encode_selfies(s: str, stoi: Dict[str, int], max_len: int) -> List[int]:
    toks = split_selfies_string(s)
    ids = [stoi[BOS]]
    for tok in toks:
        ids.append(stoi.get(tok, stoi[UNK]))
    ids.append(stoi[EOS])
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([stoi[PAD]] * (max_len - len(ids)))
    return ids


def decode_ids(ids: List[int], itos: Dict[int, str]) -> str:
    toks = []
    for idx in ids:
        tok = itos.get(int(idx), UNK)
        if tok in (PAD, BOS):
            continue
        if tok == EOS:
            break
        toks.append(tok)
    return "".join(toks)


class SequenceDataset(Dataset):
    def __init__(self, seqs: np.ndarray):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long)


class SelfiesVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        pad_idx: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.encoder = nn.GRU(
            embedding_dim, hidden_dim, batch_first=True, dropout=0.0, num_layers=1
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(
            embedding_dim, hidden_dim, batch_first=True, dropout=0.0, num_layers=1
        )
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.drop = nn.Dropout(dropout)

    def encode(self, x):
        emb = self.drop(self.embedding(x))
        _, h = self.encoder(emb)
        h = h[-1]
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        dec_in = x[:, :-1]
        target = x[:, 1:]
        emb = self.drop(self.embedding(dec_in))
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        dec_out, _ = self.decoder(emb, h0)
        logits = self.out(dec_out)
        return logits, target, mu, logvar

    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor, bos_idx: int, eos_idx: int, max_len: int):
        device = z.device
        bsz = z.shape[0]
        h = self.latent_to_hidden(z).unsqueeze(0)
        cur = torch.full((bsz, 1), bos_idx, dtype=torch.long, device=device)
        outputs = [cur]
        for _ in range(max_len - 1):
            emb = self.embedding(cur)
            out, h = self.decoder(emb, h)
            logits = self.out(out[:, -1:, :])
            nxt = torch.argmax(logits, dim=-1)
            outputs.append(nxt)
            cur = nxt
        seq = torch.cat(outputs, dim=1)
        return seq


def vae_loss(logits, target, mu, logvar, pad_idx, beta):
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=pad_idx,
    )
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.detach().item(), kl.detach().item()


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DenoiserMLP(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int, t_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        in_dim = latent_dim + t_embed_dim + cond_dim + cond_dim + latent_dim + 1
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_t, t, cond_vals, cond_mask, source_z, source_mask):
        t_emb = timestep_embedding(t, self.t_embed_dim)
        inp = torch.cat(
            [x_t, t_emb, cond_vals, cond_mask, source_z, source_mask.unsqueeze(1)], dim=1
        )
        h = F.gelu(self.fc1(inp))
        h = F.gelu(self.fc2(h))
        return self.fc3(h)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_schedule(timesteps: int, device: torch.device) -> DiffusionSchedule:
    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def q_sample(x0, t, eps, sched: DiffusionSchedule):
    ab = sched.alpha_bars[t].unsqueeze(1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps


def source_domain_score(smiles: str) -> float:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    n = atoms.count("N")
    o = atoms.count("O")
    nitro = smiles.count("[N+](=O)[O-]") + smiles.count("N(=O)=O")
    score = 1.0 / (1.0 + math.exp(-(0.35 * n + 0.28 * o + 0.9 * nitro - 2.8)))
    return float(score)


def synthetic_accessibility_proxy(smiles: str) -> float:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    mw = Descriptors.MolWt(mol)
    heavy = mol.GetNumHeavyAtoms()
    rings = mol.GetRingInfo().NumRings()
    s1 = 1.0 / (1.0 + math.exp((mw - 360.0) / 90.0))
    s2 = 1.0 / (1.0 + heavy / 28.0)
    s3 = 1.0 / (1.0 + rings / 4.0)
    return float((0.45 * s1) + (0.35 * s2) + (0.20 * s3))


def validate_basic_chemistry(smiles: str, allowed_atoms: set, min_heavy: int, max_heavy: int) -> Tuple[bool, str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return False, "invalid_smiles"
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    if any(a not in allowed_atoms for a in atoms):
        return False, "disallowed_atom"
    heavy = mol.GetNumHeavyAtoms()
    if heavy < min_heavy or heavy > max_heavy:
        return False, "heavy_atom_out_of_range"
    return True, "pass"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def train_vae(
    train_selfies: List[str],
    val_selfies: List[str],
    cfg: dict,
    device: torch.device,
    out_dir: Path,
):
    stoi, itos = build_vocab(train_selfies)
    max_len = percentile_max_len(train_selfies, cap=int(cfg["max_len_cap"]))
    train_ids = np.array([encode_selfies(s, stoi, max_len) for s in train_selfies], dtype=np.int64)
    val_ids = np.array([encode_selfies(s, stoi, max_len) for s in val_selfies], dtype=np.int64)
    train_ds = SequenceDataset(train_ids)
    val_ds = SequenceDataset(val_ids)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        drop_last=False,
    )
    model = SelfiesVAE(
        vocab_size=len(stoi),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        latent_dim=int(cfg["latent_dim"]),
        pad_idx=stoi[PAD],
        dropout=float(cfg["dropout"]),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    epochs = int(cfg["epochs"])
    beta_max = float(cfg["beta_max"])
    metrics = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        beta = beta_max * (epoch / epochs)
        for batch in tqdm(train_loader, desc=f"VAE Train {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            logits, target, mu, logvar = model(batch)
            loss, recon, kl = vae_loss(logits, target, mu, logvar, stoi[PAD], beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * batch.size(0)
            train_recon += recon * batch.size(0)
            train_kl += kl * batch.size(0)
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"VAE Val {epoch}/{epochs}", leave=False):
                batch = batch.to(device)
                logits, target, mu, logvar = model(batch)
                loss, recon, kl = vae_loss(logits, target, mu, logvar, stoi[PAD], beta)
                val_loss += loss.item() * batch.size(0)
                val_recon += recon * batch.size(0)
                val_kl += kl * batch.size(0)
        train_n = len(train_ds)
        val_n = len(val_ds)
        row = {
            "epoch": epoch,
            "beta": beta,
            "train_loss": train_loss / max(1, train_n),
            "train_recon": train_recon / max(1, train_n),
            "train_kl": train_kl / max(1, train_n),
            "val_loss": val_loss / max(1, val_n),
            "val_recon": val_recon / max(1, val_n),
            "val_kl": val_kl / max(1, val_n),
        }
        metrics.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "max_len": max_len,
                "latent_dim": int(cfg["latent_dim"]),
                "embedding_dim": int(cfg["embedding_dim"]),
                "hidden_dim": int(cfg["hidden_dim"]),
                "pad_idx": stoi[PAD],
            }
        print(
            f"[VAE] epoch={epoch} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f}"
        )
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "vae_metrics.csv", index=False)
    torch.save(best_state, out_dir / "vae_checkpoint.pt")
    return best_state, metrics_df


def load_vae_from_state(state: dict, device: torch.device) -> SelfiesVAE:
    model = SelfiesVAE(
        vocab_size=len(state["stoi"]),
        embedding_dim=int(state["embedding_dim"]),
        hidden_dim=int(state["hidden_dim"]),
        latent_dim=int(state["latent_dim"]),
        pad_idx=int(state["pad_idx"]),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def encode_selfies_to_latent(
    model: SelfiesVAE,
    selfies_list: List[str],
    stoi: Dict[str, int],
    max_len: int,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    ids = np.array([encode_selfies(s, stoi, max_len) for s in selfies_list], dtype=np.int64)
    ds = SequenceDataset(ids)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    latents = []
    for batch in tqdm(loader, desc="Encoding latents", leave=False):
        batch = batch.to(device)
        mu, _ = model.encode(batch)
        latents.append(mu.detach().cpu().numpy())
    if not latents:
        return np.zeros((0, model.latent_dim), dtype=np.float32)
    return np.concatenate(latents, axis=0).astype(np.float32)


def compute_property_stats(df_train: pd.DataFrame, props: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for p in props:
        vals = pd.to_numeric(df_train[p], errors="coerce").dropna()
        stats[p] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0) if vals.std(ddof=0) > 1e-8 else 1.0),
        }
    return stats


def normalize_props(df: pd.DataFrame, props: List[str], stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    out = []
    for p in props:
        vals = pd.to_numeric(df[p], errors="coerce").fillna(stats[p]["mean"]).to_numpy(dtype=np.float32)
        out.append(((vals - stats[p]["mean"]) / stats[p]["std"]).astype(np.float32))
    return np.stack(out, axis=1)


def denormalize_props(arr: np.ndarray, props: List[str], stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    cols = []
    for i, p in enumerate(props):
        cols.append((arr[:, i] * stats[p]["std"]) + stats[p]["mean"])
    return np.stack(cols, axis=1)


def predictor_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, props: List[str], split: str, model_type: str, seed: int
) -> List[dict]:
    rows = []
    for i, p in enumerate(props):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(math.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp))
        rows.append(
            {
                "model_type": model_type,
                "seed": seed,
                "split": split,
                "property": p,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )
    return rows


def train_predictors(
    out_dir: Path,
    props: List[str],
    latent_train: np.ndarray,
    latent_val: np.ndarray,
    latent_test: np.ndarray,
    y_train_norm: np.ndarray,
    y_val_norm: np.ndarray,
    y_test_norm: np.ndarray,
    energetic_train_smiles: List[str],
    energetic_val_smiles: List[str],
    energetic_test_smiles: List[str],
    cfg: dict,
) -> Tuple[List[MLPRegressor], List[MLPRegressor], pd.DataFrame]:
    pred_dir = out_dir / "property_predictor_checkpoints"
    ensure_dir(pred_dir)
    metrics_rows = []
    latent_models = []
    indep_models = []
    seeds = [int(cfg["random_seed_start"]) + i for i in range(int(cfg["ensemble_size"]))]

    fp_train = np.stack([fp_array_from_smiles(s) for s in energetic_train_smiles], axis=0)
    fp_val = np.stack([fp_array_from_smiles(s) for s in energetic_val_smiles], axis=0)
    fp_test = np.stack([fp_array_from_smiles(s) for s in energetic_test_smiles], axis=0)

    for seed in seeds:
        latent_model = MLPRegressor(
            hidden_layer_sizes=tuple(cfg["hidden_layer_sizes"]),
            random_state=seed,
            max_iter=int(cfg["max_iter"]),
            early_stopping=True,
            n_iter_no_change=20,
            learning_rate_init=1e-3,
            validation_fraction=0.1,
        )
        latent_model.fit(latent_train, y_train_norm)
        latent_models.append(latent_model)
        joblib.dump(latent_model, pred_dir / f"latent_seed{seed}.joblib")

        yv = latent_model.predict(latent_val)
        yt = latent_model.predict(latent_test)
        metrics_rows.extend(predictor_metrics(y_val_norm, yv, props, "validation", "latent", seed))
        metrics_rows.extend(predictor_metrics(y_test_norm, yt, props, "test", "latent", seed))

        indep_model = MLPRegressor(
            hidden_layer_sizes=tuple(cfg["hidden_layer_sizes"]),
            random_state=seed + 1000,
            max_iter=int(cfg["max_iter"]),
            early_stopping=True,
            n_iter_no_change=20,
            learning_rate_init=1e-3,
            validation_fraction=0.1,
        )
        indep_model.fit(fp_train, y_train_norm)
        indep_models.append(indep_model)
        joblib.dump(indep_model, pred_dir / f"independent_seed{seed}.joblib")
        yv_i = indep_model.predict(fp_val)
        yt_i = indep_model.predict(fp_test)
        metrics_rows.extend(
            predictor_metrics(y_val_norm, yv_i, props, "validation", "independent", seed)
        )
        metrics_rows.extend(
            predictor_metrics(y_test_norm, yt_i, props, "test", "independent", seed)
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "property_predictor_metrics.csv", index=False)
    return latent_models, indep_models, metrics_df


def ensemble_predict(models: List[MLPRegressor], x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.stack([m.predict(x) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def train_diffusion(
    out_dir: Path,
    latent_dim: int,
    cond_dim: int,
    z_generic: np.ndarray,
    z_energetic: np.ndarray,
    energetic_props_norm: np.ndarray,
    cfg: dict,
    device: torch.device,
):
    model = DenoiserMLP(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        t_embed_dim=int(cfg["timestep_embed_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    sched = make_schedule(int(cfg["timesteps"]), device)

    z_generic_t = torch.tensor(z_generic, dtype=torch.float32, device=device)
    z_ener_t = torch.tensor(z_energetic, dtype=torch.float32, device=device)
    props_t = torch.tensor(energetic_props_norm, dtype=torch.float32, device=device)
    bsz = int(cfg["batch_size"])
    logs = []

    def one_step(x0, cond_vals, cond_mask, source_z, source_mask):
        t = torch.randint(
            low=0, high=int(cfg["timesteps"]), size=(x0.size(0),), device=device
        )
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, sched)
        pred = model(x_t, t, cond_vals, cond_mask, source_z, source_mask)
        loss = F.mse_loss(pred, eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.item())

    # Stage A: unconditional on mixed latents
    for step in tqdm(range(int(cfg["stage_a_steps"])), desc="Diffusion Stage A"):
        idx_g = torch.randint(0, z_generic_t.size(0), (bsz // 2,), device=device)
        idx_e = torch.randint(0, z_ener_t.size(0), (bsz - bsz // 2,), device=device)
        x0 = torch.cat([z_generic_t[idx_g], z_ener_t[idx_e]], dim=0)
        cond_vals = torch.zeros((x0.size(0), cond_dim), device=device)
        cond_mask = torch.zeros((x0.size(0), cond_dim), device=device)
        source_z = torch.zeros_like(x0)
        source_mask = torch.zeros((x0.size(0),), device=device)
        loss = one_step(x0, cond_vals, cond_mask, source_z, source_mask)
        logs.append({"stage": "A", "step": step + 1, "loss": loss})

    # Stage B: property-conditioned
    for step in tqdm(range(int(cfg["stage_b_steps"])), desc="Diffusion Stage B"):
        idx = torch.randint(0, z_ener_t.size(0), (bsz,), device=device)
        x0 = z_ener_t[idx]
        cond_vals = props_t[idx].clone()
        cond_mask = torch.ones_like(cond_vals)
        drop_all = torch.rand((bsz, 1), device=device) < float(cfg["p_drop_all_properties"])
        cond_mask = cond_mask * (~drop_all).float()
        drop_each = (
            torch.rand(cond_mask.shape, device=device) < float(cfg["p_drop_each_property"])
        ).float()
        cond_mask = cond_mask * (1.0 - drop_each)
        cond_vals = cond_vals * cond_mask
        source_z = torch.zeros_like(x0)
        source_mask = torch.zeros((bsz,), device=device)
        loss = one_step(x0, cond_vals, cond_mask, source_z, source_mask)
        logs.append({"stage": "B", "step": step + 1, "loss": loss})

    # Stage C: source-conditioned editing (self + neighbor-like random energetic pair)
    neigh = NearestNeighbors(n_neighbors=2, metric="euclidean")
    neigh.fit(z_energetic)
    _, nn_idx = neigh.kneighbors(z_energetic)
    nn_partner = nn_idx[:, 1]
    nn_partner_t = torch.tensor(nn_partner, dtype=torch.long, device=device)
    for step in tqdm(range(int(cfg["stage_c_steps"])), desc="Diffusion Stage C"):
        idx = torch.randint(0, z_ener_t.size(0), (bsz,), device=device)
        x0 = z_ener_t[idx]
        cond_vals = props_t[idx].clone()
        cond_mask = torch.ones_like(cond_vals)
        use_self = (torch.rand((bsz,), device=device) < 0.5).long()
        paired = torch.where(use_self == 1, idx, nn_partner_t[idx])
        source_z = z_ener_t[paired].clone()
        source_mask = torch.ones((bsz,), device=device)
        drop_source = torch.rand((bsz,), device=device) < float(cfg["p_drop_source"])
        source_mask = source_mask * (~drop_source).float()
        source_z = source_z * source_mask.unsqueeze(1)
        drop_all = torch.rand((bsz, 1), device=device) < float(cfg["p_drop_all_properties"])
        cond_mask = cond_mask * (~drop_all).float()
        drop_each = (
            torch.rand(cond_mask.shape, device=device) < float(cfg["p_drop_each_property"])
        ).float()
        cond_mask = cond_mask * (1.0 - drop_each)
        cond_vals = cond_vals * cond_mask
        loss = one_step(x0, cond_vals, cond_mask, source_z, source_mask)
        logs.append({"stage": "C", "step": step + 1, "loss": loss})

    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(out_dir / "diffusion_training_metrics.csv", index=False)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "latent_dim": latent_dim,
        "cond_dim": cond_dim,
        "timesteps": int(cfg["timesteps"]),
        "timestep_embed_dim": int(cfg["timestep_embed_dim"]),
        "hidden_dim": int(cfg["hidden_dim"]),
    }
    torch.save(ckpt, out_dir / "diffusion_checkpoint.pt")
    return model, sched, logs_df


@torch.no_grad()
def sample_diffusion(
    model: DenoiserMLP,
    sched: DiffusionSchedule,
    n_samples: int,
    latent_dim: int,
    cond_vals: torch.Tensor,
    cond_mask: torch.Tensor,
    guidance_scale: float,
    sampling_steps: int,
    device: torch.device,
):
    model.eval()
    x = torch.randn((n_samples, latent_dim), device=device)
    timesteps = len(sched.betas)
    ts = np.linspace(timesteps - 1, 0, num=sampling_steps, dtype=np.int64)
    source_z = torch.zeros_like(x)
    source_mask = torch.zeros((n_samples,), device=device)
    for i, t_int in enumerate(ts):
        t = torch.full((n_samples,), int(t_int), dtype=torch.long, device=device)
        eps_cond = model(x, t, cond_vals, cond_mask, source_z, source_mask)
        eps_uncond = model(
            x,
            t,
            torch.zeros_like(cond_vals),
            torch.zeros_like(cond_mask),
            torch.zeros_like(source_z),
            torch.zeros_like(source_mask),
        )
        eps = (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
        ab_t = sched.alpha_bars[t].unsqueeze(1)
        x0_pred = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
        if i == len(ts) - 1:
            x = x0_pred
            break
        t_next = int(ts[i + 1])
        ab_next = sched.alpha_bars[
            torch.full((n_samples,), t_next, dtype=torch.long, device=device)
        ].unsqueeze(1)
        x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * eps
    return x.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Run EnergeticDiffusion2 MDD experiment")
    parser.add_argument("--config", type=str, default="configs/mdd_baseline.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))

    run_name = cfg["run"]["name"]
    out_root = root / cfg["run"]["output_root"]
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / f"{run_name}_{timestamp}"
    ensure_dir(out_dir)
    ensure_dir(out_dir / "property_predictor_checkpoints")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    set_seed(int(cfg["run"]["seed"]))
    if cfg["run"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[info] using device={device}")

    # Load datasets
    diffusion_train = read_csv(root / cfg["data"]["diffusion_train"])
    diffusion_val = read_csv(root / cfg["data"]["diffusion_validation"])
    diffusion_test = read_csv(root / cfg["data"]["diffusion_test"])
    energetic_train = read_csv(root / cfg["data"]["energetic_train"])
    energetic_val = read_csv(root / cfg["data"]["energetic_validation"])
    energetic_test = read_csv(root / cfg["data"]["energetic_test"])

    ss = cfg["data"]["subsets"]
    diffusion_train_s = maybe_sample(diffusion_train, ss["vae_generic_train_max"], int(cfg["run"]["seed"]))
    energetic_train_s = maybe_sample(energetic_train, ss["vae_energetic_train_max"], int(cfg["run"]["seed"]))
    diffusion_val_s = maybe_sample(diffusion_val, ss["vae_generic_validation_max"], int(cfg["run"]["seed"]))
    energetic_val_s = maybe_sample(energetic_val, ss["vae_energetic_validation_max"], int(cfg["run"]["seed"]))

    # Ensure canonical smiles field exists
    for df in [diffusion_train, diffusion_val, diffusion_test, energetic_train, energetic_val, energetic_test]:
        if "canonical_smiles" not in df.columns:
            df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
        if "split" not in df.columns:
            if "split_key" in df.columns:
                df["split"] = df["split_key"].map(lambda _: "")
            else:
                df["split"] = ""
        if "scaffold_id" not in df.columns:
            if "split_key" in df.columns:
                df["scaffold_id"] = df["split_key"]
            else:
                df["scaffold_id"] = ""

    # Train VAE
    vae_train_selfies = (
        diffusion_train_s["selfies"].dropna().astype(str).tolist()
        + energetic_train_s["selfies"].dropna().astype(str).tolist()
    )
    vae_val_selfies = (
        diffusion_val_s["selfies"].dropna().astype(str).tolist()
        + energetic_val_s["selfies"].dropna().astype(str).tolist()
    )
    vae_state, _ = train_vae(vae_train_selfies, vae_val_selfies, cfg["vae"], device, out_dir)
    vae_model = load_vae_from_state(vae_state, device)

    # Export latent dataset (subset to keep runtime practical)
    diffusion_train_e = maybe_sample(
        diffusion_train,
        min(ss["latent_export_generic_max"], len(diffusion_train)),
        int(cfg["run"]["seed"]),
    )
    diffusion_val_e = maybe_sample(
        diffusion_val,
        max(1, int(min(ss["latent_export_generic_max"] * 0.1, len(diffusion_val)))),
        int(cfg["run"]["seed"]),
    )
    diffusion_test_e = maybe_sample(
        diffusion_test,
        max(1, int(min(ss["latent_export_generic_max"] * 0.1, len(diffusion_test)))),
        int(cfg["run"]["seed"]),
    )
    energetic_train_e = maybe_sample(energetic_train, ss["latent_export_energetic_max"], int(cfg["run"]["seed"]))
    energetic_val_e = maybe_sample(energetic_val, ss["latent_export_energetic_max"], int(cfg["run"]["seed"]))
    energetic_test_e = maybe_sample(energetic_test, ss["latent_export_energetic_max"], int(cfg["run"]["seed"]))

    blocks = [
        ("generic", "train", diffusion_train_e),
        ("generic", "validation", diffusion_val_e),
        ("generic", "test", diffusion_test_e),
        ("energetic", "train", energetic_train_e),
        ("energetic", "validation", energetic_val_e),
        ("energetic", "test", energetic_test_e),
    ]

    latent_rows = []
    props = list(cfg["data"]["properties"])
    for role, split_name, df in blocks:
        zs = encode_selfies_to_latent(
            vae_model,
            df["selfies"].fillna("").astype(str).tolist(),
            vae_state["stoi"],
            int(vae_state["max_len"]),
            device,
            batch_size=512,
        )
        for i in range(len(df)):
            r = {
                "molecule_id": str(df.iloc[i].get("molecule_id", "")),
                "canonical_smiles": str(df.iloc[i].get("canonical_smiles", "")),
                "smiles": str(df.iloc[i].get("smiles", "")),
                "selfies": str(df.iloc[i].get("selfies", "")),
                "source_dataset": str(df.iloc[i].get("source_dataset", "")),
                "source_role": role,
                "split": split_name,
                "scaffold_id": str(df.iloc[i].get("scaffold_id", df.iloc[i].get("split_key", ""))),
            }
            for p in props:
                r[p] = (
                    float(df.iloc[i][p])
                    if p in df.columns and pd.notna(df.iloc[i][p])
                    else np.nan
                )
            for j in range(zs.shape[1]):
                r[f"z_{j}"] = float(zs[i, j])
            latent_rows.append(r)
    latent_df = pd.DataFrame(latent_rows)
    latent_df.to_parquet(out_dir / "latent_dataset.parquet", index=False)

    # Copy split manifest artifact
    src_manifest = root / cfg["data"]["split_manifest"]
    if src_manifest.exists():
        shutil.copy2(src_manifest, out_dir / "data_split_manifest.csv")
    else:
        latent_df[["molecule_id", "source_role", "split", "scaffold_id"]].to_csv(
            out_dir / "data_split_manifest.csv", index=False
        )

    # Property normalization from energetic train split
    property_stats = compute_property_stats(energetic_train_e, props)
    norm_payload = {
        "normalization_version": f"run_{timestamp}",
        "source_split": str(cfg["data"]["energetic_train"]),
        "properties": property_stats,
    }
    (out_dir / "property_normalization.json").write_text(
        json.dumps(norm_payload, indent=2), encoding="utf-8"
    )

    # Predictor training data
    z_cols = [c for c in latent_df.columns if c.startswith("z_")]
    e_train_lat = latent_df[
        (latent_df["source_role"] == "energetic") & (latent_df["split"] == "train")
    ].reset_index(drop=True)
    e_val_lat = latent_df[
        (latent_df["source_role"] == "energetic") & (latent_df["split"] == "validation")
    ].reset_index(drop=True)
    e_test_lat = latent_df[
        (latent_df["source_role"] == "energetic") & (latent_df["split"] == "test")
    ].reset_index(drop=True)
    x_train_lat = e_train_lat[z_cols].to_numpy(dtype=np.float32)
    x_val_lat = e_val_lat[z_cols].to_numpy(dtype=np.float32)
    x_test_lat = e_test_lat[z_cols].to_numpy(dtype=np.float32)
    y_train_norm = normalize_props(e_train_lat, props, property_stats)
    y_val_norm = normalize_props(e_val_lat, props, property_stats)
    y_test_norm = normalize_props(e_test_lat, props, property_stats)

    latent_models, indep_models, _ = train_predictors(
        out_dir=out_dir,
        props=props,
        latent_train=x_train_lat,
        latent_val=x_val_lat,
        latent_test=x_test_lat,
        y_train_norm=y_train_norm,
        y_val_norm=y_val_norm,
        y_test_norm=y_test_norm,
        energetic_train_smiles=e_train_lat["canonical_smiles"].fillna("").astype(str).tolist(),
        energetic_val_smiles=e_val_lat["canonical_smiles"].fillna("").astype(str).tolist(),
        energetic_test_smiles=e_test_lat["canonical_smiles"].fillna("").astype(str).tolist(),
        cfg=cfg["predictor"],
    )

    # Diffusion training
    g_train_lat = latent_df[
        (latent_df["source_role"] == "generic") & (latent_df["split"] == "train")
    ].reset_index(drop=True)
    z_generic = g_train_lat[z_cols].to_numpy(dtype=np.float32)
    z_ener = x_train_lat
    ener_props_norm = y_train_norm
    diff_model, diff_sched, diff_logs = train_diffusion(
        out_dir=out_dir,
        latent_dim=int(cfg["vae"]["latent_dim"]),
        cond_dim=len(props),
        z_generic=z_generic,
        z_energetic=z_ener,
        energetic_props_norm=ener_props_norm,
        cfg=cfg["diffusion"],
        device=device,
    )

    # Generation config
    target_vals_raw = []
    for p in props:
        q = float(np.percentile(pd.to_numeric(e_train_lat[p], errors="coerce").dropna(), 75))
        target_vals_raw.append(q)
    target_vals_raw_arr = np.array(target_vals_raw, dtype=np.float32).reshape(1, -1)
    target_vals_norm = normalize_props(
        pd.DataFrame([dict(zip(props, target_vals_raw))]), props, property_stats
    )
    gen_cfg = {
        "mode": "property_conditioned",
        "num_candidates": int(cfg["generation"]["num_candidates"]),
        "sampling_steps": int(cfg["generation"]["sampling_steps"]),
        "guidance_scale": float(cfg["generation"]["guidance_scale"]),
        "target_properties": {p: float(target_vals_raw[i]) for i, p in enumerate(props)},
    }
    (out_dir / "generation_config.yaml").write_text(yaml.safe_dump(gen_cfg, sort_keys=False), encoding="utf-8")

    n_cand = int(cfg["generation"]["num_candidates"])
    cond_vals = torch.tensor(
        np.repeat(target_vals_norm, n_cand, axis=0), dtype=torch.float32, device=device
    )
    cond_mask = torch.ones_like(cond_vals)
    z_gen = sample_diffusion(
        model=diff_model,
        sched=diff_sched,
        n_samples=n_cand,
        latent_dim=int(cfg["vae"]["latent_dim"]),
        cond_vals=cond_vals,
        cond_mask=cond_mask,
        guidance_scale=float(cfg["generation"]["guidance_scale"]),
        sampling_steps=int(cfg["generation"]["sampling_steps"]),
        device=device,
    )

    # Decode generated latents
    z_train_ref = x_train_lat
    nn_latent = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_latent.fit(z_train_ref)
    _, nearest_idx = nn_latent.kneighbors(z_gen)
    generated_rows = []
    train_canon = set(e_train_lat["canonical_smiles"].fillna("").astype(str).tolist())
    train_fps = []
    train_smiles_list = e_train_lat["canonical_smiles"].fillna("").astype(str).tolist()
    for s in train_smiles_list:
        train_fps.append(fp_from_smiles(s))

    z_gen_t = torch.tensor(z_gen, dtype=torch.float32, device=device)
    decoded_ids = vae_model.decode_from_latent(
        z_gen_t, vae_state["stoi"][BOS], vae_state["stoi"][EOS], int(vae_state["max_len"])
    ).cpu().numpy()

    # Predict generated properties from latent ensemble
    lat_mean_norm, lat_std_norm = ensemble_predict(latent_models, z_gen)
    lat_mean_raw = denormalize_props(lat_mean_norm, props, property_stats)
    lat_std_raw = lat_std_norm * np.array([property_stats[p]["std"] for p in props])[None, :]

    for i in range(n_cand):
        seq_ids = decoded_ids[i].tolist()
        gen_selfies = decode_ids(seq_ids, vae_state["itos"])
        decode_method = "vae"
        gen_smiles = ""
        can = ""
        if gen_selfies:
            try:
                gen_smiles = sf.decoder(gen_selfies)
                can = canonicalize_smiles(gen_smiles)
            except Exception:
                can = ""
        if not can:
            # fallback to nearest known latent to keep pipeline moving
            ref = int(nearest_idx[i][0])
            can = str(e_train_lat.iloc[ref]["canonical_smiles"])
            gen_smiles = can
            gen_selfies = str(e_train_lat.iloc[ref]["selfies"])
            decode_method = "nearest_train_fallback"

        mol = mol_from_smiles(can)
        inchi = Chem.MolToInchiKey(mol) if mol is not None else ""
        scaffold = ""
        if mol is not None:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                scaffold = ""

        # independent predictor on smiles fingerprint
        fp_vec = fp_array_from_smiles(can).reshape(1, -1)
        indep_pred_norm_all = np.stack([m.predict(fp_vec)[0] for m in indep_models], axis=0)
        indep_mean_norm = indep_pred_norm_all.mean(axis=0)
        indep_std_norm = indep_pred_norm_all.std(axis=0)
        indep_mean_raw = denormalize_props(indep_mean_norm.reshape(1, -1), props, property_stats)[0]

        # novelty
        cfp = fp_from_smiles(can)
        nn_sim = 0.0
        nn_smiles = ""
        if cfp is not None:
            sims = []
            for j, tfp in enumerate(train_fps):
                if tfp is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(cfp, tfp)
                sims.append((sim, train_smiles_list[j]))
            if sims:
                sims.sort(key=lambda x: x[0], reverse=True)
                nn_sim, nn_smiles = sims[0]
        novelty = 1.0 - float(nn_sim)
        domain = source_domain_score(can)
        synth = synthetic_accessibility_proxy(can)
        uncertainty = float(np.mean(lat_std_raw[i]))
        # normalized target error
        pred = lat_mean_raw[i]
        target = target_vals_raw_arr[0]
        err = float(np.mean(np.abs(pred - target) / (np.array([property_stats[p]["std"] for p in props]) + 1e-8)))
        final_score = float((-err) - 0.5 * uncertainty + 0.2 * novelty + 0.2 * domain + 0.1 * synth)

        generated_rows.append(
            {
                "candidate_id": f"cand_{i:05d}",
                "generated_selfies": gen_selfies,
                "generated_smiles": gen_smiles,
                "canonical_smiles": can,
                "inchi_key": inchi,
                "generation_mode": "mode_a_property_conditioned",
                "target_properties": json.dumps({p: float(target[j]) for j, p in enumerate(props)}),
                "predicted_properties_latent": json.dumps(
                    {p: float(lat_mean_raw[i, j]) for j, p in enumerate(props)}
                ),
                "predicted_properties_independent": json.dumps(
                    {p: float(indep_mean_raw[j]) for j, p in enumerate(props)}
                ),
                "uncertainty": uncertainty,
                "novelty_score": novelty,
                "nearest_train_smiles": nn_smiles,
                "nearest_train_similarity": float(nn_sim),
                "scaffold": scaffold,
                "domain_score": domain,
                "synthetic_accessibility": synth,
                "final_score": final_score,
                "filter_status": "pending",
                "database_lookup_status": "not_checked",
                "decode_method": decode_method,
                "in_train": can in train_canon,
            }
        )

    generated_df = pd.DataFrame(generated_rows)
    generated_df.to_csv(out_dir / "generated_candidates.csv", index=False)

    # Filtering
    allowed_atoms = set(cfg["filtering"]["allowed_atoms"])
    seen = set()
    statuses = []
    min_heavy = int(cfg["filtering"]["min_heavy_atoms"])
    max_heavy = int(cfg["filtering"]["max_heavy_atoms"])
    dup_thr = float(cfg["generation"]["novelty_duplicate_threshold"])
    min_domain_score = float(cfg["filtering"]["min_domain_score"])
    for _, row in generated_df.iterrows():
        can = row["canonical_smiles"]
        if not isinstance(can, str) or not can:
            statuses.append("invalid_smiles")
            continue
        ok, reason = validate_basic_chemistry(can, allowed_atoms, min_heavy, max_heavy)
        if not ok:
            statuses.append(reason)
            continue
        if can in seen:
            statuses.append("duplicate_generated")
            continue
        if bool(row["in_train"]) or float(row["nearest_train_similarity"]) >= dup_thr:
            statuses.append("duplicate_train")
            continue
        if float(row["domain_score"]) < min_domain_score:
            statuses.append("low_domain_score")
            continue
        statuses.append("pass")
        seen.add(can)

    generated_df["filter_status"] = statuses
    filtered_df = generated_df[generated_df["filter_status"] == "pass"].copy()
    filtered_df.to_csv(out_dir / "filtered_candidates.csv", index=False)

    ranked_df = filtered_df.sort_values("final_score", ascending=False).copy()
    ranked_df.to_csv(out_dir / "ranked_candidates.csv", index=False)

    # Evaluation report
    validity = float((generated_df["canonical_smiles"].astype(str) != "").mean() * 100.0)
    uniqueness = (
        float(generated_df["canonical_smiles"].nunique() / max(1, len(generated_df)) * 100.0)
    )
    novelty = float((generated_df["nearest_train_similarity"] < 1.0).mean() * 100.0)
    report = []
    report.append("# Evaluation Report")
    report.append("")
    report.append(f"- run_dir: `{out_dir}`")
    report.append(f"- generated_candidates: {len(generated_df)}")
    report.append(f"- filtered_candidates: {len(filtered_df)}")
    report.append(f"- ranked_candidates: {len(ranked_df)}")
    report.append(f"- validity_percent: {validity:.2f}")
    report.append(f"- uniqueness_percent: {uniqueness:.2f}")
    report.append(f"- novelty_percent: {novelty:.2f}")
    report.append("")
    report.append("## Top Candidates")
    top = ranked_df.head(int(cfg["generation"]["top_k"]))
    if len(top) == 0:
        report.append("No candidates passed filters.")
    else:
        for _, r in top.head(20).iterrows():
            report.append(
                f"- {r['candidate_id']}: score={r['final_score']:.4f}, smiles={r['canonical_smiles']}"
            )
    (out_dir / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")

    # Final run summary
    summary = {
        "run_dir": str(out_dir),
        "device": str(device),
        "generated_candidates": int(len(generated_df)),
        "filtered_candidates": int(len(filtered_df)),
        "ranked_candidates": int(len(ranked_df)),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[done] experiment completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
