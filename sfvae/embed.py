from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from .model import SFVAE
from .submanifold import gaussian_to_points
from ..gaussians.io import GaussianSet, load_ply
from ..utils import default_device


def load_sfvae_checkpoint(ckpt: Path, device=None) -> SFVAE:
    device = device or default_device()
    data = torch.load(ckpt, map_location=device)
    cfg = data.get("config", {})
    model = SFVAE(latent_dim=cfg.get("latent_dim", 32), n_grid=cfg.get("n_grid", 12), deterministic=True).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    return model


def embed_gaussians(gs: GaussianSet, ckpt: Path, n_grid: int = 12, device=None) -> Tuple[np.ndarray, np.ndarray]:
    device = device or default_device()
    model = load_sfvae_checkpoint(ckpt, device=device)
    with torch.no_grad():
        pts, _ = gaussian_to_points(gs.to(device), n_grid=n_grid, device=device)
        emb = model.encode(pts)
    return emb.cpu().numpy(), gs.means.cpu().numpy()


def embed_from_path(path: Path, ckpt: Path, n_grid: int = 12, device=None) -> Path:
    device = device or default_device()
    gs = load_ply(path)
    emb, means = embed_gaussians(gs, ckpt=ckpt, n_grid=n_grid, device=device)
    out_path = path.with_suffix(".npz")
    np.savez(out_path, embeddings=emb, means=means)
    return out_path


__all__ = ["embed_gaussians", "embed_from_path", "load_sfvae_checkpoint"]
