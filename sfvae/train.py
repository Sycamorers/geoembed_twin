from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple

import torch
from torch import optim
from torch.cuda import amp

from .model import SFVAE
from .submanifold import gaussian_to_points
from ..gaussians.io import GaussianSet
from ..utils import ensure_dir, default_device, set_deterministic, get_repo_root
from .loss import total_vae_loss


def _k_to_l(k: int) -> int:
    return int(math.floor(math.sqrt(k)))


def random_gaussian_batch(batch_size: int, lmax: int = 3, smin: float = -8.0, smax: float = 0.0, beta_sh: float = 4.0, omin: float = -5.0, omax: float = 10.0, device=None) -> GaussianSet:
    device = device or default_device()
    means = torch.zeros((batch_size, 3), device=device)
    q = torch.randn((batch_size, 4), device=device)
    q = torch.nn.functional.normalize(q, dim=1)
    q[q[:, 0] < 0] *= -1

    log_scales = torch.rand((batch_size, 3), device=device) * (smax - smin) + smin
    scales = torch.exp(log_scales)

    K = (lmax + 1) ** 2
    sh = torch.zeros((batch_size, 3, K), device=device)
    for k in range(K):
        l = _k_to_l(k)
        std = beta_sh ** (-2 * l)
        sh[:, :, k] = torch.randn((batch_size, 3), device=device) * std
    opacity_logit = torch.rand((batch_size, 1), device=device) * (omax - omin) + omin
    opacity = torch.sigmoid(opacity_logit)

    return GaussianSet(means=means, scales=scales, quats=q, opacity=opacity, sh=sh)


def train_sfvae(
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_grid: int = 12,
    latent_dim: int = 32,
    num_batches: int = 200,
    fast: bool = False,
    output_path: Path | None = None,
    device=None,
) -> Path:
    device = device or default_device()
    set_deterministic(42)
    if fast:
        epochs = max(epochs // 2, 5)
        num_batches = max(num_batches // 2, 50)
        batch_size = min(batch_size, 96)
    if not torch.cuda.is_available():
        epochs = min(epochs, 2)
        num_batches = min(num_batches, 10)
        batch_size = min(batch_size, 32)

    model = SFVAE(latent_dim=latent_dim, n_grid=n_grid, deterministic=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(epochs):
        running_loss = 0.0
        for step in range(num_batches):
            gs = random_gaussian_batch(batch_size=batch_size, lmax=3, device=device)
            pts, _ = gaussian_to_points(gs, n_grid=n_grid, device=device)
            optimizer.zero_grad()
            with amp.autocast(enabled=torch.cuda.is_available()):
                recon, mu, logvar = model(pts)
                loss, rec, kl = total_vae_loss(recon, pts, mu, logvar)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
        avg = running_loss / num_batches
        print(f"[SFVAE] epoch {epoch+1}/{epochs} loss={avg:.4f}")

    ckpt_dir = ensure_dir((output_path or get_repo_root() / "geoembed_twin" / "outputs" / "checkpoints"))
    ckpt_path = ckpt_dir / "sfvae_fallback.pt"
    torch.save({"model": model.state_dict(), "config": {"latent_dim": latent_dim, "n_grid": n_grid}}, ckpt_path)
    print(f"Saved fallback SF-VAE checkpoint to {ckpt_path}")
    return ckpt_path


__all__ = ["train_sfvae", "random_gaussian_batch"]
