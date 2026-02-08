from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

from .submanifold import spherical_grid


def mlp(sizes, activation=nn.ReLU, final_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else final_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class PointNetEncoder(nn.Module):
    def __init__(self, feat_dim: int = 7, hidden: int = 256, latent_dim: int = 32):
        super().__init__()
        self.mlp1 = mlp([feat_dim, 64, 128, hidden], activation=nn.ReLU)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,P,F)
        h = self.mlp1(x)
        h = torch.max(h, dim=1).values
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden: int = 256, n_grid: int = 12):
        super().__init__()
        self.n_grid = n_grid
        self.g_c = mlp([latent_dim + 3, hidden, hidden, 128, 3], activation=nn.ReLU, final_activation=None)
        self.g_f = mlp([latent_dim + 3, hidden, hidden, 64, 3], activation=nn.ReLU, final_activation=None)

    def forward(self, z: torch.Tensor, dirs: torch.Tensor = None):
        # z: (B,D)
        if dirs is None:
            dirs = spherical_grid(self.n_grid, device=z.device, dtype=z.dtype)
        dirs = dirs[None, :, :]  # (1,P,3)
        B = z.shape[0]
        P = dirs.shape[1]
        z_rep = z[:, None, :].expand(-1, P, -1)
        inp = torch.cat([dirs.expand(B, -1, -1), z_rep], dim=-1)
        coords = self.g_c(inp)
        colors = torch.sigmoid(self.g_f(torch.cat([coords, z_rep], dim=-1)))
        opacity = torch.ones((B, P, 1), device=z.device, dtype=z.dtype)
        return torch.cat([coords, colors, opacity], dim=-1)


class SFVAE(nn.Module):
    def __init__(self, latent_dim: int = 32, n_grid: int = 12, deterministic: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_grid = n_grid
        self.deterministic = deterministic
        self.encoder = PointNetEncoder(feat_dim=7, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, n_grid=n_grid)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        if self.deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, logvar

    def decode(self, z: torch.Tensor, dirs: torch.Tensor = None) -> torch.Tensor:
        return self.decoder(z, dirs=dirs)


__all__ = ["SFVAE"]
