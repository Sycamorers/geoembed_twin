from __future__ import annotations
import math
from functools import lru_cache
from typing import Tuple

import torch

from .sh import sh_basis
from ..gaussians.io import GaussianSet, quat_to_rotmat


def spherical_grid(n: int = 12, device=None, dtype=None) -> torch.Tensor:
    """Deterministic grid on the sphere (longitude/latitude mesh)."""
    theta = torch.linspace(0, 2 * math.pi, n, device=device, dtype=dtype, endpoint=False)
    phi = torch.linspace(1e-3, math.pi - 1e-3, n, device=device, dtype=dtype)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    dirs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    return dirs


@lru_cache(maxsize=4)
def _cached_dirs(n: int) -> torch.Tensor:
    return spherical_grid(n)


def gaussian_to_points(gs: GaussianSet, n_grid: int = 12, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert GaussianSet to submanifold point clouds.

    Returns (points, dirs):
      points: (N, P, 7) with xyzrgbα
      dirs: (P,3) unit vectors used
    """
    device = device or gs.means.device
    dirs = _cached_dirs(n_grid).to(device=device, dtype=gs.means.dtype)
    P = dirs.shape[0]
    N = gs.N

    # Build sqrt covariance matrices
    if gs.covs is not None:
        cov = gs.covs.to(device)
        try:
            sqrt_cov = torch.linalg.cholesky(cov)
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = torch.clamp(eigvals, min=1e-5)
            sqrt_cov = eigvecs @ torch.diag_embed(torch.sqrt(eigvals)) @ eigvecs.transpose(-1, -2)
    else:
        if gs.scales is None or gs.quats is None:
            raise ValueError("GaussianSet missing scales/quaternions for sqrt covariance")
        R = quat_to_rotmat(gs.quats.to(device))
        sqrt_cov = R @ torch.diag_embed(gs.scales.to(device))

    # points: mu + sqrt_cov @ dir
    dirs_exp = dirs[None, :, :, None]  # (1,P,3,1)
    sqrt_cov_exp = sqrt_cov[:, None, :, :]  # (N,1,3,3)
    offset = torch.matmul(sqrt_cov_exp, dirs_exp).squeeze(-1)  # (N,P,3)
    points_xyz = gs.means.to(device)[:, None, :] + offset

    # Colors via SH if available else gray
    if gs.sh is not None:
        basis = sh_basis(dirs, lmax=int(round((gs.sh.shape[2] ** 0.5) - 1)))  # (P,K)
        sh = gs.sh.to(device)  # (N,3,K)
        colors = torch.einsum('nck,pk->npc', sh, basis)
        colors = torch.clamp(torch.sigmoid(colors), 0.0, 1.0)
    else:
        colors = torch.full((N, P, 3), 0.5, device=device, dtype=gs.means.dtype)

    opacity = gs.opacity.to(device) if gs.opacity is not None else torch.ones((N, 1), device=device, dtype=gs.means.dtype)
    opacity_rep = opacity[:, None, :].expand(-1, P, -1)
    points = torch.cat([points_xyz, colors, opacity_rep], dim=-1)
    return points, dirs


__all__ = ["gaussian_to_points", "spherical_grid"]
