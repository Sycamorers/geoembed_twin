from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .io import GaussianSet
from ..utils import set_deterministic


@dataclass
class SyntheticConfig:
    num_surface: int = 2500
    num_floaters: int = 300
    cube_size: float = 1.2
    sphere_radius: float = 0.7
    opacity_range: Tuple[float, float] = (0.5, 0.95)
    floater_opacity: Tuple[float, float] = (0.3, 0.7)
    lmax: int = 3


def _sample_on_cube(n: int, size: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return positions and normals on cube surface."""
    faces = torch.randint(0, 6, (n,), device=device)
    coords = torch.rand((n, 2), device=device) * 2 - 1
    pos = torch.zeros((n, 3), device=device)
    normal = torch.zeros_like(pos)
    half = size / 2
    for f in range(6):
        mask = faces == f
        if not mask.any():
            continue
        uv = coords[mask]
        nval = None
        if f == 0:  # +x
            pos[mask] = torch.stack([torch.full_like(uv[:, 0], half), uv[:, 0] * half, uv[:, 1] * half], dim=1)
            nval = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=pos.dtype)
        elif f == 1:  # -x
            pos[mask] = torch.stack([-torch.full_like(uv[:, 0], half), uv[:, 0] * half, uv[:, 1] * half], dim=1)
            nval = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=pos.dtype)
        elif f == 2:  # +y
            pos[mask] = torch.stack([uv[:, 0] * half, torch.full_like(uv[:, 0], half), uv[:, 1] * half], dim=1)
            nval = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=pos.dtype)
        elif f == 3:  # -y
            pos[mask] = torch.stack([uv[:, 0] * half, -torch.full_like(uv[:, 0], half), uv[:, 1] * half], dim=1)
            nval = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=pos.dtype)
        elif f == 4:  # +z
            pos[mask] = torch.stack([uv[:, 0] * half, uv[:, 1] * half, torch.full_like(uv[:, 0], half)], dim=1)
            nval = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=pos.dtype)
        else:  # -z
            pos[mask] = torch.stack([uv[:, 0] * half, uv[:, 1] * half, -torch.full_like(uv[:, 0], half)], dim=1)
            nval = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=pos.dtype)
        if nval is not None:
            normal[mask] = nval
    return pos, normal


def _sample_on_sphere(n: int, radius: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    u = torch.rand(n, device=device)
    v = torch.rand(n, device=device)
    theta = 2 * math.pi * u
    phi = torch.acos(2 * v - 1)
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    normal = torch.stack([x, y, z], dim=1)
    pos = normal * radius
    return pos, normal


def _colors_from_normals(normals: torch.Tensor) -> torch.Tensor:
    return (normals + 1) / 2  # simple mapping to RGB in [0,1]


def generate_synthetic_scene(cfg: SyntheticConfig = SyntheticConfig(), device=None) -> GaussianSet:
    device = device or torch.device("cpu")
    set_deterministic(123)

    n_surf_cube = cfg.num_surface // 2
    n_surf_sphere = cfg.num_surface - n_surf_cube
    cube_pos, cube_n = _sample_on_cube(n_surf_cube, cfg.cube_size, device)
    sphere_pos, sphere_n = _sample_on_sphere(n_surf_sphere, cfg.sphere_radius, device)

    pos = torch.cat([cube_pos, sphere_pos], dim=0)
    normals = torch.cat([cube_n, sphere_n], dim=0)

    colors = _colors_from_normals(normals)

    # Gaussian parameters
    scales = torch.exp(torch.rand((cfg.num_surface, 3), device=device) * -2.5 - 2.0)  # anisotropic small
    # Slight flattening along normals
    scales = scales * 0.5
    opacity = torch.rand((cfg.num_surface, 1), device=device) * (cfg.opacity_range[1] - cfg.opacity_range[0]) + cfg.opacity_range[0]

    # floaters
    floaters_pos = (torch.rand((cfg.num_floaters, 3), device=device) - 0.5) * (cfg.cube_size * 1.5)
    floaters_scales = torch.exp(torch.rand((cfg.num_floaters, 3), device=device) * -1.5 - 1.5)
    floaters_opacity = torch.rand((cfg.num_floaters, 1), device=device) * (cfg.floater_opacity[1] - cfg.floater_opacity[0]) + cfg.floater_opacity[0]
    floaters_colors = torch.rand((cfg.num_floaters, 3), device=device) * 0.8 + 0.1

    means = torch.cat([pos, floaters_pos], dim=0)
    scales = torch.cat([scales, floaters_scales], dim=0)
    opacity = torch.cat([opacity, floaters_opacity], dim=0)
    colors = torch.cat([colors, floaters_colors], dim=0)

    # random quaternions
    q = torch.randn((means.shape[0], 4), device=device)
    q = torch.nn.functional.normalize(q, dim=1)
    # ensure w>=0
    q[q[:, 0] < 0] *= -1

    # SH coeffs: set DC to color, others small noise
    K = (cfg.lmax + 1) ** 2
    sh = torch.randn((means.shape[0], 3, K), device=device) * 0.02
    sh[:, :, 0] = colors

    return GaussianSet(means=means, scales=scales, quats=q, opacity=opacity, sh=sh)


def apply_changes(scene: GaussianSet) -> Tuple[GaussianSet, GaussianSet, Dict[str, torch.Tensor]]:
    before = scene.clone()
    after = scene.clone()

    # Define three regions
    remove_mask = (before.means[:, 0] > -0.3) & (before.means[:, 0] < 0.2) & (before.means[:, 1] > 0) & (before.means[:, 1] < 0.5)
    move_mask = (before.means[:, 0] < -0.4) & (before.means[:, 1] < -0.1)

    # Remove cluster A
    for tensor_name in ["means", "scales", "quats", "covs", "opacity", "sh"]:
        t = getattr(after, tensor_name)
        if t is not None:
            setattr(after, tensor_name, t[~remove_mask])

    # Move cluster B
    translation = torch.tensor([0.35, 0.0, 0.1], device=after.means.device)
    kept_move_mask = move_mask[~remove_mask]
    if kept_move_mask.any():
        after.means[kept_move_mask] += translation

    # Add new cluster C
    num_new = max(80, scene.N // 20)
    base_len_before_add = after.means.shape[0]
    new_means = torch.randn((num_new, 3), device=after.means.device) * 0.05 + torch.tensor([0.0, -0.2, 0.5], device=after.means.device)
    new_scales = torch.exp(torch.randn((num_new, 3), device=after.means.device) * -1.5 - 2.5)
    new_quats = torch.nn.functional.normalize(torch.randn((num_new, 4), device=after.means.device), dim=1)
    new_quats[new_quats[:, 0] < 0] *= -1
    new_opacity = torch.rand((num_new, 1), device=after.means.device) * 0.3 + 0.6
    new_sh = torch.randn((num_new, 3, scene.sh.shape[2] if scene.sh is not None else 16), device=after.means.device) * 0.02
    new_sh[:, :, 0] = torch.rand((num_new, 3), device=after.means.device) * 0.8 + 0.2

    for name, new_val in [("means", new_means), ("scales", new_scales), ("quats", new_quats), ("opacity", new_opacity), ("sh", new_sh)]:
        old = getattr(after, name)
        if old is not None:
            setattr(after, name, torch.cat([old, new_val], dim=0))

    final_len = after.means.shape[0]
    added_mask = torch.zeros(final_len, dtype=torch.bool, device=after.means.device)
    added_mask[base_len_before_add:] = True

    labels = {
        "removed_mask": remove_mask,
        "moved_mask": move_mask & ~remove_mask,
        "added_mask": added_mask,
    }
    return before, after, labels


__all__ = ["SyntheticConfig", "generate_synthetic_scene", "apply_changes"]
