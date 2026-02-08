from __future__ import annotations
from typing import List

import torch

from ..depth.camera import Camera
from ..gaussians.io import GaussianSet


def depth_residual_map(depth_a: torch.Tensor, depth_b: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor, threshold: float = 0.05):
    valid = mask_a & mask_b & (depth_a > 0) & (depth_b > 0)
    residual = torch.abs(depth_a - depth_b)
    change_mask = valid & (residual > threshold)
    return residual, change_mask


def backproject(camera: Camera, depth: torch.Tensor) -> torch.Tensor:
    origin, dirs = camera.generate_rays(device=depth.device)
    pts = origin[None, None, :] + dirs * depth[..., None]
    return pts


def mark_gaussians_changed(gs: GaussianSet, cameras: List[Camera], change_masks: List[torch.Tensor], depths: List[torch.Tensor], radius: float = 0.12) -> torch.Tensor:
    means = gs.means.to(depths[0].device)
    changed = torch.zeros((means.shape[0],), dtype=torch.bool, device=means.device)
    radius2 = radius * radius
    for cam, mask, depth in zip(cameras, change_masks, depths):
        pts = backproject(cam.to(means.device), depth)
        pts_flat = pts[mask]
        if pts_flat.numel() == 0:
            continue
        dist2 = torch.cdist(means, pts_flat)
        close = (dist2 < radius2).any(dim=1)
        changed |= close
    return changed


__all__ = ["depth_residual_map", "backproject", "mark_gaussians_changed"]
