from __future__ import annotations
from typing import List

import torch

from ..gaussians.io import GaussianSet
from ..depth.camera import Camera


def project_point(cam: Camera, pts: torch.Tensor) -> torch.Tensor:
    # pts: (...,3)
    cam = cam.to(pts.device)
    pts_cam = pts @ cam.R.T + cam.t
    z = pts_cam[..., 2]
    u = cam.fx * (pts_cam[..., 0] / z) + cam.cx
    v = cam.fy * (pts_cam[..., 1] / z) + cam.cy
    return torch.stack([u, v, z], dim=-1)


def detect_floaters(gs: GaussianSet, cameras: List[Camera], depth_maps: List[torch.Tensor], margin: float = 0.05, min_votes: int = 2) -> torch.Tensor:
    means = gs.means
    device = depth_maps[0].device
    means = means.to(device)
    votes = torch.zeros((means.shape[0],), device=device)
    for cam, depth in zip(cameras, depth_maps):
        proj = project_point(cam, means)
        u = proj[:, 0].long()
        v = proj[:, 1].long()
        z = proj[:, 2]
        H, W = depth.shape
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        depth_samples = depth[v[valid_idx], u[valid_idx]]
        floater_mask = z[valid_idx] + margin < depth_samples
        votes[valid_idx[floater_mask]] += 1
    return votes >= min_votes


__all__ = ["detect_floaters", "project_point"]
