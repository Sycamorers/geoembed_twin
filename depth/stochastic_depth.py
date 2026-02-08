from __future__ import annotations
from typing import Dict, List, Tuple
import math

import torch

from ..gaussians.io import GaussianSet, quat_to_rotmat
from .camera import Camera
from ..utils import default_device


def _inverse_cov(gs: GaussianSet, device=None) -> torch.Tensor:
    device = device or gs.means.device
    if gs.covs is not None:
        return torch.linalg.inv(gs.covs.to(device) + torch.eye(3, device=device) * 1e-6)
    if gs.scales is None or gs.quats is None:
        raise ValueError("GaussianSet missing scales/quats for inverse covariance")
    R = quat_to_rotmat(gs.quats.to(device))
    inv_scale = torch.diag_embed(1.0 / (gs.scales.to(device) ** 2 + 1e-9))
    return R @ inv_scale @ R.transpose(-1, -2)


def _vacancy_log(t: torch.Tensor, dirs: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
    # t: (B,K) scalar per ray per gaussian
    # dirs: (B,3)
    # delta: (B,K,3) where delta = mu - origin
    # A: (B,K,3,3)
    # opacity: (B,K,1)
    dirs_exp = dirs[:, None, :]  # (B,1,3)
    r = dirs_exp * t[:, :, None] - delta  # (B,K,3)
    Ar = torch.einsum('bkj,bkji->bki', r, A)
    mahal = torch.einsum('bki,bki->bk', r, Ar)
    G = opacity.squeeze(-1) * torch.exp(-mahal)
    G = torch.clamp(G, max=0.999999)
    logv = 0.5 * torch.log(torch.clamp(1.0 - G, min=1e-9))
    return logv


def _logT(t: torch.Tensor, dirs: torch.Tensor, delta_sel: torch.Tensor, A_sel: torch.Tensor, opacity_sel: torch.Tensor, t_star: torch.Tensor) -> torch.Tensor:
    logv_t = _vacancy_log(t, dirs, delta_sel, A_sel, opacity_sel)
    logv_star = _vacancy_log(t_star, dirs, delta_sel, A_sel, opacity_sel)
    logTi = torch.where(t[:, :, None] <= t_star[:, :, None], logv_t[:, :, None], 2 * logv_star[:, :, None] - logv_t[:, :, None]).squeeze(-1)
    return torch.sum(logTi, dim=1)


def render_depth_median(
    gs: GaussianSet,
    cameras: List[Camera],
    image_size: Tuple[int, int],
    topk: int = 32,
    near: float = 0.1,
    far: float = 6.0,
    chunk: int = 2048,
    return_expected: bool = True,
    device=None,
) -> List[Dict[str, torch.Tensor]]:
    device = device or default_device()
    results = []

    means = gs.means.to(device)
    opacity = gs.opacity.to(device) if gs.opacity is not None else torch.ones((gs.N, 1), device=device)
    A = _inverse_cov(gs, device=device)  # (N,3,3)

    for cam in cameras:
        cam = cam.to(device)
        origin, dirs = cam.generate_rays(device=device)  # origin (3,), dirs (H,W,3)
        H, W = image_size
        dirs = dirs.reshape(-1, 3)
        M = dirs.shape[0]

        delta = means - origin[None, :]  # (N,3)

        depth_med = torch.full((M,), -1.0, device=device)
        depth_exp = torch.full((M,), -1.0, device=device)
        valid_mask = torch.zeros((M,), dtype=torch.bool, device=device)

        for start in range(0, M, chunk):
            end = min(start + chunk, M)
            d_chunk = dirs[start:end]  # (B,3)
            B = d_chunk.shape[0]

            # distance of gaussian centers to each ray
            dot = torch.einsum('ni,bi->nb', delta, d_chunk)  # (N,B)
            proj = dot[:, :, None] * d_chunk[None, :, :]  # (N,B,3)
            ortho = delta[:, None, :] - proj  # (N,B,3)
            dist2 = (ortho * ortho).sum(dim=-1).transpose(0, 1)  # (B,N)
            k = min(topk, gs.N)
            dist_vals, idx = torch.topk(dist2, k=k, dim=1, largest=False)

            A_sel = A[idx]  # (B,k,3,3)
            delta_sel = delta[idx]  # (B,k,3)
            opacity_sel = opacity[idx]  # (B,k,1)

            denom = torch.einsum('bi,bkij,bj->bk', d_chunk, A_sel, d_chunk)
            num = torch.einsum('bi,bkij,bkj->bk', d_chunk, A_sel, delta_sel)
            t_star = num / (denom + 1e-8)
            t_star = torch.clamp(t_star, min=1e-4)

            target = math_log_half = torch.log(torch.tensor(0.5, device=device))

            t_low = torch.full((B, k), near, device=device)
            t_high = torch.full((B, k), far, device=device)
            # Validity check using summed logT at bounds
            logT_low = _logT(t_low, d_chunk, delta_sel, A_sel, opacity_sel, t_star)
            logT_high = _logT(t_high, d_chunk, delta_sel, A_sel, opacity_sel, t_star)
            valid = (logT_low > math_log_half) & (logT_high < math_log_half)
            if valid.ndim == 2:
                valid_any = valid.any(dim=1)
            else:
                valid_any = valid

            t_min = torch.full((B,), near, device=device)
            t_max = torch.full((B,), far, device=device)
            # Binary search on valid rays
            for _ in range(20):
                t_mid = (t_min + t_max) * 0.5
                t_mid_expand = t_mid[:, None].expand(-1, k)
                logT_mid = _logT(t_mid_expand, d_chunk, delta_sel, A_sel, opacity_sel, t_star)
                go_deeper = logT_mid > math_log_half
                t_min = torch.where(go_deeper, t_mid, t_min)
                t_max = torch.where(go_deeper, t_max, t_mid)
            depth = t_max
            depth_med[start:end] = depth
            valid_mask[start:end] = valid_any

            if return_expected:
                w = opacity_sel.squeeze(-1)
                exp_depth = (t_star * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)
                depth_exp[start:end] = exp_depth

        depth_med = depth_med.reshape(H, W)
        depth_exp_img = depth_exp.reshape(H, W)
        valid_img = valid_mask.reshape(H, W)
        results.append({
            "depth_med": depth_med,
            "depth_expected": depth_exp_img,
            "mask": valid_img,
        })
    return results


__all__ = ["render_depth_median"]
