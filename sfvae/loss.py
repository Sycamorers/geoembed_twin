from __future__ import annotations
import torch
from typing import Tuple

try:
    from geomloss import SamplesLoss  # type: ignore
    _HAS_GEOMLOSS = True
except Exception:
    _HAS_GEOMLOSS = False


def chamfer_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: (B,P,D)
    diff = a[:, :, None, :] - b[:, None, :, :]
    dist2 = (diff * diff).sum(dim=-1)
    min_a = dist2.min(dim=2).values
    min_b = dist2.min(dim=1).values
    return (min_a.mean(dim=1) + min_b.mean(dim=1)).mean()


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, lambda_color: float = 0.1) -> torch.Tensor:
    # Combine xyz + color weighting
    xyz_pred, rgb_pred = pred[..., :3], pred[..., 3:6]
    xyz_tgt, rgb_tgt = target[..., :3], target[..., 3:6]
    if _HAS_GEOMLOSS:
        feat_pred = torch.cat([xyz_pred, rgb_pred * lambda_color], dim=-1)
        feat_tgt = torch.cat([xyz_tgt, rgb_tgt * lambda_color], dim=-1)
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.01)
        return loss_fn(feat_pred, feat_tgt)
    else:
        dist_xyz = chamfer_loss(xyz_pred, xyz_tgt)
        dist_rgb = chamfer_loss(rgb_pred, rgb_tgt)
        return dist_xyz + lambda_color * dist_rgb


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def total_vae_loss(pred: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1e-3, lambda_color: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rec = reconstruction_loss(pred, target, lambda_color=lambda_color)
    kl = kl_divergence(mu, logvar)
    loss = rec + beta * kl
    return loss, rec, kl


__all__ = ["reconstruction_loss", "kl_divergence", "total_vae_loss"]
