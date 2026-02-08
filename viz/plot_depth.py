from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_depth_png(depth: torch.Tensor, mask: torch.Tensor, path: Path, cmap: str = "viridis") -> None:
    d = _to_numpy(depth)
    m = _to_numpy(mask)
    valid = d[m]
    if valid.size == 0:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = np.percentile(valid, [2, 98])
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_residual_png(residual: torch.Tensor, path: Path, cmap: str = "magma") -> None:
    r = _to_numpy(residual)
    vmax = np.percentile(r, 98) if r.size > 0 else 1.0
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(r, cmap=cmap, vmin=0, vmax=vmax)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


__all__ = ["save_depth_png", "save_residual_png"]
