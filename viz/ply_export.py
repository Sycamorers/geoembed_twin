from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from plyfile import PlyData, PlyElement
import torch

from ..gaussians.io import GaussianSet


def export_labeled_points(gs: GaussianSet, colors: torch.Tensor, path: Path, labels: Optional[np.ndarray] = None) -> None:
    N = gs.N
    pts = gs.means.cpu().numpy()
    cols = colors.clamp(0, 1).cpu().numpy()
    cols_uint = (cols * 255).astype(np.uint8)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data_list = [
        (pts[i, 0], pts[i, 1], pts[i, 2], cols_uint[i, 0], cols_uint[i, 1], cols_uint[i, 2]) for i in range(N)
    ]
    if labels is not None:
        dtype.append(("label", "i4"))
        data_list = [t + (int(labels[i]),) for i, t in enumerate(data_list)]
    arr = np.array(data_list, dtype=dtype)
    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(path))


__all__ = ["export_labeled_points"]
