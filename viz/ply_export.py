from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch

try:
    from plyfile import PlyData, PlyElement  # type: ignore
    _HAS_PLYFILE = True
except Exception:
    _HAS_PLYFILE = False

from ..gaussians.io import GaussianSet


def export_labeled_points(gs: GaussianSet, colors: torch.Tensor, path: Path, labels: Optional[np.ndarray] = None) -> None:
    N = gs.N
    pts = gs.means.cpu().numpy()
    cols = colors.clamp(0, 1).cpu().numpy()
    cols_uint = (cols * 255).astype(np.uint8)
    if _HAS_PLYFILE:
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
    else:
        # Minimal ASCII writer
        fields = ["x", "y", "z", "red", "green", "blue"]
        if labels is not None:
            fields.append("label")
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            if labels is not None:
                f.write("property int label\n")
            f.write("end_header\n")
            for i in range(N):
                line = [pts[i, 0], pts[i, 1], pts[i, 2], cols_uint[i, 0], cols_uint[i, 1], cols_uint[i, 2]]
                if labels is not None:
                    line.append(int(labels[i]))
                f.write(" ".join(map(str, line)) + "\n")


__all__ = ["export_labeled_points"]
