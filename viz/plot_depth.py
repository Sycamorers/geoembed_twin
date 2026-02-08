from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

try:
    import imageio.v2 as imageio  # type: ignore

    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False

try:
    from PIL import Image

    _HAS_PIL = True
except Exception:
    Image = None
    _HAS_PIL = False


def _write_png(arr: np.ndarray, path: Path) -> None:
    if _HAS_IMAGEIO:
        imageio.imwrite(path, arr)
    elif _HAS_PIL:
        Image.fromarray(arr).save(path)
    else:  # pragma: no cover - requires missing deps
        raise ImportError("Install imageio or pillow to write PNGs.")


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_depth_png(depth: torch.Tensor, mask: torch.Tensor, path: Path, cmap: str = "viridis") -> None:
    np.Inf = np.inf  # ensure present for numpy>=2.0
    d = np.ascontiguousarray(_to_numpy(depth), dtype=np.float32)
    m = np.asarray(_to_numpy(mask), dtype=bool)
    valid = d[m]
    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(valid, [2, 98])
        if vmax <= vmin:
            vmax = vmin + 1e-3
    d_norm = (np.clip(d, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)
    # simple colormap: viridis-like approximation using matplotlib values avoided; use grayscale fallback
    img = (d_norm * 255).astype(np.uint8)
    _write_png(img, path)


def save_residual_png(residual: torch.Tensor, path: Path, cmap: str = "magma") -> None:
    np.Inf = np.inf  # ensure present for numpy>=2.0
    r = np.ascontiguousarray(_to_numpy(residual), dtype=np.float32)
    vmax = np.percentile(r, 98) if r.size > 0 else 1.0
    vmax = vmax if vmax > 1e-6 else 1.0
    r_norm = np.clip(r / vmax, 0, 1)
    img = (r_norm * 255).astype(np.uint8)
    _write_png(img, path)


__all__ = ["save_depth_png", "save_residual_png"]
