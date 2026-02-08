from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement


@dataclass
class GaussianSet:
    means: torch.Tensor  # (N,3)
    scales: Optional[torch.Tensor] = None  # (N,3)
    quats: Optional[torch.Tensor] = None  # (N,4) wxyz
    covs: Optional[torch.Tensor] = None  # (N,3,3)
    opacity: Optional[torch.Tensor] = None  # (N,1)
    sh: Optional[torch.Tensor] = None  # (N,3,K)

    def to(self, device: torch.device) -> "GaussianSet":
        return GaussianSet(
            means=self.means.to(device),
            scales=None if self.scales is None else self.scales.to(device),
            quats=None if self.quats is None else self.quats.to(device),
            covs=None if self.covs is None else self.covs.to(device),
            opacity=None if self.opacity is None else self.opacity.to(device),
            sh=None if self.sh is None else self.sh.to(device),
        )

    @property
    def N(self) -> int:
        return int(self.means.shape[0])

    def clone(self) -> "GaussianSet":
        return GaussianSet(
            means=self.means.clone(),
            scales=None if self.scales is None else self.scales.clone(),
            quats=None if self.quats is None else self.quats.clone(),
            covs=None if self.covs is None else self.covs.clone(),
            opacity=None if self.opacity is None else self.opacity.clone(),
            sh=None if self.sh is None else self.sh.clone(),
        )

    def canonicalize_quaternions(self) -> None:
        if self.quats is None:
            return
        w = self.quats[:, 0:1]
        mask = w < 0
        self.quats[mask] *= -1

    def covariance_matrices(self) -> torch.Tensor:
        if self.covs is not None:
            return self.covs
        if self.scales is None or self.quats is None:
            raise ValueError("Cannot build covariance without scales and quaternions")
        R = quat_to_rotmat(self.quats)  # (N,3,3)
        scale_mat = torch.diag_embed(self.scales ** 2)
        cov = R @ scale_mat @ R.transpose(-1, -2)
        return cov


# ----------------------------- Quaternion utils ----------------------------

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert normalized quaternion wxyz to rotation matrix."""
    q = torch.nn.functional.normalize(quat, dim=-1)
    w, x, y, z = q.unbind(-1)
    B = q.shape[0]
    R = torch.empty((B, 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ----------------------------- PLY IO --------------------------------------

_PROP_ALIASES = {
    "x": ["x"],
    "y": ["y"],
    "z": ["z"],
    "opacity": ["opacity", "alpha"],
}


_SH_DC_RE = re.compile(r"f_dc_(\d+)")
_SH_REST_RE = re.compile(r"f_rest_(\d+)")


def _extract_property(data: PlyData, aliases) -> Optional[np.ndarray]:
    props = data.elements[0].data.dtype.names
    for alias in aliases:
        if alias in props:
            return np.asarray(data.elements[0].data[alias])
    return None


def _extract_prefixed(data: PlyData, prefix: str) -> Dict[int, np.ndarray]:
    props = data.elements[0].data.dtype.names
    out: Dict[int, np.ndarray] = {}
    for p in props:
        if p.startswith(prefix):
            try:
                idx = int(p[len(prefix):])
            except ValueError:
                continue
            out[idx] = np.asarray(data.elements[0].data[p])
    return out


def _maybe_exp_scales(scales: torch.Tensor) -> torch.Tensor:
    # Heuristic: log-scales often negative and small.
    if float(scales.mean()) < 0.5 and float(scales.max()) < 5:
        return torch.exp(scales)
    return scales


def _maybe_sigmoid_opacity(opacity: torch.Tensor) -> torch.Tensor:
    if opacity.min() < 0 or opacity.max() > 1:
        return torch.sigmoid(opacity)
    return opacity.clamp(0.0, 1.0)


def load_ply(path: Path) -> GaussianSet:
    ply = PlyData.read(path)
    props = ply.elements[0].data.dtype.names
    arr = ply.elements[0].data

    def get_vec(prefixes):
        for pfx in prefixes:
            matches = _extract_prefixed(ply, pfx)
            if matches:
                ordered = [matches[i] for i in sorted(matches.keys())]
                return np.stack(ordered, axis=1)
        return None

    means = np.stack([arr[p] for p in ["x", "y", "z"]], axis=1)

    scales_np = get_vec(["scale_", "scaling_", "scale"])
    quats_np = get_vec(["rot_", "rotation_", "q", "quat_"])

    opacity_np = None
    for name in ["opacity", "alpha", "op"]:
        if name in props:
            opacity_np = arr[name]
            break
    if opacity_np is None:
        opacity_np = np.ones((means.shape[0],), dtype=np.float32)

    # SH
    sh_dc = _extract_prefixed(ply, "f_dc_")
    sh_rest = _extract_prefixed(ply, "f_rest_")
    sh = None
    if sh_dc:
        dc = np.stack([sh_dc[i] for i in sorted(sh_dc.keys())], axis=1)  # (N,3)
        if sh_rest:
            rest = np.stack([sh_rest[i] for i in sorted(sh_rest.keys())], axis=1)
            # rest is flattened; reshape to (N, K-1, 3)
            K_minus1 = rest.shape[1] // 3
            rest = rest.reshape(means.shape[0], K_minus1, 3)
            coeffs = np.concatenate([dc[:, None, :], rest], axis=1)  # (N,K,3)
        else:
            coeffs = dc[:, None, :]
        coeffs = np.transpose(coeffs, (0, 2, 1))  # (N,3,K)
        sh = coeffs

    # Covariance optional
    cov_props = [p for p in props if p.startswith("cov")]
    covs = None
    if cov_props:
        # Expect cov_xx, cov_xy, ...
        def has_field(name: str) -> bool:
            return name in arr.dtype.names
        if has_field("cov_xx") and has_field("cov_yy") and has_field("cov_zz"):
            covs = np.zeros((means.shape[0], 3, 3), dtype=np.float32)
            covs[:, 0, 0] = arr["cov_xx"]
            covs[:, 1, 1] = arr["cov_yy"]
            covs[:, 2, 2] = arr["cov_zz"]
            zero_fallback = np.zeros_like(arr["cov_xx"])
            covs[:, 0, 1] = covs[:, 1, 0] = arr["cov_xy"] if has_field("cov_xy") else zero_fallback
            covs[:, 0, 2] = covs[:, 2, 0] = arr["cov_xz"] if has_field("cov_xz") else zero_fallback
            covs[:, 1, 2] = covs[:, 2, 1] = arr["cov_yz"] if has_field("cov_yz") else zero_fallback

    t = torch.from_numpy
    means_t = t(means).float()
    scales_t = t(scales_np).float() if scales_np is not None else None
    if scales_t is not None:
        scales_t = _maybe_exp_scales(scales_t)
    quats_t = t(quats_np).float() if quats_np is not None else None
    if quats_t is not None and quats_t.shape[1] == 4:
        # Ensure w is first; if last, roll.
        if np.abs(quats_np[:, 0]).mean() < np.abs(quats_np[:, -1]).mean():
            quats_t = torch.roll(quats_t, shifts=1, dims=1)
    covs_t = t(covs).float() if covs is not None else None
    opacity_t = _maybe_sigmoid_opacity(t(opacity_np).float().view(-1, 1))
    sh_t = t(sh).float() if sh is not None else None

    gs = GaussianSet(means=means_t, scales=scales_t, quats=quats_t, covs=covs_t, opacity=opacity_t, sh=sh_t)
    gs.canonicalize_quaternions()
    return gs


def save_ply(gaussians: GaussianSet, path: Path, extra_fields: Optional[Dict[str, np.ndarray]] = None) -> None:
    N = gaussians.N
    means = gaussians.means.cpu().numpy()
    scales = gaussians.scales.cpu().numpy() if gaussians.scales is not None else None
    quats = gaussians.quats.cpu().numpy() if gaussians.quats is not None else None
    opacity = gaussians.opacity.cpu().numpy().reshape(-1) if gaussians.opacity is not None else np.ones((N,), dtype=np.float32)
    sh = gaussians.sh.cpu().numpy() if gaussians.sh is not None else None

    elem_data: Dict[str, np.ndarray] = {
        "x": means[:, 0],
        "y": means[:, 1],
        "z": means[:, 2],
        "opacity": opacity,
    }

    if scales is not None:
        elem_data.update({f"scale_{i}": scales[:, i] for i in range(scales.shape[1])})
    if quats is not None:
        elem_data.update({f"rot_{i}": quats[:, i] for i in range(quats.shape[1])})
    if sh is not None:
        # sh shape (N,3,K)
        K = sh.shape[2]
        elem_data.update({f"f_dc_{i}": sh[:, i, 0] for i in range(3)})
        if K > 1:
            rest = sh[:, :, 1:]  # (N,3,K-1)
            flat = rest.transpose(0, 2, 1).reshape(N, -1)  # (N,(K-1)*3) interleaving channels per coeff
            for k in range(flat.shape[1]):
                elem_data[f"f_rest_{k}"] = flat[:, k]

    if extra_fields:
        for k, v in extra_fields.items():
            arr = np.asarray(v)
            if arr.shape[0] != N:
                raise ValueError(f"extra field {k} has wrong length {arr.shape[0]} != {N}")
            elem_data[k] = arr

    dtype = [(k, "f4") for k in elem_data.keys()]
    structured = np.empty(N, dtype=dtype)
    for k, v in elem_data.items():
        structured[k] = v
    el = PlyElement.describe(structured, "vertex")
    PlyData([el]).write(str(path))


__all__ = ["GaussianSet", "load_ply", "save_ply", "quat_to_rotmat"]
