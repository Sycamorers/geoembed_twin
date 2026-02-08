from __future__ import annotations
from typing import List

from ..gaussians.io import GaussianSet
from ..depth.stochastic_depth import render_depth_median
from ..depth.camera import Camera
from ..utils import default_device


class GeometryGroundedDepthAdapter:
    """Self-contained stochastic-solids median depth renderer."""

    def __init__(self, device=None):
        self.device = device or default_device()

    def render_median_depth(
        self,
        gs: GaussianSet,
        cameras: List[Camera],
        image_size: tuple[int, int],
        **kwargs,
    ):
        return render_depth_median(gs, cameras, image_size=image_size, device=self.device, **kwargs)


def load_depth_adapter(device=None) -> GeometryGroundedDepthAdapter:  # pragma: no cover compatibility alias
    return GeometryGroundedDepthAdapter(device=device)


__all__ = ["GeometryGroundedDepthAdapter", "load_depth_adapter"]
