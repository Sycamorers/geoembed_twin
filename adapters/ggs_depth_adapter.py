from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import torch

from .repo_scan import scan_repos
from ..gaussians.io import GaussianSet
from ..depth.stochastic_depth import render_depth_median
from ..depth.camera import Camera
from ..utils import default_device


class GeometryGroundedDepthAdapter:
    def __init__(self, prefer_upstream: bool = True, device=None):
        self.device = device or default_device()
        self.upstream_available = False
        self._upstream_render = None
        self._gaussian_model_cls = None
        self._camera_loader = None
        if prefer_upstream:
            self._try_load_upstream()

    def _try_load_upstream(self):
        repos = scan_repos()
        if repos.ggs is None:
            return
        try:
            import sys
            import importlib
            if str(repos.ggs.resolve()) not in sys.path:
                sys.path.append(str(repos.ggs.resolve()))
            gr = importlib.import_module("gaussian_renderer")
            scene_gaussian = importlib.import_module("scene.gaussian_model")
            self._upstream_render = gr.render
            self._gaussian_model_cls = scene_gaussian.GaussianModel
            self.upstream_available = True
            print(f"[ggs_depth_adapter] Upstream geometry-grounded renderer detected at {repos.ggs}")
        except Exception as exc:
            print(f"[ggs_depth_adapter] Upstream import failed, fallback active. Reason: {exc}")
            self.upstream_available = False

    def _convert_to_model(self, gs: GaussianSet):
        # Minimal conversion to GaussianModel; relies on upstream APIs remaining compatible.
        if self._gaussian_model_cls is None:
            return None
        model = self._gaussian_model_cls(sh_degree=3, scale_init=1.0)
        device = self.device
        model._xyz = gs.means.to(device)
        if gs.scales is not None:
            model._scaling = gs.scales.to(device)
        if gs.quats is not None:
            model._rotation = gs.quats.to(device)
        if gs.opacity is not None:
            model._opacity = gs.opacity.to(device)
        if gs.sh is not None:
            # Expect shape (N,3,K)
            model._features_dc = gs.sh[:, :, :1].permute(0, 2, 1).contiguous()  # (N,1,3)
            if gs.sh.shape[2] > 1:
                rest = gs.sh[:, :, 1:].reshape(gs.N, -1)
                model._features_rest = rest
        model.cuda() if torch.cuda.is_available() else None
        model.active_sh_degree = 3
        return model

    def render_median_depth(
        self,
        gs: GaussianSet,
        cameras: List[Camera],
        image_size: tuple[int, int],
        use_upstream: bool = False,
        **kwargs,
    ):
        if use_upstream and self.upstream_available:
            try:
                model = self._convert_to_model(gs)
                if model is None:
                    raise RuntimeError("conversion failed")
                depths = []
                for cam in cameras:
                    # Convert Camera to upstream camera utils if available
                    # Fallback to stochastic depth if conversion fails
                    raise NotImplementedError("Upstream camera conversion not implemented; using fallback")
                # If implemented, would return depths
            except Exception as exc:
                print(f"[ggs_depth_adapter] Upstream rendering failed, fallback. Reason: {exc}")
        # Fallback stochastic median depth
        return render_depth_median(gs, cameras, image_size=image_size, **kwargs)


def load_depth_adapter(prefer_upstream: bool = True, device=None) -> GeometryGroundedDepthAdapter:
    return GeometryGroundedDepthAdapter(prefer_upstream=prefer_upstream, device=device)


__all__ = ["GeometryGroundedDepthAdapter", "load_depth_adapter"]
