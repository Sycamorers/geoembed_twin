from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import torch

from .repo_scan import scan_repos
from ..sfvae.embed import embed_gaussians, load_sfvae_checkpoint
from ..sfvae.submanifold import gaussian_to_points
from ..gaussians.io import GaussianSet
from ..utils import default_device


class SFVAEAdapter:
    def __init__(self, prefer_upstream: bool = True, device=None):
        self.device = device or default_device()
        self.mode = "fallback"
        self.upstream_model = None
        self.upstream_n_grid = 12
        if prefer_upstream:
            self._try_load_upstream()

    def _try_load_upstream(self):
        repos = scan_repos()
        if repos.sfvae is None:
            return
        ckpt_candidates = [
            repos.sfvae / "checkpoints" / "checkpoint_sfvae_sh0_144.pth",
            repos.sfvae / "checkpoints" / "checkpoint_sfvae_sh0.pth",
        ]
        ckpt_path = next((c for c in ckpt_candidates if c.exists()), None)
        if ckpt_path is None:
            return
        try:
            # Lazy import inside try
            import importlib
            spec = importlib.util.spec_from_file_location("upstream_embedding_model", repos.sfvae / "embedding_model" / "embedding_model.py")
            if spec is None or spec.loader is None:
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            UpSFVAE = getattr(module, "SFVAE")
            self.upstream_model = UpSFVAE(embedding_dim=32, grid_dim=24, deterministic=True)
            state = torch.load(ckpt_path, map_location=self.device)
            sd = state.get("state_dict", state)
            # Remove _orig_mod prefix if present
            fixed = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("_orig_mod."):
                    nk = nk[len("_orig_mod."):]
                fixed[nk] = v
            self.upstream_model.load_state_dict(fixed, strict=False)
            self.upstream_model = self.upstream_model.to(self.device)
            self.upstream_model.eval()
            # Heuristic: checkpoint name with 144 -> n_grid 12 (12*12)
            if "144" in ckpt_path.name:
                self.upstream_n_grid = 12
            self.mode = "upstream"
            print(f"[sfvae_adapter] Using upstream SFVAE from {ckpt_path}")
        except Exception as exc:
            print(f"[sfvae_adapter] Upstream load failed, fallback to internal. Reason: {exc}")
            self.upstream_model = None
            self.mode = "fallback"

    def encode(self, gs: GaussianSet, ckpt_fallback: Optional[Path] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.upstream_model is not None:
            with torch.no_grad():
                pts, _ = gaussian_to_points(gs.to(self.device), n_grid=self.upstream_n_grid, device=self.device)
                emb = self.upstream_model.encode(pts)
            return emb, gs.means
        if ckpt_fallback is None:
            raise ValueError("Fallback checkpoint path is required when upstream model is unavailable.")
        emb_np, means_np = embed_gaussians(gs, ckpt_fallback, n_grid=12, device=self.device)
        return torch.from_numpy(emb_np), torch.from_numpy(means_np)


def load_upstream_or_fallback(prefer_upstream: bool = True, device=None) -> SFVAEAdapter:
    return SFVAEAdapter(prefer_upstream=prefer_upstream, device=device)


__all__ = ["SFVAEAdapter", "load_upstream_or_fallback"]
