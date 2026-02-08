from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import torch

from ..sfvae.embed import embed_gaussians
from ..gaussians.io import GaussianSet
from ..utils import default_device


class SFVAEAdapter:
    """Lightweight wrapper around the internal SF-VAE checkpoint.

    Design goal: never depend on external repos; always fall back to the in-repo model.
    """

    def __init__(self, device=None):
        self.device = device or default_device()

    def encode(self, gs: GaussianSet, ckpt_fallback: Optional[Path]) -> Tuple[torch.Tensor, torch.Tensor]:
        if ckpt_fallback is None:
            raise ValueError("Fallback checkpoint path is required.")
        emb_np, means_np = embed_gaussians(gs, ckpt_fallback, n_grid=12, device=self.device)
        return torch.from_numpy(emb_np), torch.from_numpy(means_np)


def load_adapter(device=None) -> SFVAEAdapter:  # pragma: no cover - compatibility alias
    return SFVAEAdapter(device=device)


__all__ = ["SFVAEAdapter", "load_adapter"]
