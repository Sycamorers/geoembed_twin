import os
from pathlib import Path
import torch

from geoembed_twin.sfvae.model import SFVAE
from geoembed_twin.sfvae.submanifold import gaussian_to_points
from geoembed_twin.gaussians.io import GaussianSet
from geoembed_twin.sfvae.embed import load_sfvae_checkpoint


CKPT = Path(__file__).resolve().parents[1] / "outputs" / "checkpoints" / "sfvae_fallback.pt"

def test_quaternion_sign_invariance():
    device = torch.device("cpu")
    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    scales = torch.tensor([[0.2, 0.1, 0.1], [0.2, 0.1, 0.1]])
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    opacity = torch.ones((2, 1)) * 0.8
    sh = torch.zeros((2, 3, 16))
    gs = GaussianSet(means=means, scales=scales, quats=q, opacity=opacity, sh=sh)
    pts, _ = gaussian_to_points(gs, n_grid=8, device=device)
    if CKPT.exists():
        model = load_sfvae_checkpoint(CKPT, device=device)
    else:
        model = SFVAE(deterministic=True)
    with torch.no_grad():
        emb = model.encode(pts)
    sim = torch.nn.functional.cosine_similarity(emb[0:1], emb[1:2]).item()
    if CKPT.exists():
        assert sim > 0.85
    else:
        assert emb.shape[0] == 2 and emb.shape[1] == model.latent_dim
