import torch

from geoembed_twin.gaussians.io import GaussianSet
from geoembed_twin.depth.camera import sample_sphere_cameras
from geoembed_twin.depth.stochastic_depth import render_depth_median


def test_depth_monotonic():
    means = torch.tensor([[0.0, 0.0, 0.0]])
    scales = torch.tensor([[0.2, 0.2, 0.2]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    opacity = torch.tensor([[0.8]])
    sh = torch.zeros((1, 3, 16))
    gs = GaussianSet(means=means, scales=scales, quats=quats, opacity=opacity, sh=sh)
    cams = sample_sphere_cameras(num=1, width=32, height=32, radius=1.5)
    outputs = render_depth_median(gs, cams, image_size=(32, 32), topk=1, near=0.01, far=3.0)
    depth = outputs[0]["depth_med"]
    assert torch.all(depth >= 0)
    assert torch.max(depth) <= 3.0 + 1e-3
