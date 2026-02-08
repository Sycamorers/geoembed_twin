from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import time

import torch

from .gaussians.io import GaussianSet
from .sfvae.model import SFVAE
from .sfvae.embed import load_sfvae_checkpoint
from .sfvae.submanifold import gaussian_to_points
from .depth.camera import sample_sphere_cameras
from .depth.stochastic_depth import render_depth_median, _inverse_cov, _logT
from .gaussians.synthetic import SyntheticConfig
from .pipeline import run_demo
from .utils import default_device, set_deterministic, get_repo_root


@dataclass
class TestCaseResult:
    name: str
    ok: bool
    detail: str


@dataclass
class SelfTestStatus:
    ok: bool
    cases: List[TestCaseResult]


def _quat_sign_invariance(ckpt: Path | None = None, device=None) -> TestCaseResult:
    device = device or default_device()
    set_deterministic(123)
    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.2, 0.1, 0.1], [0.2, 0.1, 0.1]], device=device)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], device=device)
    opacity = torch.ones((2, 1), device=device) * 0.8
    sh = torch.zeros((2, 3, 16), device=device)
    gs = GaussianSet(means=means, scales=scales, quats=q, opacity=opacity, sh=sh)
    pts, _ = gaussian_to_points(gs, n_grid=8, device=device)
    if ckpt and ckpt.exists():
        model = load_sfvae_checkpoint(ckpt, device=device)
    else:
        model = SFVAE(deterministic=True).to(device)
    with torch.no_grad():
        emb = model.encode(pts)
    sim = torch.nn.functional.cosine_similarity(emb[0:1], emb[1:2]).item()
    ok = sim > 0.99
    detail = f"cosine similarity {sim:.4f}"
    return TestCaseResult("quaternion_sign_invariance", ok, detail)


def _transmittance_bracket(device=None) -> TestCaseResult:
    device = device or default_device()
    set_deterministic(77)
    origin = torch.tensor([0.0, 0.0, -0.6], device=device)
    center_dir = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    gs = GaussianSet(
        means=torch.tensor([[0.0, 0.0, 0.0]], device=device),
        scales=torch.tensor([[0.18, 0.18, 0.18]], device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        opacity=torch.tensor([[0.95]], device=device),
        sh=torch.zeros((1, 3, 16), device=device),
    )
    A_sel = _inverse_cov(gs, device=device).view(1, 1, 3, 3)
    delta = (gs.means - origin[None, :]).view(1, 1, 3)
    opacity_sel = gs.opacity.view(1, 1, 1)
    denom = torch.einsum("bi,bkij,bj->bk", center_dir, A_sel, center_dir)
    num = torch.einsum("bi,bkij,bkj->bk", center_dir, A_sel, delta)
    t_star = torch.clamp(num / (denom + 1e-8), min=1e-4)

    t_vals = torch.linspace(0.0, 1.2, steps=24, device=device)
    logT = torch.stack([
        _logT(t.view(1, 1), center_dir, delta, A_sel, opacity_sel, t_star).squeeze() for t in t_vals
    ])
    monotonic = bool(torch.all(logT[1:] <= logT[:-1] + 1e-6))
    log_half = torch.log(torch.tensor(0.5, device=device))
    bracket = bool((logT[0] > log_half) and (logT[-1] < log_half))
    ok = monotonic and bracket
    detail = f"logT start {logT[0].item():.3f} end {logT[-1].item():.3f}"
    return TestCaseResult("transmittance_bracketing", ok, detail)


def _demo_smoke(quick: bool = False, device=None) -> TestCaseResult:
    device = device or default_device()
    cfg = SyntheticConfig(
        num_surface=800 if quick else 1400,
        num_floaters=80 if quick else 160,
        cube_size=1.1,
        sphere_radius=0.6,
        opacity_range=(0.5, 0.95),
        floater_opacity=(0.3, 0.7),
        lmax=3,
    )
    start = time.time()
    out_root = run_demo(
        fast=True,
        filter_floaters=False,
        num_cams=2 if quick else 3,
        width=96 if quick else 128,
        height=96 if quick else 128,
        fov=60.0,
        residual_thr=0.08 if quick else 0.05,
        synthetic_cfg_override=cfg,
        device=device,
    )
    took = time.time() - start
    required_files = [
        out_root / "before_labeled.ply",
        out_root / "after_labeled.ply",
        out_root / "change_summary.json",
        out_root / "depth_before" / "cam0_median.png",
        out_root / "depth_after" / "cam0_median.png",
        out_root / "depth_residual" / "cam0_residual.png",
    ]
    ok = all(p.exists() for p in required_files)
    missing = [str(p) for p in required_files if not p.exists()]
    detail = f"outputs at {out_root}, duration {took:.1f}s"
    if missing:
        detail += f"; missing: {missing}"
    return TestCaseResult("demo_smoke", ok, detail)


def run_selftest(quick: bool = False, skip_demo: bool = False) -> SelfTestStatus:
    root = get_repo_root() / "geoembed_twin" / "outputs" / "checkpoints" / "sfvae_fallback.pt"
    cases: List[TestCaseResult] = []
    cases.append(_quat_sign_invariance(ckpt=root))
    cases.append(_transmittance_bracket())
    if skip_demo:
        cases.append(TestCaseResult("demo_smoke", True, "skipped by flag"))
    else:
        cases.append(_demo_smoke(quick=quick))
    ok = all(c.ok for c in cases)

    print("GeoEmbedTwin selftest")
    for c in cases:
        state = "PASS" if c.ok else "FAIL"
        print(f"- [{state}] {c.name}: {c.detail}")
    return SelfTestStatus(ok=ok, cases=cases)


__all__ = ["run_selftest", "SelfTestStatus", "TestCaseResult"]
