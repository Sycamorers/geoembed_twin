from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from .utils import get_repo_root, ensure_dir, set_deterministic, default_device
from .gaussians.synthetic import generate_synthetic_scene, apply_changes, SyntheticConfig
from .adapters.sfvae_adapter import SFVAEAdapter
from .adapters.ggs_depth_adapter import load_depth_adapter
from .twin.match import match_embeddings
from .twin.change import classify_changes, save_change_summary
from .viz.ply_export import export_labeled_points
from .depth.camera import sample_sphere_cameras
from .viz.plot_depth import save_depth_png, save_residual_png
from .twin.depth_residual import depth_residual_map, mark_gaussians_changed
from .twin.floater_filter import detect_floaters
from .sfvae.train import train_sfvae


def _ensure_ckpt(ckpt_path: Path, fast: bool):
    if ckpt_path.exists():
        return ckpt_path
    print(f"Checkpoint {ckpt_path} not found; training fallback SF-VAE...")
    return train_sfvae(fast=fast, output_path=ckpt_path.parent)


def run_demo(
    fast: bool = False,
    force_fallback: bool = False,
    use_upstream_depth: bool = False,
    filter_floaters: bool = False,
    num_cams: int = 3,
    width: int = 128,
    height: int = 128,
    fov: float = 60.0,
    residual_thr: float = 0.05,
    synthetic_cfg_override: Optional[SyntheticConfig] = None,
    device=None,
) -> Path:
    """Full synthetic demo pipeline. Returns output root path."""
    device = device or default_device()
    set_deterministic(99)
    root = get_repo_root() / "geoembed_twin"
    out_root = ensure_dir(root / "outputs")
    depth_before_dir = ensure_dir(out_root / "depth_before")
    depth_after_dir = ensure_dir(out_root / "depth_after")
    depth_res_dir = ensure_dir(out_root / "depth_residual")
    ckpt_path = out_root / "checkpoints" / "sfvae_fallback.pt"

    cfg = synthetic_cfg_override
    if cfg is None:
        cfg = SyntheticConfig(
            num_surface=1400 if fast else SyntheticConfig().num_surface,
            num_floaters=160 if fast else SyntheticConfig().num_floaters,
            cube_size=SyntheticConfig().cube_size,
            sphere_radius=SyntheticConfig().sphere_radius,
            opacity_range=SyntheticConfig().opacity_range,
            floater_opacity=SyntheticConfig().floater_opacity,
            lmax=SyntheticConfig().lmax,
        )

    print("Generating synthetic scenes...")
    gs_before = generate_synthetic_scene(cfg=cfg, device=device)
    gs_before, gs_after, labels = apply_changes(gs_before)

    # Train or load SFVAE
    ckpt_path = _ensure_ckpt(ckpt_path, fast=fast)

    adapter = SFVAEAdapter(prefer_upstream=not force_fallback, device=device)
    emb_before, means_before = adapter.encode(gs_before, ckpt_fallback=ckpt_path)
    emb_after, means_after = adapter.encode(gs_after, ckpt_fallback=ckpt_path)

    print("Matching embeddings...")
    match_out = match_embeddings(emb_before, emb_after, means_before, means_after)
    change_res = classify_changes(match_out, torch.as_tensor(means_before), torch.as_tensor(means_after))

    # Cameras and depth
    cams = sample_sphere_cameras(num=num_cams, width=width, height=height, fov=fov, device=device)
    depth_adapter = load_depth_adapter(prefer_upstream=use_upstream_depth, device=device)

    print("Rendering depth before/after...")
    depth_before = depth_adapter.render_median_depth(gs_before, cams, image_size=(height, width))
    depth_after = depth_adapter.render_median_depth(gs_after, cams, image_size=(height, width))

    for i, out in enumerate(depth_before):
        save_depth_png(out["depth_med"], out["mask"], depth_before_dir / f"cam{i}_median.png")
    for i, out in enumerate(depth_after):
        save_depth_png(out["depth_med"], out["mask"], depth_after_dir / f"cam{i}_median.png")

    print("Computing residuals...")
    change_masks = []
    for i, (b, a) in enumerate(zip(depth_before, depth_after)):
        residual, mask_change = depth_residual_map(b["depth_med"], a["depth_med"], b["mask"], a["mask"], threshold=residual_thr)
        save_residual_png(residual, depth_res_dir / f"cam{i}_residual.png")
        change_masks.append(mask_change)

    geo_changed = mark_gaussians_changed(gs_after, cams, change_masks, [d["depth_med"] for d in depth_after])

    # Optional floater filter
    if filter_floaters:
        floater_mask = detect_floaters(gs_after, cams, [d["depth_med"] for d in depth_after])
        keep = ~floater_mask
        for field in ["means", "scales", "quats", "covs", "opacity", "sh"]:
            t = getattr(gs_after, field)
            if t is not None:
                setattr(gs_after, field, t[keep])
        emb_after = emb_after[keep]
        means_after = means_after[keep]
        change_res.colors = change_res.colors[keep]
        geo_changed = geo_changed[keep]
        print(f"Filtered {floater_mask.sum().item()} floaters")

    # Export labeled point clouds
    colors_before = torch.full((gs_before.N, 3), 0.6)
    colors_after = change_res.colors
    removed_mask = labels.get("removed_mask", torch.zeros(gs_before.N, dtype=torch.bool))
    moved_mask_before = labels.get("moved_mask", torch.zeros(gs_before.N, dtype=torch.bool))
    colors_before[removed_mask] = torch.tensor([0.9, 0.1, 0.1])
    colors_before[moved_mask_before] = torch.tensor([0.9, 0.6, 0.1])
    if geo_changed.numel() == colors_after.shape[0]:
        colors_after[geo_changed] = torch.tensor([0.8, 0.2, 0.8])

    export_labeled_points(gs_before, colors_before, out_root / "before_labeled.ply")
    export_labeled_points(gs_after, colors_after, out_root / "after_labeled.ply")
    save_change_summary(out_root / "change_summary.json", match_out, change_res)

    # Also persist embeddings
    np.savez(out_root / "embeddings_before.npz", embeddings=emb_before.cpu().numpy(), means=means_before.cpu().numpy())
    np.savez(out_root / "embeddings_after.npz", embeddings=emb_after.cpu().numpy(), means=means_after.cpu().numpy())

    print("Demo complete. Outputs in geoembed_twin/outputs")
    return out_root


__all__ = ["run_demo"]
