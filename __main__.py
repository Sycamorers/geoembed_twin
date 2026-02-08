from __future__ import annotations
import argparse
from pathlib import Path
import json

import torch

from .utils import get_repo_root, ensure_dir, set_deterministic, default_device
from .gaussians.synthetic import generate_synthetic_scene, apply_changes
from .sfvae.train import train_sfvae
from .sfvae.embed import embed_gaussians
from .adapters.sfvae_adapter import SFVAEAdapter
from .adapters.ggs_depth_adapter import load_depth_adapter
from .twin.match import match_embeddings
from .twin.change import classify_changes, save_change_summary
from .viz.ply_export import export_labeled_points
from .depth.camera import sample_sphere_cameras
from .viz.plot_depth import save_depth_png, save_residual_png
from .twin.depth_residual import depth_residual_map, mark_gaussians_changed
from .twin.floater_filter import detect_floaters


def cmd_sfvae_train(args):
    train_sfvae(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, n_grid=args.n_grid, latent_dim=args.latent_dim, num_batches=args.num_batches, fast=args.fast)


def cmd_sfvae_embed(args):
    from .gaussians.io import load_ply
    gs = load_ply(Path(args.input))
    ckpt = Path(args.ckpt)
    emb_np, means_np = embed_gaussians(gs, ckpt, n_grid=args.n_grid)
    out = Path(args.output) if args.output else Path(args.input).with_suffix('.npz')
    out.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez(out, embeddings=emb_np, means=means_np)
    print(f"Saved embeddings to {out}")


def _ensure_ckpt(ckpt_path: Path, fast: bool):
    if ckpt_path.exists():
        return ckpt_path
    print(f"Checkpoint {ckpt_path} not found; training fallback SF-VAE...")
    return train_sfvae(fast=fast)


def cmd_render_depth(args):
    from .gaussians.io import load_ply
    from .depth.stochastic_depth import render_depth_median
    gs = load_ply(Path(args.input))
    cams = sample_sphere_cameras(num=args.num_cams, width=args.width, height=args.height, fov=args.fov)
    adapter = load_depth_adapter(prefer_upstream=args.use_upstream)
    outputs = adapter.render_median_depth(gs, cams, image_size=(args.height, args.width), topk=args.topk)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, out in enumerate(outputs):
        save_depth_png(out["depth_med"], out["mask"], outdir / f"depth_med_{i}.png")
        if out.get("depth_expected") is not None:
            save_depth_png(out["depth_expected"], out["mask"], outdir / f"depth_expected_{i}.png")
    print(f"Depth renders saved to {outdir}")


def cmd_demo(args):
    device = default_device()
    set_deterministic(99)
    root = get_repo_root() / "geoembed_twin"
    out_root = ensure_dir(root / "outputs")
    depth_before_dir = ensure_dir(out_root / "depth_before")
    depth_after_dir = ensure_dir(out_root / "depth_after")
    depth_res_dir = ensure_dir(out_root / "depth_residual")
    ckpt_path = out_root / "checkpoints" / "sfvae_fallback.pt"

    print("Generating synthetic scenes...")
    gs_before = generate_synthetic_scene(device=device)
    gs_before, gs_after, labels = apply_changes(gs_before)

    # Train or load SFVAE
    ckpt_path = _ensure_ckpt(ckpt_path, fast=args.fast)

    adapter = SFVAEAdapter(prefer_upstream=not args.force_fallback, device=device)
    emb_before, means_before = adapter.encode(gs_before, ckpt_fallback=ckpt_path)
    emb_after, means_after = adapter.encode(gs_after, ckpt_fallback=ckpt_path)

    print("Matching embeddings...")
    match_out = match_embeddings(emb_before, emb_after, means_before, means_after)
    change_res = classify_changes(match_out, torch.as_tensor(means_before), torch.as_tensor(means_after))

    # Cameras and depth
    cams = sample_sphere_cameras(num=args.num_cams, width=args.width, height=args.height, fov=args.fov, device=device)
    depth_adapter = load_depth_adapter(prefer_upstream=args.use_upstream_depth, device=device)

    print("Rendering depth before/after...")
    depth_before = depth_adapter.render_median_depth(gs_before, cams, image_size=(args.height, args.width))
    depth_after = depth_adapter.render_median_depth(gs_after, cams, image_size=(args.height, args.width))

    for i, out in enumerate(depth_before):
        save_depth_png(out["depth_med"], out["mask"], depth_before_dir / f"cam{i}_median.png")
    for i, out in enumerate(depth_after):
        save_depth_png(out["depth_med"], out["mask"], depth_after_dir / f"cam{i}_median.png")

    print("Computing residuals...")
    change_masks = []
    for i, (b, a) in enumerate(zip(depth_before, depth_after)):
        residual, mask_change = depth_residual_map(b["depth_med"], a["depth_med"], b["mask"], a["mask"], threshold=args.residual_thr)
        save_residual_png(residual, depth_res_dir / f"cam{i}_residual.png")
        change_masks.append(mask_change)

    geo_changed = mark_gaussians_changed(gs_after, cams, change_masks, [d["depth_med"] for d in depth_after])

    # Optional floater filter
    if args.filter_floaters:
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
    import numpy as np
    np.savez(out_root / "embeddings_before.npz", embeddings=emb_before.cpu().numpy(), means=means_before.cpu().numpy())
    np.savez(out_root / "embeddings_after.npz", embeddings=emb_after.cpu().numpy(), means=means_after.cpu().numpy())

    print("Demo complete. Outputs in geoembed_twin/outputs")


def build_parser():
    parser = argparse.ArgumentParser(description="GeoEmbedTwin CLI")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("sfvae-train", help="Train fallback SF-VAE")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--n-grid", type=int, default=12)
    p_train.add_argument("--latent-dim", type=int, default=32)
    p_train.add_argument("--num-batches", type=int, default=200)
    p_train.add_argument("--fast", action="store_true")
    p_train.set_defaults(func=cmd_sfvae_train)

    p_embed = sub.add_parser("sfvae-embed", help="Embed gaussians using trained SF-VAE")
    p_embed.add_argument("--input", required=True)
    p_embed.add_argument("--ckpt", default=str(get_repo_root() / "geoembed_twin" / "outputs" / "checkpoints" / "sfvae_fallback.pt"))
    p_embed.add_argument("--n-grid", type=int, default=12)
    p_embed.add_argument("--output", default=None)
    p_embed.set_defaults(func=cmd_sfvae_embed)

    p_depth = sub.add_parser("render-depth", help="Render stochastic median depth")
    p_depth.add_argument("--input", required=True)
    p_depth.add_argument("--outdir", required=True)
    p_depth.add_argument("--num-cams", type=int, default=3)
    p_depth.add_argument("--width", type=int, default=128)
    p_depth.add_argument("--height", type=int, default=128)
    p_depth.add_argument("--fov", type=float, default=60.0)
    p_depth.add_argument("--topk", type=int, default=32)
    p_depth.add_argument("--use-upstream", action="store_true")
    p_depth.set_defaults(func=cmd_render_depth)

    p_demo = sub.add_parser("demo", help="Run full synthetic demo")
    p_demo.add_argument("--fast", action="store_true")
    p_demo.add_argument("--force-fallback", action="store_true", help="Skip upstream SF-VAE even if present")
    p_demo.add_argument("--use-upstream-depth", action="store_true")
    p_demo.add_argument("--filter-floaters", action="store_true")
    p_demo.add_argument("--num-cams", type=int, default=3)
    p_demo.add_argument("--width", type=int, default=128)
    p_demo.add_argument("--height", type=int, default=128)
    p_demo.add_argument("--fov", type=float, default=60.0)
    p_demo.add_argument("--residual-thr", type=float, default=0.05)
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
