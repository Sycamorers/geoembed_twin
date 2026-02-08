from __future__ import annotations
import argparse
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = PACKAGE_ROOT / "outputs" / "checkpoints" / "sfvae_fallback.pt"


def cmd_sfvae_train(args):
    from .sfvae.train import train_sfvae

    train_sfvae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_grid=args.n_grid,
        latent_dim=args.latent_dim,
        num_batches=args.num_batches,
        fast=args.fast,
    )


def cmd_sfvae_embed(args):
    import numpy as np

    from .gaussians.io import load_ply
    from .sfvae.embed import embed_gaussians

    gs = load_ply(Path(args.input))
    ckpt = Path(args.ckpt)
    emb_np, means_np = embed_gaussians(gs, ckpt, n_grid=args.n_grid)
    out = Path(args.output) if args.output else Path(args.input).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, embeddings=emb_np, means=means_np)
    print(f"Saved embeddings to {out}")


def cmd_render_depth(args):
    from .gaussians.io import load_ply
    from .depth.stochastic_depth import render_depth_median
    from .depth.camera import sample_sphere_cameras
    from .adapters.ggs_depth_adapter import load_depth_adapter
    from .viz.plot_depth import save_depth_png

    from .gaussians.io import load_ply
    from .depth.stochastic_depth import render_depth_median
    gs = load_ply(Path(args.input))
    cams = sample_sphere_cameras(num=args.num_cams, width=args.width, height=args.height, fov=args.fov)
    adapter = load_depth_adapter()
    outputs = adapter.render_median_depth(gs, cams, image_size=(args.height, args.width), topk=args.topk)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, out in enumerate(outputs):
        save_depth_png(out["depth_med"], out["mask"], outdir / f"depth_med_{i}.png")
        if out.get("depth_expected") is not None:
            save_depth_png(out["depth_expected"], out["mask"], outdir / f"depth_expected_{i}.png")
    print(f"Depth renders saved to {outdir}")


def cmd_demo(args):
    from .pipeline import run_demo

    run_demo(
        fast=args.fast,
        filter_floaters=args.filter_floaters,
        num_cams=args.num_cams,
        width=args.width,
        height=args.height,
        fov=args.fov,
        residual_thr=args.residual_thr,
        synthetic_cfg_override=None,
    )


def cmd_doctor(args):
    from .diagnostics import run_doctor

    status = run_doctor(as_json=args.json)
    sys.exit(0 if status.ok else 1)


def cmd_selftest(args):
    from .selftest import run_selftest

    status = run_selftest(quick=args.quick, skip_demo=args.skip_demo)
    sys.exit(0 if status.ok else 1)


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
    p_embed.add_argument("--ckpt", default=str(DEFAULT_CKPT))
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
    p_depth.set_defaults(func=cmd_render_depth)

    p_demo = sub.add_parser("demo", help="Run full synthetic demo")
    p_demo.add_argument("--fast", action="store_true")
    p_demo.add_argument("--force-fallback", action="store_true", help="(deprecated) internal model is always used")
    p_demo.add_argument("--filter-floaters", action="store_true")
    p_demo.add_argument("--num-cams", type=int, default=3)
    p_demo.add_argument("--width", type=int, default=128)
    p_demo.add_argument("--height", type=int, default=128)
    p_demo.add_argument("--fov", type=float, default=60.0)
    p_demo.add_argument("--residual-thr", type=float, default=0.05)
    p_demo.set_defaults(func=cmd_demo)

    p_doc = sub.add_parser("doctor", help="Inspect current environment and optional deps")
    p_doc.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    p_doc.set_defaults(func=cmd_doctor)

    p_self = sub.add_parser("selftest", help="Run built-in invariance and smoke tests")
    p_self.add_argument("--quick", action="store_true", help="Use the fastest settings")
    p_self.add_argument("--skip-demo", action="store_true", help="Skip the end-to-end demo smoke (not recommended)")
    p_self.set_defaults(func=cmd_selftest)

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
