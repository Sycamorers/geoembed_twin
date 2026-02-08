# GeoEmbedTwin

Dynamic digital-twin prototype for 3D Gaussian Splatting that stays **stable in identity** (embeddings) and **robust in geometry** (median depth). Runs fully offline with synthetic data; optionally reuses nearby upstream repos (`gs-embedding`, `Geometry-Grounded-Gaussian-Splatting`).

## Why it matters
- 3DGS parameter spaces are non-unique (e.g., quaternion sign, log-scale heterogeneity) → unstable learning.
- Submanifold Field VAE (SF-VAE) embeds each Gaussian via an ellipsoid point-cloud view, giving consistent latent codes.
- Geometry-grounded “stochastic solids” depth uses vacancy \(v(x)=\sqrt{1-G(x)}\) and transmittance \(T(t)=\prod_i T_i(t)\); median depth \(t_{med}=T^{-1}(0.5)\) is multi-view consistent and less sensitive to floaters.
- Combined, we can track per-primitive identity while keeping geometry reliable for change detection and cleanup.

## Contents
- `adapters/` — auto-detect upstream repos, fall back to internal code.
- `sfvae/` — fallback SF-VAE (PointNet VAE, submanifold sampling, OT/Chamfer loss).
- `depth/` — stochastic-solids median depth renderer + camera utils.
- `gaussians/` — PLY IO, synthetic scene generation, quaternion handling.
- `twin/` — embedding match, change classification, depth residual fusion, floater filter.
- `viz/` — depth/residual PNGs, PLY export, optional Open3D viewer.
- `scripts/` — convenience launchers; `outputs/` (gitignored) for artifacts.

## Requirements
- GPU: single RTX 3090 (24 GB) target; CPU runs for tests.
- OS: Linux (conda + CUDA 11.8 stack).
- Local sibling repos (optional):
  - `../gs-embedding`
  - `../Geometry-Grounded-Gaussian-Splatting`
  Both are automatically added to `sys.path`; editable installs are listed in the env.

## Setup
```bash
cd geoembed_twin
conda env create -f environment.yml
conda activate geoembed_twin
```
`environment.yml` includes PyTorch 2.2 + CUDA 11.8, Open3D, geomloss, torch-scatter, pytorch3d, gsplat, and editable installs of the two upstream repos (assumes this repo sits beside them).

If conda creation fails on optional heavy packages (torch-scatter / pytorch3d / gsplat), remove those lines or install from prebuilt wheels manually; fallbacks do not require them.

## Quickstart (offline synthetic demo)
```bash
python -m geoembed_twin demo --fast
```
Produces in `outputs/`:
- `before_labeled.ply`, `after_labeled.ply`
- `change_summary.json`
- Depth PNGs: `depth_before/*`, `depth_after/*`, `depth_residual/*`
- `embeddings_before.npz`, `embeddings_after.npz`

## CLI reference
- Train fallback SF-VAE: `python -m geoembed_twin sfvae-train --fast`
- Embed a PLY: `python -m geoembed_twin sfvae-embed --input <ply> [--ckpt outputs/checkpoints/sfvae_fallback.pt]`
- Render depth only: `python -m geoembed_twin render-depth --input <ply> --outdir outputs/depth_custom`
- Full demo: `python -m geoembed_twin demo --fast [--filter-floaters] [--use-upstream-depth]`

## Algorithm sketches
### SF-VAE (submanifold field)
1) Sample ellipsoid level-set points: \(x_k = \mu + \Sigma^{1/2} u_k\) with deterministic \(u_k\) grid.
2) Color via real SH: \(c_k = f_{SH}(u_k)\); broadcast opacity.
3) PointNet encoder → \(\mu, \log\sigma^2\); reparameterized latent \(z\); decoder reconstructs coords/colors on canonical sphere grid.
4) Loss: Sinkhorn OT on \([x, \lambda c]\) (geomloss) or Chamfer fallback + \(\beta\)-KL. Latents are invariant to parameter non-uniqueness.

### Stochastic-solids median depth
- Density: \(G(x)=\alpha e^{-(x-\mu)^T\Sigma^{-1}(x-\mu)}\); vacancy \(v(x)=\sqrt{1-G(x)}\).
- Per-Gaussian transmittance along ray \(x(t)=o+\omega t\):
  - \(T_i(t)=v_i(t)\) for \(t\le t_*\); \(T_i(t)=v_i(t_*)^2 / v_i(t)\) for \(t>t_*\), where \(t_*\) is stationary point.
- Global \(T(t)=\prod_i T_i(t)\); median depth found via log-space binary search on \([t_{near}, t_{far}]\) using top-K nearest Gaussians per ray.

### Change detection & floaters
- Matching: cosine distance in latent space + spatial gate; mutual-NN pairs → added / removed / moved.
- Geometry cue: |depth_before − depth_after| residuals → backproject to mark geometry-changed Gaussians.
- Floater filter (optional): Gaussians consistently in front of median depth across views are suppressed.

## Synthetic demo pipeline
1) Build synthetic scene (cube + sphere surfaces + floaters).
2) Apply changes: remove cluster A, translate cluster B, add cluster C.
3) Train/reuse SF-VAE fallback; embed before/after scenes.
4) Match embeddings → change labels.
5) Render stochastic median depth for both states; compute residual heatmaps.
6) Backproject residuals, fuse with embedding labels; optionally filter floaters.
7) Export PLYs, PNGs, NPZs to `outputs/`.

## Tips & troubleshooting
- Checkpoint missing? Demo auto-trains fallback to `outputs/checkpoints/sfvae_fallback.pt`.
- Upstream CUDA build fails? Run demo without `--use-upstream-depth`; fallback PyTorch renderer will be used.
- Depth looks empty: raise `--far` or lower `--topk`.
- To skip heavy extras, edit `environment.yml` pip list; core demo depends only on PyTorch, numpy, matplotlib, plyfile.

## Testing
Minimal CPU-friendly checks:
```bash
pytest geoembed_twin/tests -q
```
- Quaternion sign invariance for embeddings.
- Median-depth solver range/monotonicity on a toy scene.

## Repo cleanliness
- Generated artifacts are kept in `outputs/` (gitignored).
- Editable installs for sibling repos are declared; no changes are made to upstream code.
