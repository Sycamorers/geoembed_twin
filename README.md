# GeoEmbedTwin

Dynamic digital-twin prototype for 3D Gaussian Splatting that stays **stable in identity** (embeddings) and **robust in geometry** (median depth). Runs fully offline with synthetic data; optionally reuses nearby upstream repos (`gs-embedding`, `Geometry-Grounded-Gaussian-Splatting`).

## Prerequisites (use your existing conda env)
- Python >= 3.10 with a working PyTorch build (CUDA 12.1 if you have a GPU; CPU works for tests).
- Core deps: torch, numpy, matplotlib, imageio **or** pillow, plyfile.
- Optional extras (auto-used when present): geomloss, faiss-cpu, open3d, rich, pytest.
- Optional sibling repos in the parent folder: `../gs-embedding`, `../Geometry-Grounded-Gaussian-Splatting` (auto-detected; fall back to internal code).
- `environment.yml` exists as a reference only—do **not** create a fresh env; just `conda activate <your_env>`.

## Check your environment
From the repo root (parent of `geoembed_twin/`):
```bash
PYTHONPATH=. python -m geoembed_twin doctor
```
Prints Python/Torch build info, CUDA availability, and which core/optional deps are missing. Fails if a core dep is absent.

## Optional dependencies
```
geoembed_twin/requirements-optional.txt
geoembed_twin/scripts/install_optional_deps.sh
```
Run the script any time; it installs only the missing optional packages in-place:
```bash
PYTHONPATH=. geoembed_twin/scripts/install_optional_deps.sh
```

## Run & self-test (no new envs)
```bash
cd /home/jing/Desktop/GS_2026              # repo root (parent of geoembed_twin/)
PYTHONPATH=. python -m geoembed_twin selftest --quick   # invariance + depth + smoke demo
PYTHONPATH=. python -m geoembed_twin demo --fast        # full demo, deterministic seed
# helper wrapper: geoembed_twin/scripts/run_demo.sh
```
Outputs (in `geoembed_twin/outputs/`):
- `before_labeled.ply`, `after_labeled.ply`
- `change_summary.json`
- Depth PNGs: `depth_before/*`, `depth_after/*`, `depth_residual/*`
- `embeddings_before.npz`, `embeddings_after.npz`

## CLI reference
- `python -m geoembed_twin doctor [--json]`
- `python -m geoembed_twin selftest [--quick] [--skip-demo]`
- `python -m geoembed_twin demo [--fast] [--filter-floaters] [--use-upstream-depth]`
- `python -m geoembed_twin sfvae-train [--fast]`
- `python -m geoembed_twin sfvae-embed --input <ply> [--ckpt outputs/checkpoints/sfvae_fallback.pt]`
- `python -m geoembed_twin render-depth --input <ply> --outdir <dir>`

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
- Global \(T(t)=\prod_i T_i(t)\); median depth via log-space binary search on \([t_{near}, t_{far}]\) using top-K nearest Gaussians per ray.

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

## Testing
- `PYTHONPATH=. python -m geoembed_twin selftest --quick`
- `pytest geoembed_twin/tests -q` (optional; same checks used by selftest)

## Troubleshooting
- Missing core dep? Run `python -m geoembed_twin doctor` for install hints.
- Checkpoint missing? Demo/selftest auto-trains to `outputs/checkpoints/sfvae_fallback.pt`.
- GPU build flaky? Run without `--use-upstream-depth`; fallback PyTorch renderer is default.
- Depth looks empty: increase `--far` or decrease `--topk` on `render-depth`/`demo`.

## Repo cleanliness
- Generated artifacts live in `outputs/` (gitignored).
- No edits are applied to optional upstream repos; they are only imported if present.
