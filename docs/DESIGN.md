# GeoEmbedTwin Design

## Problem
3D Gaussian Splatting (3DGS) scenes drift in both identity and geometry: latent codes are unstable under parameter non-uniqueness (e.g., quaternion sign, log-scale permutations) and depth estimates fluctuate due to floaters or thin structures. For change detection and digital-twin upkeep we need embeddings that are invariant yet discriminative, plus a depth estimator that resists floaters and supplies a robust geometry prior.

## Approach
- **Submanifold Field VAE (SF-VAE):** deterministically samples each Gaussian’s ellipsoid on a spherical grid, evaluates SH colors, and feeds a PointNet encoder/decoder. The deterministic sampling makes embeddings invariant to quaternion sign flips and log-scale symmetries. Geomloss OT is preferred; Chamfer fallback keeps it dependency-light.
- **Stochastic-solids median depth:** models per-Gaussian vacancy \(v=\sqrt{1-G}\) and transmittance \(T(t)\); performs log-space binary search for \(T=0.5\) using the top-K nearest Gaussians per ray. Works on CPU and matches CUDA if available.
- **Change fusion:** mutual-nearest matching in latent space gated by spatial distance, fused with depth residual backprojections; optional floater filtering by multi-view depth voting. Everything stays offline and deterministic.

## Advantages
- Runs with a tiny dependency set (torch, numpy, matplotlib, imageio/pillow, plyfile); optional extras only enhance speed/visuals.
- Deterministic sampling + sign-canonical quaternions make embeddings stable without heavy training.
- Depth median search is monotonic/bracketed, giving reliable residuals even with few views.
- Single self-contained implementation; adapters are thin wrappers over the in-repo SF-VAE and stochastic-solids depth.

## Trade-offs & limitations
- Fallback SF-VAE is shallow and trains on synthetic blobs; quality is sufficient for invariance tests but not photorealistic tasks.
- Depth renderer uses top-K nearest Gaussians; very large scenes may need tiling or GPU for speed.
- Open3D viewer is optional; without it, visualization is limited to PNG/PLY exports.

## Future work
- Replace PointNet encoder with a lightweight transformer while keeping deterministic sampling.
- Add adaptive top-K selection driven by opacity/variance statistics.
- Integrate multi-scale depth grids for faster far-range search.
- Support multi-GPU batching and mixed-precision benchmarking in doctor/selftest outputs.
