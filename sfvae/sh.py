from __future__ import annotations
import torch

# Real SH basis up to l=3, adapted from nerfacc/instant-ngp style.


def sh_basis(dirs: torch.Tensor, lmax: int = 3) -> torch.Tensor:
    assert lmax <= 3, "This lightweight implementation supports l<=3"
    x, y, z = dirs.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, zx = x * y, y * z, z * x
    K = (lmax + 1) ** 2
    out = dirs.new_empty((*dirs.shape[:-1], K))
    # l=0
    out[..., 0] = 0.28209479177387814  # 1/(2sqrt(pi))
    if lmax >= 1:
        out[..., 1] = -0.4886025119029199 * y
        out[..., 2] = 0.4886025119029199 * z
        out[..., 3] = -0.4886025119029199 * x
    if lmax >= 2:
        out[..., 4] = 1.0925484305920792 * xy
        out[..., 5] = -1.0925484305920792 * yz
        out[..., 6] = 0.31539156525252005 * (3 * zz - 1)
        out[..., 7] = -1.0925484305920792 * zx
        out[..., 8] = 0.5462742152960396 * (xx - yy)
    if lmax >= 3:
        out[..., 9] = -0.5900435899266435 * y * (3 * xx - yy)
        out[...,10] = 2.890611442640554 * xy * z
        out[...,11] = -0.4570457994644658 * y * (5 * zz - 1)
        out[...,12] = 0.3731763325901154 * z * (5 * zz - 3)
        out[...,13] = -0.4570457994644658 * x * (5 * zz - 1)
        out[...,14] = 1.445305721320277 * z * (xx - yy)
        out[...,15] = -0.5900435899266435 * x * (xx - 3 * yy)
    return out


__all__ = ["sh_basis"]
