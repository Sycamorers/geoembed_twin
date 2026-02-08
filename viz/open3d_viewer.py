from __future__ import annotations
try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover
    raise ImportError("open3d is required for interactive viewing") from exc


def view_ply(path: str):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])


__all__ = ["view_ply"]
