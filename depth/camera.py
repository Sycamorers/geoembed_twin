from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Camera:
    R: torch.Tensor  # (3,3)
    t: torch.Tensor  # (3,)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to(self, device) -> "Camera":
        return Camera(R=self.R.to(device), t=self.t.to(device), fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, width=self.width, height=self.height)

    @property
    def origin(self) -> torch.Tensor:
        return -self.R.transpose(0, 1) @ self.t

    def generate_rays(self, device=None):
        device = device or self.R.device
        i, j = torch.meshgrid(
            torch.arange(self.height, device=device), torch.arange(self.width, device=device), indexing="ij"
        )
        x = (j - self.cx) / self.fx
        y = (i - self.cy) / self.fy
        dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)
        dirs_world = dirs_cam @ self.R  # (H,W,3)
        orig = self.origin.to(device)
        return orig, dirs_world


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor = None) -> torch.Tensor:
    up = up if up is not None else torch.tensor([0.0, 1.0, 0.0], device=eye.device)
    forward = (target - eye)
    forward = forward / torch.norm(forward)
    right = torch.cross(forward, up, dim=0)
    right = right / torch.norm(right)
    up_vec = torch.cross(right, forward, dim=0)
    R = torch.stack([right, up_vec, -forward], dim=1)
    return R


def sample_sphere_cameras(num: int = 3, radius: float = 2.5, width: int = 128, height: int = 128, fov: float = 60.0, device=None) -> List[Camera]:
    device = device or torch.device("cpu")
    cams: List[Camera] = []
    for k in range(num):
        phi = 2 * math.pi * k / num
        theta = math.pi / 3  # fixed elevation
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.cos(theta)
        z = radius * math.sin(theta) * math.sin(phi)
        eye = torch.tensor([x, y, z], device=device)
        target = torch.zeros(3, device=device)
        R = look_at(eye, target)
        t = -R @ eye
        fx = fy = 0.5 * width / math.tan(0.5 * math.radians(fov))
        cx = (width - 1) / 2
        cy = (height - 1) / 2
        cams.append(Camera(R=R, t=t, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height))
    return cams


__all__ = ["Camera", "sample_sphere_cameras", "look_at"]
