from __future__ import annotations
import random
import os
from pathlib import Path
import numpy as np
import torch


def get_repo_root() -> Path:
    """Return the repository root (folder containing the geoembed_twin/ package).

    Assumes the package lives at <root>/geoembed_twin/.
    """
    return Path(__file__).resolve().parents[1]


def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def add_sys_path(p: Path) -> None:
    if p.exists():
        s = str(p.resolve())
        if s not in os.sys.path:
            os.sys.path.append(s)
