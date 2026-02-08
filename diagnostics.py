from __future__ import annotations
import importlib
import json
import os
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DepReport:
    name: str
    present: bool
    version: Optional[str]
    note: str


@dataclass
class DoctorStatus:
    ok: bool
    core: List[DepReport]
    optional: List[DepReport]
    torch_info: Dict[str, object]
    env: Dict[str, str]


def _try_import(module: str) -> Tuple[bool, Optional[str]]:
    try:
        mod = importlib.import_module(module)
        ver = getattr(mod, "__version__", None)
        return True, ver
    except Exception:
        return False, None


def _format_line(prefix: str, status: DepReport) -> str:
    state = "OK" if status.present else "MISSING"
    ver = f" ({status.version})" if status.version else ""
    return f"{prefix} [{state}] {status.name}{ver}{' - ' + status.note if status.note else ''}"


def _check_group(label: str, options: List[Tuple[str, str]], note: str) -> DepReport:
    for mod, pip_name in options:
        present, ver = _try_import(mod)
        if present:
            return DepReport(name=label, present=True, version=ver, note=f"found via {mod}")
    # None matched
    pkg_hint = " or ".join(pkg for _, pkg in options)
    miss_note = note or f"install with: pip install {pkg_hint}"
    return DepReport(name=label, present=False, version=None, note=miss_note)


def _torch_summary() -> Dict[str, object]:
    info: Dict[str, object] = {"imported": False}
    ok, _ = _try_import("torch")
    if not ok:
        info["error"] = "torch not importable"
        return info
    import torch

    info["imported"] = True
    info["version"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        try:
            info["device_0"] = torch.cuda.get_device_name(0)
            info["capability"] = torch.cuda.get_device_capability(0)
        except Exception as exc:  # pragma: no cover - defensive
            info["device_error"] = str(exc)
    return info


def run_doctor(as_json: bool = False) -> DoctorStatus:
    repo_root = Path(__file__).resolve().parents[1]
    env = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "<unknown>"),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "repo_root": str(repo_root),
    }

    core_reports: List[DepReport] = []
    core_reports.append(_check_group("torch", [("torch", "torch")], "pip install torch --index-url https://download.pytorch.org/whl/cu121"))
    core_reports.append(_check_group("numpy", [("numpy", "numpy")], "pip install numpy"))
    core_reports.append(_check_group("matplotlib", [("matplotlib", "matplotlib")], "pip install matplotlib"))
    core_reports.append(_check_group("imageio|pillow", [("imageio", "imageio"), ("PIL", "pillow")], "pip install imageio pillow"))
    core_reports.append(_check_group("plyfile", [("plyfile", "plyfile")], "pip install plyfile"))

    optional_reports: List[DepReport] = []
    optional_reports.append(_check_group("geomloss", [("geomloss", "geomloss")], "pip install geomloss"))
    optional_reports.append(_check_group("faiss", [("faiss", "faiss-cpu")], "pip install faiss-cpu"))
    optional_reports.append(_check_group("open3d", [("open3d", "open3d")], "pip install open3d"))
    optional_reports.append(_check_group("rich", [("rich", "rich")], "pip install rich"))
    optional_reports.append(_check_group("pytest", [("pytest", "pytest")], "pip install pytest"))

    for r in core_reports:
        if r.name == "plyfile" and not r.present:
            try:  # attempt to detect built-in fallback
                from geoembed_twin.gaussians import io as gio  # type: ignore

                r.present = True
                r.note = "using built-in ASCII/binary PLY fallback; pip install plyfile for wider support"
            except Exception:
                pass

    torch_info = _torch_summary()
    ok = all(r.present for r in core_reports)

    status = DoctorStatus(ok=ok, core=core_reports, optional=optional_reports, torch_info=torch_info, env=env)

    if as_json:
        print(json.dumps(asdict(status), indent=2))
        return status

    print("GeoEmbedTwin doctor — current environment")
    print(f"- Python {env['python']} @ {env['executable']}")
    print(f"- Conda env: {env['conda_env']} ({env['conda_prefix']})")
    print(f"- Repo root: {env['repo_root']}")

    if torch_info.get("imported"):
        build = torch_info.get("cuda_build") or "cpu"
        print(f"- torch {torch_info.get('version')} (build CUDA {build})")
        print(f"  CUDA available: {torch_info.get('cuda_available')}")
        if torch_info.get("device_0"):
            cap = torch_info.get("capability")
            cap_str = f" capability={cap}" if cap else ""
            print(f"  Device 0: {torch_info['device_0']}{cap_str}")
    else:
        print("- torch not importable; install PyTorch matching your CUDA 12.1 stack.")

    print("\nCore dependencies (must exist):")
    for r in core_reports:
        print(_format_line(" *", r))

    print("\nOptional dependencies (used if available; safe to skip):")
    for r in optional_reports:
        print(_format_line(" -", r))
    print("   Use scripts/install_optional_deps.sh to install missing optional packages safely.")

    return status


__all__ = ["run_doctor", "DoctorStatus", "DepReport"]
