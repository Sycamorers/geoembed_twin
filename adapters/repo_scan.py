from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ..utils import get_repo_root, add_sys_path

SF_KEYWORDS = ["submanifold", "sfvae", "embedding", "gs-embedding", "gaussian_embedding"]
GGS_KEYWORDS = ["geometry_grounded", "stochastic", "median depth", "gaussian_splatting", "geometry-grounded"]


@dataclass
class RepoMatches:
    sfvae: Optional[Path]
    ggs: Optional[Path]

    def as_json(self) -> str:
        return json.dumps({
            "sfvae": str(self.sfvae) if self.sfvae else None,
            "geometry_grounded": str(self.ggs) if self.ggs else None,
        }, indent=2)


def _has_keyword(text: str, keywords) -> bool:
    tl = text.lower()
    return any(k in tl for k in keywords)


def _dir_matches(path: Path, keywords) -> bool:
    if _has_keyword(path.name, keywords):
        return True
    for child in path.iterdir():
        name = child.name.lower()
        if _has_keyword(name, keywords):
            return True
        if child.is_file() and child.suffix in {".md", ".txt"}:
            try:
                sample = child.read_text(errors="ignore")
            except Exception:
                continue
            if _has_keyword(sample, keywords):
                return True
    return False


def scan_repos(repo_root: Optional[Path] = None) -> RepoMatches:
    root = repo_root or get_repo_root()
    sf_path: Optional[Path] = None
    ggs_path: Optional[Path] = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name == "geoembed_twin":
            continue
        if sf_path is None and _dir_matches(child, SF_KEYWORDS):
            sf_path = child
        if ggs_path is None and _dir_matches(child, GGS_KEYWORDS):
            ggs_path = child
        if sf_path and ggs_path:
            break
    return RepoMatches(sfvae=sf_path, ggs=ggs_path)


def register_repos_on_sys_path(matches: Optional[RepoMatches] = None) -> RepoMatches:
    matches = matches or scan_repos()
    if matches.sfvae:
        add_sys_path(matches.sfvae)
    if matches.ggs:
        add_sys_path(matches.ggs)
    return matches


__all__ = ["RepoMatches", "scan_repos", "register_repos_on_sys_path"]
