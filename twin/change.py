from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path
import numpy as np
import torch


@dataclass
class ChangeResult:
    removed: List[int]
    added: List[int]
    moved: List[int]
    unchanged: List[int]
    colors: torch.Tensor

    def summary(self) -> Dict[str, int]:
        return {
            "removed": len(self.removed),
            "added": len(self.added),
            "moved": len(self.moved),
            "unchanged": len(self.unchanged),
        }


def classify_changes(match_out: Dict[str, List[Tuple[int, int]]], means_a: torch.Tensor, means_b: torch.Tensor, move_thr: float = 0.15) -> ChangeResult:
    pairs = match_out["pairs"]
    unmatched_a = match_out["unmatched_a"]
    unmatched_b = match_out["unmatched_b"]

    removed = unmatched_a
    added = unmatched_b
    moved: List[int] = []
    unchanged: List[int] = []
    colors_b = torch.zeros((means_b.shape[0], 3))
    for a, b in pairs:
        dist = torch.norm(means_a[a] - means_b[b])
        if dist > move_thr:
            moved.append(b)
        else:
            unchanged.append(b)
    colors_b[:] = torch.tensor([0.7, 0.7, 0.7])
    for idx in added:
        colors_b[idx] = torch.tensor([0.1, 0.8, 0.1])
    for idx in moved:
        colors_b[idx] = torch.tensor([0.9, 0.6, 0.1])
    for _, b in pairs:
        if b not in moved:
            colors_b[b] = torch.tensor([0.1, 0.5, 0.9])
    return ChangeResult(removed=removed, added=added, moved=moved, unchanged=unchanged, colors=colors_b)


def save_change_summary(path: Path, match_out: Dict[str, List[Tuple[int, int]]], change_res: ChangeResult):
    data = {
        "pairs": match_out["pairs"],
        "unmatched_a": match_out["unmatched_a"],
        "unmatched_b": match_out["unmatched_b"],
        "summary": change_res.summary(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


__all__ = ["classify_changes", "ChangeResult", "save_change_summary"]
