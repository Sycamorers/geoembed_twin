from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import numpy as np


def match_embeddings(
    emb_a: np.ndarray | torch.Tensor,
    emb_b: np.ndarray | torch.Tensor,
    means_a: np.ndarray | torch.Tensor,
    means_b: np.ndarray | torch.Tensor,
    k: int = 5,
    gate: float = 0.3,
    w_e: float = 1.0,
    w_p: float = 0.2,
) -> Dict[str, List[Tuple[int, int]]]:
    ea = torch.as_tensor(emb_a, dtype=torch.float32)
    eb = torch.as_tensor(emb_b, dtype=torch.float32)
    ma = torch.as_tensor(means_a, dtype=torch.float32)
    mb = torch.as_tensor(means_b, dtype=torch.float32)

    ea_n = torch.nn.functional.normalize(ea, dim=1)
    eb_n = torch.nn.functional.normalize(eb, dim=1)

    sim = ea_n @ eb_n.T  # (Na,Nb)
    Na, Nb = sim.shape

    # spatial distances
    dists = torch.cdist(ma, mb)

    pairs: List[Tuple[int, int]] = []
    for i in range(Na):
        vals, idxs = torch.topk(sim[i], k=min(k, Nb))
        for s, j in zip(vals, idxs):
            j = int(j)
            spatial = dists[i, j]
            if spatial > gate:
                continue
            cost = w_e * (1 - s) + w_p * spatial
            # mutual check
            sim_back = sim[:, j]
            i_back = int(torch.argmax(sim_back))
            if i_back == i:
                pairs.append((i, j))
                break
    matched_a = {a for a, _ in pairs}
    matched_b = {b for _, b in pairs}
    unmatched_a = [i for i in range(Na) if i not in matched_a]
    unmatched_b = [j for j in range(Nb) if j not in matched_b]
    return {"pairs": pairs, "unmatched_a": unmatched_a, "unmatched_b": unmatched_b}


__all__ = ["match_embeddings"]
