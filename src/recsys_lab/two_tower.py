"""Two-tower retrieval model trained with in-batch sampled softmax.

User tower: id embedding + projection of the user's genre-affinity vector.
Item tower: id embedding + projection of the item's genre flags.
Both go through a small MLP and are L2-normalised; the score is a dot product
divided by a temperature. Negatives are the other positives in the batch, with
the logQ popularity correction (Yi et al., 2019) so popular items are not
over-penalised just because they show up as in-batch negatives more often.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TwoTowerConfig:
    dim: int = 32
    epochs: int = 15
    batch_size: int = 512
    lr: float = 5e-3
    weight_decay: float = 1e-6
    temperature: float = 0.1
    logq_correction: bool = True
    seed: int = 0


class Tower(nn.Module):
    def __init__(self, n_ids: int, n_side: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_ids, dim)
        self.side = nn.Linear(n_side, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        nn.init.normal_(self.emb.weight, std=0.05)

    def forward(self, ids: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        h = self.emb(ids) + self.side(side)
        return F.normalize(h + self.mlp(h), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_user_side: int, n_item_side: int, dim: int = 32):
        super().__init__()
        self.user = Tower(n_users, n_user_side, dim)
        self.item = Tower(n_items, n_item_side, dim)


def train_two_tower(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    user_side: np.ndarray,
    item_side: np.ndarray,
    cfg: TwoTowerConfig = TwoTowerConfig(),
) -> tuple[TwoTower, list[float]]:
    """Train on (user, positive item) pairs. Returns the model and per-epoch mean loss."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = TwoTower(user_side.shape[0], item_side.shape[0], user_side.shape[1], item_side.shape[1], cfg.dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    us_t = torch.as_tensor(user_side, dtype=torch.float32)
    is_t = torch.as_tensor(item_side, dtype=torch.float32)
    counts = np.bincount(item_idx, minlength=item_side.shape[0]).astype(np.float64)
    log_q = torch.as_tensor(np.log(np.maximum(counts, 1.0) / max(len(item_idx), 1)), dtype=torch.float32)
    n = len(user_idx)
    losses: list[float] = []
    model.train()
    for _ in range(cfg.epochs):
        order = rng.permutation(n)
        tot, batches = 0.0, 0
        for s in range(0, n, cfg.batch_size):
            b = order[s : s + cfg.batch_size]
            if len(b) < 2:
                continue
            u = torch.as_tensor(user_idx[b], dtype=torch.long)
            i = torch.as_tensor(item_idx[b], dtype=torch.long)
            uv = model.user(u, us_t[u])
            iv = model.item(i, is_t[i])
            logits = uv @ iv.T / cfg.temperature
            if cfg.logq_correction:
                logits = logits - log_q[i].unsqueeze(0)
            # The same item can appear twice in a batch: mask those duplicates out of the negatives.
            dup = (i.unsqueeze(0) == i.unsqueeze(1)) & ~torch.eye(len(b), dtype=torch.bool)
            logits = logits.masked_fill(dup, -1e9)
            loss = F.cross_entropy(logits, torch.arange(len(b)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            batches += 1
        losses.append(tot / max(batches, 1))
    model.eval()
    return model, losses


@torch.no_grad()
def embed_all(model: TwoTower, user_side: np.ndarray, item_side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    us = torch.as_tensor(user_side, dtype=torch.float32)
    its = torch.as_tensor(item_side, dtype=torch.float32)
    u = model.user(torch.arange(us.shape[0]), us).numpy()
    i = model.item(torch.arange(its.shape[0]), its).numpy()
    return u, i


def retrieve(
    user_vecs: np.ndarray, item_vecs: np.ndarray, users: np.ndarray, exclude: list[set[int]], k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by dot product (brute force is fine for ~10k items).

    ``exclude[j]`` is the set of item *indices* to drop for ``users[j]``.
    Returns (item indices, scores), both shaped (len(users), k).
    """
    scores = user_vecs[users] @ item_vecs.T
    for row, ex in enumerate(exclude):
        if ex:
            scores[row, list(ex)] = -np.inf
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, top, axis=1)
    order = np.argsort(-top_scores, axis=1, kind="stable")
    return np.take_along_axis(top, order, axis=1), np.take_along_axis(top_scores, order, axis=1)
