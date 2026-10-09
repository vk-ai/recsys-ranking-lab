"""Second-stage re-ranker: a small MLP trained with a listwise softmax loss.

Each (user, candidate) row gets hand-built features: the two-tower score and
retrieval rank, item popularity and smoothed rating (from the PySpark/pandas
feature job), user activity, and user-genre affinity x item genre flags. The
loss is a softmax cross-entropy over each user's candidate list against the
(normalised) relevance labels, i.e. ListNet with top-1 probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class RankerConfig:
    hidden: int = 32
    max_epochs: int = 200
    patience: int = 20
    holdout_frac: float = 0.2  # share of lists held out for early stopping on NDCG@10
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 0


def build_ranker_features(
    tt_scores: np.ndarray,  # (U, K)
    cand: np.ndarray,  # (U, K) item indices
    user_rows: np.ndarray,  # (U,) row index into user tables
    user_num: np.ndarray,  # (n_users, 4) log1p(user_n), user_mean, log1p(user_n_pos), user_last_year
    user_aff: np.ndarray,  # (n_users, G)
    item_num: np.ndarray,  # (n_items, 4) log1p(item_n), item_mean_smoothed, item_pos_rate, release year
    item_genres: np.ndarray,  # (n_items, G)
) -> np.ndarray:
    n_users, k = cand.shape
    rank_feat = np.log1p(np.broadcast_to(np.arange(k), (n_users, k)))
    aff = user_aff[user_rows][:, None, :]  # (U, 1, G)
    g = item_genres[cand]  # (U, K, G)
    match = (aff * g).sum(-1) / np.maximum(g.sum(-1), 1.0)
    # release year relative to the user's latest activity: catches "new release" / era effects
    age = item_num[cand][..., 3] - user_num[user_rows][:, None, 3]
    feats = [
        age[..., None],
        np.abs(age)[..., None],
        tt_scores[..., None],
        rank_feat[..., None],
        item_num[cand],
        np.broadcast_to(user_num[user_rows][:, None, :], (n_users, k, user_num.shape[1])),
        match[..., None],
        aff * g,
    ]
    return np.concatenate(feats, axis=-1).astype(np.float32)


class Ranker(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sigma", torch.ones(n_features))
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mu) / self.sigma).squeeze(-1)


def _ndcg10(logits: torch.Tensor, y: torch.Tensor) -> float:
    k = min(10, y.shape[1])
    top = torch.topk(logits, k, dim=1).indices
    disc = 1.0 / torch.log2(torch.arange(k, dtype=torch.float32) + 2)
    dcg = (torch.gather(y, 1, top) * disc).sum(1)
    ideal = (torch.sort(y, dim=1, descending=True).values[:, :k] * disc).sum(1)
    return float((dcg / ideal.clamp_min(1e-9)).mean())


def train_ranker(x: np.ndarray, y: np.ndarray, cfg: RankerConfig = RankerConfig()) -> tuple[Ranker, dict]:
    """x: (U, K, F) features, y: (U, K) 0/1 labels. Lists without a positive are skipped.

    A random ``holdout_frac`` of the lists is held out; training stops when
    their NDCG@10 has not improved for ``patience`` epochs and the best
    weights are restored. With only a few hundred lists the MLP overfits fast
    without this. Returns the model and a training log.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    keep = np.where(y.sum(1) > 0)[0]
    perm = rng.permutation(keep)
    n_hold = int(round(len(perm) * cfg.holdout_frac)) if len(perm) >= 5 else 0
    hold, fit = perm[:n_hold], perm[n_hold:]
    xt, yt = torch.as_tensor(x[fit]), torch.as_tensor(y[fit], dtype=torch.float32)
    xh, yh = torch.as_tensor(x[hold]), torch.as_tensor(y[hold], dtype=torch.float32)
    model = Ranker(x.shape[-1], cfg.hidden)
    flat = xt.reshape(-1, xt.shape[-1])
    model.mu.copy_(flat.mean(0))
    model.sigma.copy_(flat.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    target = yt / yt.sum(1, keepdim=True)
    log = {"loss": [], "holdout_ndcg@10": [], "fit_lists": int(len(fit)), "holdout_lists": int(len(hold))}
    best, best_state, since = -1.0, None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        loss = -(target * torch.log_softmax(model(xt), dim=1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        log["loss"].append(float(loss.detach()))
        if n_hold == 0:
            continue
        model.eval()
        with torch.no_grad():
            nd = _ndcg10(model(xh), yh)
        log["holdout_ndcg@10"].append(nd)
        if nd > best + 1e-6:
            best, since = nd, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    log["best_epoch"] = int(np.argmax(log["holdout_ndcg@10"]) + 1) if log["holdout_ndcg@10"] else len(log["loss"])
    log["best_holdout_ndcg@10"] = best if n_hold else None
    model.eval()
    return model, log


@torch.no_grad()
def score(model: Ranker, x: np.ndarray) -> np.ndarray:
    return model(torch.as_tensor(x)).numpy()
