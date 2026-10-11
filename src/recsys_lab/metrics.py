"""Offline top-k metrics with binary relevance."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        raise ValueError("recall needs at least one relevant item")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(ranked[:k]) if it in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def average_precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """AP@k normalised by min(|relevant|, k), so a perfect list scores 1."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, it in enumerate(ranked[:k]):
        if it in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(relevant), k)


def intra_list_diversity(ranked: Sequence[int], item_genres: np.ndarray, k: int) -> float:
    """Mean pairwise (1 - genre Jaccard) over the top k."""
    top = list(ranked[:k])
    if len(top) < 2:
        return 0.0
    g = item_genres[top]
    inter = g @ g.T
    sizes = g.sum(1)
    union = sizes[:, None] + sizes[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    iu = np.triu_indices(len(top), 1)
    return float(np.mean(1.0 - sim[iu]))


def evaluate(
    lists: dict[int, Sequence[int]],
    relevant: dict[int, set[int]],
    k: int,
    item_genres: np.ndarray | None = None,
    n_items: int | None = None,
) -> dict[str, float]:
    """Mean metrics over users that have at least one relevant item."""
    users = [u for u in lists if relevant.get(u)]
    if not users:
        raise ValueError("no users with relevant items")
    out = {
        f"recall@{k}": float(np.mean([recall_at_k(lists[u], relevant[u], k) for u in users])),
        f"precision@{k}": float(np.mean([precision_at_k(lists[u], relevant[u], k) for u in users])),
        f"ndcg@{k}": float(np.mean([ndcg_at_k(lists[u], relevant[u], k) for u in users])),
        f"map@{k}": float(np.mean([average_precision_at_k(lists[u], relevant[u], k) for u in users])),
        "users": len(users),
    }
    if item_genres is not None:
        out[f"ild@{k}"] = float(np.mean([intra_list_diversity(lists[u], item_genres, k) for u in users]))
    if n_items:
        shown: set[int] = set()
        for u in users:
            shown.update(int(i) for i in list(lists[u])[:k])
        out[f"coverage@{k}"] = len(shown) / n_items
    return out


def beyond_accuracy(
    lists: dict[int, Sequence[int]],
    k: int,
    item_pop: np.ndarray,
    n_users: int,
    short_head_share: float = 0.2,
) -> dict[str, float]:
    """Coverage, novelty and popularity-bias metrics for top-k lists.

    ``item_pop[i]`` is how many history users interacted positively with item i.

    * ``novelty@k``  mean self-information -log2(p_i), p_i = (pop_i + 1) / (n_users + 1)
      (Zhou et al. 2010). Higher = less popular, more "surprising" items.
    * ``arp@k``      average recommendation popularity: mean pop_i of shown items.
    * ``aplt@k``     average percentage of long-tail items per list (Abdollahpouri et al. 2019):
      the short head is the most popular items that together hold ``short_head_share``
      of all positive interactions; everything else is long tail.
    * ``gini@k``     Gini index of how often each catalogue item is shown
      (0 = every item shown equally, 1 = all exposure on one item).
    * ``coverage@k`` share of the catalogue shown to at least one user.
    """
    item_pop = np.asarray(item_pop, dtype=np.float64)
    n_items = len(item_pop)
    tops = [list(map(int, lst[:k])) for lst in lists.values() if len(lst)]
    if not tops:
        raise ValueError("no lists")
    p = (item_pop + 1.0) / (n_users + 1.0)
    info = -np.log2(p)
    order = np.argsort(-item_pop, kind="stable")
    cum = np.cumsum(item_pop[order]) / max(item_pop.sum(), 1.0)
    n_head = int(np.searchsorted(cum, short_head_share) + 1)
    head = np.zeros(n_items, dtype=bool)
    head[order[:n_head]] = True
    exposure = np.zeros(n_items)
    for t in tops:
        np.add.at(exposure, t, 1.0)
    x = np.sort(exposure)
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())) if x.sum() > 0 else 0.0
    return {
        f"novelty@{k}": float(np.mean([info[t].mean() for t in tops])),
        f"arp@{k}": float(np.mean([item_pop[t].mean() for t in tops])),
        f"aplt@{k}": float(np.mean([(~head[t]).mean() for t in tops])),
        f"gini@{k}": gini,
        f"coverage@{k}": float((exposure > 0).mean()),
        "short_head_items": n_head,
    }


def per_user(
    metric, lists: dict[int, Sequence[int]], relevant: dict[int, set[int]], k: int, users: Iterable[int]
) -> np.ndarray:
    return np.array([metric(lists[u], relevant[u], k) for u in users])
