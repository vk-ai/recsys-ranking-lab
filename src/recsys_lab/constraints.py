"""Business constraints applied after scoring: availability and diversity.

Availability here is SIMULATED. MovieLens has no notion of stock or opening
hours, so ``simulated_availability`` marks a seeded random share of items as
unavailable (think "restaurant closed" or "out of stock"). It exists to show
where the filter sits in the pipeline and how it changes the metrics, not to
model any real availability signal.
"""

from __future__ import annotations

import numpy as np


def simulated_availability(n_items: int, available_rate: float = 0.85, seed: int = 7) -> np.ndarray:
    """SIMULATED: boolean mask, True = item can be recommended right now."""
    if not 0.0 < available_rate <= 1.0:
        raise ValueError("available_rate must be in (0, 1]")
    return np.random.default_rng(seed).random(n_items) < available_rate


def filter_available(
    cands: np.ndarray, scores: np.ndarray, available: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Drop unavailable items from each candidate list (keeps order)."""
    out_c, out_s = [], []
    for c, s in zip(cands, scores):
        m = available[c]
        out_c.append(c[m])
        out_s.append(s[m])
    return out_c, out_s


def jaccard_sim(genres: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard similarity of multi-hot genre rows."""
    inter = genres @ genres.T
    sizes = genres.sum(1)
    union = sizes[:, None] + sizes[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def mmr(items: np.ndarray, scores: np.ndarray, item_genres: np.ndarray, k: int, lam: float = 0.7) -> np.ndarray:
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    Greedily picks argmax lam * rel(i) - (1 - lam) * max_{j in S} sim(i, j),
    with rel min-max scaled to [0, 1] within the list and sim = genre Jaccard.
    lam = 1 reproduces the input order.
    """
    items = np.asarray(items)
    if len(items) == 0:
        return items
    s = np.asarray(scores, dtype=np.float64)
    rel = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.ones_like(s)
    sim = jaccard_sim(item_genres[items])
    chosen: list[int] = []
    max_sim = np.zeros(len(items))
    left = np.ones(len(items), dtype=bool)
    for _ in range(min(k, len(items))):
        obj = np.where(left, lam * rel - (1 - lam) * max_sim, -np.inf)
        j = int(np.argmax(obj))
        chosen.append(j)
        left[j] = False
        max_sim = np.maximum(max_sim, sim[j])
    return items[chosen]


def genre_cap(items: np.ndarray, item_genres: np.ndarray, k: int, max_per_genre: int = 4) -> np.ndarray:
    """Walk the ranked list and skip an item if any of its genres is already at the cap.

    If the caps leave fewer than k items, the list is back-filled in the
    original order so the slate is never short.
    """
    counts = np.zeros(item_genres.shape[1], dtype=int)
    picked: list[int] = []
    skipped: list[int] = []
    for it in items:
        g = item_genres[it] > 0
        if len(picked) < k and not np.any(counts[g] >= max_per_genre):
            picked.append(int(it))
            counts[g] += 1
        else:
            skipped.append(int(it))
        if len(picked) == k:
            break
    for it in skipped:
        if len(picked) >= k:
            break
        picked.append(it)
    return np.asarray(picked, dtype=np.asarray(items).dtype if len(items) else int)
