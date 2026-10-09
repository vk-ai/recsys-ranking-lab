import numpy as np
import torch

from recsys_lab import metrics, ranker, two_tower


def _toy(seed=0, n_users=60, n_items=40):
    """Users in group g like items in group g; side features carry the group."""
    rng = np.random.default_rng(seed)
    ug = rng.integers(0, 2, n_users)
    ig = np.arange(n_items) % 2
    us = np.eye(2, dtype=np.float32)[ug]
    its = np.eye(2, dtype=np.float32)[ig]
    pairs = [(u, i) for u in range(n_users) for i in rng.choice(np.where(ig == ug[u])[0], 6, replace=False)]
    u, i = map(np.array, zip(*pairs))
    return u, i, us, its, ug, ig


def test_two_tower_learns_group_preference_and_excludes_seen():
    u, i, us, its, ug, ig = _toy()
    model, losses = two_tower.train_two_tower(u, i, us, its, two_tower.TwoTowerConfig(epochs=25, batch_size=64, dim=8))
    assert losses[-1] < losses[0]
    uv, iv = two_tower.embed_all(model, us, its)
    assert np.allclose(np.linalg.norm(uv, axis=1), 1, atol=1e-5)
    users = np.arange(len(us))
    cand, sc = two_tower.retrieve(uv, iv, users, [{0, 1}] * len(users), 10)
    assert cand.shape == (len(users), 10) and np.all(np.diff(sc, axis=1) <= 1e-6)
    assert not np.isin(cand, [0, 1]).any()
    assert (ig[cand] == ug[:, None]).mean() > 0.9


def test_ranker_features_shape():
    cand = np.array([[0, 1], [1, 0]])
    x = ranker.build_ranker_features(
        np.ones((2, 2)), cand, np.array([0, 1]), np.zeros((2, 4)), np.ones((2, 3)), np.zeros((2, 4)), np.eye(2, 3)
    )
    assert x.shape == (2, 2, 2 + 2 + 4 + 4 + 1 + 3) and x.dtype == np.float32


def test_ranker_learns_planted_signal_with_early_stopping():
    rng = np.random.default_rng(0)
    y = (rng.random((300, 20)) < 0.1).astype(np.float32)
    y[:, 0] = np.maximum(y[:, 0], (y.sum(1) == 0))
    x = rng.normal(size=(300, 20, 5)).astype(np.float32)
    x[..., 2] += 2.0 * y
    model, log = ranker.train_ranker(x, y, ranker.RankerConfig(max_epochs=80, patience=10))
    assert log["holdout_lists"] == 60 and 1 <= log["best_epoch"] <= 80
    s = ranker.score(model, x)
    nd = np.mean([metrics.ndcg_at_k(list(np.argsort(-s[j])), set(np.where(y[j] > 0)[0]), 10) for j in range(300)])
    rand = np.mean([metrics.ndcg_at_k(list(range(20)), set(np.where(y[j] > 0)[0]), 10) for j in range(300)])
    assert nd > rand + 0.2
    assert isinstance(model(torch.zeros(1, 5)), torch.Tensor)
