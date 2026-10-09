import numpy as np

from recsys_lab import constraints


def test_simulated_availability_rate_and_determinism():
    a = constraints.simulated_availability(20000, 0.85, seed=1)
    assert abs(a.mean() - 0.85) < 0.01
    assert np.array_equal(a, constraints.simulated_availability(20000, 0.85, seed=1))


def test_filter_available_keeps_order():
    c, s = constraints.filter_available(
        np.array([[3, 1, 2, 0]]), np.array([[0.9, 0.8, 0.7, 0.6]]), np.array([True, False, True, True])
    )
    assert c[0].tolist() == [3, 2, 0] and s[0].tolist() == [0.9, 0.7, 0.6]


G = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


def test_mmr_lambda_one_is_identity_and_lower_lambda_diversifies():
    items = np.array([0, 1, 2, 3, 4])
    scores = np.array([1.0, 0.95, 0.9, 0.5, 0.4])
    assert constraints.mmr(items, scores, G, 5, lam=1.0).tolist() == [0, 1, 2, 3, 4]
    top3 = constraints.mmr(items, scores, G, 3, lam=0.5).tolist()
    assert top3[0] == 0 and set(top3) == {0, 3, 4}


def test_genre_cap_and_backfill():
    items = np.array([0, 1, 2, 3, 4])
    assert constraints.genre_cap(items, G, 3, max_per_genre=1).tolist() == [0, 3, 4]
    assert constraints.genre_cap(items, G, 4, max_per_genre=1).tolist() == [0, 3, 4, 1]  # back-filled
    assert constraints.genre_cap(items, G, 5, max_per_genre=9).tolist() == [0, 1, 2, 3, 4]
