import math

import numpy as np
import pytest

from recsys_lab import metrics


def test_hand_checked_metrics():
    ranked = [1, 2, 3, 4, 5]
    rel = {2, 5, 9}
    assert metrics.recall_at_k(ranked, rel, 5) == pytest.approx(2 / 3)
    assert metrics.precision_at_k(ranked, rel, 5) == pytest.approx(2 / 5)
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert metrics.ndcg_at_k(ranked, rel, 5) == pytest.approx(dcg / idcg)
    assert metrics.average_precision_at_k(ranked, rel, 5) == pytest.approx((1 / 2 + 2 / 5) / 3)


def test_perfect_and_empty_lists():
    assert metrics.ndcg_at_k([7, 8], {7, 8}, 10) == pytest.approx(1.0)
    assert metrics.average_precision_at_k([7, 8, 1], {7, 8}, 3) == pytest.approx(1.0)
    assert metrics.ndcg_at_k([1, 2], {9}, 2) == 0.0
    with pytest.raises(ValueError):
        metrics.recall_at_k([1], set(), 1)


def test_ild_and_evaluate():
    g = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    assert metrics.intra_list_diversity([0, 1], g, 2) == 0.0
    assert metrics.intra_list_diversity([0, 2], g, 2) == 1.0
    out = metrics.evaluate({1: [0, 1], 2: [2, 0], 3: [1]}, {1: {0}, 2: {0}, 3: set()}, 2, g, n_items=3)
    assert out["users"] == 2  # user 3 has nothing relevant and is skipped
    assert out["recall@2"] == 1.0 and out["coverage@2"] == 1.0
    assert out["ndcg@2"] == pytest.approx((1 + 1 / math.log2(3)) / 2)
