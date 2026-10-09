import numpy as np
import pandas as pd
import pytest

from recsys_lab import features


def test_hand_checked_values():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["A (2000)", "B (1990)", "C"],
            "genres": ["Drama", "Comedy|Drama", "(no genres listed)"],
        }
    )
    train = pd.DataFrame(
        {"userId": [1, 1, 2], "movieId": [1, 2, 1], "rating": [5.0, 3.0, 4.0], "timestamp": [0, 31_557_600, 10]}
    )
    users, items = features.build_features_pandas(train, movies)
    gm, gp = 4.0, 2 / 3
    i1 = items.set_index("movieId").loc[1]
    assert i1.item_n == 2 and i1.item_mean_smoothed == pytest.approx((9 + 5 * gm) / 7)
    assert i1.item_pos_rate == pytest.approx((2 + 5 * gp) / 7)
    i3 = items.set_index("movieId").loc[3]
    assert i3.item_n == 0 and i3.item_mean_smoothed == pytest.approx(gm) and np.isnan(i3.year)
    u1 = users.set_index("userId").loc[1]
    assert u1.user_n == 2 and u1.user_mean == 4.0 and u1.user_n_pos == 1
    assert u1.aff_drama == 1.0 and u1.aff_comedy == 0.0  # only the 5-star Drama is a positive
    assert u1.user_last_year == pytest.approx(1971.0)
    assert list(users.columns) == features.USER_COLS and list(items.columns) == features.ITEM_COLS


def test_roundtrip(tmp_path, mini_dir):
    from recsys_lab import data

    ratings, movies = data.load(mini_dir)
    u, i = features.build_features_pandas(ratings, movies)
    features.write_features(u, i, tmp_path)
    u2, i2 = features.read_features(tmp_path)
    pd.testing.assert_frame_equal(u, u2, check_dtype=False)
    pd.testing.assert_frame_equal(i, i2, check_dtype=False)
    with pytest.raises(FileNotFoundError):
        features.read_features(tmp_path / "missing")
