import io
import zipfile

import pandas as pd
import pytest

from recsys_lab import data


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ml-latest-small/ratings.csv", "userId,movieId,rating,timestamp\n1,1,4.0,1\n")
        zf.writestr("ml-latest-small/movies.csv", "movieId,title,genres\n1,A (1999),Drama\n")
        zf.writestr("ml-latest-small/tags.csv", "ignored\n")
    return buf.getvalue()


def test_extract_verified_checks_sha256(tmp_path):
    blob = _zip_bytes()
    with pytest.raises(data.ChecksumError):
        data.extract_verified(blob, tmp_path / "x", "0" * 64)
    out = data.extract_verified(blob, tmp_path / "ok", data.sha256_bytes(blob))
    assert sorted(p.name for p in out.iterdir()) == ["movies.csv", "ratings.csv"]


def test_download_uses_url_and_checksum(tmp_path):
    src = tmp_path / "ml.zip"
    blob = _zip_bytes()
    src.write_bytes(blob)
    out = data.download(tmp_path / "dl", url=src.as_uri(), expected_sha256=data.sha256_bytes(blob))
    ratings, movies = data.load(out)
    assert len(ratings) == 1 and movies.loc[0, "year"] == 1999
    with pytest.raises(data.ChecksumError):
        data.download(tmp_path / "bad", url=src.as_uri(), expected_sha256="f" * 64)


def test_pinned_checksum_is_sha256_hex():
    assert len(data.DATASET_SHA256) == 64 and int(data.DATASET_SHA256, 16)


def test_load_parses_year_and_genres(mini_dir):
    _, movies = data.load(mini_dir)
    row = movies.set_index("movieId")
    assert row.loc[5, "title"] == "Fixture, The Movie (1999)" and row.loc[5, "year"] == 1999
    assert pd.isna(row.loc[6, "year"])
    g = data.genre_matrix(movies)
    assert g.shape == (len(movies), len(data.GENRES))
    assert g[movies.index[movies.movieId == 6][0]].sum() == 0


def test_temporal_split_is_per_user_last_by_time(mini_dir):
    ratings, _ = data.load(mini_dir)
    sp = data.temporal_split(ratings, n_test=5, n_val=5, min_train=5)
    assert len(sp.train) + len(sp.val) + len(sp.test) == len(ratings)
    key = ["userId", "movieId"]
    assert not set(map(tuple, sp.test[key].values)) & set(map(tuple, sp.train[key].values))
    for u, g in sp.test.groupby("userId"):
        assert len(g) == 5
        assert (
            g.timestamp.min()
            > sp.val[sp.val.userId == u].timestamp.max()
            >= sp.train[sp.train.userId == u].timestamp.max()
        )
    light = ratings.groupby("userId").size()
    light = set(light[light < 15].index)
    assert light and not light & set(sp.test.userId) and light <= set(sp.train.userId)


def test_ranker_window_zero_matches_split_and_never_uses_test(mini_dir):
    ratings, _ = data.load(mini_dir)
    sp = data.temporal_split(ratings, 5, 5, 5)
    h0, l0 = data.ranker_window(ratings, 0, 5, 5, 5)
    pd.testing.assert_frame_equal(
        h0.sort_values(["userId", "timestamp"]).reset_index(drop=True),
        sp.train.sort_values(["userId", "timestamp"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(l0, sp.val)
    test_keys = set(map(tuple, sp.test[["userId", "movieId"]].values))
    for w in range(3):
        h, lab = data.ranker_window(ratings, w, 5, 5, 5)
        assert not test_keys & set(map(tuple, h[["userId", "movieId"]].values))
        assert not test_keys & set(map(tuple, lab[["userId", "movieId"]].values))
        for u, g in lab.groupby("userId"):
            assert g.timestamp.min() > h[h.userId == u].timestamp.max()


def test_positives_threshold():
    df = pd.DataFrame({"userId": [1, 1, 2], "movieId": [10, 11, 12], "rating": [4.0, 3.5, 5.0]})
    assert data.positives(df) == {1: {10}, 2: {12}}
    assert data.seen(df) == {1: {10, 11}, 2: {12}}
