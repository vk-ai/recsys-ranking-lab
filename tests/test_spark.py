"""Parity between the PySpark job and the pandas reference (needs pyspark + Java 17+)."""

import pandas as pd
import pytest

from recsys_lab import data, features, spark_job

pytestmark = pytest.mark.spark


def test_spark_matches_pandas(spark, tmp_path, mini_dir):
    ratings, movies = data.load(mini_dir)
    sp = data.temporal_split(ratings, 5, 5, 5)
    train_csv = tmp_path / "train.csv"
    sp.train.to_csv(train_csv, index=False)
    spark_job.run(str(train_csv), str(mini_dir / "movies.csv"), str(tmp_path / "spark"), spark=spark)
    su, si = features.read_features(tmp_path / "spark")
    pu, pi = features.build_features_pandas(sp.train, movies)
    pd.testing.assert_frame_equal(su, pu, check_dtype=False, atol=1e-9)
    pd.testing.assert_frame_equal(si, pi, check_dtype=False, atol=1e-9)
    assert si.set_index("movieId").loc[5, "year"] == 1999  # quoted title with a comma parsed correctly


def test_pipeline_with_spark_engine(spark, tmp_path, mini_dir):
    from recsys_lab.pipeline import run

    from .test_pipeline import small_cfg

    rep = run(mini_dir, tmp_path / "a", tmp_path / "r", engine="spark", cfg=small_cfg(), log=lambda *_: None)
    assert rep["feature_engine"] == "spark"
    assert (tmp_path / "a" / "features_test" / "item_features").is_dir()
