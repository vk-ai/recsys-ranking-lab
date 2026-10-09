"""PySpark feature job (runs in local mode here; the same code runs on a cluster).

    python -m recsys_lab.spark_job --train artifacts/splits/train.csv \
        --movies data/ml-latest-small/movies.csv --out artifacts/features

Builds the user and item feature tables described in ``features.py`` from the
train split only, so no val/test information leaks into the ranker.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import GENRES, POSITIVE_RATING
from .features import AFF_COLS, GENRE_COLS, ITEM_COLS, SMOOTHING_M, USER_COLS


def get_spark(app: str = "recsys-features"):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[2]")
        .appName(app)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def build_features_spark(spark, train_csv: str, movies_csv: str):
    from pyspark.sql import functions as F

    ratings = spark.read.csv(train_csv, header=True, inferSchema=True).select(
        F.col("userId").cast("long"),
        F.col("movieId").cast("long"),
        F.col("rating").cast("double"),
        F.col("timestamp").cast("double"),
    )
    movies = spark.read.csv(movies_csv, header=True, inferSchema=False, quote='"', escape='"')
    ratings = ratings.withColumn("pos", (F.col("rating") >= F.lit(POSITIVE_RATING)).cast("double"))

    glob = ratings.agg(F.avg("rating").alias("gm"), F.avg("pos").alias("gp")).first()
    gm, gp, m = float(glob["gm"]), float(glob["gp"]), SMOOTHING_M

    genre_list = F.split(F.col("genres"), r"\|")
    flags = (
        movies.select(
            F.col("movieId").cast("long").alias("movieId"),
            F.regexp_extract(F.col("title"), r"\((\d{4})\)\s*$", 1).alias("_y"),
            *[F.array_contains(genre_list, g).cast("double").alias(c) for g, c in zip(GENRES, GENRE_COLS)],
        )
        .withColumn("year", F.when(F.col("_y") == "", None).otherwise(F.col("_y").cast("double")))
        .drop("_y")
    )

    item_agg = ratings.groupBy("movieId").agg(
        F.count("*").cast("double").alias("item_n"), F.sum("rating").alias("r_sum"), F.sum("pos").alias("p_sum")
    )
    items = (
        flags.join(item_agg, "movieId", "left")
        .fillna({"item_n": 0.0, "r_sum": 0.0, "p_sum": 0.0})
        .withColumn("item_mean_smoothed", (F.col("r_sum") + m * gm) / (F.col("item_n") + m))
        .withColumn("item_pos_rate", (F.col("p_sum") + m * gp) / (F.col("item_n") + m))
        .select(*ITEM_COLS)
    )

    users_base = ratings.groupBy("userId").agg(
        F.count("*").cast("double").alias("user_n"),
        F.avg("rating").alias("user_mean"),
        F.sum("pos").alias("user_n_pos"),
        (F.lit(1970.0) + F.max("timestamp") / F.lit(365.25 * 86400)).alias("user_last_year"),
    )
    pos_g = ratings.where(F.col("pos") > 0).join(flags.drop("year"), "movieId", "left").fillna(0.0)
    aff = pos_g.groupBy("userId").agg(*[F.sum(c).alias(c) for c in GENRE_COLS])
    denom = F.greatest(F.col("user_n_pos"), F.lit(1.0))
    users = (
        users_base.join(aff, "userId", "left")
        .fillna(0.0)
        .select(
            "userId",
            "user_n",
            "user_mean",
            "user_n_pos",
            "user_last_year",
            *[(F.col(g) / denom).alias(a) for g, a in zip(GENRE_COLS, AFF_COLS)],
        )
        .select(*USER_COLS)
    )
    return users, items


def run(train_csv: str, movies_csv: str, out: str, spark=None) -> None:
    from pyspark.sql import SparkSession

    # Only stop a session this call created; reuse (and leave running) an active one.
    own = spark is None and SparkSession.getActiveSession() is None
    spark = spark or get_spark()
    try:
        users, items = build_features_spark(spark, train_csv, movies_csv)
        for name, df, key in (("user_features", users, "userId"), ("item_features", items, "movieId")):
            df.orderBy(key).coalesce(1).write.mode("overwrite").option("header", True).csv(str(Path(out) / name))
    finally:
        if own:
            spark.stop()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--movies", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    run(a.train, a.movies, a.out)


if __name__ == "__main__":
    main()
