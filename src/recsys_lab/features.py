"""User/item features for the ranker.

Two engines produce the same tables from the *train* split only:

* ``spark_job.py`` - the PySpark job (local mode in this repo), and
* ``build_features_pandas`` below - a small reference implementation used in
  unit tests and when Spark is not installed. A parity test checks the two agree.

Output layout (both engines)::

    <out>/user_features/part-*.csv   userId, user_n, user_mean, user_n_pos, user_last_year, aff_<genre>...
    <out>/item_features/part-*.csv   movieId, item_n, item_mean_smoothed, item_pos_rate, year, g_<genre>...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import GENRES, POSITIVE_RATING

SMOOTHING_M = 5.0


def col(genre: str) -> str:
    return genre.lower().replace("-", "_")


AFF_COLS = [f"aff_{col(g)}" for g in GENRES]
GENRE_COLS = [f"g_{col(g)}" for g in GENRES]
USER_COLS = ["userId", "user_n", "user_mean", "user_n_pos", "user_last_year", *AFF_COLS]
ITEM_COLS = ["movieId", "item_n", "item_mean_smoothed", "item_pos_rate", "year", *GENRE_COLS]


def _genre_flags(movies: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"movieId": movies["movieId"].astype("int64")})
    split = movies["genres"].astype(str).str.split("|")
    for g, c in zip(GENRES, GENRE_COLS):
        out[c] = split.apply(lambda gs, g=g: 1.0 if g in gs else 0.0)
    return out


def build_features_pandas(train: pd.DataFrame, movies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gm = float(train["rating"].mean())
    gp = float((train["rating"] >= POSITIVE_RATING).mean())
    flags = _genre_flags(movies)
    flags["year"] = movies["title"].astype(str).str.extract(r"\((\d{4})\)\s*$", expand=False).astype(float).values

    t = train.assign(pos=(train["rating"] >= POSITIVE_RATING).astype(float))
    agg = (
        t.groupby("movieId").agg(item_n=("rating", "size"), r_sum=("rating", "sum"), p_sum=("pos", "sum")).reset_index()
    )
    items = flags.merge(agg, on="movieId", how="left").fillna({"item_n": 0, "r_sum": 0.0, "p_sum": 0.0})
    items["item_n"] = items["item_n"].astype(float)
    items["item_mean_smoothed"] = (items["r_sum"] + SMOOTHING_M * gm) / (items["item_n"] + SMOOTHING_M)
    items["item_pos_rate"] = (items["p_sum"] + SMOOTHING_M * gp) / (items["item_n"] + SMOOTHING_M)
    items = items[ITEM_COLS].sort_values("movieId").reset_index(drop=True)

    u = (
        t.groupby("userId")
        .agg(
            user_n=("rating", "size"),
            user_mean=("rating", "mean"),
            user_n_pos=("pos", "sum"),
            last_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    # calendar year (UTC, fractional) of the user's latest train rating
    u["user_last_year"] = 1970.0 + u["last_ts"].astype(float) / (365.25 * 86400)
    pos = t[t["pos"] > 0].merge(flags[["movieId", *GENRE_COLS]], on="movieId", how="left").fillna(0.0)
    aff = pos.groupby("userId")[GENRE_COLS].sum().reset_index()
    users = u.merge(aff, on="userId", how="left").fillna(0.0)
    denom = np.maximum(users["user_n_pos"].to_numpy(), 1.0)
    for a, g in zip(AFF_COLS, GENRE_COLS):
        users[a] = users[g] / denom
    users["user_n"] = users["user_n"].astype(float)
    users = users[USER_COLS].sort_values("userId").reset_index(drop=True)
    return users, items


def write_features(users: pd.DataFrame, items: pd.DataFrame, out: Path) -> None:
    for name, df in (("user_features", users), ("item_features", items)):
        d = Path(out) / name
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("part-*.csv"):
            old.unlink()
        df.to_csv(d / "part-00000.csv", index=False)


def read_features(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _read(name: str, key: str, cols: list[str]) -> pd.DataFrame:
        parts = sorted((Path(out) / name).glob("part-*.csv"))
        if not parts:
            raise FileNotFoundError(f"no part-*.csv under {Path(out) / name}; run `recsys-lab features` first")
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        return df[cols].sort_values(key).reset_index(drop=True)

    return _read("user_features", "userId", USER_COLS), _read("item_features", "movieId", ITEM_COLS)
