"""Dataset download (with checksum), loading and a per-user temporal split.

The dataset is MovieLens ml-latest-small from GroupLens. It is downloaded by
``recsys-lab download`` and never committed; see the GroupLens usage terms.
"""

from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATASET_SHA256 = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
DEFAULT_DATA_DIR = Path("data/ml-latest-small")

# The 19 MovieLens genres (the "(no genres listed)" marker maps to no genre).
GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
POSITIVE_RATING = 4.0


class ChecksumError(RuntimeError):
    pass


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def extract_verified(blob: bytes, dest: Path, expected_sha256: str = DATASET_SHA256) -> Path:
    """Verify the zip's SHA-256 and extract ratings.csv and movies.csv into ``dest``."""
    got = sha256_bytes(blob)
    if got != expected_sha256:
        raise ChecksumError(f"sha256 mismatch: expected {expected_sha256}, got {got}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            base = name.rsplit("/", 1)[-1]
            if base in {"ratings.csv", "movies.csv", "README.txt"}:
                (dest / base).write_bytes(zf.read(name))
    return dest


def download(dest: Path = DEFAULT_DATA_DIR, url: str = DATASET_URL, expected_sha256: str = DATASET_SHA256) -> Path:
    if (dest / "ratings.csv").exists() and (dest / "movies.csv").exists():
        return dest
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
        blob = resp.read()
    return extract_verified(blob, dest, expected_sha256)


_YEAR = re.compile(r"\((\d{4})\)\s*$")


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    movies["year"] = movies["title"].str.extract(_YEAR, expand=False).astype(float)
    return ratings, movies


def genre_matrix(movies: pd.DataFrame) -> np.ndarray:
    """Multi-hot (n_movies, n_genres) in ``movies`` row order."""
    out = np.zeros((len(movies), len(GENRES)), dtype=np.float32)
    index = {g: j for j, g in enumerate(GENRES)}
    for i, gs in enumerate(movies["genres"].astype(str)):
        for g in gs.split("|"):
            if g in index:
                out[i, index[g]] = 1.0
    return out


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def eval_users(self) -> np.ndarray:
        return np.sort(self.test["userId"].unique())


def temporal_split(ratings: pd.DataFrame, n_test: int = 10, n_val: int = 10, min_train: int = 10) -> Split:
    """Per-user leave-last-out by time.

    Users with at least ``n_test + n_val + min_train`` ratings get their last
    ``n_test`` ratings in test and the ``n_val`` before that in val (used to
    train the re-ranker). Everything else, and all ratings of lighter users,
    goes to train.
    """
    df = ratings.sort_values(["userId", "timestamp", "movieId"], kind="mergesort").reset_index(drop=True)
    df["_rank_from_end"] = df.groupby("userId").cumcount(ascending=False)
    counts = df.groupby("userId")["movieId"].transform("size")
    eligible = counts >= n_test + n_val + min_train
    is_test = eligible & (df["_rank_from_end"] < n_test)
    is_val = eligible & (df["_rank_from_end"] >= n_test) & (df["_rank_from_end"] < n_test + n_val)
    df = df.drop(columns="_rank_from_end")
    return Split(
        train=df[~is_test & ~is_val].reset_index(drop=True),
        val=df[is_val].reset_index(drop=True),
        test=df[is_test].reset_index(drop=True),
    )


def positives(df: pd.DataFrame, threshold: float = POSITIVE_RATING) -> dict[int, set[int]]:
    pos = df[df["rating"] >= threshold]
    return {int(u): set(map(int, g)) for u, g in pos.groupby("userId")["movieId"]}


def seen(df: pd.DataFrame) -> dict[int, set[int]]:
    return {int(u): set(map(int, g)) for u, g in df.groupby("userId")["movieId"]}


def ranker_window(
    ratings: pd.DataFrame, w: int, n_test: int = 10, n_val: int = 10, min_train: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """History/label pair for re-ranker training window ``w`` (0 = the val period).

    Window ``w`` labels each eligible user's ratings ``n_test + w*n_val`` to
    ``n_test + (w+1)*n_val`` from the end; the history is everything before
    that for those users plus the train ratings of everyone else. Window 0
    reproduces ``temporal_split``'s (train, val). Test ratings are never used.
    """
    df = ratings.sort_values(["userId", "timestamp", "movieId"], kind="mergesort").reset_index(drop=True)
    r = df.groupby("userId").cumcount(ascending=False)
    counts = df.groupby("userId")["movieId"].transform("size")
    base_eligible = counts >= n_test + n_val + min_train  # users that have a test/val period at all
    lo, hi = n_test + w * n_val, n_test + (w + 1) * n_val
    eligible = base_eligible & (counts >= hi + min_train)
    labels = df[eligible & (r >= lo) & (r < hi)]
    history = df[
        (eligible & (r >= hi)) | (~eligible & ~base_eligible) | (~eligible & base_eligible & (r >= n_test + n_val))
    ]
    return history.reset_index(drop=True), labels.reset_index(drop=True)
