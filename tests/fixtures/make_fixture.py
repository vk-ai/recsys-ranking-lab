"""Regenerate the tiny synthetic MovieLens-format fixture used by the tests.

    python tests/fixtures/make_fixture.py

This is NOT MovieLens data (GroupLens terms don't allow redistributing it). It is
random data with a planted genre-preference structure, in the same CSV format.
"""

from pathlib import Path

import numpy as np

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Horror", "Film-Noir"]


def main(out: Path = Path(__file__).parent / "mini") -> None:
    rng = np.random.default_rng(42)
    out.mkdir(parents=True, exist_ok=True)
    n_items, n_users = 80, 40
    item_g = [sorted(rng.choice(GENRES, size=rng.integers(1, 4), replace=False).tolist()) for _ in range(n_items)]
    with open(out / "movies.csv", "w") as f:
        f.write("movieId,title,genres\n")
        for i, gs in enumerate(item_g, start=1):
            if i == 5:
                f.write(f'{i},"Fixture, The Movie (1999)",{"|".join(gs)}\n')  # comma inside a quoted title
            elif i == 6:
                f.write(f"{i},Untitled fixture item,(no genres listed)\n")  # no year, no genres
            else:
                f.write(f"{i},Fixture item {i} ({1980 + i % 40}),{'|'.join(gs)}\n")
    with open(out / "ratings.csv", "w") as f:
        f.write("userId,movieId,rating,timestamp\n")
        for u in range(1, n_users + 1):
            likes = set(rng.choice(GENRES, size=2, replace=False))
            n = int(rng.integers(15, 61)) if u % 5 else 12  # every 5th user is "light"
            items = rng.choice(np.arange(1, n_items + 1), size=n, replace=False)
            t = 1_000_000_000 + u * 1000
            for it in items:
                hit = len(likes & set(item_g[it - 1]))
                r = float(np.clip(np.round((2.5 + 1.0 * hit + rng.normal(0, 0.7)) * 2) / 2, 0.5, 5.0))
                t += int(rng.integers(60, 86_400))
                f.write(f"{u},{it},{r},{t}\n")


if __name__ == "__main__":
    main()
