"""Command line: ``recsys-lab download | features | run``."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import data


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="recsys-lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download MovieLens ml-latest-small and verify its SHA-256")
    d.add_argument("--dest", type=Path, default=data.DEFAULT_DATA_DIR)
    f = sub.add_parser("features", help="build user/item features from a train split")
    f.add_argument("--train", type=Path, required=True)
    f.add_argument("--movies", type=Path, required=True)
    f.add_argument("--out", type=Path, default=Path("artifacts/features"))
    f.add_argument("--engine", choices=["spark", "pandas"], default="spark")
    r = sub.add_parser("run", help="run the full offline pipeline and write reports/")
    r.add_argument("--data", type=Path, default=data.DEFAULT_DATA_DIR)
    r.add_argument("--out", type=Path, default=Path("artifacts"))
    r.add_argument("--reports", type=Path, default=Path("reports"))
    r.add_argument("--engine", choices=["spark", "pandas"], default="pandas")
    r.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("multiseed", help="run the pipeline under several model seeds; write reports/multiseed.json")
    m.add_argument("--data", type=Path, default=data.DEFAULT_DATA_DIR)
    m.add_argument("--out", type=Path, default=Path("artifacts"))
    m.add_argument("--reports", type=Path, default=Path("reports"))
    m.add_argument("--engine", choices=["spark", "pandas"], default="pandas")
    m.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = p.parse_args(argv)

    if a.cmd == "download":
        print(f"verified and extracted to {data.download(a.dest)}")
    elif a.cmd == "features":
        if a.engine == "spark":
            from .spark_job import run as spark_run

            spark_run(str(a.train), str(a.movies), str(a.out))
        else:
            import pandas as pd

            from . import features

            train = pd.read_csv(a.train)
            _, movies = (
                data.load(a.movies.parent)
                if (a.movies.parent / "ratings.csv").exists()
                else (None, pd.read_csv(a.movies))
            )
            features.write_features(*features.build_features_pandas(train, movies), a.out)
        print(f"features written to {a.out}")
    elif a.cmd == "run":
        from .pipeline import PipelineConfig, run

        cfg = PipelineConfig(seed=a.seed)
        run(a.data, a.out, a.reports, a.engine, cfg)
    elif a.cmd == "multiseed":
        from .multiseed import run_seeds

        run_seeds(a.data, a.seeds, a.out, a.reports, a.engine)


if __name__ == "__main__":
    main()
