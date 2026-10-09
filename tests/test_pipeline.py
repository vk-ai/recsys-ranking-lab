import json

from recsys_lab import ranker, two_tower
from recsys_lab.cli import main as cli_main
from recsys_lab.pipeline import PipelineConfig, run


def small_cfg() -> PipelineConfig:
    return PipelineConfig(
        k=5,
        n_candidates=20,
        n_test=5,
        n_val=5,
        min_train=5,
        ranker_windows=2,
        ab_users=400,
        aa_experiments=10,
        aa_users=200,
        interleave_users=200,
        bootstrap=200,
        two_tower=two_tower.TwoTowerConfig(epochs=3, batch_size=64, dim=8),
        ranker=ranker.RankerConfig(max_epochs=15, patience=5),
    )


def test_end_to_end_on_fixture(tmp_path, mini_dir):
    rep = run(mini_dir, tmp_path / "art", tmp_path / "rep", engine="pandas", cfg=small_cfg(), log=lambda *_: None)
    assert set(rep["offline"]) == {
        "popularity",
        "two_tower",
        "two_tower+ranker",
        "two_tower+ranker+mmr",
        "two_tower+ranker+genre_cap",
    }
    for m in rep["offline"].values():
        for key in ("recall@5", "ndcg@5", "map@5", "ild@5", "coverage@5"):
            assert 0.0 <= m[key] <= 1.0
    assert rep["availability"]["simulated"] is True
    assert 0 <= rep["offline_significance"]["p_value"] <= 1
    assert rep["simulated_ab_test"]["n_control"] + rep["simulated_ab_test"]["n_treatment"] == 400
    saved = json.loads((tmp_path / "rep" / "metrics.json").read_text())
    assert saved["split"] == rep["split"]
    md = (tmp_path / "rep" / "REPORT.md").read_text()
    assert "SIMULATED A/B test" in md and "| two_tower+ranker |" in md


def test_cli_features_pandas(tmp_path, mini_dir):
    cli_main(
        [
            "features",
            "--engine",
            "pandas",
            "--train",
            str(mini_dir / "ratings.csv"),
            "--movies",
            str(mini_dir / "movies.csv"),
            "--out",
            str(tmp_path / "f"),
        ]
    )
    assert (tmp_path / "f" / "user_features" / "part-00000.csv").exists()
