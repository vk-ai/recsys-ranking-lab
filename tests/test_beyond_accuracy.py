import json
import math

import numpy as np
import pytest

from recsys_lab import metrics
from recsys_lab.multiseed import render, run_seeds, summarize
from tests.test_pipeline import small_cfg

POP = np.array([50, 30, 10, 5, 3, 1, 1, 0], dtype=float)  # 100 interactions, 8 items


def test_hand_checked_novelty_arp_aplt():
    lists = {1: [0, 1], 2: [6, 7]}
    m = metrics.beyond_accuracy(lists, 2, POP, n_users=99)
    info = lambda c: -math.log2((c + 1) / 100)  # noqa: E731
    assert m["novelty@2"] == pytest.approx(((info(50) + info(30)) / 2 + (info(1) + info(0)) / 2) / 2)
    assert m["arp@2"] == pytest.approx(((50 + 30) / 2 + (1 + 0) / 2) / 2)
    # short head = items covering 20% of interactions = item 0 alone (50%)
    assert m["short_head_items"] == 1
    assert m["aplt@2"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["coverage@2"] == pytest.approx(4 / 8)


def test_gini_extremes():
    same = {u: [0] for u in range(10)}
    spread = {u: [u % 8] for u in range(8)}
    g_same = metrics.beyond_accuracy(same, 1, POP, 99)["gini@1"]
    g_spread = metrics.beyond_accuracy(spread, 1, POP, 99)["gini@1"]
    assert g_spread == pytest.approx(0.0)
    assert g_same == pytest.approx(1 - 1 / 8)  # all exposure on one of 8 items


def test_popular_lists_score_less_novel_than_tail_lists():
    head = {u: [0, 1, 2] for u in range(5)}
    tail = {u: [5, 6, 7] for u in range(5)}
    h = metrics.beyond_accuracy(head, 3, POP, 99)
    t = metrics.beyond_accuracy(tail, 3, POP, 99)
    assert t["novelty@3"] > h["novelty@3"] and t["aplt@3"] > h["aplt@3"] and t["arp@3"] < h["arp@3"]


def test_pipeline_reports_beyond_accuracy(tmp_path, mini_dir):
    from recsys_lab.pipeline import run

    rep = run(mini_dir, tmp_path / "art", tmp_path / "rep", engine="pandas", cfg=small_cfg(), log=lambda *_: None)
    pop, tt = rep["offline"]["popularity"], rep["offline"]["two_tower"]
    for m in rep["offline"].values():
        for key in ("aplt@5", "gini@5", "coverage@5"):
            assert 0.0 <= m[key] <= 1.0
        assert m["novelty@5"] > 0
    # the popularity baseline is by construction the most popularity-biased policy
    assert pop["arp@5"] >= tt["arp@5"] and pop["novelty@5"] <= tt["novelty@5"]
    assert "Beyond accuracy" in (tmp_path / "rep" / "REPORT.md").read_text()


def test_multiseed_varies_only_model_seeds(tmp_path, mini_dir):
    s = run_seeds(mini_dir, [0, 1, 2], tmp_path / "art", tmp_path / "rep", "pandas", small_cfg(), log=lambda *_: None)
    assert s["seeds"] == [0, 1, 2]
    row = s["metrics"]["two_tower"]["ndcg@5"]
    assert len(row["per_seed"]) == 3 and row["min"] <= row["mean"] <= row["max"]
    # popularity does not depend on model seeds -> zero spread
    assert s["metrics"]["popularity"]["ndcg@5"]["sd"] == pytest.approx(0.0)
    d = s["ranker_vs_two_tower_ndcg"]
    assert 0 <= d["seeds_ranker_better"] <= 3 and len(d["per_seed_paired_t_p"]) == 3
    saved = json.loads((tmp_path / "rep" / "multiseed.json").read_text())
    assert saved["metrics"].keys() == s["metrics"].keys()
    assert "± " in (tmp_path / "rep" / "MULTISEED.md").read_text()


def test_summarize_maths():
    def rep(nd_tt, nd_rk, p):
        base = {f"{m}@5": 0.5 for m in ("recall", "map", "ild", "coverage", "novelty", "aplt", "gini")}
        return {
            "config": {"k": 5},
            "dataset": "x",
            "feature_engine": "pandas",
            "env": {"seconds": 1},
            "offline": {"two_tower": {**base, "ndcg@5": nd_tt}, "two_tower+ranker": {**base, "ndcg@5": nd_rk}},
            "offline_significance": {"p_value": p},
        }

    s = summarize([rep(0.10, 0.12, 0.01), rep(0.10, 0.09, 0.4)], [0, 1])
    d = s["ranker_vs_two_tower_ndcg"]
    assert d["mean_delta"] == pytest.approx(0.005) and d["seeds_ranker_better"] == 1
    assert d["seeds_significant_at_0.05"] == 1
    assert s["metrics"]["two_tower+ranker"]["ndcg@5"]["sd"] == pytest.approx(np.std([0.12, 0.09], ddof=1))
    assert "2 model seeds" in render(s)
