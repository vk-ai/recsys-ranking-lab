import numpy as np

from recsys_lab import abtest


def test_team_draft_interleaving_properties():
    rng = np.random.default_rng(0)
    a, b = [1, 2, 3, 4, 5], [5, 6, 1, 7, 8]
    for _ in range(50):
        slate, teams = abtest.team_draft_interleave(a, b, 6, rng)
        assert len(slate) == len(set(slate)) == 6
        assert abs(teams.count("a") - teams.count("b")) <= 1
        assert all((s in a) if t == "a" else (s in b) for s, t in zip(slate, teams))


def test_interleaving_runs_out_gracefully():
    slate, _ = abtest.team_draft_interleave([1], [1], 5, np.random.default_rng(0))
    assert slate == [1]


def _setup():
    users = np.arange(50)
    rel = {int(u): {100 + int(u)} for u in users}
    good = {int(u): [100 + int(u), 1, 2, 3] for u in users}
    bad = {int(u): [1, 2, 3, 100 + int(u)] for u in users}
    return users, rel, good, bad


def test_ab_test_detects_a_real_difference():
    users, rel, good, bad = _setup()
    r = abtest.ab_test(bad, good, users, rel, {}, n_users=6000, seed=0, n_bootstrap=300)
    cps = r["clicks_per_session"]
    assert cps["delta"] > 0 and cps["p_value"] < 0.001 and cps["bootstrap_ci_low"] > 0
    assert r["srm_p_value"] > 0.001 and r["n_control"] + r["n_treatment"] == 6000


def test_aa_and_interleaving():
    users, rel, good, bad = _setup()
    aa = abtest.aa_false_positive_rate(good, users, rel, {}, n_experiments=60, n_users=400, seed=0)
    assert aa["false_positive_rate"] <= 0.2
    il = abtest.interleaving_test(bad, good, users, rel, {}, k=4, n_users=1500, seed=0)
    assert il["wins_treatment"] > il["wins_control"] and il["sign_test_p_value"] < 0.01
