"""Reference values computed with scipy 1.x (scipy is not a dependency)."""

import numpy as np
import pytest

from recsys_lab import stats


def test_student_t_tail_matches_scipy():
    assert stats.t_two_sided_p(2.0, 10) == pytest.approx(0.07338803477074037, rel=1e-9)
    assert stats.t_two_sided_p(1.96, 1e6) == pytest.approx(0.04999606758526985, rel=1e-6)
    assert stats.t_two_sided_p(3.5, 4.5) == pytest.approx(0.020541689969385577, rel=1e-8)
    assert stats.betainc(2, 3, 0.4) == pytest.approx(0.5248, rel=1e-10)
    assert stats._t_quantile(0.975, 7) == pytest.approx(2.364624251592784, rel=1e-6)


def test_welch_and_paired_t_match_scipy():
    r = stats.welch_t_test([1.0, 2, 3, 4, 5, 6], [2.0, 4, 5, 7, 9, 8, 10])
    assert r.statistic == pytest.approx(2.2031126811954658, rel=1e-9)
    assert r.p_value == pytest.approx(0.05128046041960558, rel=1e-6)
    assert r.ci_low < r.delta < r.ci_high
    p = stats.paired_t_test([0.1, 0.4, 0.35, 0.8, 0.5], [0.2, 0.5, 0.3, 0.9, 0.7])
    assert p.statistic == pytest.approx(2.25, rel=1e-9)
    assert p.p_value == pytest.approx(0.08764517650339461, rel=1e-6)


def test_sign_test_matches_scipy_binomtest():
    assert stats.binomial_sign_test(3, 17) == pytest.approx(0.0025768280029296875, rel=1e-9)
    assert stats.binomial_sign_test(0, 0) == 1.0


def test_z_test_bootstrap_power_srm():
    z = stats.two_proportion_z_test(100, 1000, 130, 1000)
    assert z.delta == pytest.approx(0.03) and z.p_value < 0.05 and z.ci_low > 0
    assert stats.two_proportion_z_test(100, 1000, 100, 1000).p_value == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 400), rng.normal(0.3, 1, 400)
    lo, hi = stats.bootstrap_diff_ci(a, b, 1000)
    w = stats.welch_t_test(a, b)
    assert lo < w.delta < hi and abs(lo - w.ci_low) < 0.05 and abs(hi - w.ci_high) < 0.05
    plo, phi = stats.paired_bootstrap_ci(a, a + 0.1, 500)
    assert plo == pytest.approx(0.1) and phi == pytest.approx(0.1)
    # classic textbook figure: ~3,841 per arm for 10% -> 12% at alpha .05, power .8
    assert 3800 < stats.sample_size_two_proportions(0.10, 0.02) < 3900
    assert stats.srm_check(5000, 5000) == pytest.approx(1.0)
    assert stats.srm_check(5300, 4700) < 1e-8


def test_permutation_detects_shift_and_not_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 200)
    assert stats.paired_permutation_test(a, a + 0.5, 999) < 0.01
    assert stats.paired_permutation_test(a, a + rng.normal(0, 0.05, 200), 999) > 0.01


def test_type_one_error_is_close_to_alpha():
    rng = np.random.default_rng(3)
    p = np.array([stats.welch_t_test(rng.normal(0, 1, 30), rng.normal(0, 2, 40)).p_value for _ in range(600)])
    assert 0.02 < (p < 0.05).mean() < 0.09
