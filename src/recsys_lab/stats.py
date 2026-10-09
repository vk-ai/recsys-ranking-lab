"""Significance tests, numpy + stdlib only (mirrors the approach in vk-ai/llm-eval-lab).

* Welch's two-sample t-test and the paired t-test, with an exact Student-t
  tail from the regularised incomplete beta function.
* Two-proportion z-test.
* Percentile bootstrap CIs (unpaired, for A/B arms; paired, for offline
  comparisons of two rankers on the same users).
* Paired sign-flip permutation test (the llm-eval-lab approach).
* Sample size per arm for a two-proportion test, and an SRM chi-square check.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def _betacf(a: float, b: float, x: float, max_iter: int = 300, eps: float = 1e-14) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    if math.isnan(t):
        return 1.0
    if math.isinf(t):
        return 0.0
    return betainc(df / 2.0, 0.5, df / (df + t * t))


@dataclass
class TestResult:
    test: str
    mean_a: float
    mean_b: float
    delta: float  # b - a
    statistic: float
    p_value: float
    ci_low: float  # 95% CI for delta
    ci_high: float
    df: float | None = None

    @property
    def relative_lift(self) -> float:
        return self.delta / self.mean_a if self.mean_a else float("nan")

    def to_dict(self) -> dict:
        d = {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in asdict(self).items()}
        d["relative_lift"] = float(f"{self.relative_lift:.6g}")
        return d


def welch_t_test(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> TestResult:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se = math.sqrt(va + vb)
    delta = float(b.mean() - a.mean())
    if se == 0:
        return TestResult(
            "welch_t", float(a.mean()), float(b.mean()), delta, 0.0, 1.0, delta, delta, float(na + nb - 2)
        )
    t = delta / se
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    tcrit = _t_quantile(1 - alpha / 2, df)
    return TestResult(
        "welch_t",
        float(a.mean()),
        float(b.mean()),
        delta,
        t,
        t_two_sided_p(t, df),
        delta - tcrit * se,
        delta + tcrit * se,
        df,
    )


def paired_t_test(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> TestResult:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired test needs equal-length arrays")
    d = b - a
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    mean = float(d.mean())
    if se == 0:
        return TestResult("paired_t", float(a.mean()), float(b.mean()), mean, 0.0, 1.0, mean, mean, float(n - 1))
    t = mean / se
    tcrit = _t_quantile(1 - alpha / 2, n - 1)
    return TestResult(
        "paired_t",
        float(a.mean()),
        float(b.mean()),
        mean,
        t,
        t_two_sided_p(t, n - 1),
        mean - tcrit * se,
        mean + tcrit * se,
        float(n - 1),
    )


def _t_quantile(q: float, df: float) -> float:
    """Inverse Student-t CDF by bisection on the exact tail (fine for CIs)."""
    target = 2 * (1 - q)  # two-sided p at the quantile
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def two_proportion_z_test(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> TestResult:
    pa, pb = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se0 if se0 > 0 else 0.0
    p = 2 * (1 - _N.cdf(abs(z)))
    se = math.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    zc = _N.inv_cdf(1 - alpha / 2)
    return TestResult("two_proportion_z", pa, pb, pb - pa, z, p, pb - pa - zc * se, pb - pa + zc * se)


def bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Unpaired percentile bootstrap CI for mean(b) - mean(a); arms resampled independently."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = a[rng.integers(0, len(a), (n_resamples, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n_resamples, len(b)))].mean(1)
    d = db - da
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Paired percentile bootstrap CI for mean(b - a): resample units, keep pairs together."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, float) - np.asarray(a, float)
    m = d[rng.integers(0, len(d), (n_resamples, len(d)))].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_resamples: int = 5000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value for mean(b - a) = 0."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, float) - np.asarray(a, float)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(d)))
    hits = int(np.sum(np.abs((signs * d).mean(1)) >= obs - 1e-12))
    return (hits + 1) / (n_resamples + 1)


def sample_size_two_proportions(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per arm to detect an absolute lift ``mde_abs`` on a base rate (normal approximation)."""
    p1, p2 = p_base, p_base + mde_abs
    pbar = (p1 + p2) / 2
    za, zb = _N.inv_cdf(1 - alpha / 2), _N.inv_cdf(power)
    n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / mde_abs**2
    return math.ceil(n)


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> float:
    """Sample-ratio-mismatch chi-square (1 dof) p-value. A tiny p means broken assignment."""
    n = n_a + n_b
    ea, eb = n * expected_share_a, n * (1 - expected_share_a)
    chi2 = (n_a - ea) ** 2 / ea + (n_b - eb) ** 2 / eb
    return math.erfc(math.sqrt(chi2 / 2))


def binomial_sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped), e.g. for interleaving wins vs losses."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    # log-space sum for large n
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    mx = max(logs)
    tail = math.exp(mx) * sum(math.exp(v - mx) for v in logs)
    return min(1.0, 2 * tail)
