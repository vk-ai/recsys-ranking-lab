"""SIMULATED online experiment: A/B test, A/A calibration and team-draft interleaving.

Nothing here touches real users. Simulated users are drawn (with replacement)
from the held-out MovieLens test users; each one "sees" the top-k slate that a
policy produced for that real user offline, and clicks are drawn from a
position-based click model:

    P(click at position r) = examine(r) * attract(item)
    examine(r) = 1 / log2(r + 2)
    attract    = p_relevant  if the user rated the item >= 4 in the held-out period
                 p_watched   if the user rated it < 4 in the held-out period
                 p_noise     otherwise

The click-model parameters are made up. The point is the experiment
machinery (randomisation, SRM check, Welch t, z-test, bootstrap CIs, A/A
false-positive rate, power, interleaving), not the absolute numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import stats


@dataclass
class ClickModel:
    p_relevant: float = 0.35
    p_watched: float = 0.10
    p_noise: float = 0.02
    sessions_mean: float = 2.0  # extra sessions per user ~ Poisson(mean); every user has >= 1

    def click_probs(self, slate: Sequence[int], relevant: set[int], watched: set[int]) -> np.ndarray:
        r = np.arange(len(slate))
        examine = 1.0 / np.log2(r + 2)
        attract = np.array(
            [self.p_relevant if i in relevant else self.p_watched if i in watched else self.p_noise for i in slate]
        )
        return examine * attract


def _simulate_arm(
    users: np.ndarray, slates: dict[int, Sequence[int]], relevant, watched, cm: ClickModel, rng
) -> tuple[np.ndarray, np.ndarray]:
    clicks = np.zeros(len(users))
    sessions = 1 + rng.poisson(cm.sessions_mean, len(users))
    cache: dict[int, np.ndarray] = {}
    for j, u in enumerate(users):
        u = int(u)
        if u not in cache:
            cache[u] = cm.click_probs(slates[u], relevant.get(u, set()), watched.get(u, set()))
        p = cache[u]
        clicks[j] = (rng.random((sessions[j], len(p))) < p).sum()
    return clicks, sessions


def ab_test(
    control: dict[int, Sequence[int]],
    treatment: dict[int, Sequence[int]],
    base_users: np.ndarray,
    relevant: dict[int, set[int]],
    watched: dict[int, set[int]],
    n_users: int = 20000,
    cm: ClickModel = ClickModel(),
    seed: int = 0,
    n_bootstrap: int = 2000,
) -> dict:
    rng = np.random.default_rng(seed)
    pop = rng.choice(base_users, size=n_users, replace=True)
    assign = rng.random(n_users) < 0.5  # True = treatment
    ua, ub = pop[~assign], pop[assign]
    ca, sa = _simulate_arm(ua, control, relevant, watched, cm, rng)
    cb, sb = _simulate_arm(ub, treatment, relevant, watched, cm, rng)
    per_session_a, per_session_b = ca / sa, cb / sb
    welch = stats.welch_t_test(per_session_a, per_session_b)
    boot = stats.bootstrap_diff_ci(per_session_a, per_session_b, n_resamples=n_bootstrap, seed=seed)
    conv_a, conv_b = int((ca > 0).sum()), int((cb > 0).sum())
    z = stats.two_proportion_z_test(conv_a, len(ua), conv_b, len(ub))
    mde = max(abs(z.delta), 1e-4)
    return {
        "n_control": int(len(ua)),
        "n_treatment": int(len(ub)),
        "srm_p_value": float(f"{stats.srm_check(len(ua), len(ub)):.6g}"),
        "clicks_per_session": {
            **welch.to_dict(),
            "bootstrap_ci_low": float(f"{boot[0]:.6g}"),
            "bootstrap_ci_high": float(f"{boot[1]:.6g}"),
        },
        "any_click_rate": z.to_dict(),
        "users_per_arm_for_80pct_power_at_observed_lift": stats.sample_size_two_proportions(z.mean_a, mde),
    }


def aa_false_positive_rate(
    policy: dict[int, Sequence[int]],
    base_users: np.ndarray,
    relevant: dict[int, set[int]],
    watched: dict[int, set[int]],
    n_experiments: int = 200,
    n_users: int = 2000,
    cm: ClickModel = ClickModel(),
    seed: int = 1,
    alpha: float = 0.05,
) -> dict:
    """Run many A/A tests (same policy in both arms). About ``alpha`` of them should be 'significant'."""
    rng = np.random.default_rng(seed)
    pvals = []
    for _ in range(n_experiments):
        pop = rng.choice(base_users, size=n_users, replace=True)
        assign = rng.random(n_users) < 0.5
        ca, sa = _simulate_arm(pop[~assign], policy, relevant, watched, cm, rng)
        cb, sb = _simulate_arm(pop[assign], policy, relevant, watched, cm, rng)
        pvals.append(stats.welch_t_test(ca / sa, cb / sb).p_value)
    p = np.array(pvals)
    return {
        "experiments": n_experiments,
        "users_per_experiment": n_users,
        "alpha": alpha,
        "false_positive_rate": float((p < alpha).mean()),
    }


def team_draft_interleave(
    a: Sequence[int], b: Sequence[int], k: int, rng: np.random.Generator
) -> tuple[list[int], list[str]]:
    """Team-draft interleaving (Radlinski et al., 2008). Returns the slate and the team of each slot."""
    slate: list[int] = []
    teams: list[str] = []
    in_slate: set[int] = set()
    ia = ib = 0
    count_a = count_b = 0
    while len(slate) < k and (ia < len(a) or ib < len(b)):
        pick_a = count_a < count_b or (count_a == count_b and rng.random() < 0.5)
        src, idx = (a, ia) if pick_a else (b, ib)
        while idx < len(src) and src[idx] in in_slate:
            idx += 1
        if idx >= len(src):
            # this team ran out; let the other one pick
            pick_a = not pick_a
            src, idx = (a, ia) if pick_a else (b, ib)
            while idx < len(src) and src[idx] in in_slate:
                idx += 1
            if idx >= len(src):
                break
        slate.append(int(src[idx]))
        in_slate.add(int(src[idx]))
        teams.append("a" if pick_a else "b")
        if pick_a:
            ia, count_a = idx + 1, count_a + 1
        else:
            ib, count_b = idx + 1, count_b + 1
    return slate, teams


def interleaving_test(
    a: dict[int, Sequence[int]],
    b: dict[int, Sequence[int]],
    base_users: np.ndarray,
    relevant: dict[int, set[int]],
    watched: dict[int, set[int]],
    k: int = 10,
    n_users: int = 5000,
    cm: ClickModel = ClickModel(),
    seed: int = 2,
) -> dict:
    rng = np.random.default_rng(seed)
    wins_a = wins_b = ties = 0
    for u in rng.choice(base_users, size=n_users, replace=True):
        u = int(u)
        slate, teams = team_draft_interleave(list(a[u])[:k], list(b[u])[:k], k, rng)
        p = cm.click_probs(slate, relevant.get(u, set()), watched.get(u, set()))
        clicked = rng.random(len(p)) < p
        ca = sum(1 for c, t in zip(clicked, teams) if c and t == "a")
        cb = sum(1 for c, t in zip(clicked, teams) if c and t == "b")
        if ca > cb:
            wins_a += 1
        elif cb > ca:
            wins_b += 1
        else:
            ties += 1
    decided = wins_a + wins_b
    return {
        "users": n_users,
        "wins_control": wins_a,
        "wins_treatment": wins_b,
        "ties": ties,
        "delta_ab": float(f"{(wins_b - wins_a) / decided:.6g}") if decided else 0.0,
        "sign_test_p_value": float(f"{stats.binomial_sign_test(wins_b, wins_a):.6g}"),
    }
