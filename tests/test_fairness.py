"""FR-LM-21 to FR-LM-24: fair daily lists, Nash social welfare, the exposure report, hold-back."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from lastfirstdate.fairness import (
    effectively_booked,
    exposure_cap,
    fair_day_lists,
    guarantee_bonus,
    held_back,
    nsw_day_lists,
)
from lastfirstdate.fairness_report import (
    exposure_report,
    gini,
    holdback_experiment,
    member_groups,
)
from lastfirstdate.simulate import run_month


def _skewed_day(n_choosers=12, n_cands=20, seed=0):
    """Choosers 0..11, candidates 12..31. A few candidates are liked by everyone."""
    rng = np.random.default_rng(seed)
    n = n_choosers + n_cands
    scores = np.zeros((n, n))
    popular = rng.random(n_cands) ** 3  # most candidates modest, a few very popular
    for i in range(n_choosers):
        scores[i, n_choosers:] = np.clip(popular + rng.normal(0, 0.05, n_cands), 0.01, 1.0)
    order = list(range(n_choosers))
    available = {i: np.r_[np.zeros(n_choosers, bool), np.ones(n_cands, bool)] for i in order}
    return order, available, scores, n


def _top_k(order, available, scores, bonus, size):
    out = {}
    for i in order:
        idx = np.flatnonzero(available[i])
        out[i] = idx[np.lexsort((idx, -(scores[i, idx] + bonus[idx])))][:size]
    return out


def _exposure(lists, n):
    return np.bincount(np.concatenate(list(lists.values())), minlength=n)


# ---------------------------------------------------------------- FR-LM-21


def test_guarantee_bonus_and_booked_rules():
    """FR-LM-21: bonus grows with missing dates and the month's end; booked means no room left."""
    dates = np.array([0, 2, 3, 0])
    paying = np.array([True, True, True, False])
    early = guarantee_bonus(dates, paying, 3, weeks_left=4, weight=1.0)
    late = guarantee_bonus(dates, paying, 3, weeks_left=1, weight=1.0)
    np.testing.assert_allclose(early, [0.75, 0.25, 0.0, 0.0])
    np.testing.assert_allclose(late, [3.0, 1.0, 0.0, 0.0])
    booked = effectively_booked(np.array([2, 3, 0]), np.array([2, 0, 1]), np.array([4, 4, 1]))
    assert booked.tolist() == [True, False, True]


def test_exposure_cap_formula():
    """FR-LM-21 / D7: cap = ceil(multiple * slots / candidates), at least 1."""
    assert exposure_cap(300, 100, 1.0) == 3
    assert exposure_cap(300, 100, 2.0) == 6
    assert exposure_cap(10, 100, 1.0) == 1
    assert exposure_cap(0, 0, 1.0) == 0


def test_fair_lists_respect_the_cap_and_availability():
    """FR-LM-21: nobody exceeds the daily cap; lists hold only available people, best first."""
    order, available, scores, n = _skewed_day()
    available[0][15] = False
    bonus = np.zeros(n)
    lists, cap = fair_day_lists(order, available, scores, bonus, size=5, multiple=1.0)
    assert _exposure(lists, n).max() <= cap
    for i, lst in lists.items():
        assert len(lst) <= 5 and len(set(lst.tolist())) == len(lst)
        assert all(available[i][j] for j in lst)
        s = scores[i, lst]
        assert np.all(np.diff(s) <= 1e-12)  # best first
    assert 15 not in lists[0]


def test_fair_lists_spread_attention_compared_with_top_k():
    """FR-LM-21: with popular candidates, the cap shares slots out; plain top-k piles them up."""
    order, available, scores, n = _skewed_day()
    bonus = np.zeros(n)
    fair, _ = fair_day_lists(order, available, scores, bonus, size=5, multiple=1.0)
    top = _top_k(order, available, scores, bonus, 5)
    cands = np.arange(12, n)
    assert gini(_exposure(fair, n)[cands]) < gini(_exposure(top, n)[cands])


def test_priority_moves_a_paying_member_who_is_behind_up_the_list():
    """FR-LM-21: the guarantee bonus lifts a behind paying member above an equal rival."""
    order, available, scores, n = _skewed_day()
    bonus = np.zeros(n)
    j = 12 + int(np.argmin(scores[0, 12:]))  # the least liked candidate
    bonus[j] = 2.0
    lists, _ = fair_day_lists(order, available, scores, bonus, size=5, multiple=2.0)
    assert j in lists[0]


@pytest.fixture(scope="module")
def months(cfg, default_city):
    pop, sc = default_city
    sprint3 = dataclasses.replace(
        cfg,
        lists=dataclasses.replace(cfg.lists, policy="basic"),
        dates=dataclasses.replace(cfg.dates, late_month_bonus=0.2),
    )
    return {
        "sprint3": run_month(pop, sprint3, cfg.simulation.seed, sc.mutual),
        "sprint4": run_month(pop, cfg, cfg.simulation.seed, sc.mutual),
    }


def test_fair_month_never_breaks_the_cap(months):
    """FR-LM-21: across a whole simulated month, nobody is ever shown above the daily cap."""
    assert months["sprint4"].cap_breaches == 0


def test_fair_lists_improve_the_guarantee_and_spread(cfg, default_city, months):
    """FR-LM-21: on the default city, Sprint 4 beats Sprint 3 on guarantee and exposure spread."""
    pop, _ = default_city
    groups = member_groups(pop)
    s3, s4 = months["sprint3"], months["sprint4"]
    assert s4.guarantee_share() > s3.guarantee_share()
    r3 = exposure_report(s3, groups, cfg.fairness.min_rate_ratio)
    r4 = exposure_report(s4, groups, cfg.fairness.min_rate_ratio)
    assert r4.gini < r3.gini


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FR-LM-21 target not met: about 82% of paying members with 10+ possible matches reach "
        "3 dates. Nearly all misses get too few mutual yeses, which no list rule can create. "
        "95% needs a business rule (offer the guarantee at 50+ possible matches) or a bigger city."
    ),
)
def test_guarantee_target_is_met(cfg, months):
    """FR-LM-21 (acceptance, D8): at least 95% of workable paying members get 3 dates.

    `strict=True` makes the suite fail the day this starts passing, so the marker is removed
    as soon as the target is reached.
    """
    assert months["sprint4"].guarantee_share() >= cfg.simulation.guarantee_target


# ---------------------------------------------------------------- FR-LM-22


def test_nsw_spreads_exposure_on_a_skewed_day():
    """FR-LM-22: on a day with a few very popular candidates, Nash welfare shares attention out."""
    order, available, scores, n = _skewed_day()
    bonus = np.zeros(n)
    nsw = nsw_day_lists(order, available, scores, bonus, size=5, epsilon=0.1)
    top = _top_k(order, available, scores, bonus, 5)
    cands = np.arange(12, n)
    assert gini(_exposure(nsw, n)[cands]) < gini(_exposure(top, n)[cands])
    assert all(len(lst) == 5 for lst in nsw.values())


def test_nsw_month_spreads_exposure_at_a_small_cost_in_dates(cfg, city600):
    """FR-LM-22: over a month, Nash welfare lists spread exposure more evenly than basic lists
    while keeping at least 95% of the dates."""
    pop, sc = city600
    groups = member_groups(pop)
    out = {}
    for policy in ("basic", "nsw"):
        c = dataclasses.replace(cfg, lists=dataclasses.replace(cfg.lists, policy=policy))
        m = run_month(pop, c, cfg.simulation.seed, sc.mutual)
        out[policy] = (exposure_report(m, groups, cfg.fairness.min_rate_ratio).gini, m.booked)
    assert out["nsw"][0] < out["basic"][0]
    assert out["nsw"][1] >= 0.95 * out["basic"][1]


def test_nsw_lazy_greedy_matches_plain_greedy():
    """FR-LM-22: the fast lazy version picks exactly what recomputing every gain would pick."""
    order, available, scores, n = _skewed_day(n_choosers=4, n_cands=6, seed=3)
    bonus = np.zeros(n)
    eps = 0.1
    fast = nsw_day_lists(order, available, scores, bonus, size=2, epsilon=eps)
    e = np.zeros(n)
    plain = {i: [] for i in order}
    while True:
        best, pick = -np.inf, None
        for i in order:
            if len(plain[i]) >= 2:
                continue
            for j in np.flatnonzero(available[i]):
                if j in plain[i]:
                    continue
                s = scores[i, j]
                g = (
                    np.log(eps + e[i] + s)
                    - np.log(eps + e[i])
                    + np.log(eps + e[j] + s)
                    - np.log(eps + e[j])
                )
                if g > best + 1e-12:
                    best, pick = g, (i, int(j))
        if pick is None:
            break
        i, j = pick
        plain[i].append(j)
        e[i] += scores[i, j]
        e[j] += scores[i, j]
    for i in order:
        assert set(fast[i].tolist()) == set(plain[i])


# ---------------------------------------------------------------- FR-LM-23


def test_gini_bounds():
    """FR-LM-23: Gini is 0 for equal exposure and (n - 1) / n when one person gets it all."""
    assert gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 8])) == pytest.approx(0.75)
    assert gini(np.array([])) == 0.0


def test_report_flags_a_group_shown_far_less():
    """FR-LM-23: a group shown at half the rate of others, for the same audience, is flagged."""
    n = 40
    group = np.arange(n) < 10
    month = SimpleNamespace(
        matchable=np.ones(n, bool),
        audience=np.full(n, 50),
        times_shown=np.where(group, 50, 100),
        dates=np.full(n, 3),
    )
    report = exposure_report(month, {"A": group, "B": ~group}, threshold=0.8)
    assert report.flagged == ["A"]
    assert "FLAG" in report.text()
    fair = SimpleNamespace(**{**vars(month), "times_shown": np.full(n, 100)})
    assert exposure_report(fair, {"A": group, "B": ~group}, threshold=0.8).flagged == []


def test_report_catches_a_deliberately_biased_engine(cfg, city600):
    """FR-LM-23: penalizing one fake group in the scores gets that group flagged."""
    pop, sc = city600
    groups = member_groups(pop)
    short = dataclasses.replace(cfg, lists=dataclasses.replace(cfg.lists, daily_size=5))
    biased = sc.mutual - 2.0 * groups["G1"][None, :]  # G1 candidates pushed to the bottom
    fair_month = run_month(pop, short, cfg.simulation.seed, sc.mutual)
    biased_month = run_month(pop, short, cfg.simulation.seed, biased)
    fair_flags = exposure_report(fair_month, groups, cfg.fairness.min_rate_ratio).flagged
    biased_flags = exposure_report(biased_month, groups, cfg.fairness.min_rate_ratio).flagged
    assert "G1" in biased_flags
    assert "G1" not in fair_flags


# ---------------------------------------------------------------- FR-LM-24


def test_held_back_rule():
    """FR-LM-24: threshold 0 is off; otherwise members at or above the threshold are hidden."""
    yeses = np.array([0, 2, 3, 5])
    assert not held_back(yeses, 0).any()
    assert held_back(yeses, 3).tolist() == [False, False, True, True]


def test_holdback_experiment_reports_the_difference(cfg, city600):
    """FR-LM-24: the experiment runs the same month with the rule off and on and reports both."""
    pop, sc = city600
    result = holdback_experiment(pop, cfg, cfg.simulation.seed, sc.mutual, threshold=3)
    assert cfg.lists.holdback_yeses == 0  # off by default
    text = result.text()
    assert "off" in text and "on" in text and "difference in dates" in text
    assert result.on.booked != result.off.booked or result.on.accepts != result.off.accepts
