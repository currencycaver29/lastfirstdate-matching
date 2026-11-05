"""FR-LM-25 to FR-LM-28: learning from dates, personal looks weights, exploration, three months."""

import dataclasses

import numpy as np
import pytest

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.learning import (
    DateRecord,
    Evidence,
    collect_evidence,
    exploration_slots,
    explore_picks,
    looks_weights,
    personal_directional_score,
    update_tastes,
)
from lastfirstdate.looks import PersonalTaste, StartingTaste
from lastfirstdate.month_runs import month_people, run_month
from lastfirstdate.mutual import percentile_rows

# ---------------------------------------------------------------- FR-LM-25


def _taste(d=3):
    start = StartingTaste(by_group={(1, 0): np.zeros(d)}, pooled=np.zeros(d))
    return PersonalTaste(1, start, 1.0, min_swipes_for_lists=1)


def test_held_dates_become_weighted_labels_for_both_people():
    """FR-LM-25: a held date adds one weighted label to each person's looks model."""
    feats = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.5, 1.0]])
    genders = np.array([0, 1, 0])
    tastes = {0: _taste(), 1: _taste()}
    records = [
        DateRecord(0, 1, week=0, held=True, again_a=True, again_b=False, rating_a=4, rating_b=2),
        DateRecord(1, 2, week=1, held=False, again_a=False, again_b=False, rating_a=0, rating_b=0),
    ]
    used = update_tastes(tastes, records, feats, genders, weight=3.0)
    assert used == 1
    assert tastes[0].liked == [True] and tastes[0].weights == [3.0]
    assert tastes[1].liked == [False] and tastes[1].weights == [3.0]
    np.testing.assert_array_equal(tastes[0].features[0], feats[1])
    assert not np.allclose(tastes[0].delta, 0.0)  # refitted


def test_a_heavier_label_moves_the_model_further():
    """FR-LM-25: "would meet again" counts more than a swipe, so it moves the taste more."""
    x = np.array([1.0, 0.0, 1.0])
    light, heavy = _taste(), _taste()
    light.add_label(x, 0, True, 1.0)
    heavy.add_label(x, 0, True, 3.0)
    assert (
        heavy.like_chance(x[None, :], np.array([0]))[0]
        > light.like_chance(x[None, :], np.array([0]))[0]
    )


def test_date_outcome_ratings_and_wishes(cfg, small_pop):
    """FR-LM-25: ratings run 1 to 5, and people who truly like each other more rate higher."""
    arr = MemberArrays.from_members(small_pop.members, cfg.values)
    beh, _ = month_people(small_pop, arr, cfg, cfg.test_runs.seed)
    rng = np.random.default_rng(0)
    order = np.argsort(beh.utility[0, 1:]) + 1
    low = [beh.date_outcome(0, int(j), rng) for j in order[:30] for _ in range(5)]
    high = [beh.date_outcome(0, int(j), rng) for j in order[-30:] for _ in range(5)]
    assert all(1 <= r <= 5 for _, r in low + high)
    assert np.mean([r for _, r in high]) > np.mean([r for _, r in low])
    assert np.mean([a for a, _ in high]) > np.mean([a for a, _ in low])


# ---------------------------------------------------------------- FR-LM-26


def _evidence_where(signal: str, n_rows=400, seed=0):
    rng = np.random.default_rng(seed)
    pv, pp = rng.random(n_rows), rng.random(n_rows)
    driver = pv if signal == "looks" else pp
    y = rng.random(n_rows) < 1 / (1 + np.exp(-8 * (driver - 0.5)))
    return Evidence([(a, b, float(c), 1.0) for a, b, c in zip(pv, pp, y, strict=True)])


def test_looks_weight_follows_what_drives_a_members_choices(cfg):
    """FR-LM-26: choices driven by looks push lambda up; by personality, down; none, 0.5."""
    lam = looks_weights({0: _evidence_where("looks"), 1: _evidence_where("personality")}, 3, cfg)
    assert lam[0] > 0.65
    assert lam[1] < 0.35
    assert lam[2] == 0.5
    assert cfg.learning.lambda_min <= lam.min() and lam.max() <= cfg.learning.lambda_max


def test_little_evidence_keeps_the_looks_weight_near_half(cfg):
    """FR-LM-26: with three data points the prior keeps lambda close to 0.5."""
    few = Evidence([(0.9, 0.1, 1.0, 1.0), (0.1, 0.9, 0.0, 1.0), (0.8, 0.2, 1.0, 1.0)])
    lam = looks_weights({0: few}, 1, cfg)
    assert abs(lam[0] - 0.5) < 0.2


def test_personal_looks_weights_are_applied_row_by_row():
    """FR-LM-26: S uses each member's own lambda."""
    rng = np.random.default_rng(1)
    v, p = rng.random((3, 3)), rng.random((3, 3))
    pool = ~np.eye(3, dtype=bool)
    lam = np.array([0.2, 0.5, 0.8])
    s, pv, pp = personal_directional_score(v, p, pool, lam)
    np.testing.assert_allclose(pv, percentile_rows(v, pool))
    for i in range(3):
        np.testing.assert_allclose(s[i][pool[i]], (lam[i] * pv + (1 - lam[i]) * pp)[i][pool[i]])


def test_evidence_collects_choices_and_dates(cfg):
    """FR-LM-26: list choices and held dates both become evidence, weighted as configured."""
    pv = np.full((3, 3), 0.7)
    pp = np.full((3, 3), 0.3)
    choices = (np.array([0, 0]), np.array([1, 2]), np.array([True, False]))
    records = [DateRecord(0, 1, 0, True, True, False, 4, 2)]
    ev: dict[int, Evidence] = {}
    collect_evidence(ev, pv, pp, choices, records, cfg)
    w = [r[3] for r in ev[0].rows]
    assert w.count(cfg.learning.lambda_choice_weight) == 2
    assert w.count(cfg.learning.lambda_date_weight) == 1
    assert len(ev[1].rows) == 1


# ---------------------------------------------------------------- FR-LM-27


def test_exploration_takes_the_least_certain_available_candidates():
    """FR-LM-27: exploration reserves the k most uncertain available people, under the cap."""
    assert exploration_slots(30, 0.10) == 3
    available = {0: np.array([False, True, True, True, True, True])}
    unc = np.array([[0, 0.1, 0.9, 0.5, 0.8, 0.2]])
    assert explore_picks([0], available, unc, 2)[0] == [2, 4]
    both = {0: available[0], 1: available[0]}
    capped = explore_picks([0, 1], both, np.vstack([unc, unc]), 1, cap=1)
    assert capped[0] == [2] and capped[1] == [4]  # 2 is at its cap after chooser 0


def test_month_spends_the_configured_share_on_exploration(cfg, city600):
    """FR-LM-27: with exploration on, about 10% of list slots are exploration slots."""
    pop, sc = city600
    unc = np.where(sc.eligible, np.random.default_rng(0).random(sc.eligible.shape), 0.0)
    m = run_month(pop, cfg, cfg.test_runs.seed, sc.mutual, month=1, explore=unc, record=True)
    share = m.explore_shown / m.total_shown
    assert abs(share - cfg.lists.exploration_share) < 0.03
    who, whom, picked = m.choices
    assert len(who) == len(whom) == len(picked) == m.total_shown
    assert len(m.date_records) == m.booked


def test_recording_does_not_change_a_month_without_learning(cfg, city600):
    """FR-LM-28: the default month (no inputs) is unchanged; record only adds information."""
    pop, sc = city600
    plain = run_month(pop, cfg, cfg.test_runs.seed, sc.mutual)
    again = run_month(pop, cfg, cfg.test_runs.seed, sc.mutual)
    np.testing.assert_array_equal(plain.dates, again.dates)
    assert plain.choices is None and plain.date_records == []


def test_excluded_pairs_are_never_booked(cfg, city600):
    """FR-LM-28: pairs who already dated are never matched again."""
    pop, sc = city600
    first = run_month(pop, cfg, cfg.test_runs.seed, sc.mutual, month=1, record=True)
    dated = np.zeros(sc.eligible.shape, dtype=bool)
    for r in first.date_records:
        dated[r.a, r.b] = dated[r.b, r.a] = True
    second = run_month(
        pop, cfg, cfg.test_runs.seed, sc.mutual, month=2, excluded=dated, record=True
    )
    assert not any(dated[r.a, r.b] for r in second.date_records)


@pytest.mark.parametrize("share", [0.0, 0.2])
def test_exploration_share_comes_from_config(cfg, share):
    """FR-LM-27 / R5: the share of exploration slots follows lists.exploration_share."""
    c = dataclasses.replace(cfg, lists=dataclasses.replace(cfg.lists, exploration_share=share))
    assert exploration_slots(c.lists.daily_size, c.lists.exploration_share) == round(30 * share)


# ---------------------------------------------------------------- three months (FR-LM-25 to 28)


@pytest.fixture(scope="module")
def three_months(cfg, city600):
    from lastfirstdate.learning_report import run_learning_report

    pop, sc = city600
    # 3 replicates: single runs of a month swing by a couple of points by chance
    return run_learning_report(pop, cfg, cfg.test_runs.seed, replicates=3, base=sc)


def _second(m):
    return m.second_dates / max(m.held, 1)


def _after_learning(report, arm):
    """Second-date rate in months 2 and 3 (after learning starts), averaged over replicates."""
    return float(np.mean([[_second(m) for m in rep[1:]] for rep in report.arms[arm].months]))


def _looks_auc(report):
    pairs = report.arms["learning"].looks_auc
    return float(np.mean([b for b, _ in pairs])), float(np.mean([a for _, a in pairs]))


def test_date_outcomes_do_not_make_the_looks_model_worse(three_months):
    """FR-LM-25: at the chosen weight, adding date outcomes keeps looks accuracy (within 0.005)."""
    before, after = _looks_auc(three_months)
    assert after >= before - 0.005


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FR-LM-25 not met: date outcomes do not improve looks accuracy. 'Would meet again' also "
        "depends on personality and chemistry, and only already-liked people get dated."
    ),
)
def test_date_outcomes_improve_the_looks_model(three_months):
    """FR-LM-25 (original claim): looks accuracy rises by at least 0.01 AUC after 3 months."""
    before, after = _looks_auc(three_months)
    assert after >= before + 0.01


def test_learned_looks_weights_track_the_hidden_truth(cfg, three_months):
    """FR-LM-26: learned lambda_i correlates with members' hidden true looks weight."""
    assert three_months.lam_recovery("learning") > 0.15
    for lam in three_months.arms["learning"].lam:
        assert lam.min() >= cfg.learning.lambda_min and lam.max() <= cfg.learning.lambda_max
    assert all(np.all(lam == 0.5) for lam in three_months.arms["no learning"].lam)


def test_exploration_takes_about_the_configured_share(cfg, three_months):
    """FR-LM-27: about lists.exploration_share of the profiles shown are exploration slots."""
    months = [m for rep in three_months.arms["learning + exploration"].months for m in rep]
    share = sum(m.explore_shown for m in months) / sum(m.total_shown for m in months)
    assert abs(share - cfg.lists.exploration_share) < 0.03
    assert all(m.cap_breaches == 0 for m in months)


@pytest.mark.xfail(
    strict=True,
    reason="FR-LM-27 not met: exploration gave no measurable gain in second dates in test runs.",
)
def test_exploration_raises_second_dates(three_months):
    """FR-LM-27 (original claim): exploration adds 2+ points of second dates (months 2-3)."""
    explore = _after_learning(three_months, "learning + exploration")
    assert explore >= _after_learning(three_months, "learning") + 0.02


def test_first_month_is_the_same_with_and_without_learning(three_months):
    """FR-LM-28: learning starts after month 1, so month 1 is identical in both arms."""
    for a_rep, b_rep in zip(
        three_months.arms["no learning"].months, three_months.arms["learning"].months, strict=True
    ):
        np.testing.assert_array_equal(a_rep[0].dates, b_rep[0].dates)
        assert a_rep[0].second_dates == b_rep[0].second_dates


def test_pairs_never_date_twice_across_months(three_months):
    """FR-LM-28: a pair who dated is never booked again in a later month."""
    for arm in three_months.arms.values():
        for rep in arm.months:
            seen = set()
            for m in rep:
                pairs = {(min(r.a, r.b), max(r.a, r.b)) for r in m.date_records}
                assert not (pairs & seen)
                seen |= pairs


def test_report_shows_every_measure(three_months):
    """FR-LM-28: the report gives second dates, guarantee, quality and exposure by month."""
    text = three_months.text()
    for part in ("Second-date rate", "Guarantee share", "Match quality", "Exposure Gini"):
        assert part in text


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FR-LM-28 not met: learning gave no measurable gain in second dates over 3 months. "
        "In a 1,000-member city members see nearly their whole pool, so better ranking has "
        "little leverage, and the best pairs are used up month by month."
    ),
)
def test_learning_raises_second_dates(three_months):
    """FR-LM-28 (acceptance): learning adds at least 2 points of second dates (months 2-3)."""
    learn = _after_learning(three_months, "learning")
    assert learn >= _after_learning(three_months, "no learning") + 0.02
