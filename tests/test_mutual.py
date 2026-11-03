"""FR-LM-18 to FR-LM-20: percentile scaling, the 50/50 directional score and the mutual score."""

import dataclasses

import numpy as np
import pytest

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.lists import pick_rights
from lastfirstdate.looks import CalibrationIncomplete, PersonalTaste, StartingTaste
from lastfirstdate.mutual import directional_score, looks_scores, mutual_score, percentile_rows
from lastfirstdate.mutual_report import RANKINGS, build_scores, top_quality
from lastfirstdate.simulate import month_people
from lastfirstdate.synthetic import generate

# ---------------------------------------------------------------- FR-LM-18


def test_percentiles_are_evenly_spaced_inside_each_pool():
    """FR-LM-18: k candidates get (0.5, 1.5, ..., k - 0.5) / k, average 0.5; others get 0."""
    rng = np.random.default_rng(0)
    x = rng.random((6, 6))
    pool = rng.random((6, 6)) < 0.7
    pct = percentile_rows(x, pool)
    for i in range(6):
        idx = np.flatnonzero(pool[i])
        if idx.size == 0:
            continue
        expected = (np.arange(idx.size) + 0.5) / idx.size
        np.testing.assert_allclose(np.sort(pct[i, idx]), expected)
        np.testing.assert_allclose(pct[i, idx].mean(), 0.5)
        # order is kept: a higher raw score never gets a lower percentile
        assert np.all(np.diff(pct[i, idx][np.argsort(x[i, idx])]) > 0)
    assert np.all(pct[~pool] == 0.0)


def test_percentiles_remove_differences_in_spread():
    """FR-LM-18: a narrow score and a wide score with the same order get the same percentiles."""
    pool = np.ones((1, 5), dtype=bool)
    wide = np.array([[0.01, 0.2, 0.5, 0.8, 0.99]])
    narrow = np.array([[0.45, 0.47, 0.5, 0.52, 0.55]])
    np.testing.assert_allclose(percentile_rows(wide, pool), percentile_rows(narrow, pool))


def test_tied_scores_share_a_percentile():
    """FR-LM-18: equal scores get equal percentiles."""
    pct = percentile_rows(np.array([[0.3, 0.3, 0.9]]), np.ones((1, 3), dtype=bool))
    assert pct[0, 0] == pct[0, 1] < pct[0, 2]


# ---------------------------------------------------------------- FR-LM-19


def test_directional_score_mixes_percentiles_by_the_looks_weight(cfg):
    """FR-LM-19: S = w * pct(V) + (1 - w) * pct(P), with w from config."""
    rng = np.random.default_rng(1)
    v, p = rng.random((5, 5)), rng.random((5, 5))
    pool = ~np.eye(5, dtype=bool)
    w = cfg.mix.looks_weight
    s = directional_score(v, p, pool, w)
    expected = w * percentile_rows(v, pool) + (1 - w) * percentile_rows(p, pool)
    np.testing.assert_allclose(s, expected)
    np.testing.assert_allclose(directional_score(v, p, pool, 1.0), percentile_rows(v, pool))
    np.testing.assert_allclose(directional_score(v, p, pool, 0.0), percentile_rows(p, pool))
    with pytest.raises(ValueError):
        directional_score(v, p, pool, 1.5)


def test_looks_scores_refuse_unfinished_calibration():
    """FR-LM-19 / R2: a member whose calibration has not finished cannot reach a list."""
    start = StartingTaste(by_group={(1, 0): np.zeros(3)}, pooled=np.zeros(3))
    taste = PersonalTaste(1, start, 1.0, min_swipes_for_lists=50)
    feats = np.ones((2, 3))
    pool = np.array([[False, True], [False, False]])
    with pytest.raises(CalibrationIncomplete):
        looks_scores({0: taste}, feats, np.array([1, 0]), pool)
    taste.complete = True
    assert looks_scores({0: taste}, feats, np.array([1, 0]), pool).shape == (2, 2)


# ---------------------------------------------------------------- FR-LM-20


def test_mutual_score_is_the_harmonic_mean():
    """FR-LM-20: M is symmetric, 0 if either side is 0, and pulled toward the smaller side."""
    s = np.array([[0.0, 0.9, 0.5], [0.1, 0.0, 0.0], [0.5, 0.7, 0.0]])
    m = mutual_score(s)
    np.testing.assert_allclose(m, m.T)
    assert np.isclose(m[0, 1], 2 * 0.9 * 0.1 / 1.0)  # 0.18, not the average 0.5
    assert m[1, 2] == 0.0  # one side is 0
    assert np.isclose(m[0, 2], 0.5)  # equal sides: the harmonic mean equals them
    with pytest.raises(ValueError):
        mutual_score(s, 1.0)
    weighted = mutual_score(np.array([[0.0, 0.9], [0.1, 0.0]]), 0.8)
    assert np.isclose(weighted[0, 1], 1 / (0.8 / 0.9 + 0.2 / 0.1))  # chooser 0 trusts herself
    rng = np.random.default_rng(2)
    r = rng.random((20, 20))
    mr = mutual_score(r)
    lo, hi = np.minimum(r, r.T), np.maximum(r, r.T)
    assert np.all(mr >= lo - 1e-12) and np.all(mr <= hi + 1e-12)
    assert np.all(mr <= (r + r.T) / 2 + 1e-12)


@pytest.fixture(scope="module")
def city(cfg):
    pop = generate(cfg, n=600)
    return pop, build_scores(pop, cfg, cfg.simulation.seed)


def test_scores_stay_inside_the_filtered_pool(city):
    """FR-LM-20: every score is 0 for pairs that failed Stage 1, and M is symmetric."""
    _, sc = city
    for x in (sc.looks, sc.directional, sc.mutual):
        assert np.all(x[~sc.eligible] == 0.0)
    np.testing.assert_allclose(sc.mutual, sc.mutual.T)
    assert sc.mutual.max() <= 1.0 and sc.n_calibrated > 0


def _top(cfg, city, matrices):
    pop, sc = city
    arr = MemberArrays.from_members(pop.members, cfg.values)
    behavior, _ = month_people(pop, arr, cfg, cfg.simulation.seed)
    rights = pick_rights(arr)
    return {k: top_quality(m, sc.eligible, rights, behavior, cfg) for k, m in matrices.items()}


def test_mutual_ranking_puts_people_who_say_yes_back_at_the_top(cfg, city):
    """FR-LM-20 (acceptance): M's top 10 are likelier to accept the chooser than one side's.

    Judged with the simulator's hidden truth. The 50/50 mutual score also beats either half
    alone on mutual yeses (she picks and he accepts), and is within 1.5 points of the
    chooser's one-sided view, which she likes slightly more (Sprint 3 study notes).
    """
    _, sc = city
    q = _top(cfg, city, {name: sc.ranking(name, cfg) for name in RANKINGS})
    assert q["mutual"].accept > q["one-sided"].accept
    assert q["mutual"].mutual > q["personality"].mutual
    assert q["mutual"].mutual > q["looks only"].mutual
    assert q["mutual"].mutual > q["one-sided"].mutual - 0.015
    assert q["one-sided"].pick > q["mutual"].pick


def test_chooser_weight_trades_her_picks_against_his_acceptance(cfg, city):
    """FR-LM-20: raising mix.chooser_weight moves the top of her list toward her own taste."""
    _, sc = city
    q = _top(cfg, city, {a: mutual_score(sc.directional, a) for a in (0.3, 0.5, 0.8)})
    assert q[0.3].pick < q[0.5].pick < q[0.8].pick
    assert q[0.3].accept > q[0.5].accept > q[0.8].accept


def test_looks_weight_comes_from_config(cfg, city):
    """FR-LM-19 / R5: changing mix.looks_weight changes S as configured."""
    _, sc = city
    only_looks = dataclasses.replace(cfg, mix=dataclasses.replace(cfg.mix, looks_weight=1.0))
    s1 = directional_score(sc.looks, sc.personality, sc.eligible, only_looks.mix.looks_weight)
    np.testing.assert_allclose(s1, percentile_rows(sc.looks, sc.eligible))
