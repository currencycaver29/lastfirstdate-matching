"""FR-LM-35: the Rust core gives the same answers as the Python reference."""

import dataclasses

import numpy as np
import pytest

from lastfirstdate import native
from lastfirstdate.arrays import MemberArrays
from lastfirstdate.fairness import fair_day_lists
from lastfirstdate.filter import member_ok
from lastfirstdate.forecast import ForecastModel, gate_decision
from lastfirstdate.looks import (
    ArrayPhotoEncoder,
    PersonalTaste,
    StartingTaste,
    Swipe,
    calibrate,
    calibration_pool,
    fit_logistic,
    photo_features,
    select_pair,
)
from lastfirstdate.member_behavior import Behavior
from lastfirstdate.month_runs import run_month

pytestmark = pytest.mark.skipif(
    not native.available(), reason="Rust core not built (pip install -e . builds it)"
)


def both(fn, *args, **kwargs):
    """(Rust result, Python result) of the same call."""
    rust = fn(*args, **kwargs)
    with native.python_only():
        python = fn(*args, **kwargs)
    return rust, python


def test_rust_core_is_built_and_switchable():
    """FR-LM-35: the compiled core is present, used by default, and python_only() turns it
    off for a block."""
    assert native.available() and native.core.__version__
    with native.python_only():
        assert native.backend() == "python"
    assert native.enabled() == (native.backend() == "rust")


@pytest.mark.parametrize("seed", range(6))
def test_logistic_fit_matches_python(seed):
    """FR-LM-35: weights and Laplace covariance agree, with offsets, weights, a prior and a
    warm start."""
    rng = np.random.default_rng(seed)
    n, d = rng.integers(5, 80), rng.integers(2, 70)
    x = rng.normal(size=(n, d))
    y = rng.random(n) < 0.4
    kwargs = {
        "prior_mean": rng.normal(size=d) * 0.3,
        "offset": rng.normal(size=n),
        "sample_weight": rng.uniform(0.5, 3.0, n),
        "start": rng.normal(size=d) * 0.1,
    }
    (w_r, cov_r), (w_p, cov_p) = both(fit_logistic, x, y, 1.5, **kwargs)
    np.testing.assert_allclose(w_r, w_p, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(cov_r, cov_p, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(cov_r, cov_r.T)


def test_both_versions_reach_the_optimum_where_plain_newton_diverged():
    """FR-LM-35: on a hard problem where plain Newton's steps grew without end, step
    halving brings both versions to the optimum (gradient zero)."""
    rng = np.random.default_rng(0)  # n = 68 examples, d = 45 features
    n, d = rng.integers(5, 80), rng.integers(2, 70)
    x = rng.normal(size=(n, d))
    y = (rng.random(n) < 0.4).astype(float)
    m, o, s = rng.normal(size=d) * 0.3, rng.normal(size=n), rng.uniform(0.5, 3.0, n)
    start = rng.normal(size=d) * 0.1
    for w, _ in both(fit_logistic, x, y, 1.5, m, o, s, start):
        p = 1 / (1 + np.exp(-(o + x @ w)))
        grad = x.T @ (s * (p - y)) + 1.5 * (w - m)
        assert np.max(np.abs(grad)) < 1e-6


def test_logistic_fit_refuses_a_singular_problem():
    """FR-LM-35: with no penalty and duplicated features the Rust fit raises a clear error."""
    x = np.ones((4, 2))
    with pytest.raises(ValueError, match="positive definite"):
        fit_logistic(x, np.array([0, 1, 0, 1]), 0.0)


def _taste(rng, d):
    start = StartingTaste({(0, 1): rng.normal(size=d) * 0.2}, rng.normal(size=d) * 0.2)
    taste = PersonalTaste(0, start, 1.0, 10)
    a = rng.normal(size=(d, d))
    taste.cov = a @ a.T / d + np.eye(d) * 0.1
    taste.delta = rng.normal(size=d) * 0.1
    return taste


def test_uncertainty_matches_python():
    """FR-LM-35: s^2 = x' cov x for many candidates at once."""
    rng = np.random.default_rng(1)
    taste = _taste(rng, 65)
    x = rng.normal(size=(300, 65))
    rust, python = both(taste.uncertainty, x)
    np.testing.assert_allclose(rust, python, rtol=1e-11)


@pytest.mark.parametrize("seed", range(5))
def test_pair_search_matches_python(seed):
    """FR-LM-35: the same most-informative pair question, genders respected."""
    rng = np.random.default_rng(seed)
    taste = _taste(rng, 12)
    features = rng.normal(size=(80, 12))
    genders = rng.integers(0, 2, 80)
    remaining = rng.permutation(80)[:50]
    rust, python = both(select_pair, taste, remaining, features, genders)
    assert rust == python and rust is not None
    assert genders[rust[0]] == genders[rust[1]]


@pytest.mark.parametrize("seed", range(5))
def test_fair_lists_match_python(seed):
    """FR-LM-35: identical lists and cap, including tied scores and reserved slots."""
    rng = np.random.default_rng(seed)
    n = 120
    order = [int(i) for i in rng.permutation(n)[:70]]
    scores = np.round(rng.random((n, n)), 1)  # rounding creates many ties
    bonus = np.where(rng.random(n) < 0.2, 0.5, 0.0)
    available = {i: rng.random(n) < 0.4 for i in order}
    reserved = {i: [int(j) for j in rng.choice(n, 2, replace=False)] for i in order[:10]}
    (lists_r, cap_r), (lists_p, cap_p) = both(
        fair_day_lists, order, available, scores, bonus, 8, 1.5, reserved
    )
    assert cap_r == cap_p
    assert list(lists_r) == list(lists_p)
    for i in order:
        np.testing.assert_array_equal(lists_r[i], lists_p[i])


def _random_market(rng, n_women, n_men, density):
    n = n_women + n_men
    e = np.zeros((n, n), dtype=bool)
    block = rng.random((n_women, n_men)) < density
    e[:n_women, n_women:] = block
    e[n_women:, :n_women] = block.T
    return e, list(range(n_women)), list(range(n_women, n))


@pytest.mark.parametrize(
    "floor, fitted, target, current",
    [(3.0, 0.5, 0.9, 0), (0.0, 0.5, 0.9, 10), (0.0, 2.0, 0.5, 0), (0.0, 0.5, 0.99, 5)],
)
def test_gate_matches_python(p1cfg, floor, fitted, target, current):
    """FR-LM-35: the same admissions, deferrals and reason for stopping, whichever rule
    binds (ratio floor, fitted range, forecast) and with men already in the market."""
    rng = np.random.default_rng(int(floor * 10 + fitted * 3 + target * 7 + current))
    e, women, men = _random_market(rng, 200, 260, 0.15)
    model = dataclasses.replace(ForecastModel.load(), fitted_women_per_man=(fitted, 6.0))
    g = dataclasses.replace(p1cfg.gate, min_women_per_man=floor, target=target)
    cfg = dataclasses.replace(p1cfg, gate=g)
    current_men, waitlist = men[:current], men[current:]
    rust, python = both(gate_decision, model, e, women, current_men, waitlist, cfg)
    assert rust.admit == python.admit
    assert rust.deferred_small_pool == python.deferred_small_pool
    assert rust.stopped_because == python.stopped_because
    assert rust.women_per_man == pytest.approx(python.women_per_man, rel=1e-12)
    assert rust.forecast == pytest.approx(python.forecast, rel=1e-12)


@pytest.fixture(scope="module")
def p1cfg():
    from lastfirstdate.config import load

    return load("config/phase1.toml")


def test_calibration_and_month_are_identical_on_both_backends(cfg, default_city):
    """FR-LM-35: in the default city, members are shown the same calibration photos and a
    synthetic month books exactly the same dates with either backend."""
    pop, sc = default_city
    arr = MemberArrays.from_members(pop.members, cfg.values)
    rng = np.random.default_rng(3)
    behavior = Behavior(pop, arr, cfg, rng)
    _, feats = photo_features(ArrayPhotoEncoder(pop.photo_embeddings), len(pop), cfg)
    viewers = np.flatnonzero(member_ok(arr, cfg))
    swipes = []
    for v in viewers[:40]:
        likes = behavior.swipe_outcomes(int(v), 7)
        for j in calibration_pool(arr, cfg, int(v))[:20]:
            swipes.append(Swipe(int(arr.gender[v]), int(arr.gender[j]), feats[j], bool(likes[j])))
    start = StartingTaste.fit(swipes, cfg.calibration.w0_l2_penalty)

    def shown(v):
        likes = behavior.swipe_outcomes(v, 7)
        taste = PersonalTaste(int(arr.gender[v]), start, cfg.calibration.l2_penalty, 50)
        pool = calibration_pool(arr, cfg, v)
        return calibrate(taste, pool, feats, arr.gender, lambda j: bool(likes[j]), cfg)

    for v in viewers[100:112]:
        rust, python = both(shown, int(v))
        assert rust == python
    month_r, month_p = both(run_month, pop, cfg, 7, sc.mutual)
    np.testing.assert_array_equal(month_r.dates, month_p.dates)
