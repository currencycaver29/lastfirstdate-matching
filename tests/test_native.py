"""FR-LM-35: the Rust core gives the same answers as the Python reference."""

import dataclasses
import sqlite3
from contextlib import nullcontext
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from lastfirstdate import app, native
from lastfirstdate.app_fill import fill_database
from lastfirstdate.arrays import MemberArrays
from lastfirstdate.booking import earliest_common, window_bits
from lastfirstdate.cycle_runs import CycleRun
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
from lastfirstdate.synthetic import generate

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


# ---------------------------------------------------------------- date windows (FR-LM-37)


@pytest.mark.parametrize("seed", range(6))
def test_window_bits_match_python(seed):
    """FR-LM-35: free windows agree for random weekly slots, days (negative too), lead times,
    horizons and blocked days."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    weekly = rng.integers(0, 1 << 21, n)
    lead = int(rng.integers(0, 3))
    horizon = int(rng.integers(lead + 1, 22))
    today = int(rng.integers(-30, 400))
    blocked = rng.random((n, horizon)) < 0.3
    for mask in (None, blocked):
        rust, python = both(window_bits, weekly, today, lead, horizon, mask)
        np.testing.assert_array_equal(rust, python)
        assert rust.dtype == python.dtype == np.uint64


@pytest.mark.parametrize("seed", range(6))
def test_earliest_common_matches_python(seed):
    """FR-LM-35: the first shared window agrees one against many, pair by pair, for single
    pairs and for every pair at once, including pairs with no window."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 50))
    bits = rng.integers(0, 1 << 63, n, dtype=np.uint64) & rng.integers(
        0, 1 << 63, n, dtype=np.uint64
    )
    bits[rng.random(n) < 0.2] = 0
    other = rng.integers(0, 1 << 63, n, dtype=np.uint64)
    cases = [(bits[0], bits), (bits, other), (bits[0], other[0]), (bits[:, None], bits[None, :])]
    for a, b in cases:
        rust, python = both(earliest_common, a, b)
        np.testing.assert_array_equal(rust, python)
        assert np.shape(rust) == np.shape(python)


@pytest.mark.parametrize("seed", range(6))
def test_blocked_days_match_python(seed, p1cfg):
    """FR-LM-35: days blocked by a date that day, a full week or a full cap period agree,
    for members with and without a subscription, in their trial and in paid cycles."""
    rng = np.random.default_rng(seed)
    n = 40
    market = SimpleNamespace(
        today=int(rng.integers(0, 120)),
        members=[
            SimpleNamespace(
                paying=bool(rng.random() < 0.5), gender=str(rng.choice(["man", "woman"]))
            )
            for _ in range(n)
        ],
    )
    signups = np.where(rng.random(n) < 0.5, rng.integers(-60, 100, n), app.NO_SIGNUP)
    cap = app.Capacity()
    for _ in range(120):
        m, day = int(rng.integers(0, n)), market.today + int(rng.integers(-10, 20))
        cap.busy[m].add(day)
        cap.weeks[m, day // 7] += 1
        cap.periods[m, app._period(m, day, signups, p1cfg)] += 1
    members = np.sort(rng.choice(n, size=25, replace=False))
    rust = app._blocked_days_rust(market, cap, signups, p1cfg, members)
    python = app._blocked_days(market, cap, signups, p1cfg, members)
    np.testing.assert_array_equal(rust, python)
    assert rust.any() and not rust.all()


def test_a_day_of_lists_and_a_cycle_run_are_identical_on_both_backends(p1cfg):
    """FR-LM-35: the app's daily lists (windows, positions, match values) and a billing cycle
    of test runs come out the same with Rust and with Python."""
    small = dataclasses.replace(
        p1cfg, population=dataclasses.replace(p1cfg.population, n_members=160)
    )
    pop = generate(small, seed=11)
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    filled = fill_database(con, pop, small, date(2026, 11, 4), 11)
    market = app.load_market(con, small, app.Calendar(date(2026, 11, 2)), date(2026, 11, 4))
    feats = app.photo_feature_matrix(pop.photo_embeddings, small)
    tastes = app.fit_tastes(
        market, feats, app.calibration_answers(con, market, small), small, start=filled.start
    )
    scores = app.nightly_scores(market, tastes, feats, small)
    women = [int(market.ids[k]) for k, m in enumerate(market.members) if m.gender == "woman"]
    rust, python = both(app.day_lists, con, market, scores, women, small)
    assert rust and rust == python
    one = dataclasses.replace(small, cycle_runs=dataclasses.replace(small.cycle_runs, cycles=1))
    runs = [None, None]
    for k, ctx in enumerate((nullcontext(), native.python_only())):
        with ctx:
            runs[k] = CycleRun(pop, one, 11, scores.mutual).run("check")
    assert runs[0].states == runs[1].states and runs[0].picks == runs[1].picks
    np.testing.assert_array_equal(runs[0].delivered, runs[1].delivered)
