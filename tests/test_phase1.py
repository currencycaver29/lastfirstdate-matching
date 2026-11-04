"""FR-LM-29 to FR-LM-34: lastfirstdate's real rules, the forecast and the ratio gate."""

import dataclasses

import numpy as np
import pytest

from lastfirstdate import forecast_fit
from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import ConfigError, ValuesQuestion, load, validate
from lastfirstdate.filter import eligibility_matrix, pair_passes
from lastfirstdate.forecast import (
    PROGRESS_FEATURES,
    START_FEATURES,
    ForecastModel,
    at_risk,
    fit_forecast,
    forecast_fulfilment,
    gate_decision,
    market_features,
    start_matrix,
)
from lastfirstdate.looks import PersonalTaste, StartingTaste
from lastfirstdate.looks_report import auc
from lastfirstdate.personality import (
    TagEncoder,
    WordEncoder,
    interest_similarity,
    personality_matrix,
    values_compatibility,
)
from lastfirstdate.phase1_report import gate_trial
from lastfirstdate.simulator import Behavior
from lastfirstdate.synthetic import generate, summary

PHASE1 = "config/phase1.toml"


@pytest.fixture(scope="module")
def p1cfg():
    return load(PHASE1)


@pytest.fixture(scope="module")
def p1pop(p1cfg):
    return generate(p1cfg)


# ---------------------------------------------------------------- FR-LM-29


def test_phase1_population_follows_the_sites_rules(p1cfg, p1pop):
    """FR-LM-29: 100 men and 300 women in one zone, man-woman only, every man pays."""
    men = [m for m in p1pop.members if m.gender == "man"]
    women = [m for m in p1pop.members if m.gender == "woman"]
    assert (len(men), len(women)) == (100, 300)
    assert all(m.seeking == {"woman"} for m in men) and all(w.seeking == {"man"} for w in women)
    assert all(m.paying for m in men) and not any(w.paying for w in women)
    assert all(m.verified for m in p1pop.members)
    zone = p1cfg.population.areas[0]
    assert len(p1cfg.population.areas) == 1
    assert all(abs(m.lat - zone.lat) < 0.08 and abs(m.lon - zone.lon) < 0.08 for m in p1pop.members)
    assert p1cfg.dates.cap_per_month == p1cfg.dates.cap_per_month_free == 3
    assert p1pop.members[0].bio.startswith("A perfect Sunday looks like")


def test_default_population_is_unchanged_by_sprint_6(cfg):
    """FR-LM-29: moving areas into the settings file leaves the default fake city as it was."""
    text = summary(generate(cfg), cfg)
    for expected in (
        "1000 members",
        "men 554, women 446",
        "paying: 283 men, 146 women",
        "straight 903, gay or lesbian 66, bi 31",
        "verified 922",
    ):
        assert expected in text


def test_areas_are_validated(cfg):
    """FR-LM-29 / R5: the settings file must list at least one area with a positive share."""
    no_areas = dataclasses.replace(cfg, population=dataclasses.replace(cfg.population, areas=()))
    with pytest.raises(ConfigError):
        validate(no_areas)


# ---------------------------------------------------------------- FR-LM-30


def _with_answers(pop, cfg, answers):
    base = pop.members[0]
    out = []
    for k, ans in enumerate(answers):
        values = {q.key: ans for q in cfg.values}
        out.append(
            dataclasses.replace(
                pop.members[k],
                id=k,
                values=values,
                accepts={q.key: frozenset(q.options) for q in cfg.values},
            )
        )
    del base
    return out


def test_closeness_scores_by_distance_between_answers(p1cfg, p1pop):
    """FR-LM-30: same answer 1, neighbors 0.5, opposite ends 0 (three-option questions)."""
    kids = next(q for q in p1cfg.values if q.key == "kids")
    assert kids.mode == "closeness"
    members = [dataclasses.replace(m, id=k) for k, m in enumerate(p1pop.members[:3])]
    members = [
        dataclasses.replace(m, values={q.key: q.options[k] for q in p1cfg.values})
        for k, m in enumerate(members)
    ]
    a = values_compatibility(MemberArrays.from_members(members, p1cfg.values), p1cfg.values)
    np.testing.assert_allclose(np.diag(a), 1.0)
    assert np.isclose(a[0, 1], 0.5) and np.isclose(a[0, 2], 0.0) and np.isclose(a[1, 2], 0.5)


def test_incompatible_answers_rule_a_pair_out(p1cfg, p1pop):
    """FR-LM-30: "Want them" with "Not for me" is a dealbreaker in both filters."""
    e = eligibility_matrix(MemberArrays.from_members(p1pop.members, p1cfg.values), p1cfg)
    kids = {m.id: m.values["kids"] for m in p1pop.members}
    bad = [
        (i, j)
        for i, j in zip(*np.nonzero(e), strict=True)
        if {kids[i], kids[j]} == {"Want them", "Not for me"}
    ]
    assert bad == []
    man = next(m for m in p1pop.members if m.gender == "man")
    woman = next(m for m in p1pop.members if m.gender == "woman" and pair_passes(man, m, p1cfg))
    flipped = dataclasses.replace(
        woman,
        values={
            **woman.values,
            "kids": "Not for me" if man.values["kids"] == "Want them" else "Want them",
        },
    )
    if {man.values["kids"], flipped.values["kids"]} == {"Want them", "Not for me"}:
        assert not pair_passes(man, flipped, p1cfg)


def test_accepts_mode_is_still_the_default(cfg):
    """FR-LM-30: the research profile keeps "accepts" questions, so earlier results hold."""
    assert all(q.mode == "accepts" for q in cfg.values)
    with pytest.raises(ConfigError):
        validate(
            dataclasses.replace(
                cfg,
                values=(ValuesQuestion("x", ("a", "b"), "accepts", (("a", "b"),)),),
            )
        )


# ---------------------------------------------------------------- FR-LM-31


def test_word_encoder_reads_free_text(p1pop):
    """FR-LM-31: shared words raise similarity; stop words don't count; rows have length 1."""
    base = p1pop.members[0]
    texts = [
        "A perfect Sunday looks like hiking and jazz.",
        "Hiking and jazz, every weekend.",
        "Opera and chess.",
        "And the of with.",
    ]
    people = [dataclasses.replace(base, id=k, bio=t) for k, t in enumerate(texts)]
    x = WordEncoder().encode(people)
    norms = np.linalg.norm(x, axis=1)
    np.testing.assert_allclose(norms[:3], 1.0)
    assert norms[3] == 0.0  # only stop words
    t = interest_similarity(x)
    assert t[0, 1] > t[0, 2] and np.isclose(t[0, 2], 0.0)


def test_text_encoder_setting_selects_the_encoder(p1cfg, p1pop, cfg):
    """FR-LM-31 / R5: personality.text_encoder chooses words or tags."""
    m = p1pop.members
    arr = MemberArrays.from_members(m, p1cfg.values)
    w = p1cfg.personality
    expected = w.w_text * interest_similarity(
        WordEncoder().encode(m)
    ) + w.w_values * values_compatibility(arr, p1cfg.values)
    np.testing.assert_allclose(personality_matrix(m, arr, p1cfg), expected, atol=1e-12)
    assert cfg.personality.text_encoder == "tags" and isinstance(TagEncoder(), TagEncoder)


# ---------------------------------------------------------------- FR-LM-32


def test_a_pair_is_one_example_on_the_photo_difference():
    """FR-LM-32: a pair adds the difference of the two photos (intercept cancels) as one label."""
    start = StartingTaste(by_group={(1, 0): np.zeros(3)}, pooled=np.zeros(3))
    taste = PersonalTaste(1, start, 1.0, 1)
    xa, xb = np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0])
    taste.add_pair(xa, xb, 0, True)
    np.testing.assert_allclose(taste.features[0], [1.0, -1.0, 0.0])
    assert taste.liked == [True]
    p = taste.like_chance(np.vstack([xa, xb]), np.array([0, 0]))
    assert p[0] > p[1]  # choosing A over B moves the taste toward A


def test_pair_answers_follow_true_looks(p1cfg, p1pop):
    """FR-LM-32: the simulator picks the better-looking-to-her photo more often, consistently."""
    arr = MemberArrays.from_members(p1pop.members, p1cfg.values)
    beh = Behavior(p1pop, arr, p1cfg, np.random.default_rng(0))
    woman = next(m.id for m in p1pop.members if m.gender == "woman")
    men = [m.id for m in p1pop.members if m.gender == "man"]
    order = sorted(men, key=lambda j: beh.looks[woman, j])
    low, high = order[:20], order[-20:]
    wins = sum(beh.pair_outcome(woman, h, lo, 7) for h, lo in zip(high, low, strict=True))
    assert wins >= 15
    a, b = high[0], low[0]
    assert beh.pair_outcome(woman, a, b, 7) == (not beh.pair_outcome(woman, b, a, 7))
    assert beh.pair_outcome(woman, a, b, 7) == beh.pair_outcome(woman, a, b, 7)


def test_pairs_teach_about_as_much_as_single_swipes(cfg):
    """FR-LM-32 (acceptance): per answer, pairs and single swipes are within 0.01 AUC."""
    import lastfirstdate.phase1_report as report

    arms = {"21 single photos": (21, 12, 0), "12 singles + 9 pairs": (30, 12, 9)}
    original = report.CALIBRATION_ARMS
    report.CALIBRATION_ARMS = arms
    try:
        result = report.calibration_comparison(generate(cfg), cfg, cfg.simulation.seed, 100)
    finally:
        report.CALIBRATION_ARMS = original
    diff, se = result.paired("12 singles + 9 pairs", "21 single photos")
    assert abs(diff) < 0.01, (diff, se)


# ---------------------------------------------------------------- FR-LM-33


def test_market_features_count_pools_and_attention():
    """FR-LM-33: pool = women he could meet; share = sum of 1 / (men each of them could meet)."""
    e = np.zeros((5, 5), dtype=bool)
    women, men = [0, 1, 2], [3, 4]
    for i, j in [(0, 3), (1, 3), (1, 4), (2, 4)]:
        e[i, j] = e[j, i] = True
    mf = market_features(e, women, men)
    np.testing.assert_array_equal(mf.pool, [2, 2])
    np.testing.assert_allclose(mf.share, [1.5, 1.5])  # woman 1 splits her attention
    assert mf.women_per_man == 1.5
    x = start_matrix(mf)
    np.testing.assert_allclose(x[0], [1.0, np.log(1.5), np.log(3.0), np.log(2.5), 0.0])


def test_shortfall_term_is_zero_until_women_are_fewer_than_men():
    """FR-LM-33: log(min(women per man, 1)) is 0 at or above one woman per man, log r below."""
    e = np.ones((6, 6), dtype=bool)
    above = start_matrix(market_features(e, [0, 1, 2, 3], [4, 5]))
    below = start_matrix(market_features(e, [0], [3, 4, 5]))
    assert above[0, -1] == 0.0
    np.testing.assert_allclose(below[:, -1], np.log(1 / 3))
    assert START_FEATURES[-1] == "log women per man below 1"


def test_forecast_model_saves_and_loads(tmp_path):
    """FR-LM-33: coefficients round-trip through the TOML file."""
    model = ForecastModel(np.arange(5) * 0.5, np.arange(8) * 0.25, "test", (0.5, 5.25))
    path = tmp_path / "model.toml"
    model.save(path)
    loaded = ForecastModel.load(path)
    np.testing.assert_allclose(loaded.start, model.start)
    np.testing.assert_allclose(loaded.progress, model.progress)
    assert loaded.fitted_women_per_man == (0.5, 5.25) and loaded.note == "test"
    assert len(START_FEATURES) == 5 and len(PROGRESS_FEATURES) == 8


def test_fit_forecast_learns_that_more_women_per_man_helps():
    """FR-LM-33: on rows where the ratio drives success, its coefficient comes out positive."""
    rng = np.random.default_rng(0)
    n = 4000
    ratio = rng.uniform(0.5, 6, n)
    x = np.column_stack([np.ones(n), np.log(ratio), rng.normal(size=n), rng.normal(size=n)])
    y = rng.random(n) < 1 / (1 + np.exp(-(-1 + 2 * np.log(ratio))))
    xp = np.column_stack([x, rng.random((n, 3))])
    model = fit_forecast(x, y, xp, y, l2=1.0)
    assert model.start[1] > 1.5 and abs(model.start[2]) < 0.3


def test_at_risk_flags_men_behind_late_in_the_cycle(p1cfg):
    """FR-LM-33: with a model that rewards progress, men with no dates late on are flagged."""
    e = np.zeros((6, 6), dtype=bool)
    for i in (0, 1, 2):
        for j in (3, 4, 5):
            e[i, j] = e[j, i] = True
    mf = market_features(e, [0, 1, 2], [3, 4, 5])
    model = ForecastModel(np.zeros(5), np.array([-2.0, 0, 0, 0, 0, 6.0, 1.0, 0.0]))
    dates = np.array([0, 0, 0, 3, 0, 2])
    waiting = np.zeros(6, dtype=int)
    flagged, p = at_risk(model, mf, dates, waiting, 1, p1cfg)
    assert flagged.tolist() == [4]
    assert p[0] > p[2] > p[1]


def test_saved_forecast_is_sensible_and_tracks_fresh_markets(p1cfg, monkeypatch):
    """FR-LM-33: the committed model has sensible signs and forecasts markets it never saw."""
    model = ForecastModel.load()
    start = dict(zip(START_FEATURES, model.start, strict=True))
    progress = dict(zip(PROGRESS_FEATURES, model.progress, strict=True))
    for name in ("log possible matches", "log attention share", "log women per man below 1"):
        assert start[name] > 0, name
    assert progress["dates so far / guarantee"] > 0 and progress["log waiting yeses"] > 0
    lo, hi = model.fitted_women_per_man
    assert lo <= 0.6 and hi >= 4.5
    monkeypatch.setattr(forecast_fit, "MONTHS", 1)
    rows = forecast_fit.Rows([], [], [], [], [])
    for ratio in (0.75, 3.0):  # seed 501 was used for neither fitting nor the held-out check
        pop = forecast_fit.market(p1cfg, 100, ratio, 501)
        forecast_fit.market_rows(pop, p1cfg, 501, rows, {"ratio": ratio})
    for mk in rows.markets:
        f = forecast_fulfilment(model, mk["mf"])
        assert abs(f - mk["actual"]) < 0.07, (mk["ratio"], f, mk["actual"])
    xs, ys, _, _ = rows.stacked()
    assert auc(model.p_start(xs), ys) > 0.85


# ---------------------------------------------------------------- FR-LM-34


def _model(intercept, ratio=0.0, fitted=(0.0, np.inf)):
    """A forecast that depends only on women per man."""
    start = np.zeros(len(START_FEATURES))
    start[:2] = intercept, ratio
    return ForecastModel(start, np.zeros(len(PROGRESS_FEATURES)), "", fitted)


def _full_market(n_women, n_men):
    n = n_women + n_men
    e = np.zeros((n, n), dtype=bool)
    e[:n_women, n_women:] = True
    e[n_women:, :n_women] = True
    return e, list(range(n_women)), list(range(n_women, n))


def test_gate_stops_at_the_ratio_limit(p1cfg):
    """FR-LM-34: with a generous forecast, admissions stop at 3 women per man."""
    e, women, men = _full_market(30, 20)
    d = gate_decision(_model(10.0), e, women, [], men, p1cfg)
    assert len(d.admit) == 10 and d.women_per_man == 3.0
    assert "women per man" in d.stopped_because
    assert d.admit == men[:10]  # waitlist order kept


def test_gate_stops_at_the_forecast_limit(p1cfg):
    """FR-LM-34: when the forecast falls below the target, admissions stop there."""
    e, women, men = _full_market(60, 20)
    # forecast depends only on the ratio: sigmoid(-4 + 4 log(r)) is 90% at r = e^(1 + ln 9 / 4)
    model = _model(-4.0, 4.0)
    d = gate_decision(model, e, women, [], men, p1cfg)
    assert d.forecast >= p1cfg.gate.target
    assert "forecast" in d.stopped_because
    nxt = market_features(e, women, d.admit + [men[len(d.admit)]])
    assert forecast_fulfilment(model, nxt) < p1cfg.gate.target


def test_gate_defers_applicants_with_small_pools(p1cfg):
    """FR-LM-34: an applicant with fewer possible matches than gate.min_pool is deferred."""
    e, women, men = _full_market(40, 3)
    lonely = men[1]
    e[:, lonely] = False
    e[lonely, :] = False
    e[0, lonely] = e[lonely, 0] = True  # one possible match
    d = gate_decision(_model(10.0), e, women, [], men, p1cfg)
    assert lonely in d.deferred_small_pool and lonely not in d.admit
    assert d.admit == [men[0], men[2]]


def test_gate_never_relies_on_the_forecast_outside_its_fitted_range(p1cfg):
    """FR-LM-34: with no ratio floor, admissions still stop where the forecast's data ends."""
    e, women, men = _full_market(30, 40)
    no_floor = dataclasses.replace(p1cfg, gate=dataclasses.replace(p1cfg.gate, min_women_per_man=0))
    d = gate_decision(_model(10.0, fitted=(1.5, 5.0)), e, women, [], men, no_floor)
    assert len(d.admit) == 20 and d.women_per_man == 1.5
    assert "fitted" in d.stopped_because


@pytest.fixture(scope="module")
def oversubscribed(p1cfg):
    """A simulated waitlist longer than the women: 200 women invited, 300 men waiting."""
    return gate_trial(p1cfg, ForecastModel.load(), 200, 300, seeds=(404,), months=1)


def test_gate_protects_fulfilment_when_men_outnumber_women(p1cfg, oversubscribed):
    """FR-LM-34: on a simulated waitlist longer than the women, the men the gate admits reach
    3 dates close to the target and close to the forecast, while admitting everyone would fail
    most of them (and the forecast says so)."""
    target = p1cfg.gate.target
    for arm in ("gate as set", "forecast only"):
        actual, forecast = oversubscribed.mean_actual(arm), oversubscribed.mean_forecast(arm)
        assert actual >= target - 0.05, (arm, actual)
        assert abs(forecast - actual) < 0.05, (arm, forecast, actual)
    assert oversubscribed.mean_actual("everyone") < 0.5
    assert oversubscribed.mean_forecast("everyone") < target


@pytest.mark.xfail(
    strict=True,
    reason="Sprint 6 finding: far below one woman per man in markets bigger than any it was "
    "fitted on (287 men), the forecast is too optimistic (45% versus 12%); refit on larger "
    "markets or real cycles",
)
def test_forecast_tracks_large_markets_with_fewer_women_than_men(oversubscribed):
    """FR-LM-33: the forecast is also accurate for admitting everyone in a large market."""
    actual = oversubscribed.mean_actual("everyone")
    assert abs(oversubscribed.mean_forecast("everyone") - actual) < 0.1
