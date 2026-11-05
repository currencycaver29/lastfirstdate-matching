"""FR-LM-43 to FR-LM-45: the forecast and the ratio gate on delivered dates."""

import dataclasses

import numpy as np
import pytest

from lastfirstdate import forecast_fit
from lastfirstdate.config import ConfigError, load, validate
from lastfirstdate.forecast import (
    DEFAULT_MODEL_PATH,
    DELIVERED_MODEL_PATH,
    DELIVERED_PROGRESS_FEATURES,
    START_FEATURES,
    ForecastModel,
    at_risk_delivered,
    delivered_progress_matrix,
    gate_model,
    market_features,
)
from lastfirstdate.gate_report import real_gate_trial
from lastfirstdate.looks_report import auc


@pytest.fixture(scope="module")
def p1cfg():
    return load("config/phase1.toml")


@pytest.fixture(scope="module")
def delivered():
    return ForecastModel.load(DELIVERED_MODEL_PATH)


@pytest.fixture(scope="module")
def fresh_market(p1cfg):
    """The real flow on a market the forecast never saw (100 men, 3.5 women per man)."""
    return forecast_fit.delivered_market(p1cfg, 100, 3.5, 501)


# ---------------------------------------------------------------- FR-LM-43


def test_the_delivered_model_records_what_it_assumes(p1cfg, delivered):
    """FR-LM-43: the model file says it forecasts delivered dates, the range of women per man
    it was fitted on, and the frictions and rate of new women it assumes."""
    assert delivered.kind == "delivered"
    lo, hi = delivered.fitted_women_per_man
    assert lo <= 1.1 and hi >= 6.5
    a = delivered.assumptions
    assert a["new_women_per_cycle"] == p1cfg.cycle_runs.new_women_per_cycle
    assert a["show_up_rate"] == p1cfg.test_runs.show_up_rate
    assert a["cancel_rate"] == p1cfg.cycle_runs.cancel_rate
    assert len(delivered.progress) == len(DELIVERED_PROGRESS_FEATURES)


def test_kind_and_assumptions_survive_saving(tmp_path):
    """FR-LM-43: kind, fitted range and assumptions round-trip through the TOML file."""
    model = ForecastModel(
        np.arange(len(START_FEATURES)) * 0.1,
        np.arange(len(DELIVERED_PROGRESS_FEATURES)) * 0.2,
        "test",
        (1.0, 7.0),
        "delivered",
        {"new_women_per_cycle": 0.15, "stagger": True},
    )
    model.save(tmp_path / "m.toml")
    loaded = ForecastModel.load(tmp_path / "m.toml")
    assert loaded.kind == "delivered" and loaded.fitted_women_per_man == (1.0, 7.0)
    assert loaded.assumptions == {"new_women_per_cycle": 0.15, "stagger": True}
    np.testing.assert_allclose(loaded.progress, model.progress)


def test_the_delivered_forecast_tracks_a_market_it_never_saw(delivered, fresh_market):
    """FR-LM-43: within 7 points of the share of man-cycles with 3 delivered dates."""
    (xs, _, _, _), report = fresh_market
    forecast = float(delivered.p_start(xs).mean())
    assert abs(forecast - report.guarantee_share) < 0.07, (forecast, report.guarantee_share)


# ---------------------------------------------------------------- FR-LM-44


def test_progress_rows_follow_each_mans_own_cycle(p1cfg):
    """FR-LM-44: start features, then delivered so far / 3, log(1 + confirmed to come) and
    days left / cycle days."""
    e = np.zeros((4, 4), dtype=bool)
    e[0, 2] = e[2, 0] = e[1, 2] = e[2, 1] = e[1, 3] = e[3, 1] = True
    mf = market_features(e, [0, 1], [2, 3])
    delivered = np.array([0, 0, 2, 0.0])
    upcoming = np.array([0, 0, 1, 0.0])
    days_left = np.array([0, 0, 7, 21.0])
    x = delivered_progress_matrix(mf, delivered, upcoming, days_left, p1cfg)
    np.testing.assert_allclose(x[0, -3:], [2 / 3, np.log(2), 7 / 28])
    np.testing.assert_allclose(x[1, -3:], [0, 0, 21 / 28])


def test_mid_cycle_alerts_rank_men_on_a_market_they_never_saw(p1cfg, delivered, fresh_market):
    """FR-LM-44: the weekly chances rank men who end their cycle with 3 delivered dates above
    those who don't (AUC above 0.85), and delivered dates so far raise the chance."""
    (_, _, xp, yp), _ = fresh_market
    assert 0 < yp.mean() < 1
    assert auc(delivered.p_progress(xp), yp) > 0.85
    progress = dict(zip(DELIVERED_PROGRESS_FEATURES, delivered.progress, strict=True))
    assert progress["delivered so far / guarantee"] > 0


def test_alerts_need_a_delivered_model(p1cfg):
    """FR-LM-44: delivered-date alerts refuse a forecast fitted on booked dates."""
    e = np.ones((2, 2), dtype=bool)
    mf = market_features(e, [0], [1])
    zeros = np.zeros(2)
    with pytest.raises(ValueError):
        at_risk_delivered(ForecastModel.load(DEFAULT_MODEL_PATH), mf, zeros, zeros, zeros, p1cfg)


# ---------------------------------------------------------------- FR-LM-45


def test_the_setting_picks_the_gates_forecast(p1cfg):
    """FR-LM-45: gate.forecast chooses the model file, and only booked or delivered pass."""
    assert gate_model(p1cfg).kind == "delivered"
    assert gate_model(load()).kind == "booked"
    bad = dataclasses.replace(p1cfg, gate=dataclasses.replace(p1cfg.gate, forecast="hope"))
    with pytest.raises(ConfigError):
        validate(bad)


def test_the_delivered_gate_protects_small_zones(p1cfg, delivered):
    """FR-LM-45: in a 300-woman zone with a long waitlist, the men the booked forecast admits
    fall more than 5 points short of the target, and the delivered forecast admits fewer men
    (or none: no number of men reaches the target there)."""
    rows = real_gate_trial(
        p1cfg,
        ForecastModel.load(DEFAULT_MODEL_PATH),
        delivered,
        n_women=300,
        n_waitlist=300,
        seeds=(404,),
        versions=("booked forecast", "delivered forecast"),
    )
    by = {r.version: r for r in rows}
    assert by["booked forecast"].actual < p1cfg.gate.target - 0.05
    assert by["delivered forecast"].admitted < by["booked forecast"].admitted


def test_the_delivered_gate_is_calibrated_in_a_large_zone(p1cfg, delivered):
    """FR-LM-45: in a 500-woman zone with a long waitlist, the men the delivered forecast
    admits reach 3 delivered dates within 5 points of the target and of the forecast."""
    (row,) = real_gate_trial(
        p1cfg,
        ForecastModel.load(DEFAULT_MODEL_PATH),
        delivered,
        n_women=500,
        n_waitlist=300,
        seeds=(404,),
        versions=("delivered forecast",),
    )
    assert row.admitted > 0
    assert abs(row.actual - p1cfg.gate.target) < 0.05, row
    assert abs(row.forecast - row.actual) < 0.05, row
