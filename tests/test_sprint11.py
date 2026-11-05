"""FR-LM-51 to FR-LM-53: outcomes from the app's records, recalibration, and when to switch."""

import dataclasses
import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from lastfirstdate import app, refit
from lastfirstdate.app_fill import fill_database, write_history
from lastfirstdate.config import load
from lastfirstdate.cycle_runs import CycleRun, join_days, market
from lastfirstdate.filter import eligibility_matrix
from lastfirstdate.forecast import DELIVERED_MODEL_PATH, ForecastModel
from lastfirstdate.looks import sigmoid
from lastfirstdate.mutual_report import build_scores

EPOCH = date(2026, 1, 5)  # a Monday: test run day 0


@pytest.fixture(scope="module")
def history():
    """Three billing cycles of the real flow in a world where people show up less often than
    the forecast assumed (80% instead of 92%), written into SQLite as the app's records."""
    base = load("config/phase1.toml")
    cfg = dataclasses.replace(
        base, test_runs=dataclasses.replace(base.test_runs, show_up_rate=0.80)
    )
    pop = market(cfg, 7, 0.15)
    sc = build_scores(pop, cfg, 7)
    sim = CycleRun(pop, cfg, 7, sc.mutual, join_days(pop, cfg, 0.15), record=True)
    report = sim.run("history")
    today = EPOCH + timedelta(days=sim.days)
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    fill_database(con, pop, cfg, today, 7)
    write_history(con, sim, EPOCH)
    mk = app.load_market(con, cfg, app.Calendar(EPOCH), today)
    eligible = eligibility_matrix(mk.arr, cfg)
    return SimpleNamespace(
        cfg=cfg,
        sim=sim,
        report=report,
        con=con,
        market=mk,
        eligible=eligible,
        rows=refit.cycle_rows(con, mk, eligible, cfg),
        model=ForecastModel.load(DELIVERED_MODEL_PATH),
    )


# ---------------------------------------------------------------- FR-LM-51


def test_cycle_outcomes_from_records_match_the_test_run(history):
    """FR-LM-51: every cycle the test run reported reached 3 delivered dates (or not)
    exactly as the records say."""
    men = np.flatnonzero(history.sim.guaranteed)
    want = {
        (1000 + int(m), k): bool(history.report.delivered[r, k] >= 3)
        for r, m in enumerate(men)
        for k in range(history.cfg.cycle_runs.cycles)
    }
    got = dict(
        zip(
            zip(history.rows.member_ids, history.rows.cycle_index, strict=True),
            history.rows.reached,
            strict=True,
        )
    )
    assert {k: got[k] for k in want} == want


def test_weekly_progress_from_records_matches_the_test_run(history):
    """FR-LM-51: delivered-so-far and confirmed-to-come at every weekly snapshot equal what the
    test run saw at that moment."""
    seen = {}
    for rows, men, k in history.sim.week_rows:
        for x, m in zip(rows, men, strict=True):
            week = int(round((1 - x[7]) * history.cfg.billing.cycle_days / 7))
            seen[(1000 + int(m), k, week)] = (int(round(x[5] * 3)), int(round(np.expm1(x[6]))))
    rebuilt = dict(
        zip(
            history.rows.week_keys,
            zip(history.rows.delivered_so_far, history.rows.to_come, strict=True),
            strict=True,
        )
    )
    assert seen and all(rebuilt[key] == value for key, value in seen.items())


# ---------------------------------------------------------------- FR-LM-52


def test_folding_a_recalibration_into_the_coefficients(history):
    """FR-LM-52: a = 0, b = 1 changes nothing; otherwise the folded model gives exactly
    sigmoid(a + b · logit) for start and mid-cycle forecasts."""
    model, rows = history.model, history.rows
    xs, _, xp, _ = rows.arrays()
    same = refit.fold(model, (0.0, 1.0), (0.0, 1.0), 10)
    np.testing.assert_array_equal(same.start, model.start)
    np.testing.assert_array_equal(same.progress, model.progress)
    moved = refit.fold(model, (-0.4, 0.8), (0.2, 1.1), 10)
    np.testing.assert_allclose(moved.p_start(xs), sigmoid(-0.4 + 0.8 * (xs @ model.start)))
    np.testing.assert_allclose(moved.p_progress(xp), sigmoid(0.2 + 1.1 * (xp @ model.progress)))
    assert moved.assumptions["recalibrated_on_cycles"] == 10


def test_recalibration_fixes_a_wrong_assumption_on_later_cycles(history):
    """FR-LM-52: when people show up less often than assumed, the model over-forecasts; a
    recalibration on the earlier 70% of cycles (the split the switch rule uses) is closer on
    the latest 30%. (Split at the median it is not: fulfilment falls as a zone wears out, so
    first-cycle data alone can't foresee the later level. Hence monthly recalibration on recent
    cycles, and the switch rule testing on the latest ones.)"""
    model, rows, rf = history.model, history.rows, history.cfg.refit
    xs, ys, _, _ = rows.arrays()
    assert model.p_start(xs).mean() > ys.mean() + 0.03  # the wrong assumption shows
    ends = np.sort(np.array(rows.ends))
    early, late = rows.split(int(ends[int(np.floor((1 - rf.holdout_share) * (len(ends) - 1)))]))
    candidate = refit.recalibrate(model, early, history.cfg)
    x_late, y_late, _, _ = late.arrays()
    assert abs(candidate.p_start(x_late).mean() - y_late.mean()) < abs(
        model.p_start(x_late).mean() - y_late.mean()
    )
    assert refit.brier(candidate, late) < refit.brier(model, late)


# ---------------------------------------------------------------- FR-LM-53


def test_switching_needs_enough_cycles(history):
    """FR-LM-53: with fewer completed cycles than refit.min_man_cycles, no switch."""
    cfg = dataclasses.replace(
        history.cfg, refit=dataclasses.replace(history.cfg.refit, min_man_cycles=10_000)
    )
    d = refit.propose_refit(history.con, history.market, history.eligible, history.model, cfg)
    assert not d.switch and d.model is None and "need" in d.reason


def test_a_worse_than_assumed_world_leads_to_a_switch(history):
    """FR-LM-53: here the recalibrated forecast beats the current one on the latest cycles by
    more than refit.min_improvement, so the switch is recommended, recalibrated on every
    cycle."""
    d = refit.propose_refit(
        history.con, history.market, history.eligible, history.model, history.cfg
    )
    assert d.switch, d
    assert d.candidate_brier < d.current_brier - history.cfg.refit.min_improvement
    assert d.model.assumptions["recalibrated_on_cycles"] == d.man_cycles == len(history.rows)
