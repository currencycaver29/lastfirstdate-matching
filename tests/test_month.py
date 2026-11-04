"""FR-LM-12: the simulated month and its report."""

import numpy as np
import pytest

from lastfirstdate.simulate import DAYS_PER_WEEK, run_month
from lastfirstdate.synthetic import generate

# Personality-only baseline on the default population (seed 7) is about 0.8. This floor only
# catches a later change that makes things clearly worse. The 95% target is checked with
# the full engine in tests/test_fairness.py (FR-LM-21).
BASELINE_FLOOR = 0.75


@pytest.fixture(scope="module")
def month(cfg):
    return run_month(generate(cfg), cfg)


def test_month_runs_and_books_dates_within_the_caps(cfg, month):
    """FR-LM-12: a full month runs, books dates, and nobody exceeds the monthly cap."""
    assert month.booked > 0
    assert month.dates.max() <= cfg.dates.cap_per_month
    assert month.dates.sum() == 2 * month.booked
    assert month.picks >= month.accepts > 0
    assert len(month.methods) == cfg.dates.weeks_per_month


def test_lists_do_not_run_dry(month):
    """FR-LM-12: with re-showing, week 4 lists keep at least 80% of their week 1 size."""
    sizes = month.list_size_by_day
    week1 = np.mean(sizes[:DAYS_PER_WEEK])
    week4 = np.mean(sizes[3 * DAYS_PER_WEEK : 4 * DAYS_PER_WEEK])
    assert week4 >= 0.8 * week1


def test_every_paying_member_who_missed_has_a_reason(cfg, month):
    """FR-LM-12: each paying member below the guarantee is listed with a reason."""
    short = month.paying & (month.dates < cfg.dates.guaranteed_per_month_paying)
    assert {m.member for m in month.misses} == set(np.flatnonzero(short).tolist())
    assert all(m.reason for m in month.misses)
    text = month.text()
    assert "reached the guarantee" in text and "missed" in text


def test_guarantee_share_does_not_fall_below_the_sprint_1_baseline(month):
    """FR-LM-12: regression floor for the personality-only matcher."""
    assert month.guarantee_share() >= BASELINE_FLOOR


def test_same_seed_gives_the_same_month(cfg, month):
    """FR-LM-12 / R7: rerunning with the same seed reproduces every date."""
    again = run_month(generate(cfg), cfg)
    np.testing.assert_array_equal(again.dates, month.dates)
