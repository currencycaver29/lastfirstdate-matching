"""Ready for real data: outcomes from the app's records, recalibration, and when to switch
(FR-LM-51 to FR-LM-53).

The guarantee forecast (Sprint 8) was fitted on synthetic markets. Once Phase 1 has run for a
few cycles, the app's records say what really happened. This module turns those records into
training rows and decides whether a refitted forecast is better.

Why recalibrate rather than refit from scratch. Phase 1 is one zone: every man in it sees the
same women per man, so real Phase 1 data cannot tell how much the ratio matters. It can
reliably correct the forecast's level and steepness:

    logit(p_real) = a + b · logit(p_model)

(Platt scaling). a > 0 means the model was too pessimistic, a < 0 too optimistic; b < 1 means
it was too sure of itself. a = 0 and b = 1 change nothing, and the fit is pulled toward that,
so a few cycles can't swing the forecast far. Because the model is linear inside the sigmoid,
the correction folds into its coefficients (b times every coefficient, plus a on the
intercept): the gate, its Rust twin and everything else keep working unchanged.

When to switch. Recalibrate on the earlier cycles, compare on the latest ones (Brier score: the
mean squared gap between forecast and outcome), and switch only if the recalibrated forecast is
better by a margin. Then recalibrate on every cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from lastfirstdate import app
from lastfirstdate.billing import counts_toward_guarantee, cycle_on
from lastfirstdate.config import Config
from lastfirstdate.forecast import (
    ForecastModel,
    delivered_progress_matrix,
    market_features,
    start_matrix,
)
from lastfirstdate.looks import fit_logistic

WEEKS = (1, 2, 3)  # weekly snapshots inside a 28-day cycle


@dataclass
class CycleRows:
    """Every completed paid cycle, and its weekly snapshots, rebuilt from the records."""

    member_ids: list[int] = field(default_factory=list)
    cycle_index: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)  # day each cycle ended
    x_start: list[np.ndarray] = field(default_factory=list)
    reached: list[bool] = field(default_factory=list)
    week_keys: list[tuple[int, int, int]] = field(default_factory=list)  # member, cycle, week
    week_ends: list[int] = field(default_factory=list)
    x_progress: list[np.ndarray] = field(default_factory=list)
    y_progress: list[bool] = field(default_factory=list)
    delivered_so_far: list[int] = field(default_factory=list)
    to_come: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.reached)

    def arrays(self):
        """(x_start, reached, x_progress, y_progress) as arrays."""
        xs = np.array(self.x_start).reshape(len(self.x_start), -1)
        xp = np.array(self.x_progress).reshape(len(self.x_progress), -1)
        return xs, np.array(self.reached, float), xp, np.array(self.y_progress, float)

    def split(self, last_day: int) -> tuple[CycleRows, CycleRows]:
        """(cycles that ended on or before `last_day`, the rest)."""
        early, late = CycleRows(), CycleRows()
        for k in range(len(self.reached)):
            out = early if self.ends[k] <= last_day else late
            for name in ("member_ids", "cycle_index", "ends", "x_start", "reached"):
                getattr(out, name).append(getattr(self, name)[k])
        for k in range(len(self.y_progress)):
            out = early if self.week_ends[k] <= last_day else late
            for name in (
                "week_keys",
                "week_ends",
                "x_progress",
                "y_progress",
                "delivered_so_far",
                "to_come",
            ):
                getattr(out, name).append(getattr(self, name)[k])
        return early, late


def _day_of(stamp: str | None, market: app.Market) -> int | None:
    """The day of a timestamp in the zone's own time (D19)."""
    if stamp is None:
        return None
    return market.calendar.day(app.local_date(stamp, market.tz))


def cycle_rows(con, market: app.Market, eligible: np.ndarray, cfg: Config) -> CycleRows:
    """Rebuild every paid cycle that has ended by today (FR-LM-51).

    The market at a day holds the women who had joined by then and the men who had
    subscribed by then (current profiles and eligibility). A booking counts as "to come" at a
    snapshot if it was confirmed before that day, falls inside the cycle on or after it, and
    hadn't been cancelled before it.
    """
    bookings = app.load_bookings(con, market)
    timing = {
        r["id"]: (_day_of(r["confirmed_at"], market), _day_of(r["cancelled_at"], market))
        for r in app.read(con, "bookings")
    }
    signups = app.signup_days(con, market, cfg)
    joined = np.array([0] * len(market.members))
    for r in app.read(con, "members"):
        joined[market.row[r["id"]]] = _day_of(r["joined_at"], market)
    women_all = np.flatnonzero((market.arr.gender == app.WOMAN) & market.arr.verified)
    subscribers = np.flatnonzero(signups != app.NO_SIGNUP)
    g, c = cfg.dates.guaranteed_per_month_paying, cfg.billing.cycle_days
    features: dict[int, object] = {}

    def market_on(day: int):
        if day not in features:
            women = women_all[joined[women_all] <= day]
            men = subscribers[signups[subscribers] <= day]
            features[day] = market_features(eligible, women, men)
        return features[day]

    mine = {int(m): [] for m in subscribers}
    for bk, bid in zip(bookings.items, bookings.ids, strict=True):
        for m in bk.members:
            if m in mine:
                mine[m].append((bk, timing[bid]))
    rows = CycleRows()
    n = len(market.members)
    for m in subscribers:
        m = int(m)
        k = 0
        while True:
            first_paid = int(signups[m]) + cfg.billing.trial_days
            cyc = cycle_on(m, int(signups[m]), first_paid + k * c, cfg)
            if cyc.end > market.today:
                break
            mf = market_on(cyc.start)
            pos = int(np.flatnonzero(mf.men == m)[0])
            got = sum(counts_toward_guarantee(bk, m, cyc, cfg) for bk, _ in mine[m])
            rows.member_ids.append(int(market.ids[m]))
            rows.cycle_index.append(k)
            rows.ends.append(cyc.end)
            rows.x_start.append(start_matrix(mf)[pos])
            rows.reached.append(got >= g)
            for w in WEEKS:
                snap = cyc.start + 7 * w
                if snap >= cyc.end:
                    continue
                done = sum(
                    counts_toward_guarantee(bk, m, cyc, cfg) for bk, _ in mine[m] if bk.day < snap
                )
                coming = sum(
                    cyc.contains(bk.day)
                    and bk.day >= snap
                    and confirmed is not None
                    and confirmed < snap
                    and (cancelled is None or cancelled >= snap)
                    for bk, (confirmed, cancelled) in mine[m]
                )
                mf_s = market_on(snap)
                pos_s = int(np.flatnonzero(mf_s.men == m)[0])
                delivered, upcoming, left = np.zeros(n), np.zeros(n), np.ones(n)
                delivered[m], upcoming[m], left[m] = done, coming, cyc.end - snap
                x = delivered_progress_matrix(mf_s, delivered, upcoming, left, cfg)[pos_s]
                rows.week_keys.append((int(market.ids[m]), k, w))
                rows.week_ends.append(cyc.end)
                rows.x_progress.append(x)
                rows.y_progress.append(got >= g)
                rows.delivered_so_far.append(done)
                rows.to_come.append(coming)
            k += 1
    return rows


# ---------------------------------------------------------------- recalibration


def fold(
    model: ForecastModel,
    start_ab: tuple[float, float],
    progress_ab: tuple[float, float],
    cycles: int,
) -> ForecastModel:
    """The model with logit' = a + b · logit folded into its coefficients (FR-LM-52)."""
    (a, b), (a2, b2) = start_ab, progress_ab
    start = b * model.start
    start[0] += a
    progress = b2 * model.progress
    progress[0] += a2
    assumptions = dict(model.assumptions)
    assumptions.update(
        recalibrated_on_cycles=cycles,
        recalibration_start_a=round(a, 6),
        recalibration_start_b=round(b, 6),
        recalibration_progress_a=round(a2, 6),
        recalibration_progress_b=round(b2, 6),
    )
    note = f"{model.note}; recalibrated on {cycles} real cycles"
    return replace(model, start=start, progress=progress, note=note, assumptions=assumptions)


def _platt(z: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, float]:
    """(a, b) for logit' = a + b · z, pulled toward (0, 1): no change."""
    if len(y) == 0:
        return 0.0, 1.0
    w, _ = fit_logistic(
        np.column_stack([np.ones_like(z), z]), y, l2, prior_mean=np.array([0.0, 1.0])
    )
    return float(w[0]), float(w[1])


def recalibrate(model: ForecastModel, rows: CycleRows, cfg: Config) -> ForecastModel:
    xs, ys, xp, yp = rows.arrays()
    l2 = cfg.gate.forecast_l2
    start_ab = _platt(xs @ model.start, ys, l2)
    progress_ab = _platt(xp @ model.progress, yp, l2) if len(yp) else (0.0, 1.0)
    return fold(model, start_ab, progress_ab, len(rows))


def brier(model: ForecastModel, rows: CycleRows) -> float:
    """Mean squared gap between the start-of-cycle forecast and what happened."""
    xs, ys, _, _ = rows.arrays()
    return float(np.mean((model.p_start(xs) - ys) ** 2))


# ---------------------------------------------------------------- when to switch


@dataclass
class RefitDecision:
    switch: bool
    reason: str
    man_cycles: int
    current_brier: float = float("nan")
    candidate_brier: float = float("nan")
    model: ForecastModel | None = None  # the recalibrated forecast, when switching


def propose_refit(
    con, market: app.Market, eligible: np.ndarray, model: ForecastModel, cfg: Config
) -> RefitDecision:
    """Should the app switch to a forecast recalibrated on its own cycles? (FR-LM-53)"""
    rf = cfg.refit
    rows = cycle_rows(con, market, eligible, cfg)
    n = len(rows)
    if n < rf.min_man_cycles:
        return RefitDecision(False, f"only {n} completed cycles; need {rf.min_man_cycles}", n)
    ends = np.sort(np.array(rows.ends))
    cut = int(ends[int(np.floor((1 - rf.holdout_share) * (n - 1)))])
    train, test = rows.split(cut)
    if len(test) == 0 or len(train) == 0:
        return RefitDecision(False, "the cycles don't spread over enough time to test on", n)
    candidate = recalibrate(model, train, cfg)
    current_score, candidate_score = brier(model, test), brier(candidate, test)
    if candidate_score < current_score - rf.min_improvement:
        return RefitDecision(
            True,
            "the recalibrated forecast is better on the latest cycles",
            n,
            current_score,
            candidate_score,
            recalibrate(model, rows, cfg),
        )
    return RefitDecision(
        False,
        "the current forecast is as good on the latest cycles",
        n,
        current_score,
        candidate_score,
    )
