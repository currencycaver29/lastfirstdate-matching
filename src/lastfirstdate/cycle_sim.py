"""Phase 1 cycles: lastfirstdate's real booking flow, simulated day by day (FR-LM-42).

Simulation only. Sprints 1 to 6 booked dates in a weekly batch and counted dates per calendar
month. lastfirstdate actually works like this, and so does this simulation:

1. Every day, women who open the app get up to 30 men, each shown with a proposed date
   window: the earliest time both are free (FR-LM-37). Men with no common window aren't
   shown. Men short of their 3 dates get a bonus that grows as their own cycle runs out
   (FR-LM-38); low trust lowers a man's scores and flagged men are paused (FR-LM-39).
2. She picks; he accepts or not; both $10 payments must arrive within 15 minutes, or the hold
   expires. Then the venue is fixed at the fair midpoint (FR-LM-41).
3. Before the date, either may cancel (early or late). On the day, each shows up or not.
   Afterwards both rate each other.
4. Members-only events add a few dates outside the lists (FR-LM-40).
5. Each paying man's billing cycles run on his own clock. At the end of each, his delivered
   dates decide his credit.
6. New women keep joining (`cycle_simulation.new_women_per_cycle`). Without them the market
   wears out: with up to 30 profiles a day and a 7-day rest, each woman sees every man in her
   pool (about 46 in Phase 1) about once a week, and each pass makes a later pick less likely.

The fake people's conduct (how respectful they are, how accurate their profile is) is hidden
truth drawn here; the trust score only ever sees their ratings.

Run `python -m lastfirstdate.cycle_sim` to print the report (about a minute).
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field

import numpy as np

from lastfirstdate import geo
from lastfirstdate.arrays import GENDERS, MemberArrays
from lastfirstdate.billing import (
    counts_toward_guarantee,
    credit,
    cycle_on,
    fee_refunds,
    refill_bonus,
)
from lastfirstdate.booking import (
    NO_WINDOW,
    Booking,
    earliest_common,
    event_booking,
    window_bits,
    window_day_part,
)
from lastfirstdate.config import Config, load
from lastfirstdate.fairness import fair_day_lists
from lastfirstdate.filter import eligibility_matrix, member_ok
from lastfirstdate.forecast import delivered_progress_matrix, market_features, start_matrix
from lastfirstdate.lists import NEVER_SHOWN, available_row, pick_rights
from lastfirstdate.mutual_report import build_scores
from lastfirstdate.simulate import month_people, run_month
from lastfirstdate.synthetic import Population, generate
from lastfirstdate.trust import Rating, TrustBook, with_trust

PHASE1 = "config/phase1.toml"
WELL_RATED = 6  # ratings after which a typical bad actor's conduct is below the flag line
SHE_PAYS_AT, HE_PAYS_AT = 2.0, 6.0  # minutes after the hold starts, when paying in time


@dataclass(frozen=True)
class Conduct:
    """Hidden truth: how each fake member behaves on dates (the engine never sees this)."""

    conduct: np.ndarray  # (n,) respectfulness, in standard deviations
    honesty: np.ndarray  # (n,) how well they match their profile
    bad: np.ndarray  # (n,) members whose conduct is poor


def hidden_conduct(n: int, cfg: Config, seed: int) -> Conduct:
    cs = cfg.cycle_simulation
    rng = np.random.default_rng([seed, 707])
    conduct, honesty = rng.normal(size=n), rng.normal(size=n)
    bad = rng.random(n) < cs.bad_actor_share
    conduct[bad] = honesty[bad] = cs.bad_actor_conduct
    return Conduct(conduct, honesty, bad)


@dataclass
class CycleReport:
    label: str
    cycles: int
    men: int
    women: int
    delivered: np.ndarray  # (men, cycles) delivered dates per paid cycle
    credits: float
    revenue: float
    fees_paid: float
    fees_refunded: float
    states: dict[str, int]
    events: int
    holds_expired: int
    picks: int
    declined: int
    no_window: int
    cancellations: dict[str, int]
    women_dates: np.ndarray  # (women, cycles) completed dates per 28-day block
    bad: int
    flagged: int
    flagged_bad: int
    bad_well_rated: int  # bad actors with WELL_RATED or more ratings (or already flagged)
    caught_well_rated: int
    ratings_at_flag: list[int]
    trust_looks_corr: float
    longer_trip_km: float
    trip_gap_km: float
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def guarantee_share(self) -> float:
        return float((self.delivered >= 3).mean())

    def per_cycle(self) -> list[float]:
        return [float((self.delivered[:, k] >= 3).mean()) for k in range(self.cycles)]

    def text(self) -> str:
        dist = np.bincount(self.delivered.ravel(), minlength=4)[:4]
        total = max(int(self.delivered.size), 1)
        lines = [
            f"{self.label}: {self.men} paying men, {self.women} women, {self.cycles} cycles each",
            "  men with 3 delivered dates, by cycle: "
            + ", ".join(f"{x:.1%}" for x in self.per_cycle())
            + f" (all cycles {self.guarantee_share:.1%})",
            "  delivered dates per man-cycle: "
            + ", ".join(f"{k}: {dist[k] / total:.0%}" for k in range(4)),
            f"  credits ${self.credits:,.0f} of ${self.revenue:,.0f} subscriptions "
            f"({self.credits / max(self.revenue, 1):.1%})",
            f"  date fees paid ${self.fees_paid:,.0f}, refunded ${self.fees_refunded:,.0f}",
            f"  picks {self.picks}: he declined {self.declined}, no window left "
            f"{self.no_window}, hold expired {self.holds_expired}",
            "  bookings by outcome: "
            + ", ".join(f"{k} {v}" for k, v in sorted(self.states.items()) if v),
            f"  cancellations: {self.cancellations['early']} early, "
            f"{self.cancellations['late']} late; event dates {self.events}",
            f"  launch women's completed dates per 28 days: average "
            f"{self.women_dates.mean():.2f}, none {np.mean(self.women_dates == 0):.0%}",
            f"  trust: {self.flagged} flagged, {self.flagged_bad} of them simulated bad actors "
            f"({self.bad} bad actors in all); caught {self.caught_well_rated} of the "
            f"{self.bad_well_rated} with {WELL_RATED}+ ratings; median ratings when flagged "
            f"{np.median(self.ratings_at_flag) if self.ratings_at_flag else float('nan'):.0f}",
            f"  trust versus true looks appeal (others): correlation {self.trust_looks_corr:+.2f}",
            f"  venues: longer trip {self.longer_trip_km:.2f} km on average, "
            f"difference between the two trips {self.trip_gap_km:.2f} km",
        ]
        return "\n".join(lines)


class CycleSimulation:
    """One run of the real flow on one market."""

    def __init__(
        self,
        pop: Population,
        cfg: Config,
        seed: int,
        scores: np.ndarray,
        join_day: np.ndarray | None = None,
        record: bool = False,
    ):
        self.cfg, self.seed, self.scores = cfg, seed, scores
        self.record = record  # keep forecast rows (FR-LM-43, FR-LM-44)
        self.launch_rows: np.ndarray | None = None
        self.week_rows: list[tuple[np.ndarray, np.ndarray, int]] = []  # rows, men, cycle
        cs, b = cfg.cycle_simulation, cfg.billing
        n = len(pop)
        self.n = n
        self.arr = arr = MemberArrays.from_members(pop.members, cfg.values)
        self.behavior, _ = month_people(pop, arr, cfg, seed)
        self.rng = np.random.default_rng([seed, 2_000_003, int(cs.stagger)])
        self.eligible = eligibility_matrix(arr, cfg)
        self.rights = pick_rights(arr)
        self.matchable = member_ok(arr, cfg)
        self.join_day = np.zeros(n, dtype=int) if join_day is None else np.asarray(join_day)
        self.guaranteed = arr.paying & self.matchable & (self.join_day == 0)
        d = cfg.dates
        capped = ~arr.paying & np.isin(arr.gender, [GENDERS.index(g) for g in d.free_cap_genders])
        self.cap = np.where(capped, d.cap_per_month_free, d.cap_per_month)
        c = b.cycle_days
        offsets = self.rng.integers(0, c, n) if cs.stagger else np.zeros(n, dtype=int)
        self.anchor = np.where(self.guaranteed, offsets, 0)  # first paid day for paying men
        self.days = (c if cs.stagger else 0) + cs.cycles * c
        h = cfg.booking.horizon_days
        self.busy = np.zeros((n, self.days + h + 1), dtype=bool)
        self.week_count = np.zeros((n, (self.days + h) // 7 + 2), dtype=int)
        self.period_count = np.zeros((n, (self.days + h) // c + 3), dtype=int)
        self.closed = np.zeros((n, n), dtype=bool)
        self.last_shown = np.full((n, n), NEVER_SHOWN, dtype=np.int64)
        self.passes = np.zeros((n, n), dtype=np.int16)
        self.bookings: list[Booking] = []
        self.booked_on: list[int] = []  # day each booking was made (for app records)
        self.cancelled_on: dict[int, int] = {}  # booking index -> day it was cancelled
        self.by_member: list[list[int]] = [[] for _ in range(n)]
        self.on_day: dict[int, list[int]] = {}
        self.cancel_plan: dict[int, tuple[int, int, float]] = {}
        self.trust = TrustBook(n, cfg)
        self.paused = np.zeros(n, dtype=bool)
        self.ratings_at_flag: dict[int, int] = {}
        self.truth = hidden_conduct(n, cfg, seed)
        self.hood = geo.nearest_place(arr.lat, arr.lon, cfg.geo.neighbourhoods)
        self.trips: list[tuple[float, float]] = []
        self.count = dict.fromkeys(
            ("picks", "declined", "no_window", "expired", "events", "early", "late"), 0
        )

    # ------------------------------------------------------------ calendar and capacity

    def signup(self, m: int) -> int:
        return int(self.anchor[m]) - self.cfg.billing.trial_days

    def period(self, members, days) -> np.ndarray:
        """Cap period of each member on each day: 0 = trial, then one per cycle or month."""
        a = self.anchor[members]
        c = self.cfg.billing.cycle_days
        return np.where(days < a, 0, (days - a) // c + 1)

    def blocked(self, today: int, members: np.ndarray) -> np.ndarray:
        """(len(members), horizon) days each member cannot take another date."""
        bk, d = self.cfg.booking, self.cfg.dates
        days = today + np.arange(bk.horizon_days)
        rows = np.asarray(members)[:, None]
        week_full = self.week_count[rows, days[None, :] // 7] >= d.max_per_week
        period = self.period(rows, days[None, :])
        cap_full = self.period_count[rows, period] >= self.cap[rows]
        return self.busy[rows, days[None, :]] | week_full | cap_full

    def bits(self, today: int, members: np.ndarray) -> np.ndarray:
        bk = self.cfg.booking
        members = np.asarray(members)
        return window_bits(
            self.arr.slots[members],
            today,
            bk.lead_days,
            bk.horizon_days,
            self.blocked(today, members),
        )

    def occupy(self, bk: Booking, sign: int, keep_cap: tuple[int, ...] = ()) -> None:
        """Take (+1) or give back (-1) the booking's day, week and cap for both members.

        `keep_cap`: members whose cap stays used when giving back, because the cancelled
        date still counts as one of their delivered dates (D17).
        """
        for m in bk.members:
            self.busy[m, bk.day] = sign > 0
            self.week_count[m, bk.day // 7] += sign
            if m not in keep_cap:
                period = int(self.period(np.array([m]), np.array([bk.day]))[0])
                self.period_count[m, period] += sign

    def release_slot(self, bk: Booking, m: int) -> None:
        """Give m back the week and cap a date used, when it didn't happen through no fault
        of theirs (the other person didn't come). The evening itself has passed."""
        self.week_count[m, bk.day // 7] -= 1
        self.period_count[m, int(self.period(np.array([m]), np.array([bk.day]))[0])] -= 1

    def counts_for(self, bk: Booking, m: int) -> bool:
        """Does this booking count as one of paying member m's delivered dates?"""
        if not self.guaranteed[m]:
            return False
        cyc = cycle_on(m, self.signup(m), bk.day, self.cfg)
        return counts_toward_guarantee(bk, m, cyc, self.cfg)

    # ------------------------------------------------------------ one day

    def run(self, label: str) -> CycleReport:
        for today in range(self.days):
            if self.record:
                self.snapshot(today)
            self.cancellations(today)
            self.dates(today)
            self.lists(today)
            self.events(today)
        return self.report(label)

    def cancellations(self, today: int) -> None:
        late_hours = self.cfg.billing.full_refund_hours
        for idx, (day, who, hours) in list(self.cancel_plan.items()):
            if day != today:
                continue
            del self.cancel_plan[idx]
            bk = self.bookings[idx]
            if bk.state != "confirmed":
                continue
            bk.cancel(who, hours)
            self.cancelled_on[idx] = today
            self.occupy(bk, -1, keep_cap=tuple(m for m in bk.members if self.counts_for(bk, m)))
            if hours < late_hours:
                self.count["late"] += 1
                self.trust.add_attendance(who, kept=False)
            else:
                self.count["early"] += 1

    def dates(self, today: int) -> None:
        for idx in self.on_day.get(today, []):
            bk = self.bookings[idx]
            if bk.state != "confirmed":
                continue
            came = self.behavior.shows_up(self.rng), self.behavior.shows_up(self.rng)
            bk.finish(*came)
            for m, c in zip(bk.members, came, strict=True):
                self.trust.add_attendance(m, kept=c)
                if c and not all(came):
                    self.release_slot(bk, m)  # stood up: the slot is refilled
            if all(came):
                for rater, rated in ((bk.a, bk.b), (bk.b, bk.a)):
                    self.trust.add_rating(self.rate(rater, rated))
                    self.check_flag(rated)

    def rate(self, rater: int, rated: int) -> Rating:
        cs = self.cfg.cycle_simulation
        attraction = float(self.behavior.utility[rater, rated])

        def score(trait: float) -> int:
            x = (
                cs.rating_center
                + cs.conduct_effect * trait
                + cs.rating_halo * attraction
                + cs.rating_noise * self.rng.normal()
            )
            return int(np.clip(np.rint(x), 1, 5))

        again, conversation = self.behavior.date_outcome(rater, rated, self.rng)
        respect = score(self.truth.conduct[rated])
        accuracy = score(self.truth.honesty[rated])
        return Rating(rater, rated, respect, accuracy, conversation, again)

    def check_flag(self, m: int) -> None:
        if not self.paused[m] and self.trust.flagged()[m]:
            self.paused[m] = True
            self.ratings_at_flag[m] = int(self.trust.n_ratings[m])

    def progress(self, today: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each paying man's dates delivered so far, confirmed dates to come and days left,
        all in his current cycle (indexed by member id)."""
        delivered, upcoming = np.zeros(self.n), np.zeros(self.n)
        days_left = np.ones(self.n)
        for m in np.flatnonzero(self.guaranteed):
            cyc = cycle_on(int(m), self.signup(m), today, self.cfg)
            for idx in self.by_member[m]:
                bk = self.bookings[idx]
                if not cyc.contains(bk.day):
                    continue
                if bk.state == "confirmed":
                    upcoming[m] += 1
                elif counts_toward_guarantee(bk, int(m), cyc, self.cfg):
                    delivered[m] += 1
            days_left[m] = cyc.days_left(today)
        return delivered, upcoming, days_left

    def missing_now(self, today: int) -> tuple[np.ndarray, np.ndarray]:
        """Each paying man's missing dates and days left in his current cycle."""
        g = self.cfg.dates.guaranteed_per_month_paying
        delivered, upcoming, days_left = self.progress(today)
        missing = np.where(self.guaranteed, np.maximum(0, g - delivered - upcoming), 0)
        return missing, days_left

    def market_now(self, today: int):
        """Market features of the women present and the paying men active today."""
        women = np.flatnonzero(
            self.matchable & ~self.arr.paying & (self.join_day <= today) & ~self.paused
        )
        men = np.flatnonzero(self.guaranteed & ~self.paused)
        return market_features(self.eligible, women, men)

    def snapshot(self, today: int) -> None:
        """Forecast rows: every man's market at launch, and his progress each week of his
        paid cycles (FR-LM-43, FR-LM-44)."""
        c, cycles = self.cfg.billing.cycle_days, self.cfg.cycle_simulation.cycles
        if today == 0:
            launch = market_features(
                self.eligible,
                np.flatnonzero(self.matchable & ~self.arr.paying & (self.join_day == 0)),
                np.flatnonzero(self.guaranteed),
            )
            self.launch_rows = start_matrix(launch)
        into = today - self.anchor
        weekly = self.guaranteed & ~self.paused & (into > 0) & (into < cycles * c)
        weekly &= (into % c > 0) & (into % 7 == 0)
        if not weekly.any():
            return
        mf = self.market_now(today)
        delivered, upcoming, days_left = self.progress(today)
        x = delivered_progress_matrix(mf, delivered, upcoming, days_left, self.cfg)
        keep = weekly[mf.men]
        for k in range(cycles):
            in_k = keep & ((into[mf.men] // c) == k)
            if in_k.any():
                self.week_rows.append((x[in_k], mf.men[in_k], k))

    def forecast_rows(self, report: CycleReport):
        """(x_start, y_start, x_progress, y_progress) from a recorded run.

        y_start: the share of each man's paid cycles that reached 3 delivered dates.
        y_progress: whether the cycle of each weekly snapshot ended with 3.
        """
        reached = report.delivered >= self.cfg.dates.guaranteed_per_month_paying
        row_of = {int(m): r for r, m in enumerate(np.flatnonzero(self.guaranteed))}
        xs, ys = [], []
        for rows, men, k in self.week_rows:
            xs.append(rows)
            ys.append(np.array([reached[row_of[int(m)], k] for m in men]))
        x_progress = np.vstack(xs) if xs else np.zeros((0, 8))
        y_progress = np.concatenate(ys) if ys else np.zeros(0)
        return self.launch_rows, reached.mean(axis=1), x_progress, y_progress

    def lists(self, today: int) -> None:
        cfg, n = self.cfg, self.n
        sim, lst = cfg.simulation, cfg.lists
        everyone = np.arange(n)
        now = self.period(everyone, np.full(n, today))
        full = (self.period_count[everyone, now] >= self.cap) | self.paused
        full |= self.join_day > today  # not joined yet
        opened = (self.rng.random(n) < sim.open_app_rate) & self.matchable & ~full
        bits = self.bits(today, everyone)
        order, avail = [], {}
        for i in (int(i) for i in self.rng.permutation(np.flatnonzero(opened))):
            a = available_row(
                i,
                self.eligible,
                self.rights,
                self.last_shown,
                today,
                lst.reshow_after_days,
                full,
                self.closed,
            )
            cand = np.flatnonzero(a)
            if cand.size == 0:
                continue
            a[cand[earliest_common(bits[i], bits[cand]) == NO_WINDOW]] = False
            if a.any():
                order.append(i)
                avail[i] = a
        if not order:
            return
        missing, days_left = self.missing_now(today)
        bonus = refill_bonus(missing, days_left, lst.guarantee_priority)
        scores = with_trust(self.scores, self.trust.ranking_factor())
        day_lists, _ = fair_day_lists(
            order, avail, scores, bonus, lst.daily_size, lst.exposure_multiple
        )
        for i in order:
            todays = day_lists[i]
            if todays.size == 0:
                continue
            self.last_shown[i, todays] = today
            chosen = self.behavior.choose(i, todays, self.rng, self.passes[i, todays])
            self.passes[i, np.setdiff1d(todays, chosen)] += 1
            for j in chosen:
                if self.closed[i, j]:
                    continue
                self.closed[i, j] = self.closed[j, i] = True
                self.count["picks"] += 1
                self.book(i, int(j), today, "list")

    def events(self, today: int) -> None:
        cs = self.cfg.cycle_simulation
        rate = cs.event_rate / self.cfg.billing.cycle_days
        for m in np.flatnonzero(self.guaranteed & ~self.paused):
            if self.rng.random() >= rate:
                continue
            others = np.flatnonzero(
                self.eligible[m]
                & self.rights[:, m]
                & ~self.closed[m]
                & ~self.paused
                & (self.join_day <= today)
            )
            if others.size == 0:
                continue
            w = int(self.rng.choice(others))
            self.closed[w, m] = self.closed[m, w] = True
            self.count["events"] += 1
            self.book(w, int(m), today, "event")

    def book(self, a: int, b: int, today: int, source: str) -> None:
        """a chose b (or they met at an event): propose, hold, pay, confirm."""
        cfg, cs = self.cfg, self.cfg.cycle_simulation
        ba, bb = self.bits(today, np.array([a, b]))
        window = int(earliest_common(ba, bb))
        if window == NO_WINDOW:
            # an earlier booking today took the last window: in the app he would just drop
            # off her list, so the pair stays open for another day
            self.count["no_window"] += 1
            self.closed[a, b] = self.closed[b, a] = False
            return
        day, part = window_day_part(window, today)
        if source == "list" and not self.behavior.accepts(b, a, self.rng):
            self.count["declined"] += 1
            return
        bk = Booking(a, b, day, part) if source == "list" else event_booking(a, b, day, part)
        hold = cfg.booking.hold_minutes
        bk.hold(0.0)
        venues = cfg.geo.venues
        hoods = cfg.geo.neighbourhoods
        v, trip_a, trip_b = geo.venue_for(hoods[self.hood[a]], hoods[self.hood[b]], venues)
        late = hold + 1.0
        on_time = self.rng.random() < cs.pay_in_time_chooser
        ok = bk.pay(a, SHE_PAYS_AT if on_time else late, hold)
        if ok:
            on_time = self.rng.random() < cs.pay_in_time_chosen
            ok = bk.pay(b, HE_PAYS_AT if on_time else late, hold, venue=venues[v].name)
        idx = len(self.bookings)
        self.bookings.append(bk)
        self.booked_on.append(today)
        if not ok:
            self.count["expired"] += 1
            self.closed[a, b] = self.closed[b, a] = False  # they can find each other again
            return
        for m in bk.members:
            self.by_member[m].append(idx)
        self.on_day.setdefault(day, []).append(idx)
        self.occupy(bk, +1)
        self.trips.append((max(trip_a, trip_b), abs(trip_a - trip_b)))
        plans = []
        for m in bk.members:
            if self.rng.random() < cs.cancel_rate:
                if self.rng.random() < cs.late_cancel_share or day - 2 < today:
                    plans.append((day, m, 6.0))  # on the day, a few hours ahead
                else:
                    plans.append((day - 2, m, 48.0))
        if plans:
            self.cancel_plan[idx] = min(plans)

    # ------------------------------------------------------------ results

    def report(self, label: str) -> CycleReport:
        cfg, cs = self.cfg, self.cfg.cycle_simulation
        c = cfg.billing.cycle_days
        g = cfg.dates.guaranteed_per_month_paying
        men = np.flatnonzero(self.guaranteed)
        delivered = np.zeros((men.size, cs.cycles), dtype=int)
        credits = revenue = 0.0
        for r, m in enumerate(men):
            for k in range(cs.cycles):
                cyc = cycle_on(int(m), self.signup(m), int(self.anchor[m]) + k * c, cfg)
                got = sum(
                    counts_toward_guarantee(self.bookings[i], int(m), cyc, cfg)
                    for i in self.by_member[m]
                )
                delivered[r, k] = got
                credits += credit(max(0, g - got), cyc, cfg)
                revenue += cyc.price
        past = [bk for bk in self.bookings if bk.day < self.days]
        fees_paid = sum(len(bk.paid) for bk in past) * cfg.billing.date_fee
        refunded = sum(sum(fee_refunds(bk, cfg).values()) for bk in past)
        states: dict[str, int] = {}
        for bk in past:
            states[bk.state] = states.get(bk.state, 0) + 1
        women = np.flatnonzero(self.matchable & ~self.arr.paying & (self.join_day == 0))
        women_dates = np.zeros((women.size, cs.cycles), dtype=int)
        pos = {int(w): r for r, w in enumerate(women)}
        for bk in past:
            if bk.state != "completed" or bk.day // c >= cs.cycles:
                continue
            for m in bk.members:
                if m in pos:
                    women_dates[pos[m], bk.day // c] += 1
        bad = self.truth.bad & self.matchable
        flagged = np.zeros(self.n, dtype=bool)
        flagged[list(self.ratings_at_flag)] = True
        enough = self.trust.n_ratings >= cfg.trust.flag_min_ratings
        well_rated = bad & ((self.trust.n_ratings >= WELL_RATED) | flagged)
        looks = self.behavior.looks
        looks_appeal = np.nanmean(np.where(np.isfinite(looks), looks, np.nan), axis=0)
        sample = enough & ~bad
        corr = (
            float(np.corrcoef(self.trust.trust()[sample], looks_appeal[sample])[0, 1])
            if sample.sum() > 2
            else float("nan")
        )
        trips = np.array(self.trips) if self.trips else np.zeros((1, 2))
        return CycleReport(
            label=label,
            cycles=cs.cycles,
            men=int(men.size),
            women=int(women.size),
            delivered=delivered,
            credits=credits,
            revenue=revenue,
            fees_paid=fees_paid,
            fees_refunded=refunded,
            states=states,
            events=self.count["events"],
            holds_expired=self.count["expired"],
            picks=self.count["picks"],
            declined=self.count["declined"],
            no_window=self.count["no_window"],
            cancellations={"early": self.count["early"], "late": self.count["late"]},
            women_dates=women_dates,
            bad=int(bad.sum()),
            flagged=int(flagged.sum()),
            flagged_bad=int((flagged & bad).sum()),
            bad_well_rated=int(well_rated.sum()),
            caught_well_rated=int((well_rated & flagged).sum()),
            ratings_at_flag=list(self.ratings_at_flag.values()),
            trust_looks_corr=corr,
            longer_trip_km=float(trips[:, 0].mean()),
            trip_gap_km=float(trips[:, 1].mean()),
        )


NEVER = 10**9  # join day of women who never arrive


def market(cfg: Config, seed: int, max_new_per_cycle: float) -> Population:
    """The Phase 1 market plus room for new women: the settings' men and women at launch,
    then enough extra women for `max_new_per_cycle` new women per cycle."""
    p, cs = cfg.population, cfg.cycle_simulation
    n_men = round(p.n_members * p.share_men)
    n_women = p.n_members - n_men
    extra = round(n_women * max_new_per_cycle * cs.cycles)
    n = n_men + n_women + extra
    pop_cfg = dataclasses.replace(p, n_members=n, share_men=n_men / n, exact_counts=True)
    return generate(dataclasses.replace(cfg, population=pop_cfg), seed=seed)


def join_days(pop: Population, cfg: Config, new_per_cycle: float) -> np.ndarray:
    """Day each member joins: everyone at launch except the newest women, who arrive at an
    even pace of `new_per_cycle` (share of launch women) per cycle; the rest never come."""
    p, cs, c = cfg.population, cfg.cycle_simulation, cfg.billing.cycle_days
    n_women = p.n_members - round(p.n_members * p.share_men)
    women = np.flatnonzero(np.array([m.gender == "woman" for m in pop.members]))
    join = np.zeros(len(pop), dtype=np.int64)
    later = women[n_women:]
    join[later] = NEVER
    days = (c if cs.stagger else 0) + cs.cycles * c
    k = min(later.size, round(n_women * new_per_cycle * days / c))
    join[later[:k]] = np.linspace(1, days, k, endpoint=False).astype(np.int64)
    return join


def simulate_cycles(
    pop: Population,
    cfg: Config,
    seed: int,
    scores: np.ndarray,
    label: str = "real flow",
    join_day: np.ndarray | None = None,
) -> CycleReport:
    return CycleSimulation(pop, cfg, seed, scores, join_day).run(label)


def frictionless(cfg: Config) -> Config:
    """Everyone pays in time, nobody cancels, everyone shows up."""
    cs = dataclasses.replace(
        cfg.cycle_simulation, pay_in_time_chooser=1.0, pay_in_time_chosen=1.0, cancel_rate=0.0
    )
    sim = dataclasses.replace(cfg.simulation, show_up_rate=1.0)
    return dataclasses.replace(cfg, cycle_simulation=cs, simulation=sim)


def synchronized(cfg: Config) -> Config:
    """Every man's cycles start on launch day."""
    return dataclasses.replace(
        cfg, cycle_simulation=dataclasses.replace(cfg.cycle_simulation, stagger=False)
    )


def batch_baseline(
    pop: Population, cfg: Config, seed: int, scores: np.ndarray, join_day: np.ndarray
) -> list[float]:
    """Sprint 6's flow on the same market: weekly batch schedule, calendar months.

    Only members present at launch take part (no new women), pairs who dated in an earlier
    month are not matched again, and booked dates count: no payment holds, cancellations or
    no-shows.
    """
    n = len(pop)
    men = np.flatnonzero(np.array([m.gender == "man" for m in pop.members]))
    excluded = np.zeros((n, n), dtype=bool)
    later = join_day > 0
    excluded[later, :] = excluded[:, later] = True
    shares = []
    for k in range(cfg.cycle_simulation.cycles):
        m = run_month(pop, cfg, seed, scores, month=k + 1, excluded=excluded, record=True)
        paying = men[m.paying[men]]
        shares.append(float((m.dates[paying] >= cfg.dates.guaranteed_per_month_paying).mean()))
        for r in m.date_records:
            excluded[r.a, r.b] = excluded[r.b, r.a] = True
    return shares


def show_up(cfg: Config, rate: float) -> Config:
    sim = dataclasses.replace(cfg.simulation, show_up_rate=rate)
    return dataclasses.replace(cfg, simulation=sim)


def pays_in_time(cfg: Config) -> Config:
    cs = dataclasses.replace(cfg.cycle_simulation, pay_in_time_chooser=1.0, pay_in_time_chosen=1.0)
    return dataclasses.replace(cfg, cycle_simulation=cs)


def scenarios(cfg: Config) -> dict[str, tuple[Config, float]]:
    """name: (settings, new women per cycle). The first is the settings as they are."""
    new = cfg.cycle_simulation.new_women_per_cycle
    return {
        f"real flow, {new:.0%} new women per cycle": (cfg, new),
        "real flow, no new women": (cfg, 0.0),
        "real flow, every cycle starts on launch day": (synchronized(cfg), new),
        "show-up 97% instead of 92%": (show_up(cfg, 0.97), new),
        "everyone pays within the 15 minutes": (pays_in_time(cfg), new),
        "no friction (all pay, nobody cancels, all show up)": (frictionless(cfg), new),
    }


def compare(cfg: Config, seeds: tuple[int, ...]) -> tuple[dict[str, list[CycleReport]], list]:
    """Every scenario on the same markets (one per seed), plus Sprint 6's batch flow."""
    runs: dict[str, list[CycleReport]] = {}
    batch = []
    for seed in seeds:
        pop = market(cfg, seed, cfg.cycle_simulation.new_women_per_cycle)
        sc = build_scores(pop, cfg, seed)
        for name, (c, new) in scenarios(cfg).items():
            join = join_days(pop, c, new)
            runs.setdefault(name, []).append(simulate_cycles(pop, c, seed, sc.mutual, name, join))
        batch.append(np.mean(batch_baseline(pop, cfg, seed, sc.mutual, join_days(pop, cfg, 0.0))))
    return runs, batch


def comparison_text(runs: dict[str, list[CycleReport]], batch: list, seeds) -> str:
    def mean_se(xs):
        xs = np.asarray(xs, dtype=float)
        se = xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else float("nan")
        return f"{xs.mean():6.1%} ± {se:4.1%}"

    lines = [
        f"Phase 1 cycles: men reaching 3 delivered dates, average of {len(seeds)} simulated "
        f"markets (seeds {', '.join(map(str, seeds))}), ± one standard error",
        f"  {'version':52s} {'3 dates':>15s} {'by cycle':>16s} {'credits':>8s}",
    ]
    for name, reps in runs.items():
        by_cycle = np.mean([r.per_cycle() for r in reps], axis=0)
        credits = np.mean([r.credits / max(r.revenue, 1) for r in reps])
        lines.append(
            f"  {name:52s} {mean_se([r.guarantee_share for r in reps]):>15s} "
            f"{' '.join(f'{x:4.0%}' for x in by_cycle):>16s} {credits:8.1%}"
        )
    lines.append(
        f"  {'Sprint 6 weekly batch, booked dates, no new women':52s} {mean_se(batch):>15s}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="lastfirstdate's real booking flow.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = parser.parse_args()
    cfg = load(PHASE1)
    runs, batch = compare(cfg, tuple(args.seeds))
    print(comparison_text(runs, batch, args.seeds))
    print()
    print(next(iter(runs.values()))[0].text())


if __name__ == "__main__":
    main()
