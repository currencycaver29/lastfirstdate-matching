"""A simulated month on fake members, and its report (FR-LM-12).

Each day, members open the app with probability `simulation.open_app_rate`. A chooser who
opens it gets their daily list (Stage 6), picks a few profiles (simulator), and each picked
person accepts or declines (simulator). Accepted pairs wait for the weekly scheduler
(Stage 7), which books dates at the end of each of the `dates.weeks_per_month` weeks.

Run `python -m lastfirstdate.simulate` to print the report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config, load
from lastfirstdate.filter import eligibility_matrix, member_ok
from lastfirstdate.lists import NEVER_SHOWN, available_row, daily_list, pick_rights
from lastfirstdate.personality import personality_matrix
from lastfirstdate.scheduler import schedule_week, urgency
from lastfirstdate.simulator import Behavior
from lastfirstdate.synthetic import Population, generate

DAYS_PER_WEEK = 7


@dataclass
class Miss:
    member: int
    gender: str
    dates: int
    pool: int
    times_shown: int
    times_picked: int
    yeses: int  # pairs where both said yes
    reason: str


@dataclass
class MonthReport:
    seed: int
    n_members: int
    n_matchable: int
    pool: np.ndarray  # (n,) possible matches per member after Stage 1, 0 if not matchable
    matchable: np.ndarray  # (n,) bool
    paying: np.ndarray  # (n,) bool, paying and matchable
    gender: list[str]
    dates: np.ndarray  # (n,) dates booked this month
    list_size_by_day: list[float]
    picks: int
    accepts: int
    booked: int
    held: int
    second_dates: int
    methods: list[str]
    guarantee: int
    min_pool: int
    pair_appeal: float  # average true appeal within booked pairs, both directions (simulation)
    both_keen: float  # share of booked pairs where each truly likes the other above average
    misses: list[Miss] = field(default_factory=list)

    @property
    def workable(self) -> np.ndarray:
        return self.paying & (self.pool >= self.min_pool)

    def guarantee_share(self, mask: np.ndarray | None = None) -> float:
        group = self.workable if mask is None else self.workable & mask
        if not group.any():
            return float("nan")
        return float((self.dates[group] >= self.guarantee).mean())

    def text(self) -> str:
        n_pay = int(self.paying.sum())
        n_work = int(self.workable.sum())
        men = np.array([g == "man" for g in self.gender])
        pool = self.pool[self.matchable]
        sizes = self.list_size_by_day
        week_avgs = [
            np.mean(sizes[w * DAYS_PER_WEEK : (w + 1) * DAYS_PER_WEEK])
            for w in range(len(sizes) // DAYS_PER_WEEK)
        ]
        dist = np.bincount(self.dates[self.paying], minlength=5)
        accept_rate = self.accepts / max(self.picks, 1)
        second_rate = self.second_dates / max(self.held, 1)
        lines = [
            f"Simulated month, seed {self.seed}",
            f"  members {self.n_members}, matchable (verified and active) {self.n_matchable}",
            f"  possible matches per member: median {np.median(pool):.0f}, "
            f"10th percentile {np.percentile(pool, 10):.0f}, "
            f"under {self.min_pool}: {(pool < self.min_pool).sum()}",
            "  average daily list size by week: " + ", ".join(f"{a:.1f}" for a in week_avgs),
            f"  picks {self.picks}, accepted {self.accepts} ({accept_rate:.0%})",
            f"  dates booked {self.booked}, held {self.held}, "
            f"both wanted a second date {self.second_dates} ({second_rate:.0%})",
            f"  booked pairs: average true appeal {self.pair_appeal:+.2f} SD, "
            f"both above average {self.both_keen:.0%}",
            f"  scheduler method by week: {', '.join(self.methods)}",
            "",
            f"Guarantee: {self.guarantee} dates for paying members",
            f"  paying and matchable {n_pay}; "
            f"with at least {self.min_pool} possible matches {n_work}",
            f"  reached the guarantee: {self.guarantee_share():.1%} "
            f"(men {self.guarantee_share(men):.1%}, women {self.guarantee_share(~men):.1%})",
            "  dates per paying member: " + ", ".join(f"{k}: {dist[k]}" for k in range(len(dist))),
        ]
        if self.misses:
            reasons: dict[str, int] = {}
            for m in self.misses:
                reasons[m.reason] = reasons.get(m.reason, 0) + 1
            lines.append(f"  missed: {len(self.misses)} paying members")
            for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {c:4d}  {r}")
        return "\n".join(lines)


def month_people(
    pop: Population, arr: MemberArrays, cfg: Config, seed: int
) -> tuple[Behavior, np.random.Generator]:
    """The simulated people for a month with this seed, and the generator their choices use.

    Anything that must be judged by the same people as run_month (for example the ranking
    quality in mutual_report) should get them from here.
    """
    rng = np.random.default_rng(seed + 1_000_003)
    return Behavior(pop, arr, cfg, rng), rng


def _miss_reason(pool: int, yeses: int, min_pool: int, guarantee: int) -> str:
    if pool < min_pool:
        return f"small pool (fewer than {min_pool} possible matches)"
    if yeses < guarantee:
        return f"fewer than {guarantee} mutual yeses all month"
    return "enough mutual yeses, but the schedule could not fit them"


def run_month(
    pop: Population | None = None,
    cfg: Config | None = None,
    seed: int | None = None,
    scores: np.ndarray | None = None,
) -> MonthReport:
    """Simulate one month. `scores` (n, n) ranks lists and weights dates; default is P."""
    cfg = cfg or load()
    seed = cfg.simulation.seed if seed is None else seed
    pop = pop or generate(cfg, seed=seed)
    members = pop.members
    n = len(members)
    arr = MemberArrays.from_members(members, cfg.values)
    behavior, rng = month_people(pop, arr, cfg, seed)

    eligible = eligibility_matrix(arr, cfg)
    if scores is None:
        scores = personality_matrix(members, arr, cfg)
    rights = pick_rights(arr)
    matchable = member_ok(arr, cfg)
    pool = eligible.sum(axis=1)

    d = cfg.dates
    sim = cfg.simulation
    dates = np.zeros(n, dtype=int)
    last_shown = np.full((n, n), NEVER_SHOWN, dtype=np.int64)
    passes = np.zeros((n, n), dtype=np.int16)
    today = 0
    closed = np.zeros((n, n), dtype=bool)
    times_shown = np.zeros(n, dtype=int)
    times_picked = np.zeros(n, dtype=int)
    yeses = np.zeros(n, dtype=int)
    pending: set[tuple[int, int]] = set()
    list_sizes: list[float] = []
    picks = accepts = held = second = 0
    methods: list[str] = []
    booked_total = 0
    booked_pairs: list[tuple[int, int]] = []

    for week in range(d.weeks_per_month):
        for _ in range(DAYS_PER_WEEK):
            full = dates >= d.cap_per_month
            opened = (rng.random(n) < sim.open_app_rate) & matchable
            sizes = []
            for i in np.flatnonzero(opened):
                avail = available_row(
                    i,
                    eligible,
                    rights,
                    last_shown,
                    today,
                    cfg.lists.reshow_after_days,
                    full,
                    closed,
                )
                if not avail.any():
                    continue
                todays = daily_list(scores[i], avail, cfg.lists.daily_size)
                last_shown[i, todays] = today
                times_shown[todays] += 1
                sizes.append(len(todays))
                chosen = behavior.choose(i, todays, rng, passes[i, todays])
                passes[i, np.setdiff1d(todays, chosen)] += 1
                for j in chosen:
                    if closed[i, j]:
                        continue
                    closed[i, j] = closed[j, i] = True
                    picks += 1
                    times_picked[j] += 1
                    if behavior.accepts(j, i, rng):
                        accepts += 1
                        yeses[[i, j]] += 1
                        pending.add((min(i, j), max(i, j)))
            list_sizes.append(float(np.mean(sizes)) if sizes else 0.0)
            today += 1

        weeks_left = d.weeks_per_month - week
        urg = urgency(
            dates, arr.paying, d.guaranteed_per_month_paying, weeks_left, d.late_month_bonus
        )
        pairs = sorted(
            p for p in pending if dates[p[0]] < d.cap_per_month and dates[p[1]] < d.cap_per_month
        )
        capacity = {
            v: min(d.max_per_week, d.cap_per_month - int(dates[v])) for p in pairs for v in p
        }
        weights = [
            d.date_value + (scores[a, b] + scores[b, a]) / 2 + urg[a] + urg[b] for a, b in pairs
        ]
        masks = [int(arr.slots[a] & arr.slots[b]) for a, b in pairs]
        result = schedule_week(pairs, weights, masks, capacity, rng)
        methods.append(result.method)
        for bk in result.bookings:
            booked_pairs.append((bk.a, bk.b))
            dates[[bk.a, bk.b]] += 1
            pending.discard((bk.a, bk.b))
            booked_total += 1
            if behavior.shows_up(rng) and behavior.shows_up(rng):
                held += 1
                if behavior.wants_again(bk.a, bk.b, rng) and behavior.wants_again(bk.b, bk.a, rng):
                    second += 1
        pending = {
            p for p in pending if dates[p[0]] < d.cap_per_month and dates[p[1]] < d.cap_per_month
        }

    paying = arr.paying & matchable
    u = behavior.utility
    appeal = [(u[a, b] + u[b, a]) / 2 for a, b in booked_pairs]
    keen = [min(u[a, b], u[b, a]) > 0 for a, b in booked_pairs]
    report = MonthReport(
        seed=seed,
        n_members=n,
        n_matchable=int(matchable.sum()),
        pool=np.where(matchable, pool, 0),
        matchable=matchable,
        paying=paying,
        gender=[m.gender for m in members],
        dates=dates,
        list_size_by_day=list_sizes,
        picks=picks,
        accepts=accepts,
        booked=booked_total,
        held=held,
        second_dates=second,
        methods=methods,
        guarantee=d.guaranteed_per_month_paying,
        min_pool=sim.min_pool_for_guarantee,
        pair_appeal=float(np.mean(appeal)) if appeal else float("nan"),
        both_keen=float(np.mean(keen)) if keen else float("nan"),
    )
    for v in np.flatnonzero(paying & (dates < d.guaranteed_per_month_paying)):
        report.misses.append(
            Miss(
                member=int(v),
                gender=members[v].gender,
                dates=int(dates[v]),
                pool=int(pool[v]),
                times_shown=int(times_shown[v]),
                times_picked=int(times_picked[v]),
                yeses=int(yeses[v]),
                reason=_miss_reason(
                    int(pool[v]),
                    int(yeses[v]),
                    sim.min_pool_for_guarantee,
                    d.guaranteed_per_month_paying,
                ),
            )
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate one month on fake members.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    cfg = load()
    seed = cfg.simulation.seed if args.seed is None else args.seed
    pop = generate(cfg, seed=seed, n=args.n)
    print(run_month(pop, cfg, seed=seed).text())


if __name__ == "__main__":
    main()
