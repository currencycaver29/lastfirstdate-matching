"""Fill an app database from fake members, to test the engine on rows (FR-LM-47).

Test runs only. It writes a synthetic population into the tables of `db/schema.sql` the way
the app would: neighbourhoods and venues, members with neighbourhoods (never coordinates),
free slots, intent answers, prompts, subscriptions, and each member's photo exercise, run with
the engine's own calibration and the synthetic people's answers.

Database ids start at 1000 so a mix-up between ids and row numbers shows up in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config
from lastfirstdate.filter import member_ok
from lastfirstdate.geo import nearest_place
from lastfirstdate.looks import (
    ArrayPhotoEncoder,
    PersonalTaste,
    StartingTaste,
    Swipe,
    calibrate,
    calibration_pool,
    photo_features,
)
from lastfirstdate.member_behavior import Behavior
from lastfirstdate.synthetic import Population

SCHEMA = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
FIRST_ID = 1000


@dataclass
class FilledDatabase:
    """What was written, for tests to compare against."""

    start: StartingTaste  # the starting taste the calibration used
    tastes: dict[int, PersonalTaste] = field(default_factory=dict)  # by row, as calibrated
    subscribers: list[int] = field(default_factory=list)  # database ids
    waitlist: list[int] = field(default_factory=list)  # database ids, in order


def db_id(row: int) -> int:
    return int(FIRST_ID + row)  # a plain int: SQLite can't store numpy integers


def birth_date(age: int, today: date) -> date:
    """A birth date that makes someone exactly `age` on `today` (birthday yesterday)."""
    return date(today.year - age, today.month, today.day) - timedelta(days=1)


def fill_database(
    con,
    pop: Population,
    cfg: Config,
    today: date,
    seed: int,
    subscribers: int | None = None,
) -> FilledDatabase:
    """Write `pop` into an empty database with the schema loaded.

    The first `subscribers` paying men (all, if None) get a subscription that began 40 days
    ago (past the free trial); the other paying men go on the waitlist, in id order.
    """
    con.executescript(SCHEMA.read_text())
    hoods = cfg.geo.neighbourhoods
    con.executemany(
        "INSERT INTO neighbourhoods VALUES (?, ?, ?, ?)",
        [(k + 1, h.name, h.lat, h.lon) for k, h in enumerate(hoods)],
    )
    venue_hood = nearest_place(
        [v.lat for v in cfg.geo.venues], [v.lon for v in cfg.geo.venues], hoods
    )
    con.executemany(
        "INSERT INTO venues VALUES (?, ?, ?, ?, ?, 1)",
        [
            (k + 1, v.name, int(h) + 1, v.lat, v.lon)
            for k, (v, h) in enumerate(zip(cfg.geo.venues, venue_hood, strict=True))
        ],
    )
    lat = np.array([m.lat for m in pop.members])
    lon = np.array([m.lon for m in pop.members])
    member_hood = nearest_place(lat, lon, hoods)
    paying_men = [k for k, m in enumerate(pop.members) if m.paying and m.gender == "man"]
    subs = paying_men if subscribers is None else paying_men[:subscribers]
    waiting = [k for k in paying_men if k not in set(subs)]
    rows = []
    for k, m in enumerate(pop.members):
        rows.append(
            (
                db_id(k),
                m.gender,
                int("man" in m.seeking),
                int("woman" in m.seeking),
                int("nonbinary" in m.seeking),
                birth_date(m.age, today).isoformat(),
                m.age_min_pref,
                m.age_max_pref,
                int(member_hood[k]) + 1,
                m.max_km,
                int(m.verified),
                int(m.paying),
                int(k in set(subs)),
                0,
                (today - timedelta(days=60)).isoformat(),
                (today - timedelta(days=m.last_active_days)).isoformat(),
            )
        )
    con.executemany(
        "INSERT INTO members (id, gender, seeks_men, seeks_women, seeks_nonbinary, birth_date, "
        "age_min_pref, age_max_pref, neighbourhood_id, max_km, verified, income_verified, "
        "paying, paused_for_review, joined_at, last_active_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.executemany(
        "INSERT INTO free_slots VALUES (?, ?)",
        [(db_id(k), s) for k, m in enumerate(pop.members) for s in sorted(m.free_slots)],
    )
    con.executemany(
        "INSERT INTO intent_answers VALUES (?, ?, ?)",
        [(db_id(k), q, a) for k, m in enumerate(pop.members) for q, a in m.values.items()],
    )
    con.executemany(
        "INSERT INTO prompts VALUES (?, 'about', ?)",
        [(db_id(k), m.bio[:500]) for k, m in enumerate(pop.members)],
    )
    signed = (today - timedelta(days=40)).isoformat()
    con.executemany(
        "INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?, 'active')",
        [
            (
                db_id(k),
                signed,
                cfg.billing.trial_days,
                cfg.billing.cycle_days,
                int(cfg.billing.price_per_cycle * 100),
            )
            for k in subs
        ],
    )
    con.executemany(
        "INSERT INTO waitlist VALUES (?, ?, 'waiting')",
        [
            (db_id(k), (today - timedelta(days=30) + timedelta(minutes=n)).isoformat())
            for n, k in enumerate(waiting)
        ],
    )
    filled = _calibration(con, pop, cfg, seed)
    filled.subscribers = [db_id(k) for k in subs]
    filled.waitlist = [db_id(k) for k in waiting]
    con.commit()
    return filled


def _calibration(con, pop: Population, cfg: Config, seed: int) -> FilledDatabase:
    """Run every matchable member's photo exercise and store the answers."""
    arr = MemberArrays.from_members(pop.members, cfg.values)
    behavior = Behavior(pop, arr, cfg, np.random.default_rng(seed + 3_000_017))
    _, feats = photo_features(ArrayPhotoEncoder(pop.photo_embeddings), len(pop), cfg)
    rng = np.random.default_rng(seed + 4_000_037)
    viewers = [int(v) for v in np.flatnonzero(member_ok(arr, cfg))]
    pools = {v: calibration_pool(arr, cfg, v) for v in viewers}
    likes = {v: behavior.swipe_outcomes(v, seed) for v in viewers}
    history = [
        Swipe(int(arr.gender[v]), int(arr.gender[j]), feats[j], bool(likes[v][j]))
        for v in viewers
        for j in rng.choice(
            pools[v], min(cfg.test_runs.training_swipes, pools[v].size), replace=False
        )
    ]
    start = StartingTaste.fit(history, cfg.calibration.w0_l2_penalty)
    filled = FilledDatabase(start)
    answers = []
    for v in viewers:
        if pools[v].size == 0:
            continue
        log: list[tuple[int, int | None, bool]] = []

        def swipe(j, v=v, log=log):
            liked = bool(likes[v][j])
            log.append((j, None, liked))
            return liked

        def pair(a, b, v=v, log=log):
            chose_a = behavior.pair_outcome(v, a, b, seed)
            log.append((a, b, chose_a))
            return chose_a

        taste = PersonalTaste(
            int(arr.gender[v]), start, cfg.calibration.l2_penalty, cfg.calibration.n_profiles
        )
        calibrate(taste, pools[v], feats, arr.gender, swipe, cfg, pair=pair)
        filled.tastes[v] = taste
        answers += [
            (db_id(v), pos, db_id(a), None if b is None else db_id(b), int(ans), "2026-01-01")
            for pos, (a, b, ans) in enumerate(log)
        ]
    con.executemany("INSERT INTO calibration_answers VALUES (?, ?, ?, ?, ?, ?)", answers)
    return filled


def write_history(con, sim, epoch: date) -> None:
    """Write a finished `CycleRun` into the database as the app's records (FR-LM-51).

    Test run day 0 is `epoch` (a Monday). Members get their join days, paying men their
    subscriptions (signed up a trial before their first paid cycle), and every booking its
    timestamps, payments and refunds, as the app would have stored them: timestamps in UTC
    with an offset, dates in the zone's own time (D19).
    """
    from zoneinfo import ZoneInfo

    from lastfirstdate.billing import fee_refunds
    from lastfirstdate.models import PARTS_OF_DAY

    cfg = sim.cfg
    tz = ZoneInfo(cfg.booking.timezone)

    def utc(local: datetime) -> str:
        return local.replace(tzinfo=tz).astimezone(UTC).isoformat(timespec="seconds")

    con.executemany(
        "UPDATE members SET joined_at = ? WHERE id = ?",
        [
            ((epoch + timedelta(days=int(min(d, 3650)) if d > 0 else -60)).isoformat(), db_id(k))
            for k, d in enumerate(sim.join_day)
        ],
    )
    con.execute("DELETE FROM waitlist")
    con.execute("DELETE FROM subscriptions")
    con.executemany(
        "INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?, 'active')",
        [
            (
                db_id(m),
                (epoch + timedelta(days=int(sim.signup(m)))).isoformat(),
                cfg.billing.trial_days,
                cfg.billing.cycle_days,
                int(cfg.billing.price_per_cycle * 100),
            )
            for m in np.flatnonzero(sim.guaranteed)
        ],
    )
    con.execute("INSERT INTO events VALUES (1, 'Members evening', 1, ?)", (epoch.isoformat(),))
    venue_id = {name: vid for vid, name in con.execute("SELECT id, name FROM venues")}
    for idx, bk in enumerate(sim.bookings):
        day = epoch + timedelta(days=bk.day)
        part = PARTS_OF_DAY[bk.part]
        made = utc(datetime.combine(epoch + timedelta(days=sim.booked_on[idx]), time(18)))
        cancelled_at = None
        if bk.cancelled_by is not None:  # true hours before the start, across clock changes
            start = datetime.combine(day, time(cfg.booking.part_starts[bk.part]), tzinfo=tz)
            when = start.astimezone(UTC) - timedelta(hours=bk.hours_before)
            cancelled_at = when.isoformat(timespec="seconds")
        confirmed = bk.state != "expired"
        con.execute(
            "INSERT INTO bookings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                idx + 1,
                db_id(bk.a),
                db_id(bk.b),
                bk.source,
                1 if bk.source == "event" else None,
                day.isoformat(),
                part,
                bk.state,
                made,
                made,
                made if confirmed else None,
                venue_id.get(bk.venue) if confirmed else None,
                None if bk.cancelled_by is None else db_id(bk.cancelled_by),
                cancelled_at,
                None if not bk.showed else int(bk.showed[bk.a]),
                None if not bk.showed else int(bk.showed[bk.b]),
            ),
        )
        refunds = fee_refunds(bk, cfg)
        for m in sorted(bk.paid):
            con.execute(
                "INSERT INTO payments (booking_id, member_id, amount_cents, paid_at, "
                "refunded_cents) VALUES (?, ?, ?, ?, ?)",
                (
                    idx + 1,
                    db_id(m),
                    int(cfg.billing.date_fee * 100),
                    made,
                    int(round(refunds.get(m, 0.0) * 100)),
                ),
            )
    con.commit()
