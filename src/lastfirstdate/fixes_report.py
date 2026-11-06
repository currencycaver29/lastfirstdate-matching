"""Sprint 12 report: what each fix changes, on synthetic members (FR-LM-54 to FR-LM-57).

Usage: python -m lastfirstdate.fixes_report

1. Times: hours before a winter evening date, as the old code read a UTC timestamp and as
   the engine reads it now.
2. Picks: one morning in a Phase 1 zone, every woman picks the top profile on her list at
   once. How many picks would land on a window or day already taken, and what the re-check
   does instead.
3. Forgetting: the cost, and that the zone still loads and refreshes.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from lastfirstdate import app
from lastfirstdate.app_fill import fill_database
from lastfirstdate.config import Config, load
from lastfirstdate.synthetic import generate

TODAY, EPOCH = date(2026, 11, 4), date(2026, 11, 2)  # a Wednesday, and the Monday before


def times_table(cfg: Config) -> list[tuple[int, float, float]]:
    """(true hours ahead, hours the old code read, hours read now) for cancellations of a
    Friday 6 November, 7 pm date, stored in UTC without an offset as the schema asked."""
    tz = ZoneInfo(cfg.booking.timezone)
    start = datetime(2026, 11, 6, cfg.booking.part_starts[1], tzinfo=tz)
    out = []
    for ahead in range(20, 33):
        stamp = (start.astimezone(UTC) - timedelta(hours=ahead)).replace(tzinfo=None).isoformat()
        old = (start.replace(tzinfo=None) - datetime.fromisoformat(stamp)).total_seconds() / 3600
        new = app.hours_between(start, app.local_time(stamp, tz))
        out.append((ahead, old, new))
    return out


def zone(cfg: Config, seed: int = 7):
    pop = generate(cfg, seed=seed)
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    filled = fill_database(con, pop, cfg, TODAY, seed, subscribers=60)
    market = app.load_market(con, cfg, app.Calendar(EPOCH), TODAY)
    feats = app.photo_feature_matrix(pop.photo_embeddings, cfg)
    answers = app.calibration_answers(con, market, cfg)
    tastes = app.fit_tastes(market, feats, answers, cfg, start=filled.start)
    return pop, con, market, app.nightly_scores(market, tastes, feats, cfg)


def picks_morning(cfg: Config, con, market, scores) -> dict[str, float]:
    women = [int(market.ids[k]) for k, m in enumerate(market.members) if m.gender == "woman"]
    t0 = time.perf_counter()
    lists = app.day_lists(con, market, scores, women, cfg)
    out: dict[str, float] = {"lists_seconds": time.perf_counter() - t0, "women": len(lists)}
    windows: dict[tuple, int] = {}
    days: dict[tuple, int] = {}
    for entries in lists.values():
        e = entries[0]
        windows[e.member_id, e.day, e.part] = windows.get((e.member_id, e.day, e.part), 0) + 1
        days[e.member_id, e.day] = days.get((e.member_id, e.day), 0) + 1
    out["taken_window"] = sum(n - 1 for n in windows.values())
    out["taken_day"] = sum(n - 1 for n in days.values())
    kept = moved = refused = 0
    seconds = []
    for woman, entries in lists.items():
        e = entries[0]
        t1 = time.perf_counter()
        got = app.recheck_pick(con, market, woman, e.member_id, e, cfg)
        seconds.append(time.perf_counter() - t1)
        if got is None:
            refused += 1
            continue
        if (got.day, got.part) == (e.day, e.part):
            kept += 1
        else:
            moved += 1
        bk, venue = app.propose(con, market, woman, e.member_id, got)
        row = app.booking_row(bk, market, venue, datetime(2026, 11, 4, 14, tzinfo=UTC))
        con.execute(
            f"INSERT INTO bookings ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
            tuple(row.values()),
        )
    out.update(kept=kept, moved=moved, refused=refused)
    out["recheck_ms"] = 1000 * sorted(seconds)[len(seconds) // 2]
    out["two_on_one_day"] = con.execute(
        "SELECT COUNT(*) FROM (SELECT m, date_on FROM (SELECT member_a AS m, date_on FROM "
        "bookings UNION ALL SELECT member_b, date_on FROM bookings) GROUP BY m, date_on "
        "HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    return out


def forgetting(cfg: Config, pop, con, market) -> dict[str, float]:
    woman = next(int(market.ids[k]) for k, m in enumerate(market.members) if m.gender == "woman")
    t0 = time.perf_counter()
    with con:
        for sql, params in app.forget_statements(woman, datetime(2026, 11, 4, 16, tzinfo=UTC)):
            con.execute(sql, params)
    out = {"forget_ms": 1000 * (time.perf_counter() - t0)}
    again = app.load_market(con, cfg, app.Calendar(EPOCH), TODAY)
    emb = pop.photo_embeddings.copy()
    emb[again.row[woman]] = 0.0
    t1 = time.perf_counter()
    scores = app.MarketCache(cfg, app.Calendar(EPOCH)).refresh(con, TODAY, emb)
    out["refresh_seconds"] = time.perf_counter() - t1
    out["her_pairs"] = int(scores.eligible[again.row[woman]].sum())
    return out


def main() -> None:
    cfg = load("config/phase1.toml")
    late = cfg.billing.full_refund_hours
    print("1. Times: a Friday 7 pm date, cancelled in UTC without an offset")
    print("   true hours ahead | old code read | read now | old verdict | verdict now")
    for ahead, old, new in times_table(cfg):
        verdict = ("late" if old < late else "early", "late" if new < late else "early")
        print(f"   {ahead:17d} | {old:13.1f} | {new:8.1f} | {verdict[0]:11s} | {verdict[1]}")
    pop, con, market, scores = zone(cfg)
    p = picks_morning(cfg, con, market, scores)
    print(
        f"\n2. Picks: {p['women']:.0f} women each pick the top profile on their list at once\n"
        f"   without the re-check: {p['taken_window']:.0f} picks land on a window already "
        f"taken, {p['taken_day']:.0f} on a day he already has a date\n"
        f"   with the re-check: kept {p['kept']:.0f}, moved {p['moved']:.0f}, refused "
        f"{p['refused']:.0f}; members with two dates on one day: {p['two_on_one_day']:.0f}\n"
        f"   time: day_lists {p['lists_seconds']:.2f} s, recheck_pick about "
        f"{p['recheck_ms']:.0f} ms each"
    )
    f = forgetting(cfg, pop, con, market)
    print(
        f"\n3. Forgetting a member: {f['forget_ms']:.0f} ms; the zone reloads and the nightly "
        f"refresh takes {f['refresh_seconds']:.2f} s; her eligible pairs: {f['her_pairs']}"
    )


if __name__ == "__main__":
    main()
