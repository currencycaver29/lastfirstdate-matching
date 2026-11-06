"""FR-LM-54 to FR-LM-58: bug fixes before Phase 1 (times, picks, blocks, forgetting, list logs)."""

import sqlite3
from datetime import UTC, date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import lastfirstdate
from lastfirstdate import app
from lastfirstdate.app_fill import fill_database
from lastfirstdate.config import ConfigError, from_dict, load
from lastfirstdate.filter import eligibility_matrix
from lastfirstdate.synthetic import generate
from lastfirstdate.trust import with_trust

TODAY, EPOCH = date(2026, 11, 4), date(2026, 11, 2)  # a Wednesday, and the Monday before
NY = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def base():
    """A Phase 1 zone in SQLite (60 subscribers), and its nightly scores, built once."""
    cfg = load("config/phase1.toml")
    pop = generate(cfg, seed=7)
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    filled = fill_database(con, pop, cfg, TODAY, 7, subscribers=60)
    return SimpleNamespace(cfg=cfg, pop=pop, con=con, filled=filled)


def zone(base):
    """A fresh copy of the base database, loaded as the app would each night."""
    con = sqlite3.connect(":memory:")
    base.con.backup(con)
    con.execute("PRAGMA foreign_keys = ON")
    market = app.load_market(con, base.cfg, app.Calendar(EPOCH), TODAY)
    feats = app.photo_feature_matrix(base.pop.photo_embeddings, base.cfg)
    answers = app.calibration_answers(con, market, base.cfg)
    tastes = app.fit_tastes(market, feats, answers, base.cfg, start=base.filled.start)
    scores = app.nightly_scores(market, tastes, feats, base.cfg)
    return SimpleNamespace(con=con, market=market, scores=scores, cfg=base.cfg)


def insert(con, table: str, row: dict) -> int:
    cols = ", ".join(row)
    marks = ", ".join("?" * len(row))
    return con.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values())
    ).lastrowid


def booking(con, a: int, b: int, on: str, part: str, state: str, **extra) -> int:
    row = {
        "member_a": a,
        "member_b": b,
        "source": "list",
        "date_on": on,
        "part": part,
        "state": state,
        "proposed_at": "2026-11-03T23:00:00+00:00",
        "venue_id": 1 if state in ("confirmed", "completed", "no_show") else None,
    }
    row.update(extra)
    return insert(con, "bookings", row)


def a_pair(z):
    """A woman and a subscribed man who appear on each other's lists today."""
    lists = app.day_lists(z.con, z.market, z.scores, women(z), z.cfg)
    woman, entries = next(iter(lists.items()))
    return woman, entries[0]


def women(z) -> list[int]:
    return [int(z.market.ids[k]) for k, m in enumerate(z.market.members) if m.gender == "woman"]


# ---------------------------------------------------------------- FR-LM-54 times


def test_timestamps_are_read_in_the_zones_own_time():
    """FR-LM-54: with an offset a time is converted, with none it is UTC, a bare date is local;
    true hours hold across the November clock change."""
    for text in ("2026-11-05T22:00:00Z", "2026-11-05T22:00:00+00:00", "2026-11-05T22:00:00"):
        assert app.local_time(text, NY) == datetime(2026, 11, 5, 17, 0, tzinfo=NY)
    assert app.local_date("2026-11-05", NY) == date(2026, 11, 5)
    assert app.local_date("2026-11-05T03:00:00Z", NY) == date(2026, 11, 4)  # 10 pm in New York
    # 1 Nov 2026: clocks go back at 2 am. 7:30 pm on 31 Oct to 7 pm on 1 Nov is 24.5 hours.
    start = datetime(2026, 11, 1, 19, tzinfo=NY)
    assert app.hours_between(start, datetime(2026, 10, 31, 19, 30, tzinfo=NY)) == 24.5


def test_late_cancellations_use_true_hours(base):
    """FR-LM-54: a cancellation 26 hours ahead (stored in UTC, any spelling) is early, 23
    hours ahead is late, and one 24.5 true hours ahead across the clock change is early."""
    z = zone(base)
    woman, man = women(z)[0], base.filled.subscribers[0]
    late = z.cfg.billing.full_refund_hours
    cases = {  # cancelled_at: true hours before a Friday 6 November, 7 pm date
        "2026-11-05T22:00:00Z": 26.0,
        "2026-11-05T22:00:00+00:00": 26.0,
        "2026-11-05T22:00:00": 26.0,
        "2026-11-06T01:00:00Z": 23.0,
    }
    ids = {}
    for stamp in cases:
        ids[stamp] = booking(
            z.con,
            woman,
            man,
            "2026-11-06",
            "evening",
            "cancelled",
            cancelled_by=man,
            cancelled_at=stamp,
        )
    # Sunday 1 November, 7 pm (EST), cancelled at 7:30 pm the evening before (EDT)
    ids["dst"] = booking(
        z.con,
        woman,
        man,
        "2026-11-01",
        "evening",
        "cancelled",
        cancelled_by=man,
        cancelled_at="2026-10-31T23:30:00Z",
    )
    cases["dst"] = 24.5
    got = app.load_bookings(z.con, z.market)
    hours = dict(zip(got.ids, (bk.hours_before for bk in got.items), strict=True))
    for key, expected in cases.items():
        assert hours[ids[key]] == pytest.approx(expected), key
        assert (hours[ids[key]] < late) == (expected < late), key


def test_last_active_and_join_dates_use_new_yorks_date(base):
    """FR-LM-54: 3 am UTC on 4 November is still 3 November in New York."""
    z = zone(base)
    member = int(z.market.ids[0])
    z.con.execute(
        "UPDATE members SET last_active_at = '2026-11-04T03:00:00Z' WHERE id = ?", (member,)
    )
    market = app.load_market(z.con, z.cfg, app.Calendar(EPOCH), TODAY)
    assert market.members[0].last_active_days == 1


def test_part_starts_and_timezone_are_settings():
    """FR-LM-54: the zone's clock and the hours each part of the day begins live in the
    settings (R5), and nonsense is refused."""
    import tomllib

    with open("config/phase1.toml", "rb") as fh:
        raw = tomllib.load(fh)
    cfg = from_dict(raw)
    assert (cfg.booking.timezone, list(cfg.booking.part_starts)) == (
        "America/New_York",
        [12, 19, 22],
    )
    for key, bad in (("timezone", "Mars/Olympus"), ("part_starts", [19, 12, 22])):
        broken = {**raw, "booking": {**raw["booking"], key: bad}}
        with pytest.raises(ConfigError):
            from_dict(broken)


# ---------------------------------------------------------------- FR-LM-55 the pick, checked again


def test_a_taken_window_is_moved_or_refused(base):
    """FR-LM-55: once one woman's pick holds a man's evening, a second woman shown the same
    window is moved to another window or refused, never booked on top of it."""
    z = zone(base)
    lists = app.day_lists(z.con, z.market, z.scores, women(z), z.cfg)
    shown: dict[tuple, list[int]] = {}
    for woman, entries in lists.items():
        for e in entries:
            shown.setdefault((e.member_id, e.day, e.part), []).append(woman)
    (man, day, part), (first, second, *_) = next((k, v) for k, v in shown.items() if len(v) > 1)
    entry = next(e for e in lists[first] if e.member_id == man)
    assert app.recheck_pick(z.con, z.market, first, man, entry, z.cfg) == entry
    bk, venue = app.propose(z.con, z.market, first, man, entry)
    held = insert(z.con, "bookings", app.booking_row(bk, z.market, venue, datetime.now(UTC)))
    other = next(e for e in lists[second] if e.member_id == man)
    again = app.recheck_pick(z.con, z.market, second, man, other, z.cfg)
    assert again is None or again.day != day  # a pending pick holds the whole day
    # the booking being accepted doesn't block itself
    assert app.recheck_pick(z.con, z.market, first, man, entry, z.cfg, booking_id=held) == entry
    # once it is held, a second check for the same pair is refused: they already have a date
    z.con.execute("UPDATE bookings SET state = 'held' WHERE id = ?", (held,))
    assert app.recheck_pick(z.con, z.market, first, man, entry, z.cfg) is None


def test_a_man_at_his_cap_is_refused(base):
    """FR-LM-55: with 3 dates already in his current cycle, no window is offered."""
    z = zone(base)
    woman, entry = a_pair(z)
    man = entry.member_id
    others = [w for w in women(z) if w != woman][:3]
    for k, w in enumerate(others):  # three completed dates earlier in his current cycle
        booking(z.con, w, man, f"2026-10-2{7 + k}", "evening", "completed", showed_a=1, showed_b=1)
    assert app.recheck_pick(z.con, z.market, woman, man, entry, z.cfg) is None


# ---------------------------------------------------------------- FR-LM-56 blocks


def test_a_blocked_pair_never_meets(base):
    """FR-LM-56: after a block, the pair is off each other's list at once (even with last
    night's scores), out of each other's photo pool, unbookable, and ineligible next night;
    upcoming bookings between them are listed for cancelling."""
    z = zone(base)
    woman, entry = a_pair(z)
    man = entry.member_id
    i, j = z.market.row[woman], z.market.row[man]
    pending = booking(z.con, woman, man, entry.day.isoformat(), entry.part, "proposed")
    assert j in app.photo_pool(z.market, z.cfg, i) and i in app.photo_pool(z.market, z.cfg, j)
    insert(
        z.con,
        "blocks",
        {"blocker_id": man, "blocked_id": woman, "blocked_at": "2026-11-04T15:00:00Z"},
    )
    assert app.upcoming_between(z.con, z.market, woman, man) == [pending]
    again = app.day_lists(z.con, z.market, z.scores, [woman], z.cfg)
    assert man not in {e.member_id for e in again.get(woman, [])}
    assert app.recheck_pick(z.con, z.market, woman, man, entry, z.cfg) is None
    market = app.load_market(z.con, z.cfg, app.Calendar(EPOCH), TODAY)
    assert market.blocked[i, j] and market.blocked[j, i] and market.blocked.sum() == 2
    assert j not in app.photo_pool(market, z.cfg, i) and i not in app.photo_pool(market, z.cfg, j)
    feats = app.photo_feature_matrix(base.pop.photo_embeddings, z.cfg)
    scores = app.nightly_scores(market, {}, feats, z.cfg)
    assert not scores.eligible[i, j] and not scores.eligible[j, i]


def test_a_block_refreshes_the_cache_and_nothing_else_moves(base):
    """FR-LM-56: the nightly cache doesn't return stale scores after a block; with no blocks,
    eligibility is exactly as before."""
    z = zone(base)
    np.testing.assert_array_equal(z.scores.eligible, eligibility_matrix(z.market.arr, z.cfg))
    cache = app.MarketCache(z.cfg, app.Calendar(EPOCH))
    first = cache.refresh(z.con, TODAY, base.pop.photo_embeddings)
    woman, entry = a_pair(z)
    i, j = z.market.row[woman], z.market.row[entry.member_id]
    assert first.eligible[i, j]
    insert(
        z.con,
        "blocks",
        {"blocker_id": woman, "blocked_id": entry.member_id, "blocked_at": "2026-11-04T15:00Z"},
    )
    second = cache.refresh(z.con, TODAY, base.pop.photo_embeddings)
    assert second is not first and not second.eligible[i, j]


# ---------------------------------------------------------------- FR-LM-57 forgetting a member


def test_forgetting_a_member_keeps_the_records_and_drops_the_profile(base):
    """FR-LM-57: a woman with history is forgotten under D18; her profile and every answer
    about her photo go, her birth year, bookings and payments stay, and the zone still runs
    without her."""
    z = zone(base)
    woman, entry = a_pair(z)
    man = entry.member_id
    insert(z.con, "photos", {"member_id": woman, "storage_key": "k1", "position": 0})
    insert(
        z.con,
        "photo_embeddings",
        {"member_id": woman, "model": "test", "encrypted_vector": b"x", "created_at": "2026"},
    )
    insert(z.con, "list_shows", app.list_show_row(woman, entry, TODAY, picked=True))
    done = booking(z.con, woman, man, "2026-10-30", "evening", "completed", showed_a=1, showed_b=1)
    insert(
        z.con,
        "payments",
        {
            "booking_id": done,
            "member_id": woman,
            "amount_cents": 1000,
            "paid_at": "2026-10-29T22:00:00Z",
        },
    )
    about_her = z.con.execute(
        "SELECT COUNT(*) FROM calibration_answers WHERE shown_a = ? OR shown_b = ?",
        (woman, woman),
    ).fetchone()[0]
    assert about_her > 0
    birth = z.con.execute("SELECT birth_date FROM members WHERE id = ?", (woman,)).fetchone()[0]

    with z.con:  # one transaction
        for sql, params in app.forget_statements(woman, datetime(2026, 11, 4, 16, tzinfo=UTC)):
            z.con.execute(sql, params)

    def count(sql: str) -> int:
        return z.con.execute(sql, (woman,)).fetchone()[0]

    for table in ("photos", "photo_embeddings", "prompts", "intent_answers", "free_slots"):
        assert count(f"SELECT COUNT(*) FROM {table} WHERE member_id = ?") == 0, table
    assert count("SELECT COUNT(*) FROM calibration_answers WHERE member_id = ?") == 0
    assert (
        z.con.execute(
            "SELECT COUNT(*) FROM calibration_answers WHERE shown_a = ? OR shown_b = ?",
            (woman, woman),
        ).fetchone()[0]
        == 0
    )
    kept = z.con.execute(
        "SELECT birth_date, forgotten_at, verified FROM members WHERE id = ?", (woman,)
    ).fetchone()
    assert kept == (birth[:4] + "-01-01", "2026-11-04T16:00:00+00:00", 0)
    assert z.con.execute("SELECT COUNT(*) FROM bookings WHERE id = ?", (done,)).fetchone()[0] == 1
    assert count("SELECT COUNT(*) FROM payments WHERE member_id = ?") == 1
    assert count("SELECT COUNT(*) FROM list_shows WHERE chooser_id = ?") == 1

    # the zone still loads and runs; she never matches, lists or appears in a photo exercise
    market = app.load_market(z.con, z.cfg, app.Calendar(EPOCH), TODAY)
    r = market.row[woman]
    assert market.forgotten[r] and market.forgotten.sum() == 1
    emb = base.pop.photo_embeddings.copy()
    emb[r] = 0.0  # the app passes a zero embedding once hers is deleted
    cache = app.MarketCache(z.cfg, app.Calendar(EPOCH))
    scores = cache.refresh(z.con, TODAY, emb)
    assert not scores.eligible[r].any() and not scores.eligible[:, r].any()
    assert all(r not in app.photo_pool(market, z.cfg, v) for v in range(len(market.members)))
    lists = app.day_lists(z.con, market, scores, women(z), z.cfg)
    assert woman not in lists
    assert all(woman != e.member_id for entries in lists.values() for e in entries)
    assert app.load_bookings(z.con, market).ids  # past bookings still resolve


def test_forgetting_a_subscriber_ends_his_subscription(base):
    """FR-LM-57: a forgotten man's subscription is cancelled, so he is out of alerts and the
    gate, which still run."""
    z = zone(base)
    man = base.filled.subscribers[0]
    with z.con:
        for sql, params in app.forget_statements(man, datetime(2026, 11, 4, 16)):
            z.con.execute(sql, params)
    status = z.con.execute("SELECT status FROM subscriptions WHERE member_id = ?", (man,))
    assert status.fetchone()[0] == "cancelled"
    market = app.load_market(z.con, z.cfg, app.Calendar(EPOCH), TODAY)
    emb = base.pop.photo_embeddings.copy()
    emb[market.row[man]] = 0.0
    scores = app.MarketCache(z.cfg, app.Calendar(EPOCH)).refresh(z.con, TODAY, emb)
    _, chances = app.weekly_alerts(z.con, market, scores, z.cfg)
    assert man not in chances
    app.gate_for_sale(z.con, market, scores, z.cfg)


# ---------------------------------------------------------------- FR-LM-58 list logs


def test_each_entry_records_its_position_and_match_value(base):
    """FR-LM-58: positions run 1, 2, 3 in list order, each match value is what ordered the list
    (mutual score times trust), and the list_shows row fits the schema with the engine's
    version (and passes the schema's R2 check on column names)."""
    z = zone(base)
    lists = app.day_lists(z.con, z.market, z.scores, women(z), z.cfg)
    bookings = app.load_bookings(z.con, z.market)
    trust = app.trust_from_db(z.con, z.market, bookings, z.cfg)
    ranked = with_trust(z.scores.mutual, trust.ranking_factor())
    for woman, entries in lists.items():
        assert [e.position for e in entries] == list(range(1, len(entries) + 1))
        i = z.market.row[woman]
        for e in entries:
            assert e.match_value == pytest.approx(float(ranked[i, z.market.row[e.member_id]]))
    woman, entries = next(iter(lists.items()))
    for e in entries:
        insert(z.con, "list_shows", app.list_show_row(woman, e, TODAY))
    rows = z.con.execute(
        "SELECT position, match_value, engine_version FROM list_shows WHERE chooser_id = ? "
        "ORDER BY position",
        (woman,),
    ).fetchall()
    assert [r[0] for r in rows] == [e.position for e in entries]
    assert {r[2] for r in rows} == {lastfirstdate.__version__}
