"""FR-LM-71 to FR-LM-78: the code review findings, fixed."""

import dataclasses
import io
import sqlite3
from datetime import UTC, date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from PIL import Image

from lastfirstdate import app
from lastfirstdate.app_fill import fill_database
from lastfirstdate.booking import N_PARTS, dated_pairs
from lastfirstdate.config import load
from lastfirstdate.encoders import PhotoModel, SentenceEncoder, photo_model_id
from lastfirstdate.filter import member_ok
from lastfirstdate.models import PARTS_OF_DAY
from lastfirstdate.synthetic import generate

TODAY, EPOCH = date(2026, 11, 4), date(2026, 11, 2)


@pytest.fixture(scope="module")
def base():
    """A Phase 1 zone in SQLite (60 subscribers), built once."""
    cfg = load("config/phase1.toml")
    pop = generate(cfg, seed=7)
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    filled = fill_database(con, pop, cfg, TODAY, 7, subscribers=60)
    return SimpleNamespace(cfg=cfg, pop=pop, con=con, filled=filled)


def zone(base):
    """A fresh copy of the base database, its market and nightly scores."""
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
    cols, marks = ", ".join(row), ", ".join("?" * len(row))
    return con.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values())
    ).lastrowid


def booking(con, a: int, b: int, on: str, state: str, part: str = "evening", **extra) -> int:
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


def women(z) -> list[int]:
    return [int(z.market.ids[k]) for k, m in enumerate(z.market.members) if m.gender == "woman"]


class Counting:
    """A connection that counts the rows it hands back and keeps the SQL it ran."""

    def __init__(self, con):
        self.con, self.rows, self.sql = con, 0, []

    def execute(self, sql, *args):
        self.sql.append(sql)
        return _CountingCursor(self.con.execute(sql, *args), self)


class _CountingCursor:
    def __init__(self, cur, owner):
        self.cur, self.owner = cur, owner

    @property
    def description(self):
        return self.cur.description

    def fetchall(self):
        rows = self.cur.fetchall()
        self.owner.rows += len(rows)
        return rows


# ---------------------------------------------------------------- FR-LM-71


def test_a_date_before_a_subscription_doesnt_crash_the_lists(base):
    """FR-LM-71: with a cancelled booking dated before a subscriber's signup, the day's lists
    are built and the rest of his capacity is unchanged; close_cycle for a day before the
    signup is refused clearly."""
    z = zone(base)
    man = base.filled.subscribers[0]
    signup = z.con.execute(
        "SELECT signup_date FROM subscriptions WHERE member_id = ?", (man,)
    ).fetchone()[0]
    before = date.fromisoformat(signup).replace(day=1).isoformat()
    m = z.market.row[man]

    def his_capacity():
        bookings = app.load_bookings(z.con, z.market)
        signups = app.signup_days(z.con, z.market, z.cfg)
        cap = app.capacity(bookings, z.market, signups, z.cfg)
        return cap.busy[m], {k: v for k, v in cap.periods.items() if k[0] == m}

    clean = his_capacity()
    booking(
        z.con,
        women(z)[0],
        man,
        before,
        "cancelled",
        cancelled_by=man,
        cancelled_at=f"{before}T12:00:00Z",
    )
    assert his_capacity() == clean
    assert app.day_lists(z.con, z.market, z.scores, women(z), z.cfg)
    with pytest.raises(app.HandoffError, match="hadn't signed up"):
        app.close_cycle(z.con, z.market, man, date.fromisoformat(before), z.cfg)


# ---------------------------------------------------------------- FR-LM-72


class FakePhotoSession:
    def __init__(self):
        self.seen = []

    def run(self, names, feeds):
        self.seen.append(feeds["pixel_values"])
        return [np.array([[1.0, 0.0]], dtype=np.float32)]


def test_photos_are_turned_upright(tmp_path):
    """FR-LM-72: a photo saved sideways with an EXIF orientation tag gives exactly the pixels
    of the upright photo."""
    upright = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (40, 24, 3), dtype=np.uint8)
    )
    sideways = upright.transpose(Image.Transpose.ROTATE_90)  # how the camera stored it
    exif = Image.Exif()
    exif[0x0112] = 6  # "turn 90 degrees clockwise to show"
    buf = io.BytesIO()
    sideways.save(buf, "PNG", exif=exif.tobytes())
    session = FakePhotoSession()
    model = PhotoModel(session, size=16)
    model.embed(upright)
    model.embed(buf.getvalue())
    model.embed(np.asarray(sideways))  # no tag: left as it is
    np.testing.assert_array_equal(session.seen[1], session.seen[0])
    assert not np.array_equal(session.seen[2], session.seen[0])


# ---------------------------------------------------------------- FR-LM-73


def test_the_cache_notices_a_changed_answer_and_forgets_the_forgotten(base):
    """FR-LM-73: replacing one answer with another refits that member; a forgotten member's
    taste is gone after the next refresh, and the shared fits equal a fresh fit without them
    (D24)."""
    z = zone(base)
    cfg = z.cfg
    cache = app.MarketCache(cfg, app.Calendar(EPOCH))
    emb = base.pop.photo_embeddings.copy()
    cache.refresh(z.con, TODAY, emb)
    member, position = z.con.execute(
        "SELECT member_id, position FROM calibration_answers WHERE shown_b IS NULL LIMIT 1"
    ).fetchone()
    z.con.execute(
        "UPDATE calibration_answers SET answer = 1 - answer WHERE member_id = ? AND position = ?",
        (member, position),
    )
    cache.refresh(z.con, TODAY, emb)
    assert cache.last_refitted == [member]

    woman = next(m for m in women(z) if m in cache.tastes_by_id and m != member)
    with z.con:
        for sql, params in app.forget_statements(woman, datetime(2026, 11, 4, 16, tzinfo=UTC)):
            z.con.execute(sql, params)
    emb[z.market.row[woman]] = 0.0
    cache.refresh(z.con, TODAY, emb)
    assert woman not in cache.tastes_by_id
    market = app.load_market(z.con, cfg, app.Calendar(EPOCH), TODAY)
    fresh_pca = app._fit_pca(emb, cfg)
    np.testing.assert_allclose(cache.pca.components, fresh_pca.components)
    feats = app.with_intercept(fresh_pca.transform(emb))
    fresh = app.starting_taste(market, feats, app.calibration_answers(z.con, market, cfg), cfg)
    for group, w in fresh.by_group.items():
        np.testing.assert_allclose(cache.start.by_group[group], w)


# ---------------------------------------------------------------- FR-LM-74


def test_times_are_written_with_their_time_zone(base):
    """FR-LM-74: a time without a zone is refused; a New York time is stored as the same
    instant in UTC (D25)."""
    z = zone(base)
    lists = app.day_lists(z.con, z.market, z.scores, women(z), z.cfg)
    woman, entries = next(iter(lists.items()))
    bk, venue = app.propose(z.con, z.market, woman, entries[0].member_id, entries[0])
    naive = datetime(2026, 11, 4, 14, 5)
    with pytest.raises(app.HandoffError, match="time zone"):
        app.booking_row(bk, z.market, venue, naive)
    with pytest.raises(app.HandoffError, match="time zone"):
        app.forget_statements(woman, naive)
    ny = datetime(2026, 11, 4, 14, 5, tzinfo=ZoneInfo("America/New_York"))
    assert app.booking_row(bk, z.market, venue, ny)["proposed_at"] == "2026-11-04T19:05:00+00:00"
    assert app.utc_text(datetime(2026, 11, 4, 19, 5, tzinfo=UTC)) == "2026-11-04T19:05:00+00:00"


# ---------------------------------------------------------------- FR-LM-75


def _window(z, entry) -> int:
    return (z.market.calendar.day(entry.day) - z.market.today) * N_PARTS + PARTS_OF_DAY.index(
        entry.part
    )


def test_a_pending_pick_holds_his_day_in_the_lists(base):
    """FR-LM-75: with a pick waiting for his answer, no list offers him that day; his other
    days are still free; when he accepts, his own pending booking doesn't block itself."""
    z = zone(base)
    lists = app.day_lists(z.con, z.market, z.scores, women(z), z.cfg)
    woman, entries = next(iter(lists.items()))
    entry = entries[0]
    man = entry.member_id
    m = z.market.row[man]

    def his_days():
        bookings = app.load_bookings(z.con, z.market)
        signups = app.signup_days(z.con, z.market, z.cfg)
        cap = app.capacity(bookings, z.market, signups, z.cfg)
        bits = int(app.free_windows(z.market, cap, signups, z.cfg, members=[m])[0])
        return {k // N_PARTS for k in range(64) if (bits >> k) & 1}

    free_before = his_days()
    pending = booking(z.con, woman, man, entry.day.isoformat(), "proposed", part=entry.part)
    held_day = z.market.calendar.day(entry.day) - z.market.today
    assert held_day not in his_days()
    assert his_days() == free_before - {held_day}
    others = [w for w in women(z) if w != woman]
    again = app.day_lists(z.con, z.market, z.scores, others, z.cfg)
    assert not any(e.member_id == man and e.day == entry.day for es in again.values() for e in es)
    accepted = app.recheck_pick(z.con, z.market, woman, man, entry, z.cfg, booking_id=pending)
    assert accepted == entry


# ---------------------------------------------------------------- FR-LM-76


def reference_recheck(con, market, chooser, chosen, entry, cfg, booking_id=None):
    """recheck_pick as it was: every row and the members-by-members block grid."""
    a, b = market.row[chooser], market.row[chosen]
    ok = member_ok(market.arr, cfg) & ~market.paused
    if not (ok[a] and ok[b]) or app.blocked_pairs(con, market)[a, b]:
        return None
    every = app.load_bookings(con, market)
    keep = [k for k, bid in enumerate(every.ids) if bid != booking_id]
    others = app.Bookings([every.items[k] for k in keep], [every.ids[k] for k in keep])
    if (min(a, b), max(a, b)) in dated_pairs(others.items):
        return None
    signups = app.signup_days(con, market, cfg)
    cap = app.capacity(others, market, signups, cfg)
    bits = app.free_windows(market, cap, signups, cfg, members=[a, b])
    common = int(bits[0]) & int(bits[1])
    k = (market.calendar.day(entry.day) - market.today) * N_PARTS + PARTS_OF_DAY.index(entry.part)
    if 0 <= k < 64 and (common >> k) & 1:
        return entry
    w = int(app.earliest_common(bits[0], bits[1]))
    if w < 0:
        return None
    day, part = app.window_day_part(w, market.today)
    return app.ListEntry(
        chosen, market.calendar.date(day), PARTS_OF_DAY[part], entry.position, entry.match_value
    )


def test_checking_a_pick_reads_only_the_two_members(base):
    """FR-LM-76: recheck_pick gives the same answers as the full-table version on many picks,
    and the rows it reads don't grow as other members' bookings are added."""
    z = zone(base)
    lists = app.day_lists(z.con, z.market, z.scores, women(z), z.cfg)
    picks = [(w, es[0]) for w, es in lists.items()]
    for w, e in picks[:15]:  # some picks taken, so later ones are moved or refused
        booking(z.con, w, e.member_id, e.day.isoformat(), "held", part=e.part)
    w0, e0 = picks[20]
    insert(
        z.con,
        "blocks",
        {"blocker_id": e0.member_id, "blocked_id": w0, "blocked_at": "2026-11-04T10:00:00Z"},
    )
    outcomes = set()
    for w, e in picks:
        got = app.recheck_pick(z.con, z.market, w, e.member_id, e, z.cfg)
        assert got == reference_recheck(z.con, z.market, w, e.member_id, e, z.cfg)
        outcomes.add("kept" if got == e else "refused" if got is None else "moved")
    assert {"kept", "refused"} <= outcomes

    w, e = picks[-1]
    counting = Counting(z.con)
    app.recheck_pick(counting, z.market, w, e.member_id, e, z.cfg)
    rows_before = counting.rows
    everyone = women(z)
    for k in range(400):
        a = everyone[k % len(everyone)]
        b = base.filled.subscribers[(k * 7) % len(base.filled.subscribers)]
        if e.member_id not in (a, b) and w not in (a, b):
            booking(z.con, a, b, f"2026-10-{1 + k % 28:02d}", "completed", showed_a=1, showed_b=1)
    counting = Counting(z.con)
    app.recheck_pick(counting, z.market, w, e.member_id, e, z.cfg)
    assert counting.rows == rows_before


# ---------------------------------------------------------------- FR-LM-77


def test_the_model_check_reads_no_embeddings(base):
    """FR-LM-77: check_photo_models passes and fails as before, and reads no embedding."""
    z = zone(base)
    want = photo_model_id(z.cfg.models)
    z.con.execute(
        "INSERT INTO photo_embeddings VALUES (1000, ?, X'00', '2026-11-04T10:00:00Z')", (want,)
    )
    counting = Counting(z.con)
    app.check_photo_models(counting, z.cfg)
    assert counting.sql and all("encrypted_vector" not in q and "*" not in q for q in counting.sql)
    z.con.execute(
        "INSERT INTO photo_embeddings VALUES (1001, 'openai/clip-vit-base-patch32', X'00', "
        "'2026-11-04T10:00:00Z')"
    )
    with pytest.raises(app.HandoffError, match="openai/clip"):
        app.check_photo_models(Counting(z.con), z.cfg)


# ---------------------------------------------------------------- FR-LM-78


class FakeTokenizer:
    def enable_truncation(self, max_length):
        self.max_length = max_length

    def enable_padding(self):
        pass

    def encode_batch(self, texts):
        ids = [[1 + (sum(map(ord, w)) % 97) for w in t.split()] for t in texts]
        width = max(map(len, ids))
        return [
            SimpleNamespace(
                ids=i + [0] * (width - len(i)),
                attention_mask=[1] * len(i) + [0] * (width - len(i)),
                type_ids=[0] * width,
            )
            for i in ids
        ]


class FakeTextSession:
    def __init__(self):
        self.table = np.random.default_rng(0).normal(size=(100, 8)).astype(np.float32)
        self.texts_run = 0

    def run(self, names, feeds):
        self.texts_run += len(feeds["input_ids"])
        return [self.table[feeds["input_ids"]]]


def test_the_text_model_forgets_old_prompts():
    """FR-LM-78: after a call without an old text, that text is no longer held; a text in both
    calls isn't run through the model again."""
    enc = SentenceEncoder(FakeTextSession(), FakeTokenizer(), max_tokens=16)
    first = enc.embed_texts(["hiking and jazz", "opera and chess"])
    second = enc.embed_texts(["opera and chess", "baking", ""])
    assert set(enc.seen) == {"opera and chess", "baking"}
    assert enc.session.texts_run == 3  # the shared text was run once
    np.testing.assert_array_equal(second[0], first[1])
    assert second.shape == (3, 8) and not second[2].any()
    assert enc.embed_texts(["", ""]).shape == (2, 8)


# ---------------------------------------------------------------- FR-LM-79


@pytest.fixture(scope="module")
def mixed():
    """A Phase 1 zone in SQLite where 7% of members are gay and 3% bisexual, with its scores."""
    cfg = load("config/phase1.toml")
    population = dataclasses.replace(
        cfg.population, orientation={"straight": 0.90, "gay": 0.07, "bi": 0.03}
    )
    cfg = dataclasses.replace(cfg, population=population)
    pop = generate(cfg, seed=7)
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    filled = fill_database(con, pop, cfg, TODAY, 7, subscribers=60)
    return zone(SimpleNamespace(cfg=cfg, pop=pop, con=con, filled=filled)), filled


def test_the_gate_counts_only_women_who_can_date_the_men(mixed):
    """FR-LM-79: no woman who seeks only women is counted, even with lists on some days; the
    sale equals the gate on women who seek men and subscribers who seek women; applicants who
    seek only men are deferred."""
    z, filled = mixed
    m, cfg = z.market, z.cfg
    is_woman = m.arr.gender == app.WOMAN
    only_women = is_woman & ~app.seeks(m, "man")
    assert only_women.sum() >= 5  # the zone has some
    fake = int(m.ids[np.flatnonzero(only_women)[0]])
    for k in range(3):  # a few list days, as a rarely-shown fake would have
        insert(
            z.con,
            "list_shows",
            {
                "chooser_id": fake,
                "candidate_id": int(m.ids[np.flatnonzero(only_women)[1]]),
                "shown_on": f"2026-10-{20 + 7 * k}",
                "picked": 0,
            },
        )
    counted = app.counted_women(z.con, m, cfg)
    assert counted.any() and not (counted & only_women).any()
    assert (counted == (is_woman & app.seeks(m, "man") & counted)).all()

    sale = app.gate_for_sale(z.con, m, z.scores, cfg)
    ok = member_ok(m.arr, cfg) & ~m.paused
    signups = app.signup_days(z.con, m, cfg)
    current = np.flatnonzero(ok & (signups != app.NO_SIGNUP) & app.seeks(m, "woman"))
    applicants = m.rows(filled.waitlist)
    d = app.gate_decision(
        app.gate_model(cfg), z.scores.eligible, np.flatnonzero(counted), current, applicants, cfg
    )
    assert sale.admit == [int(m.ids[r]) for r in d.admit]
    assert sale.deferred_small_pool == [int(m.ids[r]) for r in d.deferred_small_pool]
    only_men = [r for r in applicants if not app.seeks(m, "woman")[r]]
    assert only_men and not set(m.ids[only_men].tolist()) & set(sale.admit)
    first = app.gate_decision(  # reached first in the queue, they are deferred, never sold
        app.gate_model(cfg), z.scores.eligible, np.flatnonzero(counted), current, only_men, cfg
    )
    assert first.admit == [] and first.deferred_small_pool == only_men
