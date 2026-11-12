"""FR-LM-63 to FR-LM-70: abuse testing (synthetic attackers and the engine's defences)."""

import dataclasses
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lastfirstdate import app, attacks
from lastfirstdate.app_fill import fill_database
from lastfirstdate.config import load
from lastfirstdate.cycle_runs import join_days, market
from lastfirstdate.mutual_report import build_scores
from lastfirstdate.personality import WordEncoder, interest_similarity
from lastfirstdate.synthetic import generate
from lastfirstdate.trust import Rating, TrustBook

TODAY, EPOCH = date(2026, 11, 4), date(2026, 11, 2)
SCHEMA = (Path(__file__).resolve().parents[1] / "db" / "schema.sql").read_text()


@pytest.fixture(scope="module")
def p1():
    return load("config/phase1.toml")


@pytest.fixture(scope="module")
def small_zone(p1):
    """A small Phase 1 zone for the attackers: 150 women, 50 men, a seed nothing was tuned on."""
    c = attacks.with_market(p1, 150, 50)
    rate = c.cycle_runs.new_women_per_cycle
    pop = market(c, 2031, rate)
    join = join_days(pop, c, rate)
    women = attacks.launch_women(pop, c, join)
    return SimpleNamespace(cfg=c, pop=pop, join=join, seed=2031, women=women)


@pytest.fixture(scope="module")
def base(p1):
    """A Phase 1 zone in SQLite (60 subscribers, the other men waiting), built once."""
    pop = generate(p1, seed=7)
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    filled = fill_database(con, pop, p1, TODAY, 7, subscribers=60)
    return SimpleNamespace(cfg=p1, pop=pop, con=con, filled=filled)


def zone(base, cfg=None):
    """A fresh copy of the base database, loaded as the app would each night."""
    cfg = cfg or base.cfg
    con = sqlite3.connect(":memory:")
    base.con.backup(con)
    con.execute("PRAGMA foreign_keys = ON")
    m = app.load_market(con, cfg, app.Calendar(EPOCH), TODAY)
    return SimpleNamespace(con=con, market=m, cfg=cfg)


def reload(z):
    z.market = app.load_market(z.con, z.cfg, app.Calendar(EPOCH), TODAY)
    return z


def scores_of(z, base):
    feats = app.photo_feature_matrix(base.pop.photo_embeddings, z.cfg)
    answers = app.calibration_answers(z.con, z.market, z.cfg)
    tastes = app.fit_tastes(z.market, feats, answers, z.cfg, start=base.filled.start)
    return app.nightly_scores(z.market, tastes, feats, z.cfg)


def insert(con, table: str, row: dict) -> int:
    cols, marks = ", ".join(row), ", ".join("?" * len(row))
    return con.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values())
    ).lastrowid


def booking(con, a: int, b: int, state: str, on: str = "2026-10-30", **extra) -> int:
    row = {
        "member_a": a,
        "member_b": b,
        "source": "list",
        "date_on": on,
        "part": "evening",
        "state": state,
        "proposed_at": "2026-10-28T23:00:00+00:00",
        "venue_id": 1 if state in ("confirmed", "completed", "no_show") else None,
    }
    row.update(extra)
    return insert(con, "bookings", row)


def ids_of(z, gender: str) -> list[int]:
    return [int(z.market.ids[k]) for k, m in enumerate(z.market.members) if m.gender == gender]


# ---------------------------------------------------------------- FR-LM-63


def test_attackers_gain_without_the_defences_and_lose_it_with_them(small_zone):
    """FR-LM-63: the stuffed prompt and the answer flood each gain without the engine's
    defence and lose that gain with it."""
    z = small_zone
    attackers = np.random.default_rng(1).choice(attacks.men_of(z.pop), 5, replace=False)
    text = attacks.stuffed_texts(z.pop, z.women)["everyone's words"]
    stuffed = attacks.with_prompts(z.pop, attackers, text)
    for on, rises in ((False, True), (True, False)):
        c = attacks.centered(z.cfg, on)
        gain = attacks.personality_rank(stuffed, c, attackers) - attacks.personality_rank(
            z.pop, c, attackers
        )
        assert (gain > 0.1) if rises else (gain < 0.03)

    target = attacks.popular_target(z.pop, z.cfg, z.seed)

    def rank(attack):
        return attacks.taste_attack(z.pop, z.cfg, z.seed, target, attack)[0]

    clean = rank(attacks.TasteAttack(0, 1))
    assert rank(attacks.TasteAttack(1, 20, one_exercise=False)) > clean + 0.02
    assert rank(attacks.TasteAttack(1, 20)) < clean + 0.01


# ---------------------------------------------------------------- FR-LM-64


def test_centring_takes_out_what_everyone_writes(p1):
    """FR-LM-64: words in everyone's prompt count for nothing once centred; a shared interest
    still counts; an empty prompt scores 0; off, the similarity is exactly as before."""
    base = generate(p1, seed=3, n=10).members[0]
    others = (
        "chess and poetry",
        "baking and tennis",
        "opera and sailing",
        "yoga and karaoke",
        "surfing and wine",
        "gardening and techno",
    )
    texts = [
        "A perfect Sunday looks like hiking and jazz.",
        "A perfect Sunday looks like hiking and opera.",
        *(f"A perfect Sunday looks like {t}." for t in others),
        "",
    ]
    people = [dataclasses.replace(base, id=k, bio=t) for k, t in enumerate(texts)]
    x = WordEncoder().encode(people)
    plain, centred = interest_similarity(x), interest_similarity(x, center=True)
    assert plain[2, 3] > 0.2 and centred[2, 3] == 0.0  # only the template in common
    assert centred[0, 1] > 0.2  # both hike
    empty = len(texts) - 1
    assert not centred[empty].any() and not centred[:, empty].any()
    np.testing.assert_array_equal(plain, np.clip(x @ x.T, 0.0, 1.0))
    assert load().personality.center_text is False and p1.personality.center_text is True


def test_a_stuffed_prompt_gains_nothing_once_centred(small_zone):
    """FR-LM-64: five men with stuffed prompts climb in women's personality percentiles
    without centring and gain at most 0.03 with it, whichever stuffing they use."""
    z = small_zone
    attackers = np.random.default_rng(2).choice(attacks.men_of(z.pop), 5, replace=False)
    gains: dict[bool, list[float]] = {False: [], True: []}
    for text in attacks.stuffed_texts(z.pop, z.women).values():
        stuffed = attacks.with_prompts(z.pop, attackers, text)
        for on in (False, True):
            c = attacks.centered(z.cfg, on)
            gains[on].append(
                attacks.personality_rank(stuffed, c, attackers)
                - attacks.personality_rank(z.pop, c, attackers)
            )
    assert max(gains[False]) > 0.1
    assert max(gains[True]) < 0.03


# ---------------------------------------------------------------- FR-LM-65


def test_one_account_counts_as_one_exercise(base):
    """FR-LM-65: rows past the exercise, repeats and answers about oneself change nothing;
    a normal exercise is read exactly as before."""
    z = zone(base)
    rows = z.con.execute("SELECT * FROM calibration_answers ORDER BY member_id, position")
    raw = {}
    for mid, _, a, b, ans, _ in rows.fetchall():
        raw.setdefault(z.market.row[mid], []).append(
            (z.market.row[a], None if b is None else z.market.row[b], bool(ans))
        )
    before = app.calibration_answers(z.con, z.market, z.cfg)
    assert before == raw and app.check_rows(z.con, z.market, z.cfg) == []

    def photos(v):
        return {p for a, b, _ in before[v] for p in (a, b) if p is not None}

    full = next(v for v in before if len(photos(v)) == z.cfg.calibration.n_profiles)
    member = int(z.market.ids[full])
    nxt = len(before[full])
    first = int(z.market.ids[before[full][0][0]])
    others = [
        int(z.market.ids[j])
        for j in range(len(z.market.members))
        if j not in photos(full) and j != full
    ]
    flood = [(member, nxt, member, None, 1), (member, nxt + 1, first, None, 1)]
    flood += [(member, nxt + 2 + k, j, None, 1) for k, j in enumerate(others[:300])]
    z.con.executemany(
        "INSERT INTO calibration_answers VALUES (?, ?, ?, ?, ?, '2026-11-01T12:00:00Z')", flood
    )
    assert app.calibration_answers(z.con, z.market, z.cfg) == before
    problems = app.check_rows(z.con, z.market, z.cfg)
    assert len(problems) == len(flood)
    assert "own photo" in problems[0] and "repeats" in problems[1] and "end" in problems[2]


def test_only_verified_members_teach_the_starting_taste(base):
    """FR-LM-65: an unverified member's answers don't change the starting taste."""
    z = zone(base)
    feats = app.photo_feature_matrix(base.pop.photo_embeddings, z.cfg)
    answers = app.calibration_answers(z.con, z.market, z.cfg)
    woman = next(v for v in answers if z.market.members[v].gender == "woman")
    group = (int(z.market.arr.gender[woman]), int(z.market.arr.gender[answers[woman][0][0]]))
    everyone = app.starting_taste(z.market, feats, answers, z.cfg)
    without = app.starting_taste(
        z.market, feats, {v: r for v, r in answers.items() if v != woman}, z.cfg
    )
    z.con.execute("UPDATE members SET verified = 0 WHERE id = ?", (int(z.market.ids[woman]),))
    reload(z)
    unverified = app.starting_taste(z.market, feats, answers, z.cfg)
    np.testing.assert_allclose(unverified.by_group[group], without.by_group[group])
    assert not np.allclose(everyone.by_group[group], without.by_group[group])


# ---------------------------------------------------------------- FR-LM-66


def test_the_gate_skips_silent_and_unverified_women(base):
    """FR-LM-66: silent and unverified women aren't counted, so the gate sees fewer women; a
    woman who has paid for a date (FR-LM-81), or who has had fewer list days, still counts."""
    z = zone(base)
    scores = scores_of(z, base)
    counted = app.counted_women(z.con, z.market, z.cfg)
    payers = {m for (m,) in z.con.execute("SELECT member_id FROM payments").fetchall()}
    women = [int(z.market.ids[r]) for r in np.flatnonzero(counted)]
    women = [w for w in women if w not in payers]  # no paid date yet in the test history
    man = ids_of(z, "man")[0]
    days = z.cfg.gate.silent_after_days
    before = app.gate_for_sale(z.con, z.market, scores, z.cfg)

    def shown(woman: int, n_days: int, pick_on_last: bool = False) -> None:
        for k in range(n_days):
            insert(
                z.con,
                "list_shows",
                {
                    "chooser_id": woman,
                    "candidate_id": man,
                    "shown_on": (TODAY - timedelta(days=k + 1)).isoformat(),
                    "picked": int(pick_on_last and k == 0),
                },
            )

    silent, almost, payer = women[:60], women[60], women[61]
    for w in silent:
        shown(w, days)
    shown(almost, days - 1)
    shown(payer, days, pick_on_last=True)
    paid = booking(z.con, payer, man, "expired", held_at="2026-10-28T23:01:00+00:00")
    insert(
        z.con,
        "payments",
        {"booking_id": paid, "member_id": payer, "amount_cents": 1000, "paid_at": "2026-10-28"},
    )
    after = app.counted_women(z.con, z.market, z.cfg)
    row = z.market.row
    assert not any(after[row[w]] for w in silent)
    assert after[row[almost]] and after[row[payer]]
    sale = app.gate_for_sale(z.con, z.market, scores, z.cfg)
    assert len(sale.admit) <= len(before.admit)
    assert sale.women_per_man < before.women_per_man

    loose = dataclasses.replace(
        z.cfg, filter=dataclasses.replace(z.cfg.filter, require_verified=False)
    )
    unverified = women[62]
    z.con.execute("UPDATE members SET verified = 0 WHERE id = ?", (unverified,))
    loose_zone = reload(SimpleNamespace(con=z.con, cfg=loose, market=None))
    assert app.member_ok(loose_zone.market.arr, loose)[row[unverified]]  # still matchable
    assert not app.counted_women(z.con, loose_zone.market, loose)[row[unverified]]


# ---------------------------------------------------------------- FR-LM-67


def test_only_ratings_from_real_dates_count(base):
    """FR-LM-67: a rating on a cancelled booking, from someone not on the booking, or naming
    the wrong person changes no one's trust; a rating on a completed date does."""
    z = zone(base)
    woman, other = ids_of(z, "woman")[:2]
    man, man2 = ids_of(z, "man")[:2]
    done = booking(z.con, woman, man, "completed", showed_a=1, showed_b=1)
    called_off = booking(
        z.con, woman, man2, "cancelled", cancelled_by=woman, cancelled_at="2026-10-29T12:00:00Z"
    )

    def rate(bid, rater, rated, score):
        insert(
            z.con,
            "ratings",
            {
                "booking_id": bid,
                "rater_id": rater,
                "rated_id": rated,
                "respect": score,
                "accuracy": score,
                "conversation": score,
                "would_meet_again": 0,
                "rated_at": "2026-10-31T12:00:00Z",
            },
        )

    def conduct():
        bookings = app.load_bookings(z.con, z.market)
        return app.trust_from_db(z.con, z.market, bookings, z.cfg).conduct()

    clean = conduct()
    rate(called_off, woman, man2, 1)  # not a completed date
    rate(done, other, man, 1)  # not on the booking
    rate(done, man, other, 1)  # names the wrong person
    np.testing.assert_array_equal(conduct(), clean)
    assert len(app.check_rows(z.con, z.market, z.cfg)) == 3
    rate(done, woman, man, 1)
    assert conduct()[z.market.row[man]] < clean[z.market.row[man]]


# ---------------------------------------------------------------- FR-LM-68


def _two_members():
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    con.executescript(
        """
        INSERT INTO neighbourhoods VALUES (1, 'SoHo', 40.7233, -74.0030);
        INSERT INTO venues VALUES (1, 'Venue A', 1, 40.7310, -73.9970, 1);
        INSERT INTO members (id, gender, seeks_men, seeks_women, seeks_nonbinary, birth_date,
            age_min_pref, age_max_pref, neighbourhood_id, max_km, verified, paying, joined_at,
            last_active_at)
        VALUES (1, 'woman', 1, 0, 0, '1994-05-01', 28, 40, 1, 5, 1, 0, '2026-11-01', '2026-11-04'),
               (2, 'man', 0, 1, 0, '1990-02-01', 26, 36, 1, 5, 1, 1, '2026-11-01', '2026-11-04');
        """
    )
    return con


MALFORMED = [
    ("cancelled", {}),  # without who and when
    ("cancelled", {"cancelled_by": 1}),  # without when
    ("no_show", {}),  # without who came
    ("no_show", {"showed_a": 1, "showed_b": 1}),  # both came: that's a completed date
    ("completed", {"showed_a": 0, "showed_b": 1}),  # someone marked absent
    ("completed", {"showed_a": 1}),  # half known
    ("confirmed", {"cancelled_by": 1, "cancelled_at": "2026-11-05T12:00:00Z"}),
]


@pytest.mark.parametrize("state, extra", MALFORMED)
def test_the_schema_refuses_bookings_missing_their_facts(state, extra):
    """FR-LM-68: the schema refuses a booking row whose state is missing its facts."""
    con = _two_members()
    with pytest.raises(sqlite3.IntegrityError):
        booking(con, 1, 2, state, **extra)


def test_the_schema_keeps_whole_bookings():
    """FR-LM-68: complete rows for every state still load."""
    con = _two_members()
    booking(con, 1, 2, "completed", showed_a=1, showed_b=1)
    booking(con, 1, 2, "completed")
    booking(con, 1, 2, "no_show", showed_a=1, showed_b=0)
    booking(con, 1, 2, "cancelled", cancelled_by=2, cancelled_at="2026-10-29T12:00:00Z")
    assert con.execute("SELECT count(*) FROM bookings").fetchone() == (4,)


@pytest.mark.parametrize("state, extra", MALFORMED)
def test_the_engine_names_a_malformed_booking(base, state, extra):
    """FR-LM-68: in a database without the schema's checks, the engine refuses the row with
    an error naming the booking, and check_rows lists it."""
    z = zone(base)
    woman, man = ids_of(z, "woman")[0], ids_of(z, "man")[0]
    extra = {k: (woman if k == "cancelled_by" else v) for k, v in extra.items()}
    z.con.execute("PRAGMA ignore_check_constraints = ON")
    bid = booking(z.con, woman, man, state, **extra)
    z.con.execute("PRAGMA ignore_check_constraints = OFF")
    with pytest.raises(app.HandoffError, match=f"booking {bid} "):
        app.load_bookings(z.con, z.market)
    assert any(p.startswith(f"booking {bid} ") for p in app.check_rows(z.con, z.market, z.cfg))


def test_payments_from_strangers_are_ignored(base):
    """FR-LM-68: a payment from someone not on the booking is ignored and reported."""
    z = zone(base)
    woman, other = ids_of(z, "woman")[:2]
    man = ids_of(z, "man")[0]
    bid = booking(z.con, woman, man, "confirmed")
    for who in (woman, other):
        insert(
            z.con,
            "payments",
            {"booking_id": bid, "member_id": who, "amount_cents": 1000, "paid_at": "2026-10-28"},
        )
    bk = app.load_bookings(z.con, z.market).items[-1]
    assert bk.paid == {z.market.row[woman]}
    assert any("isn't on booking" in p for p in app.check_rows(z.con, z.market, z.cfg))


# ---------------------------------------------------------------- FR-LM-69


def test_no_rating_buys_rank(p1):
    """FR-LM-69: perfect ratings and perfect attendance never raise anyone's ranking factor
    above 1, so a ring can't buy rank."""
    book = TrustBook(3, p1)
    for rater in range(20):
        book.add_rating(Rating(1 + rater % 2, 0, 5, 5, 5, True))
        book.add_attendance(0, True)
    factor = book.ranking_factor()
    assert factor[0] == 1.0 and factor[1] == 1.0 and factor.max() == 1.0


def test_too_few_ratings_never_flag(p1):
    """FR-LM-69: fewer than trust.flag_min_ratings ratings never flag, whatever they say."""
    n = p1.trust.flag_min_ratings
    book = TrustBook(n + 2, p1)
    for rater in range(1, n):
        book.add_rating(Rating(rater, 0, 1, 1, 1, False))
    assert not book.flagged().any()
    book.add_rating(Rating(n, 0, 1, 1, 1, False))
    assert book.flagged()[0]


# ---------------------------------------------------------------- FR-LM-70


def test_lists_dont_reveal_a_ranking(small_zone):
    """FR-LM-70: the ranking spy accounts read from two weeks of lists correlates with the
    zone's consensus looks below 0.2."""
    z = small_zone
    sc = build_scores(z.pop, z.cfg, z.seed)
    rho = attacks.list_leak(z.pop, z.cfg, z.seed, sc.mutual, z.join, n_spies=10, days=14)
    assert rho < 0.2
