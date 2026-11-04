"""FR-LM-36 to FR-LM-42: booking, billing, trust, events, geography and the cycle simulation."""

import dataclasses
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from lastfirstdate import geo
from lastfirstdate.billing import (
    counts_toward_guarantee,
    credit,
    cycle_on,
    fee_refunds,
    missing,
    refill_bonus,
)
from lastfirstdate.booking import (
    NO_WINDOW,
    SOURCES,
    STATES,
    Booking,
    BookingError,
    dated_pairs,
    earliest_common,
    event_booking,
    window_bits,
    window_day_part,
)
from lastfirstdate.config import Place, load
from lastfirstdate.cycle_sim import CycleSimulation, frictionless, join_days, market
from lastfirstdate.models import PARTS_OF_DAY
from lastfirstdate.mutual_report import build_scores
from lastfirstdate.trust import Rating, TrustBook

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (ROOT / "db" / "schema.sql").read_text()


@pytest.fixture(scope="module")
def p1cfg():
    return load("config/phase1.toml")


# ---------------------------------------------------------------- FR-LM-36 app data model


def _db():
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
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


BOOKING = (
    "INSERT INTO bookings (id, member_a, member_b, source, event_id, date_on, part, state, "
    "proposed_at, venue_id) VALUES (?, 1, 2, ?, ?, '2026-11-10', 'evening', ?, '2026-11-04', ?)"
)


def test_schema_loads_and_keeps_a_whole_date():
    """FR-LM-36: the schema loads and holds a confirmed date, both payments and a rating."""
    con = _db()
    con.execute(BOOKING, (1, "list", None, "confirmed", 1))
    con.execute("INSERT INTO payments VALUES (1, 1, 1, 1000, '2026-11-04T19:00', 0)")
    con.execute("INSERT INTO payments VALUES (2, 1, 2, 1000, '2026-11-04T19:06', 0)")
    con.execute("UPDATE bookings SET state = 'completed', showed_a = 1, showed_b = 1")
    con.execute("INSERT INTO ratings VALUES (1, 1, 2, 5, 4, 3, 1, '2026-11-11')")
    assert con.execute("SELECT count(*) FROM ratings").fetchone() == (1,)


@pytest.mark.parametrize(
    "sql, args",
    [
        (BOOKING, (1, "list", None, "pending", 1)),  # unknown state
        (BOOKING, (1, "list", None, "confirmed", None)),  # confirmed without a venue
        (BOOKING, (1, "event", None, "proposed", None)),  # event date without an event
        ("INSERT INTO ratings VALUES (9, 1, 2, 6, 4, 3, 1, 'x')", ()),  # rating above 5
        ("INSERT INTO ratings VALUES (9, 1, 1, 5, 4, 3, 1, 'x')", ()),  # rating oneself
        ("INSERT INTO free_slots VALUES (1, 21)", ()),  # no such weekly slot
    ],
)
def test_schema_rejects_invalid_rows(sql, args):
    """FR-LM-36: the schema's checks refuse rows the rules forbid."""
    con = _db()
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(sql, args)


def test_schema_matches_the_engine_and_keeps_the_rules():
    """FR-LM-36: booking states, sources and parts of day match the engine; no column holds
    looks or a rank (R2); photo embeddings have their own table (R3)."""

    def listed(column):
        m = re.search(rf"{column}\s+TEXT NOT NULL CHECK \({column} IN \(([^)]*)\)\)", SCHEMA)
        return tuple(re.findall(r"'([^']+)'", m.group(1)))

    assert listed("state") == STATES
    assert listed("source") == SOURCES
    assert listed("part") == PARTS_OF_DAY
    con = _db()
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        for row in con.execute(f"PRAGMA table_info({table})"):
            assert not re.search(r"attract|rank|elo|looks|score", row[1]), (table, row[1])
    columns = [r[1] for r in con.execute("PRAGMA table_info(members)")]
    assert "encrypted_vector" not in columns
    assert [r[1] for r in con.execute("PRAGMA table_info(photo_embeddings)")][0] == "member_id"


# ---------------------------------------------------------------- FR-LM-37 windows and holds

MON_EVENING, TUE_EVENING, MON_LUNCH = 1 << 1, 1 << 4, 1 << 0


def test_the_earliest_common_window_is_proposed():
    """FR-LM-37: the first evening both are free, from the lead time on, as (day, part)."""
    bits = window_bits(np.array([MON_EVENING, MON_EVENING | TUE_EVENING, MON_LUNCH]), 0, 1, 14)
    w = earliest_common(bits[:, None], bits[None, :])
    assert window_day_part(int(w[0, 1]), 0) == (7, 1)  # today's Monday is too soon: next one
    assert window_day_part(int(w[1, 1]), 0) == (1, 1)  # alone: Tuesday evening
    assert w[0, 2] == NO_WINDOW  # free at different times of day


def test_blocked_days_are_skipped():
    """FR-LM-37: a day with a date (or a full week or month) is never proposed."""
    blocked = np.zeros((2, 14), dtype=bool)
    blocked[0, 1] = True  # member 0 already has a date on Tuesday
    bits = window_bits(np.array([TUE_EVENING, TUE_EVENING]), 0, 1, 14, blocked)
    assert window_day_part(int(earliest_common(bits[0], bits[1])), 0) == (8, 1)


def test_a_booking_follows_the_legal_moves():
    """FR-LM-37: proposed, held, both pay within the hold, confirmed with the venue,
    completed; illegal moves are refused."""
    bk = Booking(1, 2, day=3, part=1)
    with pytest.raises(BookingError):
        bk.pay(1, 0.0, 15)  # no hold yet
    bk.hold(0.0)
    assert bk.pay(1, 2.0, 15) and bk.state == "held"
    assert bk.pay(2, 9.0, 15, venue="Venue A") and bk.state == "confirmed"
    assert bk.venue == "Venue A"
    bk.finish(True, True)
    assert bk.state == "completed" and bk.is_final
    with pytest.raises(BookingError):
        bk.cancel(1, 48)
    with pytest.raises(BookingError):
        Booking(1, 1, 3, 1)
    with pytest.raises(BookingError):
        Booking(1, 2, 3, 1, source="speed dating")


def test_a_late_payment_expires_the_hold_and_frees_the_pair():
    """FR-LM-37: a payment after hold_minutes expires the booking; the pair is not 'dated'."""
    bk = Booking(1, 2, day=3, part=1)
    bk.hold(0.0)
    assert bk.pay(1, 2.0, 15)
    assert not bk.pay(2, 16.0, 15)
    assert bk.state == "expired"
    assert dated_pairs([bk]) == set()


# ---------------------------------------------------------------- FR-LM-38 billing


def _done(state, day=40, cancelled_by=None, hours=None, showed=None):
    bk = Booking(1, 2, day=day, part=1)
    bk.hold(0.0)
    bk.pay(1, 1.0, 15)
    bk.pay(2, 2.0, 15, venue="V")
    if state == "cancelled":
        bk.cancel(cancelled_by, hours)
    elif state in ("completed", "no_show"):
        bk.finish(*(showed or (True, True)))
    return bk


def test_cycles_start_after_the_trial(p1cfg):
    """FR-LM-38: a 30-day free trial, then 28-day paid cycles back to back."""
    trial = cycle_on(2, 0, 10, p1cfg)
    assert (trial.index, trial.start, trial.end, trial.price) == (-1, 0, 30, 0.0)
    first = cycle_on(2, 0, 30, p1cfg)
    assert (first.index, first.start, first.end, first.price) == (0, 30, 58, 200.0)
    assert cycle_on(2, 0, 60, p1cfg).start == 58


def test_what_counts_as_a_delivered_date(p1cfg):
    """FR-LM-38: completed dates count; her cancellations and no-shows don't; his own late
    cancellation or no-show does (D17), unless own_absence_counts is off; dates outside
    the cycle never count."""
    cyc = cycle_on(2, 0, 40, p1cfg)  # man 2's cycle, days 30 to 57
    cases = {
        "completed": (_done("completed"), True),
        "she cancels late": (_done("cancelled", cancelled_by=1, hours=6), False),
        "he cancels late": (_done("cancelled", cancelled_by=2, hours=6), True),
        "he cancels early": (_done("cancelled", cancelled_by=2, hours=48), False),
        "she doesn't come": (_done("no_show", showed=(False, True)), False),
        "he doesn't come": (_done("no_show", showed=(True, False)), True),
        "next cycle": (_done("completed", day=70), False),
    }
    for name, (bk, expected) in cases.items():
        assert counts_toward_guarantee(bk, 2, cyc, p1cfg) is expected, name
    strict = dataclasses.replace(
        p1cfg, billing=dataclasses.replace(p1cfg.billing, own_absence_counts=False)
    )
    assert not counts_toward_guarantee(cases["he doesn't come"][0], 2, cyc, strict)


def test_a_cancelled_date_is_refilled_with_rising_priority():
    """FR-LM-38: a date that falls through raises his missing count, and the list bonus
    grows as his cycle runs out."""
    assert missing(2, 1, 3) == 0
    assert missing(2, 0, 3) == 1  # the upcoming one was cancelled by her
    early, late = refill_bonus(np.array([1, 1]), np.array([21, 5]), 1.0)
    assert 0 < early < late


def test_credits_and_date_fee_refunds(p1cfg):
    """FR-LM-38: credit = price * missing / 3 (none in the trial); fees refunded in full 24h+
    ahead, half later, nothing to a no-show, nothing after a completed date."""
    assert credit(1, cycle_on(2, 0, 40, p1cfg), p1cfg) == pytest.approx(200 / 3)
    assert credit(3, cycle_on(2, 0, 5, p1cfg), p1cfg) == 0.0
    assert fee_refunds(_done("cancelled", cancelled_by=2, hours=48), p1cfg) == {1: 10, 2: 10}
    assert fee_refunds(_done("cancelled", cancelled_by=2, hours=6), p1cfg) == {1: 10, 2: 5}
    assert fee_refunds(_done("no_show", showed=(True, False)), p1cfg) == {1: 10, 2: 0}
    assert fee_refunds(_done("completed"), p1cfg) == {1: 0, 2: 0}


# ---------------------------------------------------------------- FR-LM-39 ratings and trust


def test_trust_starts_at_the_prior_and_moves_with_conduct(p1cfg):
    """FR-LM-39: no ratings means the prior; conduct = (k * prior + sum) / (k + n)."""
    book = TrustBook(3, p1cfg)
    np.testing.assert_allclose(book.trust(), p1cfg.trust.prior_mean)
    book.add_rating(Rating(0, 1, respect=1, accuracy=3, conversation=5, again=True))
    k, prior = p1cfg.trust.prior_strength, p1cfg.trust.prior_mean
    assert book.conduct()[1] == pytest.approx((k * prior + 0.25) / (k + 1))


def test_conversation_and_again_never_move_trust(p1cfg):
    """FR-LM-39: chemistry is personal, so only respect, accuracy and reliability count (R2)."""
    a, b = TrustBook(2, p1cfg), TrustBook(2, p1cfg)
    a.add_rating(Rating(0, 1, 4, 4, conversation=1, again=False))
    b.add_rating(Rating(0, 1, 4, 4, conversation=5, again=True))
    np.testing.assert_array_equal(a.trust(), b.trust())


def test_missed_dates_lower_reliability(p1cfg):
    """FR-LM-39: no-shows and late cancellations lower trust through reliability."""
    book = TrustBook(2, p1cfg)
    for kept in (True, False, False):
        book.add_attendance(1, kept)
    assert book.reliability()[1] < book.reliability()[0] == pytest.approx(p1cfg.trust.prior_mean)
    assert book.trust()[1] < book.trust()[0]


def test_low_trust_lowers_ranking_and_enough_poor_ratings_flag(p1cfg):
    """FR-LM-39: the ranking factor is 1 at or above the prior and lower below it; a member is
    flagged only after flag_min_ratings poor ratings."""
    book = TrustBook(3, p1cfg)
    book.add_rating(Rating(0, 2, 5, 5, 3, True))
    book.add_rating(Rating(0, 1, 1, 1, 3, False))
    factor = book.ranking_factor()
    assert factor[0] == 1.0 and factor[2] == 1.0 and factor[1] < 1.0
    assert not book.flagged()[1]  # one bad date is not enough
    for _ in range(p1cfg.trust.flag_min_ratings + 2):
        book.add_rating(Rating(0, 1, 1, 1, 3, False))
    assert book.flagged()[1] and not book.flagged()[2]
    with pytest.raises(ValueError):
        Rating(0, 1, 6, 4, 3, True)
    with pytest.raises(ValueError):
        Rating(1, 1, 4, 4, 3, True)


# ---------------------------------------------------------------- FR-LM-40 event dates


def test_an_event_date_counts_like_a_list_date(p1cfg):
    """FR-LM-40: an event connection booked as a date follows the same rules, counts toward
    his guarantee and closes the pair."""
    bk = event_booking(1, 2, day=40, part=1)
    assert bk.source == "event"
    bk.hold(0.0)
    bk.pay(1, 1.0, 15)
    bk.pay(2, 2.0, 15, venue="V")
    bk.finish(True, True)
    assert counts_toward_guarantee(bk, 2, cycle_on(2, 0, 40, p1cfg), p1cfg)
    assert dated_pairs([bk]) == {(1, 2)}


# ---------------------------------------------------------------- FR-LM-41 geography


def test_members_are_placed_in_their_nearest_neighbourhood(p1cfg):
    """FR-LM-41: a point in SoHo maps to SoHo."""
    hoods = p1cfg.geo.neighbourhoods
    assert hoods[int(geo.nearest_place(40.7231, -74.0025, hoods))].name == "SoHo"


def test_the_venue_keeps_the_longer_trip_short():
    """FR-LM-41: the venue minimises the longer of the two trips, ties by the total; two people
    in one place get the venue nearest to them."""
    a, b = Place("A", 40.70, -74.00), Place("B", 40.72, -74.00)
    near_a, middle = Place("near A", 40.701, -74.00), Place("middle", 40.71, -74.00)
    best, trip_a, trip_b = geo.venue_for(a, b, [near_a, middle])
    assert best == 1 and abs(trip_a - trip_b) < 0.01
    east, west = Place("east", 40.71, -73.99), Place("west", 40.71, -74.004)
    assert geo.venue_for(a, b, [east, west])[0] == 1  # same longer trip? the shorter total
    assert geo.venue_for(a, a, [middle, near_a])[0] == 1


# ---------------------------------------------------------------- FR-LM-42 cycle simulation


@pytest.fixture(scope="module")
def phase1_cycles(p1cfg):
    """Seed 7 Phase 1 market with new women arriving: the real flow and a frictionless one."""
    pop = market(p1cfg, 7, p1cfg.cycle_simulation.new_women_per_cycle)
    sc = build_scores(pop, p1cfg, 7)
    join = join_days(pop, p1cfg, p1cfg.cycle_simulation.new_women_per_cycle)
    real = CycleSimulation(pop, p1cfg, 7, sc.mutual, join)
    smooth = CycleSimulation(pop, frictionless(p1cfg), 7, sc.mutual, join)
    return real, real.run("real"), smooth.run("frictionless")


def test_the_books_balance(p1cfg, phase1_cycles):
    """FR-LM-42: every past booking ends in a final state, nobody exceeds a cap, delivered
    dates never pass 3 a cycle, credits match missing dates, refunds never exceed fees."""
    sim, rep, _ = phase1_cycles
    past = [bk for bk in sim.bookings if bk.day < sim.days]
    assert past and all(bk.is_final for bk in past)
    assert sum(rep.states.values()) == len(past)
    counts: dict[tuple[int, int], int] = {}
    weeks: dict[tuple[int, int], int] = {}
    for bk in sim.bookings:
        if bk.state in ("expired", "cancelled"):
            continue
        for m in bk.members:
            if bk.state == "no_show" and bk.showed[m]:
                continue  # stood up: the slot went back to them
            period = int(sim.period(np.array([m]), np.array([bk.day]))[0])
            counts[m, period] = counts.get((m, period), 0) + 1
            weeks[m, bk.day // 7] = weeks.get((m, bk.day // 7), 0) + 1
    assert all(v <= sim.cap[m] for (m, _), v in counts.items())
    assert max(weeks.values()) <= p1cfg.dates.max_per_week
    assert rep.delivered.max() <= 3
    expected = p1cfg.billing.price_per_cycle * np.maximum(0, 3 - rep.delivered).sum() / 3
    assert rep.credits == pytest.approx(expected)
    assert 0 <= rep.fees_refunded <= rep.fees_paid


def test_friction_costs_men_their_dates(phase1_cycles):
    """FR-LM-42: with everyone paying in time, nobody cancelling and everyone showing up, at
    least 5 points more men reach 3 delivered dates. (Sprint 7 claimed 10; that came from a
    simulation bug that never refilled a slot after a no-show, fixed in Sprint 8.)"""
    _, real, smooth = phase1_cycles
    assert smooth.guarantee_share >= real.guarantee_share + 0.05


def test_trust_flags_bad_actors_and_not_looks(phase1_cycles):
    """FR-LM-42: at least 80% of simulated bad actors with 6+ ratings are flagged (about 6
    poor ratings bring a typical one below the line), at least 80% of flags are bad actors,
    and trust does not track how attractive people find someone (R2)."""
    _, rep, _ = phase1_cycles
    assert rep.bad_well_rated > 0
    assert rep.caught_well_rated >= 0.8 * rep.bad_well_rated
    assert rep.flagged_bad >= 0.8 * rep.flagged
    assert abs(rep.trust_looks_corr) < 0.2


@pytest.mark.xfail(
    strict=True,
    reason="Sprint 7 finding (corrected in Sprint 8): at 3 women per man, with payment holds, "
    "cancellations and no-shows, about 82% of men (3 simulated markets) get 3 delivered dates "
    "a cycle; reaching 90% takes about 4.5 women per man",
)
def test_the_real_flow_reaches_the_north_star(p1cfg, phase1_cycles):
    """FR-LM-42: at least gate.target of men get 3 delivered dates in the real flow."""
    _, real, _ = phase1_cycles
    assert real.guarantee_share >= p1cfg.gate.target
