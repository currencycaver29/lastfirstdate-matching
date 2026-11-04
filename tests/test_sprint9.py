"""FR-LM-46 to FR-LM-48: the engine on the app's database, and the handoff."""

import dataclasses
import importlib
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lastfirstdate import app
from lastfirstdate.app_sim import fill_database
from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import IMPACTS, load, setting_impact, setting_names, settings_table
from lastfirstdate.filter import eligibility_matrix, member_ok
from lastfirstdate.forecast import gate_decision, gate_model
from lastfirstdate.geo import nearest_place
from lastfirstdate.models import PARTS_OF_DAY
from lastfirstdate.personality import personality_matrix
from lastfirstdate.synthetic import generate

HANDOFF = Path(__file__).resolve().parents[1] / "docs" / "handoff" / "README.md"
TODAY, EPOCH = date(2026, 11, 4), date(2026, 11, 2)  # a Wednesday, and the Monday before


@pytest.fixture(scope="module")
def world():
    """A Phase 1 zone written into SQLite (60 subscribers, the other men on the waitlist), then
    loaded back by the engine as the app would every night."""
    cfg = load("config/phase1.toml")
    pop = generate(cfg, seed=7)
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    filled = fill_database(con, pop, cfg, TODAY, 7, subscribers=60)
    market = app.load_market(con, cfg, app.Calendar(EPOCH), TODAY)
    feats = app.photo_feature_matrix(pop.photo_embeddings, cfg)
    answers = app.calibration_answers(con, market)
    tastes = app.fit_tastes(market, feats, answers, cfg, start=filled.start)
    scores = app.nightly_scores(market, tastes, feats, cfg)
    return SimpleNamespace(
        cfg=cfg,
        pop=pop,
        con=con,
        filled=filled,
        market=market,
        feats=feats,
        tastes=tastes,
        scores=scores,
    )


def insert(con, table: str, row: dict) -> int:
    cols = ", ".join(row)
    marks = ", ".join("?" * len(row))
    cur = con.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
    return cur.lastrowid


# ---------------------------------------------------------------- FR-LM-46 call sequence


def test_every_call_in_the_handoff_exists():
    """FR-LM-46: every engine function the handoff tells the app to call can be imported."""
    calls = HANDOFF.read_text().split("<!-- settings:start -->")[0]  # not the settings table
    names = set(
        re.findall(
            r"`((?:app|looks|billing)\.[a-z_]+|(?:Booking|TrustBook|PersonalTaste)\.[a-z_]+)`",
            calls,
        )
    )
    assert len(names) >= 15
    owners = {
        "app": importlib.import_module("lastfirstdate.app"),
        "looks": importlib.import_module("lastfirstdate.looks"),
        "billing": importlib.import_module("lastfirstdate.billing"),
        "Booking": importlib.import_module("lastfirstdate.booking").Booking,
        "TrustBook": importlib.import_module("lastfirstdate.trust").TrustBook,
        "PersonalTaste": importlib.import_module("lastfirstdate.looks").PersonalTaste,
    }
    for name in names:
        owner, attr = name.split(".")
        assert callable(getattr(owners[owner], attr, None)), name


# ---------------------------------------------------------------- FR-LM-47 the engine from rows


def test_members_from_rows_match_the_profiles(world):
    """FR-LM-47: profiles loaded from rows match the originals, placed at their neighbourhood's
    centre; only subscribers count as paying."""
    hoods = world.cfg.geo.neighbourhoods
    subscribed = {i - 1000 for i in world.filled.subscribers}
    for k, (got, m) in enumerate(zip(world.market.members, world.pop.members, strict=True)):
        hood = hoods[int(nearest_place(m.lat, m.lon, hoods))]
        assert (got.lat, got.lon) == (hood.lat, hood.lon)
        for f in (
            "gender",
            "seeking",
            "age",
            "age_min_pref",
            "age_max_pref",
            "max_km",
            "verified",
            "last_active_days",
            "free_slots",
            "values",
        ):
            assert getattr(got, f) == getattr(m, f), (k, f)
        assert got.bio == m.bio[:500]
        assert got.paying == (k in subscribed)
    assert int(world.market.ids[0]) == 1000


def test_scores_from_rows_match_the_same_profiles(world):
    """FR-LM-47: eligibility and personality from rows equal those computed directly on the
    same profiles at their neighbourhood centres."""
    direct = [
        dataclasses.replace(m, lat=g.lat, lon=g.lon, paying=g.paying)
        for m, g in zip(world.pop.members, world.market.members, strict=True)
    ]
    arr = MemberArrays.from_members(direct, world.cfg.values)
    np.testing.assert_array_equal(world.scores.eligible, eligibility_matrix(arr, world.cfg))
    np.testing.assert_array_equal(
        world.scores.personality, personality_matrix(direct, arr, world.cfg)
    )


def test_taste_models_from_rows_match_the_photo_exercise(world):
    """FR-LM-47: the nightly refit from stored answers gives each member the same taste model
    the photo exercise built, answer by answer."""
    assert set(world.tastes) == set(world.filled.tastes)
    for v, taste in world.filled.tastes.items():
        np.testing.assert_allclose(world.tastes[v].delta, taste.delta, atol=1e-9)


def test_a_full_day_runs_from_rows(world):
    """FR-LM-47: lists show only eligible, admitted men, each with a window both can make;
    a booking written back passes the schema's checks and blocks that week next time."""
    cfg, market, con = world.cfg, world.market, world.con
    women = [int(market.ids[k]) for k, m in enumerate(market.members) if m.gender == "woman"]
    lists = app.day_lists(con, market, world.scores, women[:60], cfg)
    assert lists
    waiting = set(world.filled.waitlist)
    for woman, entries in lists.items():
        i = market.row[woman]
        assert 0 < len(entries) <= cfg.lists.daily_size
        for e in entries:
            j = market.row[e.member_id]
            assert world.scores.eligible[i, j] and e.member_id not in waiting
            slot = e.day.weekday() * 3 + PARTS_OF_DAY.index(e.part)
            assert slot in market.members[i].free_slots and slot in market.members[j].free_slots
            ahead = (e.day - TODAY).days
            assert cfg.booking.lead_days <= ahead < cfg.booking.horizon_days

    woman, entries = next(iter(lists.items()))
    entry = entries[0]
    bk, venue_id = app.propose(con, market, woman, entry.member_id, entry)
    bk.hold(0.0)
    assert bk.pay(bk.a, 1.0, cfg.booking.hold_minutes)
    assert bk.pay(bk.b, 4.0, cfg.booking.hold_minutes, venue=bk.venue)
    row = app.booking_row(bk, market, venue_id, datetime(2026, 11, 4, 19))
    booking_id = insert(con, "bookings", row)
    for member in (woman, entry.member_id):
        insert(
            con,
            "payments",
            {
                "booking_id": booking_id,
                "member_id": member,
                "amount_cents": 1000,
                "paid_at": "2026-11-04T19:05:00",
            },
        )
    for e in entries:
        insert(
            con,
            "list_shows",
            {
                "chooser_id": woman,
                "candidate_id": e.member_id,
                "shown_on": TODAY.isoformat(),
                "picked": int(e.member_id == entry.member_id),
            },
        )

    again = app.day_lists(con, market, world.scores, [woman], cfg).get(woman, [])
    week = entry.day.isocalendar()[:2]
    assert all(e.member_id not in {x.member_id for x in entries} for e in again)  # rested/closed
    assert all(e.day.isocalendar()[:2] != week for e in again)  # her week is full


def test_the_gate_on_rows_matches_the_gate_on_arrays(world):
    """FR-LM-47: the sale decision from rows equals gate_decision on the same market."""
    cfg, market = world.cfg, world.market
    sale = app.gate_for_sale(world.con, market, world.scores, cfg)
    ok = member_ok(market.arr, cfg) & ~market.paused
    women = np.flatnonzero(ok & (market.arr.gender == app.WOMAN))
    current = [r for r in market.rows(world.filled.subscribers) if ok[r]]
    applicants = market.rows(world.filled.waitlist)
    d = gate_decision(gate_model(cfg), world.scores.eligible, women, current, applicants, cfg)
    assert sale.admit == [int(market.ids[m]) for m in d.admit]
    assert sale.stopped_because == d.stopped_because


def test_alerts_and_cycle_end_from_rows(world):
    """FR-LM-47: every subscribed man gets a chance; a cycle with no dates is credited in full."""
    flagged, chances = app.weekly_alerts(world.con, world.market, world.scores, world.cfg)
    subscribed_ok = [
        i
        for i in world.filled.subscribers
        if member_ok(world.market.arr, world.cfg)[world.market.row[i]]
    ]
    assert set(chances) == set(subscribed_ok) and set(flagged) <= set(chances)
    man = world.filled.subscribers[-1]
    assert app.close_cycle(world.con, world.market, man, date(2026, 12, 20), world.cfg) == (
        0,
        world.cfg.billing.price_per_cycle,
    )


def test_loading_refuses_settings_the_database_cannot_serve(world):
    """FR-LM-47: Big Five weights or tag-based interests need data the schema doesn't hold."""
    cfg = world.cfg
    with pytest.raises(app.HandoffError):
        app.load_market(
            world.con,
            dataclasses.replace(cfg, personality=dataclasses.replace(cfg.personality, w_big5=0.2)),
            app.Calendar(EPOCH),
            TODAY,
        )
    with pytest.raises(app.HandoffError):
        app.Calendar(TODAY)  # not a Monday


# ---------------------------------------------------------------- FR-LM-48 operator settings


def test_every_setting_is_classified():
    """FR-LM-48: each of the settings has one of the five classes."""
    names = setting_names()
    assert len(names) > 100 and len(names) == len(set(names))
    assert all(setting_impact(n) in IMPACTS for n in names)
    with pytest.raises(KeyError):
        setting_impact("nonsense.key")


def test_the_handoff_lists_the_settings_as_classified():
    """FR-LM-48: the handoff's settings table is exactly the classification in the code."""
    text = HANDOFF.read_text()
    between = text.split("<!-- settings:start -->")[1].split("<!-- settings:end -->")[0]
    assert between.strip() == settings_table().strip()
