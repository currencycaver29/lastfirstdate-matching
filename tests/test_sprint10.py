"""FR-LM-49 and FR-LM-50: step-by-step calibration, and the nightly cache."""

import sqlite3
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from lastfirstdate import app
from lastfirstdate.app_fill import fill_database
from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import load
from lastfirstdate.filter import member_ok
from lastfirstdate.looks import (
    ArrayPhotoEncoder,
    CalibrationSession,
    PersonalTaste,
    StartingTaste,
    Swipe,
    calibrate,
    calibration_pool,
    photo_features,
    select_pair,
    select_varied,
)
from lastfirstdate.member_behavior import Behavior
from lastfirstdate.synthetic import generate

TODAY, EPOCH = date(2026, 11, 4), date(2026, 11, 2)


def original_calibrate(taste, pool_ids, features, genders, swipe, cfg, pair=None):
    """The photo exercise exactly as written before Sprint 10 (the reference)."""
    pool_ids = np.asarray(pool_ids, dtype=int)
    k = min(cfg.calibration.n_profiles, pool_ids.size)
    shown = []
    for row in select_varied(features[pool_ids][:, :-1], min(cfg.calibration.n_variety, k)):
        j = int(pool_ids[row])
        taste.add_swipe(features[j], genders[j], swipe(j), refit=False)
        shown.append(j)
    if shown:
        taste.refit()
    remaining = np.setdiff1d(pool_ids, shown)
    for _ in range(cfg.calibration.n_pairs if pair is not None else 0):
        if len(shown) + 2 > k:
            break
        chosen = select_pair(taste, remaining, features, genders)
        if chosen is None:
            break
        a, b = chosen
        taste.add_pair(features[a], features[b], genders[a], pair(a, b))
        shown += [a, b]
        remaining = remaining[(remaining != a) & (remaining != b)]
    while len(shown) < k and remaining.size:
        info = taste.information(features[remaining], genders[remaining])
        j = int(remaining[int(np.argmax(info))])
        taste.add_swipe(features[j], genders[j], swipe(j))
        shown.append(j)
        remaining = remaining[remaining != j]
    taste.complete = True
    return shown


@pytest.fixture(scope="module", params=["config/phase1.toml", "config/lastfirstdate.toml"])
def exercise(request):
    """A population, its people's answers and a starting taste: Phase 1 (with pair questions)
    and the research default (singles only)."""
    cfg = load(request.param)
    pop = generate(cfg, seed=11, n=300)
    arr = MemberArrays.from_members(pop.members, cfg.values)
    behavior = Behavior(pop, arr, cfg, np.random.default_rng(5))
    _, feats = photo_features(ArrayPhotoEncoder(pop.photo_embeddings), len(pop), cfg)
    viewers = [int(v) for v in np.flatnonzero(member_ok(arr, cfg))][:25]
    likes = {v: behavior.swipe_outcomes(v, 11) for v in viewers}
    swipes = [
        Swipe(int(arr.gender[v]), int(arr.gender[j]), feats[j], bool(likes[v][j]))
        for v in viewers
        for j in calibration_pool(arr, cfg, v)[:15]
    ]
    start = StartingTaste.fit(swipes, cfg.calibration.w0_l2_penalty)
    return SimpleNamespace(
        cfg=cfg, arr=arr, behavior=behavior, feats=feats, viewers=viewers, likes=likes, start=start
    )


def _oracle(ex, v):
    swipe = lambda j: bool(ex.likes[v][j])  # noqa: E731
    pair = lambda a, b: ex.behavior.pair_outcome(v, a, b, 11)  # noqa: E731
    return swipe, pair


def _taste(ex, v):
    c = ex.cfg.calibration
    return PersonalTaste(int(ex.arr.gender[v]), ex.start, c.l2_penalty, c.n_profiles)


# ---------------------------------------------------------------- FR-LM-49


def test_a_session_asks_what_the_original_exercise_asked(exercise):
    """FR-LM-49: question by question, the session (and calibrate, built on it) asks exactly
    what the pre-Sprint 10 exercise asked, and ends with the same taste model."""
    ex = exercise
    for v in ex.viewers:
        swipe, pair = _oracle(ex, v)
        pool = calibration_pool(ex.arr, ex.cfg, v)
        before = _taste(ex, v)
        expected = original_calibrate(before, pool, ex.feats, ex.arr.gender, swipe, ex.cfg, pair)
        now = _taste(ex, v)
        assert calibrate(now, pool, ex.feats, ex.arr.gender, swipe, ex.cfg, pair=pair) == expected
        np.testing.assert_array_equal(now.delta, before.delta)
        assert now.complete


def test_an_interrupted_session_resumes_where_it_stopped(exercise):
    """FR-LM-49: stop after any number of answers, resume from the stored answers, and the
    remaining questions are the same as if nobody had stopped."""
    ex = exercise
    for v in ex.viewers[:8]:
        swipe, pair = _oracle(ex, v)
        pool = calibration_pool(ex.arr, ex.cfg, v)
        full = CalibrationSession(_taste(ex, v), pool, ex.feats, ex.arr.gender, ex.cfg)
        while (q := full.next_question()) is not None:
            full.answer(swipe(q.a) if q.b is None else pair(q.a, q.b))
        for stop in (3, 14, len(full.answers) - 1):
            again = CalibrationSession(_taste(ex, v), pool, ex.feats, ex.arr.gender, ex.cfg)
            again.resume(full.answers[:stop])
            while (q := again.next_question()) is not None:
                again.answer(swipe(q.a) if q.b is None else pair(q.a, q.b))
            assert again.shown == full.shown, (v, stop)


def test_a_session_refuses_stale_or_extra_answers(exercise):
    """FR-LM-49: answering an already-answered question, or a finished exercise, is refused."""
    ex = exercise
    v = ex.viewers[0]
    swipe, pair = _oracle(ex, v)
    s = CalibrationSession(
        _taste(ex, v), calibration_pool(ex.arr, ex.cfg, v), ex.feats, ex.arr.gender, ex.cfg
    )
    first = s.next_question()
    assert s.next_question() == first  # the same question until it is answered
    s.answer(swipe(first.a), question=first)
    with pytest.raises(ValueError):
        s.answer(True, question=first)
    while (q := s.next_question()) is not None:
        s.answer(swipe(q.a) if q.b is None else pair(q.a, q.b))
    assert s.done
    with pytest.raises(ValueError):
        s.answer(True)


@pytest.fixture(scope="module")
def zone():
    """A Phase 1 zone in SQLite, as in Sprint 9."""
    cfg = load("config/phase1.toml")
    pop = generate(cfg, seed=7)
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    filled = fill_database(con, pop, cfg, TODAY, 7, subscribers=60)
    market = app.load_market(con, cfg, app.Calendar(EPOCH), TODAY)
    feats = app.photo_feature_matrix(pop.photo_embeddings, cfg)
    return SimpleNamespace(cfg=cfg, pop=pop, con=con, filled=filled, market=market, feats=feats)


def test_the_app_resumes_an_exercise_from_the_database(zone):
    """FR-LM-49: with only the first answers stored, `resume_calibration` asks exactly the
    question the member was asked next, in database ids, and the answer row passes the
    schema's checks."""
    con, market = zone.con, zone.market
    member = zone.filled.subscribers[0]
    stored = con.execute(
        "SELECT position, shown_a, shown_b, answer FROM calibration_answers "
        "WHERE member_id = ? ORDER BY position",
        (member,),
    ).fetchall()
    con.execute("DELETE FROM calibration_answers WHERE member_id = ? AND position >= 15", (member,))
    session = app.resume_calibration(con, market, zone.feats, member, zone.filled.start, zone.cfg)
    q = session.next_question()
    expected = tuple(x for x in stored[15][1:3] if x is not None)
    assert app.question_photo_ids(market, q) == expected
    row = app.answer_row(market, member, 15, q, bool(stored[15][3]), datetime(2026, 11, 4, 9))
    con.execute(
        "INSERT INTO calibration_answers VALUES (:member_id, :position, :shown_a, "
        ":shown_b, :answer, :asked_at)",
        row,
    )
    session.answer(bool(stored[15][3]), question=q)
    assert len(session.answers) == 16


# ---------------------------------------------------------------- FR-LM-50


def test_the_cache_refits_only_what_changed_and_matches_a_full_rebuild(zone):
    """FR-LM-50: a refresh after one member's answers arrive refits only that member and
    equals a full rebuild with the same starting taste; a refresh with no changes refits no
    one."""
    con, cfg, market = zone.con, zone.cfg, zone.market
    member = zone.filled.subscribers[1]
    saved = con.execute(
        "SELECT * FROM calibration_answers WHERE member_id = ?", (member,)
    ).fetchall()
    con.execute("DELETE FROM calibration_answers WHERE member_id = ?", (member,))
    cache = app.MarketCache(cfg, app.Calendar(EPOCH))
    cache.refresh(con, TODAY, zone.pop.photo_embeddings, refit_start=True)
    assert member not in cache.tastes_by_id
    con.executemany("INSERT INTO calibration_answers VALUES (?, ?, ?, ?, ?, ?)", saved)
    scores = cache.refresh(con, TODAY, zone.pop.photo_embeddings)
    assert cache.last_refitted == [member]
    answers = app.calibration_answers(con, cache.market, cfg)
    full = app.fit_tastes(cache.market, cache.features, answers, cfg, start=cache.start)
    assert set(full) == {cache.market.row[m] for m in cache.tastes_by_id}
    for v, taste in full.items():
        np.testing.assert_array_equal(
            cache.tastes_by_id[int(cache.market.ids[v])].delta, taste.delta
        )
    rebuilt = app.nightly_scores(cache.market, full, cache.features, cfg)
    np.testing.assert_array_equal(scores.mutual, rebuilt.mutual)
    same = cache.refresh(con, TODAY, zone.pop.photo_embeddings)
    assert cache.last_refitted == [] and same is scores  # nothing changed: the cached scores
    assert market.ids.tolist() == cache.market.ids.tolist()
    # a profile edit with no new answers must still recompute the scores
    con.execute("UPDATE members SET max_km = 0.1 WHERE id = ?", (member,))
    edited = cache.refresh(con, TODAY, zone.pop.photo_embeddings)
    assert edited is not scores and cache.last_refitted == []
    row = cache.market.row[member]
    assert edited.eligible[row].sum() <= scores.eligible[row].sum()
