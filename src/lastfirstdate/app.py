"""The engine as the app calls it: from database rows to decisions (FR-LM-46, FR-LM-47).

Everything here reads the tables in `db/schema.sql` and never the test harness. `docs/handoff/`
says when the app calls each function.

Reading: functions take a DB-API connection and only run `SELECT * FROM <table>`, so the same
code works on PostgreSQL and SQLite. Writing stays with the app: functions return engine
objects (for example a `Booking`) and `booking_row` turns one into column values.

Days: the engine counts days from a Monday, `Calendar.epoch` (day 0), because weekly free
slots and date windows are numbered from Monday.

Locations: members give a neighbourhood, never an address, so the engine places each member at
their neighbourhood's centre.

Times (D19): timestamps are UTC, with or without an offset; a value with no time of day is a
date in the zone's own time (`booking.timezone`), as are date windows. `local_time` converts.

What the database doesn't hold, and so isn't used:
- Big Five scores and interest tags (Phase 1 has neither; `w_big5` must be 0 and the text
  encoder "words");
- accepted-answer lists (Phase 1 intent questions are single-answer "closeness" questions).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import astuple, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from lastfirstdate import __version__, native
from lastfirstdate.arrays import GENDERS, MemberArrays
from lastfirstdate.billing import counts_toward_guarantee, credit, cycle_on, delivered, refill_bonus
from lastfirstdate.booking import (
    N_PARTS,
    Booking,
    dated_pairs,
    earliest_common,
    window_bits,
    window_day_part,
)
from lastfirstdate.config import Config, Place
from lastfirstdate.fairness import fair_day_lists
from lastfirstdate.filter import eligibility_matrix, member_ok
from lastfirstdate.forecast import (
    GateDecision,
    at_risk_delivered,
    gate_decision,
    gate_model,
    market_features,
)
from lastfirstdate.geo import nearest_place, venue_for
from lastfirstdate.lists import NEVER_SHOWN, available_row, pick_rights
from lastfirstdate.looks import (
    PCA,
    ArrayPhotoEncoder,
    CalibrationSession,
    PersonalTaste,
    Question,
    StartingTaste,
    Swipe,
    calibration_pool,
    photo_features,
    with_intercept,
)
from lastfirstdate.models import PARTS_OF_DAY, Member
from lastfirstdate.mutual import directional_score, looks_scores, mutual_score
from lastfirstdate.personality import personality_matrix
from lastfirstdate.trust import Rating, TrustBook, with_trust

WOMAN = GENDERS.index("woman")
NOT_ADMITTED = ("waiting", "deferred_small_pool", "withdrawn")  # waitlist states
NEUTRAL_BIG5 = (0.5, 0.5, 0.5, 0.5, 0.5)
NO_SIGNUP = np.iinfo(np.int64).min  # signup day of members without a subscription
PENDING = ("proposed", "held", "confirmed")  # bookings that still lie ahead


class HandoffError(ValueError):
    """The database or the settings don't support what was asked."""


# ---------------------------------------------------------------- reading rows


def read(con, table: str) -> list[dict]:
    """Every row of a table as dicts (works with any DB-API connection)."""
    cur = con.execute(f"SELECT * FROM {table}")
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]


@dataclass(frozen=True)
class Calendar:
    epoch: date  # day 0; must be a Monday

    def __post_init__(self) -> None:
        if self.epoch.weekday() != 0:
            raise HandoffError("the calendar's day 0 must be a Monday")

    def day(self, when: date | str) -> int:
        when = date.fromisoformat(when[:10]) if isinstance(when, str) else when
        return (when - self.epoch).days

    def date(self, day: int) -> date:
        return self.epoch + timedelta(days=day)


def age_on(birth: date, today: date) -> int:
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


def local_time(stamp: str | datetime, tz: ZoneInfo) -> datetime:
    """An instant in the zone's own time (D19). With an offset (`Z`, `+00:00`, ...) it is
    converted; with none it is read as UTC; a date with no time of day is midnight there."""
    if isinstance(stamp, str):
        text = stamp.strip()
        when = datetime.fromisoformat(text)
        if len(text) == 10:  # a date, already in the zone's own time
            return when.replace(tzinfo=tz)
    else:
        when = stamp
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(tz)


def local_date(stamp: str | datetime, tz: ZoneInfo) -> date:
    return local_time(stamp, tz).date()


def hours_between(later: datetime, earlier: datetime) -> float:
    """True hours between two aware instants (right across a daylight-saving change)."""
    return (later.astimezone(UTC) - earlier.astimezone(UTC)).total_seconds() / 3600


# ---------------------------------------------------------------- the market


@dataclass
class Market:
    """Everyone in a zone, as the engine sees them, with their database ids."""

    members: list[Member]
    ids: np.ndarray  # database id of each row
    row: dict[int, int]  # database id -> row
    arr: MemberArrays
    paused: np.ndarray  # paused for review
    neighbourhood: list[Place]  # each member's neighbourhood
    calendar: Calendar
    today: int  # day index
    tz: ZoneInfo  # the zone's clock (booking.timezone)
    part_starts: tuple[int, ...]  # hour lunch, evening and late begin, local time
    blocked: np.ndarray  # (n, n) pairs where either blocked the other (FR-LM-56)
    forgotten: np.ndarray  # (n,) members who deleted their account (FR-LM-57)

    def rows(self, ids) -> list[int]:
        return [self.row[int(i)] for i in ids]


def load_market(con, cfg: Config, calendar: Calendar, today: date) -> Market:
    """Members, free slots, intent answers, prompts and blocks, as the engine's profiles.

    A forgotten member (D18) loads as permanently inactive, so past bookings still resolve.
    """
    if cfg.personality.w_big5 > 0:
        raise HandoffError("the database holds no Big Five scores: set personality.w_big5 = 0")
    if cfg.personality.text_encoder not in ("words", "sentence"):
        raise HandoffError("the database holds prompts, not tags: use 'sentence' or 'words'")
    if any(q.mode != "closeness" for q in cfg.values):
        raise HandoffError("the database holds one answer per question: use closeness questions")
    tz = ZoneInfo(cfg.booking.timezone)
    hoods = {r["id"]: Place(r["name"], r["lat"], r["lon"]) for r in read(con, "neighbourhoods")}
    slots: dict[int, set[int]] = defaultdict(set)
    for r in read(con, "free_slots"):
        slots[r["member_id"]].add(r["slot"])
    answers: dict[int, dict[str, str]] = defaultdict(dict)
    for r in read(con, "intent_answers"):
        answers[r["member_id"]][r["question"]] = r["answer"]
    prompts: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for r in read(con, "prompts"):
        prompts[r["member_id"]].append((r["prompt"], r["answer"]))
    members, ids, places, paused, forgotten = [], [], [], [], []
    for k, r in enumerate(sorted(read(con, "members"), key=lambda r: r["id"])):
        mid = r["id"]
        gone = r.get("forgotten_at") is not None
        missing = [q.key for q in cfg.values if q.key not in answers[mid]]
        if missing and not gone:
            raise HandoffError(f"member {mid} has not answered: {', '.join(missing)}")
        hood = hoods[r["neighbourhood_id"]]
        seeking = {
            g
            for g, flag in (
                ("man", r["seeks_men"]),
                ("woman", r["seeks_women"]),
                ("nonbinary", r["seeks_nonbinary"]),
            )
            if flag
        }
        members.append(
            Member(
                id=k,
                gender=r["gender"],
                seeking=frozenset(seeking),
                age=age_on(date.fromisoformat(r["birth_date"]), today),
                age_min_pref=r["age_min_pref"],
                age_max_pref=r["age_max_pref"],
                lat=hood.lat,
                lon=hood.lon,
                max_km=r["max_km"],
                verified=bool(r["verified"]) and not gone,
                last_active_days=(today - local_date(r["last_active_at"], tz)).days,
                paying=bool(r["paying"]) and not gone,
                free_slots=frozenset() if gone else frozenset(slots[mid]),
                interests=(),
                bio=" ".join(text for _, text in sorted(prompts[mid])),
                values={q.key: answers[mid].get(q.key, q.options[0]) for q in cfg.values},
                accepts={q.key: frozenset(q.options) for q in cfg.values},
                dealbreakers=frozenset(),
                big5=NEUTRAL_BIG5,
                audit_group="not collected",
            )
        )
        ids.append(mid)
        places.append(hood)
        paused.append(bool(r["paused_for_review"]) or gone)
        forgotten.append(gone)
    ids_arr = np.array(ids, dtype=np.int64)
    row = {int(m): k for k, m in enumerate(ids_arr)}
    return Market(
        members=members,
        ids=ids_arr,
        row=row,
        arr=MemberArrays.from_members(members, cfg.values),
        paused=np.array(paused, dtype=bool),
        neighbourhood=places,
        calendar=calendar,
        today=calendar.day(today),
        tz=tz,
        part_starts=tuple(cfg.booking.part_starts),
        blocked=_blocked(con, row, len(members)),
        forgotten=np.array(forgotten, dtype=bool),
    )


def _blocked(con, row: dict[int, int], n: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=bool)
    for r in read(con, "blocks"):
        a, b = row.get(r["blocker_id"]), row.get(r["blocked_id"])
        if a is not None and b is not None:
            out[a, b] = out[b, a] = True
    return out


def blocked_pairs(con, market: Market) -> np.ndarray:
    """(n, n) pairs where either member blocked the other, read now (FR-LM-56): a block
    counts at once, before the next nightly refresh."""
    return _blocked(con, market.row, len(market.members))


# ---------------------------------------------------------------- nightly: tastes and scores


def calibration_answers(con, market: Market) -> dict[int, list[tuple[int, int | None, bool]]]:
    """Each member's photo-exercise answers in order: (shown, other in a pair or None, answer)."""
    out: dict[int, list] = defaultdict(list)
    for r in sorted(
        read(con, "calibration_answers"), key=lambda r: (r["member_id"], r["position"])
    ):
        b = None if r["shown_b"] is None else market.row[r["shown_b"]]
        out[market.row[r["member_id"]]].append((market.row[r["shown_a"]], b, bool(r["answer"])))
    return dict(out)


def check_photo_models(con, cfg: Config) -> None:
    """Refuse photo embeddings made by a different model than the settings name (FR-LM-62):
    re-embed every photo, then refresh with `refit_start=True`, before switching models."""
    from lastfirstdate.encoders import photo_model_id

    want = photo_model_id(cfg.models)
    found = {r["model"] for r in read(con, "photo_embeddings")}
    if found - {want}:
        others = ", ".join(sorted(found - {want}))
        raise HandoffError(f"photo embeddings from {others}; the settings name {want}")


def photo_feature_matrix(embeddings: np.ndarray, cfg: Config) -> np.ndarray:
    """Photo features from the decrypted embeddings, one row per market row (a forgotten
    member's row is all zeros and is left out of the compression's fit)."""
    raw = np.asarray(embeddings)
    if _has_photo(raw).all():
        return photo_features(ArrayPhotoEncoder(raw), len(raw), cfg)[1]
    return with_intercept(_fit_pca(raw, cfg).transform(raw))


def _has_photo(raw: np.ndarray) -> np.ndarray:
    return np.linalg.norm(raw, axis=1) > 0


def _fit_pca(raw: np.ndarray, cfg: Config) -> PCA:
    """The photo compression, fitted on members who still have a photo (FR-LM-57)."""
    has = _has_photo(raw)
    return PCA.fit(raw[has] if has.any() else raw, cfg.calibration.pca_dim)


def starting_taste(
    market: Market,
    features: np.ndarray,
    answers: dict[int, list[tuple[int, int | None, bool]]],
    cfg: Config,
) -> StartingTaste:
    """The starting taste w_0, fitted on everyone's single-photo answers."""
    g = market.arr.gender
    swipes = [
        Swipe(int(g[v]), int(g[a]), features[a], liked)
        for v, rows in answers.items()
        for a, b, liked in rows
        if b is None
    ]
    return StartingTaste.fit(swipes, cfg.calibration.w0_l2_penalty)


def fit_tastes(
    market: Market,
    features: np.ndarray,
    answers: dict[int, list[tuple[int, int | None, bool]]],
    cfg: Config,
    start: StartingTaste | None = None,
) -> dict[int, PersonalTaste]:
    """Each member's looks taste from their answers (Stages 2 and 3).

    The starting taste w_0 is fitted on everyone's single-photo answers unless given. A member
    counts as calibrated after `n_profiles - n_pairs` answers (a pair shows two photos), or
    after seeing every photo in their calibration pool.
    """
    g = market.arr.gender
    c = cfg.calibration
    if start is None:
        start = starting_taste(market, features, answers, cfg)
    tastes = {}
    for v, rows in answers.items():
        taste = PersonalTaste(int(g[v]), start, c.l2_penalty, c.n_profiles)
        seen: set[int] = set()
        for a, b, choice in rows:
            if b is None:
                taste.add_swipe(features[a], g[a], choice, refit=False)
                seen.add(a)
            else:
                taste.add_pair(features[a], features[b], g[a], choice, refit=False)
                seen |= {a, b}
        taste.refit()
        pool = photo_pool(market, cfg, v)
        taste.complete = len(rows) >= c.n_profiles - c.n_pairs or seen >= set(pool.tolist())
        if taste.complete:
            tastes[v] = taste
    return tastes


@dataclass
class NightlyScores:
    eligible: np.ndarray  # (n, n) Stage 1
    personality: np.ndarray  # (n, n) P
    looks: np.ndarray  # (n, n) V
    directional: np.ndarray  # (n, n) S
    mutual: np.ndarray  # (n, n) M, which ranks the lists


def nightly_scores(
    market: Market, tastes: dict[int, PersonalTaste], features: np.ndarray, cfg: Config
) -> NightlyScores:
    """Stages 1, 4 and 5 for everyone, with the looks tastes from `fit_tastes`."""
    arr = market.arr
    eligible = eligibility_matrix(arr, cfg, excluded=market.blocked)
    p = personality_matrix(market.members, arr, cfg)
    v = looks_scores(tastes, features, arr.gender, eligible)
    s = directional_score(v, p, eligible, cfg.mix.looks_weight)
    return NightlyScores(eligible, p, v, s, mutual_score(s, cfg.mix.chooser_weight))


# ---------------------------------------------------------------- the photo exercise, step by step


def photo_pool(market: Market, cfg: Config, v: int) -> np.ndarray:
    """Rows whose photos member v may be shown (decision D12), never someone either of them
    blocked (FR-LM-56)."""
    pool = calibration_pool(market.arr, cfg, v)
    return pool[~market.blocked[v, pool]]


def start_calibration(
    market: Market, features: np.ndarray, member_id: int, start: StartingTaste, cfg: Config
) -> CalibrationSession:
    """A new member's photo exercise (FR-LM-49): call `next_question`, show it, `answer` it."""
    v = market.row[member_id]
    c = cfg.calibration
    taste = PersonalTaste(int(market.arr.gender[v]), start, c.l2_penalty, c.n_profiles)
    pool = photo_pool(market, cfg, v)
    return CalibrationSession(taste, pool, features, market.arr.gender, cfg, pairs=c.n_pairs > 0)


def resume_calibration(
    con, market: Market, features: np.ndarray, member_id: int, start: StartingTaste, cfg: Config
) -> CalibrationSession:
    """The member's photo exercise, continued from the answers stored so far."""
    session = start_calibration(market, features, member_id, start, cfg)
    session.resume(calibration_answers(con, market).get(market.row[member_id], []))
    return session


def question_photo_ids(market: Market, q: Question) -> tuple[int, ...]:
    """The database ids of the members whose photos the question shows (one, or two)."""
    rows = (q.a,) if q.b is None else (q.a, q.b)
    return tuple(int(market.ids[r]) for r in rows)


def answer_row(
    market: Market, member_id: int, position: int, q: Question, value: bool, asked_at: datetime
) -> dict:
    """Column values for `calibration_answers` (answer 1: liked the photo, or chose the
    first photo of a pair)."""
    return {
        "member_id": member_id,
        "position": position,
        "shown_a": int(market.ids[q.a]),
        "shown_b": None if q.b is None else int(market.ids[q.b]),
        "answer": int(bool(value)),
        "asked_at": asked_at.isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- the nightly cache


class MarketCache:
    """The nightly work kept between calls (FR-LM-50).

    The photo features' compression (PCA) and the starting taste are frozen until
    `refresh(..., refit_start=True)` (for example once a week), because refitting either one
    changes every taste model. In between, a refresh refits only the members whose answers
    changed and recomputes the scores; if neither the members nor any answers changed, it
    returns the cached scores.
    """

    def __init__(self, cfg: Config, calendar: Calendar):
        self.cfg, self.calendar = cfg, calendar
        self.pca: PCA | None = None
        self.start: StartingTaste | None = None
        self.market: Market | None = None
        self.features: np.ndarray | None = None
        self.scores: NightlyScores | None = None
        self.tastes_by_id: dict[int, PersonalTaste] = {}
        self.answer_counts: dict[int, int] = {}
        self.last_refitted: list[int] = []  # database ids refitted by the last refresh

    def refresh(
        self, con, today: date, embeddings: np.ndarray, refit_start: bool = False
    ) -> NightlyScores:
        """Reload members; refit what changed; recompute the scores. `embeddings` are the
        decrypted photo embeddings, one row per member in database-id order."""
        cfg = self.cfg
        check_photo_models(con, cfg)
        market = load_market(con, cfg, self.calendar, today)
        raw = np.asarray(embeddings)
        if self.pca is None or refit_start:
            self.pca = _fit_pca(raw, cfg)
        features = with_intercept(self.pca.transform(raw))
        answers = calibration_answers(con, market)
        if self.start is None or refit_start:
            self.start = starting_taste(market, features, answers, cfg)
            self.tastes_by_id, self.answer_counts = {}, {}
        changed = {
            v: rows
            for v, rows in answers.items()
            if self.answer_counts.get(int(market.ids[v])) != len(rows)
        }
        same_members = (
            self.market is not None
            and market.ids.tolist() == self.market.ids.tolist()
            and all(
                astuple(a) == astuple(b)
                for a, b in zip(market.members, self.market.members, strict=True)
            )
            and np.array_equal(market.blocked, self.market.blocked)
        )
        if not changed and same_members and not refit_start and self.scores is not None:
            self.market, self.last_refitted = market, []
            return self.scores  # nothing changed since the last refresh
        fresh = fit_tastes(market, features, changed, cfg, start=self.start)
        for v, rows in changed.items():
            mid = int(market.ids[v])
            self.answer_counts[mid] = len(rows)
            self.tastes_by_id.pop(mid, None)
            if v in fresh:
                self.tastes_by_id[mid] = fresh[v]
        tastes = {market.row[m]: t for m, t in self.tastes_by_id.items() if m in market.row}
        self.market, self.features = market, features
        self.scores = nightly_scores(market, tastes, features, cfg)
        self.last_refitted = sorted(int(market.ids[v]) for v in changed)
        return self.scores


# ---------------------------------------------------------------- bookings, history, subscriptions


@dataclass
class Bookings:
    items: list[Booking]
    ids: list[int]  # database id of each booking


def load_bookings(con, market: Market) -> Bookings:
    """Every booking as the engine's `Booking`, with its payments and outcome."""
    venues = {r["id"]: r["name"] for r in read(con, "venues")}
    paid: dict[int, set[int]] = defaultdict(set)
    for r in read(con, "payments"):
        paid[r["booking_id"]].add(market.row[r["member_id"]])
    items, ids = [], []
    for r in sorted(read(con, "bookings"), key=lambda r: r["id"]):
        a, b = market.row[r["member_a"]], market.row[r["member_b"]]
        bk = Booking(
            a,
            b,
            market.calendar.day(r["date_on"]),
            PARTS_OF_DAY.index(r["part"]),
            source=r["source"],
            state=r["state"],
        )
        bk.paid = paid[r["id"]]
        bk.venue = venues.get(r["venue_id"])
        if r["cancelled_by"] is not None:
            bk.cancelled_by = market.row[r["cancelled_by"]]
            bk.hours_before = hours_between(
                date_start(market, r["date_on"], r["part"]),
                local_time(r["cancelled_at"], market.tz),
            )
        if r["showed_a"] is not None:
            bk.showed = {a: bool(r["showed_a"]), b: bool(r["showed_b"])}
        items.append(bk)
        ids.append(r["id"])
    return Bookings(items, ids)


def date_start(market: Market, day: date | str, part: str) -> datetime:
    """When a date window begins, in the zone's own time."""
    on = date.fromisoformat(day) if isinstance(day, str) else day
    hour = market.part_starts[PARTS_OF_DAY.index(part)]
    return datetime.combine(on, time(hour), tzinfo=market.tz)


def list_history(con, market: Market, bookings: Bookings) -> tuple[np.ndarray, np.ndarray]:
    """(last_shown, closed): the day j last appeared on i's list, and pairs already picked or
    dated (never shown again)."""
    n = len(market.members)
    last = np.full((n, n), NEVER_SHOWN, dtype=np.int64)
    closed = np.zeros((n, n), dtype=bool)
    for r in read(con, "list_shows"):
        i, j = market.row[r["chooser_id"]], market.row[r["candidate_id"]]
        last[i, j] = max(last[i, j], market.calendar.day(r["shown_on"]))
        if r["picked"]:
            closed[i, j] = closed[j, i] = True
    for a, b in dated_pairs(bookings.items):
        closed[a, b] = closed[b, a] = True
    return last, closed


def signup_days(con, market: Market, cfg: Config) -> np.ndarray:
    """(n,) the day each subscriber signed up (may be before day 0), or NO_SIGNUP. Cycles
    follow `[billing]`."""
    out = np.full(len(market.members), NO_SIGNUP, dtype=np.int64)
    for r in read(con, "subscriptions"):
        if (r["trial_days"], r["cycle_days"]) != (cfg.billing.trial_days, cfg.billing.cycle_days):
            raise HandoffError(f"subscription {r['member_id']} doesn't match [billing]")
        if r["status"] in ("trial", "active"):
            out[market.row[r["member_id"]]] = market.calendar.day(r["signup_date"])
    return out


# ---------------------------------------------------------------- capacity and windows


def _counts_against(bk: Booking, m: int, market: Market, signups: np.ndarray, cfg: Config) -> bool:
    """Does this booking use one of m's dates (their cap, week and day)?"""
    if bk.state in ("proposed", "expired"):
        return False
    if bk.state == "cancelled":
        if signups[m] == NO_SIGNUP:
            return False
        return counts_toward_guarantee(bk, m, cycle_on(m, int(signups[m]), bk.day, cfg), cfg)
    if bk.state == "no_show":
        return not bk.showed[m]  # whoever was stood up gets the slot back
    return True  # held, confirmed, completed


def _period(m: int, day: int, signups: np.ndarray, cfg: Config) -> int:
    """Cap period: a subscriber's billing cycle, otherwise blocks of cycle_days from day 0."""
    if signups[m] != NO_SIGNUP and day >= signups[m]:
        return cycle_on(m, int(signups[m]), day, cfg).start
    return (day // cfg.billing.cycle_days) * cfg.billing.cycle_days


@dataclass
class Capacity:
    busy: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    weeks: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    periods: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))


def capacity(bookings: Bookings, market: Market, signups: np.ndarray, cfg: Config) -> Capacity:
    cap = Capacity()
    for bk in bookings.items:
        for m in bk.members:
            if _counts_against(bk, m, market, signups, cfg):
                cap.busy[m].add(bk.day)
                cap.weeks[m, bk.day // 7] += 1
                cap.periods[m, _period(m, bk.day, signups, cfg)] += 1
    return cap


def _date_cap(m: int, market: Market, cfg: Config) -> int:
    d = cfg.dates
    free_capped = not market.members[m].paying and market.members[m].gender in d.free_cap_genders
    return d.cap_per_month_free if free_capped else d.cap_per_month


def free_windows(
    market: Market, cap: Capacity, signups: np.ndarray, cfg: Config, members=None
) -> np.ndarray:
    """Each member's free date windows over the booking horizon, as bits (FR-LM-37)."""
    bk = cfg.booking
    members = np.arange(len(market.members)) if members is None else np.asarray(members)
    if native.enabled():
        blocked = _blocked_days_rust(market, cap, signups, cfg, members)
    else:
        blocked = _blocked_days(market, cap, signups, cfg, members)
    return window_bits(
        market.arr.slots[members], market.today, bk.lead_days, bk.horizon_days, blocked
    )


def _blocked_days(market: Market, cap: Capacity, signups: np.ndarray, cfg: Config, members):
    """(members, horizon) days a member can't date: a date that day, a full week, or a full
    cap period. The reference loop; `_blocked_days_rust` is its twin."""
    bk = cfg.booking
    blocked = np.zeros((members.size, bk.horizon_days), dtype=bool)
    for r, m in enumerate(members):
        limit = _date_cap(int(m), market, cfg)
        for k in range(bk.horizon_days):
            day = market.today + k
            blocked[r, k] = (
                day in cap.busy[int(m)]
                or cap.weeks[int(m), day // 7] >= cfg.dates.max_per_week
                or cap.periods[int(m), _period(int(m), day, signups, cfg)] >= limit
            )
    return blocked


def _blocked_days_rust(market: Market, cap: Capacity, signups: np.ndarray, cfg: Config, members):
    """`_blocked_days` in Rust: the dates already taken go over as flat arrays."""
    pos = {int(m): r for r, m in enumerate(members)}
    busy = [(pos[m], d) for m, days in cap.busy.items() if m in pos for d in days]
    weeks = [(pos[m], w, c) for (m, w), c in cap.weeks.items() if m in pos]
    periods = [(pos[m], p, c) for (m, p), c in cap.periods.items() if m in pos]

    def cols(rows, k):
        return [np.array([row[i] for row in rows], dtype=np.int64) for i in range(k)]

    return native.core.blocked_days(
        *cols(busy, 2),
        *cols(weeks, 3),
        *cols(periods, 3),
        market.today,
        cfg.booking.horizon_days,
        cfg.dates.max_per_week,
        cfg.billing.trial_days,
        cfg.billing.cycle_days,
        np.array([_date_cap(int(m), market, cfg) for m in members], dtype=np.int64),
        np.ascontiguousarray(signups[members], dtype=np.int64),
    )


# ---------------------------------------------------------------- the daily list


@dataclass(frozen=True)
class ListEntry:
    member_id: int  # database id of the profile shown
    day: date  # the proposed date
    part: str  # lunch, evening or late
    position: int = 0  # place on her list, from 1 (FR-LM-58)
    match_value: float = 0.0  # what ordered it: the pair's mutual score times the trust factor


def trust_from_db(con, market: Market, bookings: Bookings, cfg: Config) -> TrustBook:
    """Ratings and attendance as a `TrustBook` (FR-LM-39)."""
    book = TrustBook(len(market.members), cfg)
    for r in read(con, "ratings"):
        book.add_rating(
            Rating(
                market.row[r["rater_id"]],
                market.row[r["rated_id"]],
                r["respect"],
                r["accuracy"],
                r["conversation"],
                bool(r["would_meet_again"]),
            )
        )
    late = cfg.billing.full_refund_hours
    for bk in bookings.items:
        if bk.state == "completed":
            for m in bk.members:
                book.add_attendance(m, True)
        elif bk.state == "no_show":
            for m in bk.members:
                book.add_attendance(m, bk.showed[m])
        elif bk.state == "cancelled" and bk.hours_before < late:
            book.add_attendance(bk.cancelled_by, False)
    return book


def missing_dates(
    market: Market, bookings: Bookings, signups: np.ndarray, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Each subscriber's missing dates and days left in his current cycle (FR-LM-38)."""
    g = cfg.dates.guaranteed_per_month_paying
    n = len(market.members)
    missing, days_left = np.zeros(n), np.ones(n)
    for m in np.flatnonzero(signups != NO_SIGNUP):
        if market.today < signups[m]:
            continue
        cyc = cycle_on(int(m), int(signups[m]), market.today, cfg)
        mine = [bk for bk in bookings.items if m in bk.members]
        done = delivered(mine, int(m), cyc, cfg)
        upcoming = sum(bk.state == "confirmed" and cyc.contains(bk.day) for bk in mine)
        missing[m] = max(0, g - done - upcoming)
        days_left[m] = cyc.days_left(market.today)
    return missing, days_left


def day_lists(
    con, market: Market, scores: NightlyScores, opened: list[int], cfg: Config
) -> dict[int, list[ListEntry]]:
    """Today's lists, each profile with a window both can make (FR-LM-37, Stage 6).

    `opened`: database ids of every member who can choose today, in the order they should take
    turns (the app shuffles it each day). Call it once a day for all of them: the exposure cap
    shares attention across everyone's lists, so lists built one at a time aren't fair.
    Returns lists keyed by database id.
    """
    bookings = load_bookings(con, market)
    last, closed = list_history(con, market, bookings)
    signups = signup_days(con, market, cfg)
    cap = capacity(bookings, market, signups, cfg)
    trust = trust_from_db(con, market, bookings, cfg)
    bits = free_windows(market, cap, signups, cfg)
    blocked = blocked_pairs(con, market)
    n = len(market.members)
    full = market.paused.copy()
    for m in range(n):
        full[m] |= cap.periods[m, _period(m, market.today, signups, cfg)] >= _date_cap(
            m, market, cfg
        )
    full |= ~member_ok(market.arr, cfg)
    for r in read(con, "waitlist"):  # applicants are not shown until admitted
        if r["status"] in NOT_ADMITTED and r["member_id"] in market.row:
            full[market.row[r["member_id"]]] = True
    rights = pick_rights(market.arr)
    order, avail = [], {}
    for i in market.rows(opened):
        a = available_row(
            i,
            scores.eligible,
            rights,
            last,
            market.today,
            cfg.lists.reshow_after_days,
            full,
            closed,
        )
        a &= ~blocked[i]
        cand = np.flatnonzero(a)
        if cand.size:
            a[cand[earliest_common(bits[i], bits[cand]) < 0]] = False
        if a.any():
            order.append(i)
            avail[i] = a
    if not order:
        return {}
    missing, days_left = missing_dates(market, bookings, signups, cfg)
    bonus = refill_bonus(missing, days_left, cfg.lists.guarantee_priority)
    ranked = with_trust(scores.mutual, trust.ranking_factor())
    lists, _ = fair_day_lists(
        order, avail, ranked, bonus, cfg.lists.daily_size, cfg.lists.exposure_multiple
    )
    out = {}
    for i, js in lists.items():
        entries = []
        for position, j in enumerate(js, start=1):
            day, part = window_day_part(int(earliest_common(bits[i], bits[int(j)])), market.today)
            entries.append(
                ListEntry(
                    int(market.ids[j]),
                    market.calendar.date(day),
                    PARTS_OF_DAY[part],
                    position,
                    float(ranked[i, int(j)]),
                )
            )
        out[int(market.ids[i])] = entries
    return out


def list_show_row(chooser_id: int, entry: ListEntry, shown_on: date, picked: bool = False) -> dict:
    """Column values for `list_shows`, with where the profile was shown and its match value
    (FR-LM-58)."""
    return {
        "chooser_id": chooser_id,
        "candidate_id": entry.member_id,
        "shown_on": shown_on.isoformat(),
        "picked": int(picked),
        "position": entry.position,
        "match_value": entry.match_value,
        "engine_version": __version__,
    }


# ---------------------------------------------------------------- picks and bookings


def venues_from_db(con) -> tuple[list[Place], list[int]]:
    rows = sorted(read(con, "venues"), key=lambda r: r["id"])
    rows = [r for r in rows if r["active"]]
    return [Place(r["name"], r["lat"], r["lon"]) for r in rows], [r["id"] for r in rows]


def recheck_pick(
    con,
    market: Market,
    chooser_id: int,
    chosen_id: int,
    entry: ListEntry,
    cfg: Config,
    booking_id: int | None = None,
) -> ListEntry | None:
    """Is the window still free? (FR-LM-55) Call it when she picks and again, with the
    booking's id, when he accepts, in the same transaction as the write that follows.

    Reads the bookings now. Returns `entry` if its window is still free for both and both are
    under their caps; otherwise the earliest window both can still make; or None (no window,
    a block, they already have a date together, or either can't be matched).
    """
    a, b = market.row[chooser_id], market.row[chosen_id]
    ok = member_ok(market.arr, cfg) & ~market.paused
    if not (ok[a] and ok[b]) or blocked_pairs(con, market)[a, b]:
        return None
    every = load_bookings(con, market)
    others = Bookings(
        [bk for bid, bk in zip(every.ids, every.items, strict=True) if bid != booking_id],
        [bid for bid in every.ids if bid != booking_id],
    )
    if (min(a, b), max(a, b)) in dated_pairs(others.items):
        return None
    signups = signup_days(con, market, cfg)
    cap = capacity(others, market, signups, cfg)
    for bk in others.items:  # a pending pick holds its day for both people
        if bk.state == "proposed":
            for m in bk.members:
                cap.busy[m].add(bk.day)
    bits = free_windows(market, cap, signups, cfg, members=[a, b])
    common = int(bits[0]) & int(bits[1])
    k = (market.calendar.day(entry.day) - market.today) * N_PARTS + PARTS_OF_DAY.index(entry.part)
    if 0 <= k < 64 and (common >> k) & 1:
        return entry
    w = int(earliest_common(bits[0], bits[1]))
    if w < 0:
        return None
    day, part = window_day_part(w, market.today)
    return ListEntry(
        chosen_id, market.calendar.date(day), PARTS_OF_DAY[part], entry.position, entry.match_value
    )


def upcoming_between(con, market: Market, member_a: int, member_b: int) -> list[int]:
    """Database ids of bookings still ahead between two members, for the app to cancel when
    one blocks the other (FR-LM-56)."""
    pair = {market.row[member_a], market.row[member_b]}
    every = load_bookings(con, market)
    return [
        bid
        for bid, bk in zip(every.ids, every.items, strict=True)
        if set(bk.members) == pair and bk.state in PENDING and bk.day >= market.today
    ]


def forget_statements(member_id: int, when: datetime) -> list[tuple[str, tuple]]:
    """The SQL that forgets a member under D18 (FR-LM-57), as (statement, parameters) for the
    app to run in one transaction. Delete their photo files from storage first (the `photos`
    rows give the keys); then pass a zero embedding for them each night.

    Deleted: photos, photo embeddings, prompts, intent answers, free slots, their photo-exercise
    answers and everyone's answers about their photo. Kept, under the same id: the member row
    (birth year only, marked forgotten), bookings, payments, ratings, blocks, list history.
    """
    stamp = (when if when.tzinfo else when.replace(tzinfo=UTC)).astimezone(UTC)
    at = stamp.isoformat(timespec="seconds")
    m = (member_id,)
    return [
        ("DELETE FROM photo_embeddings WHERE member_id = ?", m),
        ("DELETE FROM photos WHERE member_id = ?", m),
        ("DELETE FROM prompts WHERE member_id = ?", m),
        ("DELETE FROM intent_answers WHERE member_id = ?", m),
        ("DELETE FROM free_slots WHERE member_id = ?", m),
        (
            "DELETE FROM calibration_answers WHERE member_id = ? OR shown_a = ? OR shown_b = ?",
            m * 3,
        ),
        (
            "UPDATE subscriptions SET status = 'cancelled' "
            "WHERE member_id = ? AND status IN ('trial', 'active', 'paused')",
            m,
        ),
        ("UPDATE waitlist SET status = 'withdrawn' WHERE member_id = ? AND status = 'waiting'", m),
        (
            "UPDATE members SET birth_date = substr(birth_date, 1, 4) || '-01-01', verified = 0, "
            "income_verified = 0, paying = 0, paused_for_review = 1, forgotten_at = ? "
            "WHERE id = ?",
            (at, member_id),
        ),
    ]


def propose(
    con, market: Market, chooser_id: int, chosen_id: int, entry: ListEntry, source: str = "list"
) -> tuple[Booking, int]:
    """The booking for a pick (or an event connection), at the window shown, with the fair
    venue's database id. The app saves it with `booking_row`, then runs the payment hold."""
    places, venue_ids = venues_from_db(con)
    a, b = market.row[chooser_id], market.row[chosen_id]
    v, _, _ = venue_for(market.neighbourhood[a], market.neighbourhood[b], places)
    bk = Booking(
        a, b, market.calendar.day(entry.day), PARTS_OF_DAY.index(entry.part), source=source
    )
    bk.venue = places[v].name
    return bk, venue_ids[v]


def booking_row(
    bk: Booking,
    market: Market,
    venue_id: int | None,
    proposed_at: datetime,
    event_id: int | None = None,
) -> dict:
    """Column values for the `bookings` table."""
    keep_venue = bk.state in ("confirmed", "completed", "no_show", "cancelled")
    return {
        "member_a": int(market.ids[bk.a]),
        "member_b": int(market.ids[bk.b]),
        "source": bk.source,
        "event_id": event_id,
        "date_on": market.calendar.date(bk.day).isoformat(),
        "part": PARTS_OF_DAY[bk.part],
        "state": bk.state,
        "proposed_at": proposed_at.isoformat(timespec="seconds"),
        "venue_id": venue_id if keep_venue else None,
        "cancelled_by": None if bk.cancelled_by is None else int(market.ids[bk.cancelled_by]),
        "showed_a": None if not bk.showed else int(bk.showed[bk.a]),
        "showed_b": None if not bk.showed else int(bk.showed[bk.b]),
    }


# ---------------------------------------------------------------- weekly alerts, cycle end, gate


def weekly_alerts(con, market: Market, scores: NightlyScores, cfg: Config):
    """Paying men unlikely to end their cycle with 3 delivered dates (FR-LM-44): their
    database ids, and every man's chance as {database id: chance}."""
    bookings = load_bookings(con, market)
    signups = signup_days(con, market, cfg)
    ok = member_ok(market.arr, cfg) & ~market.paused
    men = np.flatnonzero(ok & (signups != NO_SIGNUP) & (signups <= market.today))
    women = np.flatnonzero(ok & (market.arr.gender == WOMAN))
    mf = market_features(scores.eligible, women, men)
    n = len(market.members)
    done, upcoming, left = np.zeros(n), np.zeros(n), np.ones(n)
    for m in men:
        cyc = cycle_on(int(m), int(signups[m]), market.today, cfg)
        mine = [bk for bk in bookings.items if m in bk.members]
        done[m] = delivered(mine, int(m), cyc, cfg)
        upcoming[m] = sum(bk.state == "confirmed" and cyc.contains(bk.day) for bk in mine)
        left[m] = cyc.days_left(market.today)
    flagged, p = at_risk_delivered(gate_model(cfg), mf, done, upcoming, left, cfg)
    return [int(market.ids[m]) for m in flagged], {
        int(market.ids[m]): float(x) for m, x in zip(mf.men, p, strict=True)
    }


def close_cycle(con, market: Market, member_id: int, day: date, cfg: Config) -> tuple[int, float]:
    """(delivered dates, credit in dollars) for the subscriber's cycle containing `day`."""
    bookings = load_bookings(con, market)
    signups = signup_days(con, market, cfg)
    m = market.row[member_id]
    if signups[m] == NO_SIGNUP:
        raise HandoffError(f"member {member_id} has no subscription")
    cyc = cycle_on(m, int(signups[m]), market.calendar.day(day), cfg)
    got = delivered([bk for bk in bookings.items if m in bk.members], m, cyc, cfg)
    return got, credit(max(0, cfg.dates.guaranteed_per_month_paying - got), cyc, cfg)


@dataclass
class SaleDecision:
    admit: list[int]  # database ids, in waitlist order
    deferred_small_pool: list[int]
    women_per_man: float
    forecast: float
    stopped_because: str


def gate_for_sale(con, market: Market, scores: NightlyScores, cfg: Config) -> SaleDecision:
    """How many waitlisted men to admit now (FR-LM-34, FR-LM-45), with the forecast chosen by
    `gate.forecast`."""
    ok = member_ok(market.arr, cfg) & ~market.paused
    signups = signup_days(con, market, cfg)
    women = np.flatnonzero(ok & (market.arr.gender == WOMAN))
    current = np.flatnonzero(ok & (signups != NO_SIGNUP))
    waiting = sorted(
        (r for r in read(con, "waitlist") if r["status"] == "waiting"),
        key=lambda r: (r["applied_at"], r["member_id"]),
    )
    applicants = [market.row[r["member_id"]] for r in waiting if r["member_id"] in market.row]
    d: GateDecision = gate_decision(
        gate_model(cfg), scores.eligible, women, current, applicants, cfg
    )
    return SaleDecision(
        admit=[int(market.ids[m]) for m in d.admit],
        deferred_small_pool=[int(market.ids[m]) for m in d.deferred_small_pool],
        women_per_man=d.women_per_man,
        forecast=d.forecast,
        stopped_because=d.stopped_because,
    )


def nearest_neighbourhood_id(con, lat: float, lon: float) -> int:
    """The neighbourhood a sign-up belongs to (the app stores only this, never the address)."""
    rows = sorted(read(con, "neighbourhoods"), key=lambda r: r["id"])
    places = [Place(r["name"], r["lat"], r["lon"]) for r in rows]
    return rows[int(nearest_place(lat, lon, places))]["id"]
