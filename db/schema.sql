-- lastfirstdate: the data the app keeps for matching, booking, billing, ratings, events and
-- geography (FR-LM-36). The engine's rules in src/lastfirstdate/ read and write these.
--
-- Portable SQL: runs on PostgreSQL and SQLite (the tests load it into SQLite). Times are
-- ISO-8601 text in UTC; money is in cents. On PostgreSQL, production can tighten the types
-- (timestamptz, identity columns, enum types) without changing any column.
--
-- Rules this schema keeps:
--   R2  no column anywhere holds an attractiveness score, rank or rating of looks.
--   R3  photo embeddings are biometric data: their own table, encrypted, deleted with the
--       member. Identity and income checks store the result only, never the documents.
--   R6  dealbreakers are answers here; the engine turns them into hard filters.

-- ---------------------------------------------------------------- geography (FR-LM-41)

CREATE TABLE neighbourhoods (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE,
    lat     REAL NOT NULL CHECK (lat BETWEEN -90 AND 90),
    lon     REAL NOT NULL CHECK (lon BETWEEN -180 AND 180)
);

CREATE TABLE venues (
    id               INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    neighbourhood_id INTEGER NOT NULL REFERENCES neighbourhoods (id),
    lat              REAL NOT NULL CHECK (lat BETWEEN -90 AND 90),
    lon              REAL NOT NULL CHECK (lon BETWEEN -180 AND 180),
    active           INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
);

-- ---------------------------------------------------------------- members

CREATE TABLE members (
    id                INTEGER PRIMARY KEY,
    gender            TEXT NOT NULL CHECK (gender IN ('man', 'woman', 'nonbinary')),
    seeks_men         INTEGER NOT NULL CHECK (seeks_men IN (0, 1)),
    seeks_women       INTEGER NOT NULL CHECK (seeks_women IN (0, 1)),
    seeks_nonbinary   INTEGER NOT NULL CHECK (seeks_nonbinary IN (0, 1)),
    birth_date        TEXT NOT NULL,
    age_min_pref      INTEGER NOT NULL CHECK (age_min_pref >= 18),
    age_max_pref      INTEGER NOT NULL,
    neighbourhood_id  INTEGER NOT NULL REFERENCES neighbourhoods (id),
    max_km            REAL NOT NULL CHECK (max_km > 0),
    verified          INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    income_verified   INTEGER NOT NULL DEFAULT 0 CHECK (income_verified IN (0, 1)),
    paying            INTEGER NOT NULL DEFAULT 0 CHECK (paying IN (0, 1)),
    paused_for_review INTEGER NOT NULL DEFAULT 0 CHECK (paused_for_review IN (0, 1)),
    joined_at         TEXT NOT NULL,
    last_active_at    TEXT NOT NULL,
    CHECK (age_min_pref <= age_max_pref)
);

-- Weekly free slots: slot = weekday * 3 + part (weekday 0 = Monday; part 0 lunch,
-- 1 evening, 2 late). The engine proposes date windows from these (FR-LM-37).
CREATE TABLE free_slots (
    member_id INTEGER NOT NULL REFERENCES members (id) ON DELETE CASCADE,
    slot      INTEGER NOT NULL CHECK (slot BETWEEN 0 AND 20),
    PRIMARY KEY (member_id, slot)
);

-- Intent questions: one answer each (FR-LM-30).
CREATE TABLE intent_answers (
    member_id INTEGER NOT NULL REFERENCES members (id) ON DELETE CASCADE,
    question  TEXT NOT NULL,
    answer    TEXT NOT NULL,
    PRIMARY KEY (member_id, question)
);

-- Free-text prompts (FR-LM-31).
CREATE TABLE prompts (
    member_id INTEGER NOT NULL REFERENCES members (id) ON DELETE CASCADE,
    prompt    TEXT NOT NULL,
    answer    TEXT NOT NULL CHECK (length(answer) <= 500),
    PRIMARY KEY (member_id, prompt)
);

CREATE TABLE photos (
    id          INTEGER PRIMARY KEY,
    member_id   INTEGER NOT NULL REFERENCES members (id) ON DELETE CASCADE,
    storage_key TEXT NOT NULL,
    position    INTEGER NOT NULL CHECK (position >= 0),
    UNIQUE (member_id, position)
);

-- Biometric data (R3): kept apart from the profile, encrypted, deleted with the member.
CREATE TABLE photo_embeddings (
    member_id        INTEGER PRIMARY KEY REFERENCES members (id) ON DELETE CASCADE,
    model            TEXT NOT NULL,
    encrypted_vector BYTEA NOT NULL,
    created_at       TEXT NOT NULL
);

-- The photo exercise (FR-LM-16, FR-LM-32): a single photo (shown_b NULL, answer 1 = liked)
-- or a pair (answer 1 = chose shown_a).
CREATE TABLE calibration_answers (
    member_id INTEGER NOT NULL REFERENCES members (id) ON DELETE CASCADE,
    position  INTEGER NOT NULL CHECK (position >= 0),
    shown_a   INTEGER NOT NULL REFERENCES members (id),
    shown_b   INTEGER REFERENCES members (id),
    answer    INTEGER NOT NULL CHECK (answer IN (0, 1)),
    asked_at  TEXT NOT NULL,
    PRIMARY KEY (member_id, position),
    CHECK (shown_b IS NULL OR shown_b <> shown_a)
);

-- ---------------------------------------------------------------- the men's waitlist and gate

CREATE TABLE waitlist (
    member_id  INTEGER PRIMARY KEY REFERENCES members (id),
    applied_at TEXT NOT NULL,
    status     TEXT NOT NULL CHECK (status IN ('waiting', 'admitted', 'deferred_small_pool',
                                                'withdrawn'))
);

-- Every gate decision, for audit (FR-LM-34).
CREATE TABLE gate_decisions (
    id              INTEGER PRIMARY KEY,
    decided_at      TEXT NOT NULL,
    admitted        INTEGER NOT NULL CHECK (admitted >= 0),
    deferred        INTEGER NOT NULL CHECK (deferred >= 0),
    women_per_man   REAL NOT NULL CHECK (women_per_man >= 0),
    forecast        REAL NOT NULL CHECK (forecast BETWEEN 0 AND 1),
    stopped_because TEXT NOT NULL
);

-- ---------------------------------------------------------------- billing (FR-LM-38)

CREATE TABLE subscriptions (
    member_id   INTEGER PRIMARY KEY REFERENCES members (id),
    signup_date TEXT NOT NULL,
    trial_days  INTEGER NOT NULL CHECK (trial_days >= 0),
    cycle_days  INTEGER NOT NULL CHECK (cycle_days > 0),
    price_cents INTEGER NOT NULL CHECK (price_cents >= 0),
    status      TEXT NOT NULL CHECK (status IN ('trial', 'active', 'paused', 'cancelled'))
);

CREATE TABLE cycles (
    id           INTEGER PRIMARY KEY,
    member_id    INTEGER NOT NULL REFERENCES members (id),
    cycle_index  INTEGER NOT NULL CHECK (cycle_index >= 0),
    starts_on    TEXT NOT NULL,
    ends_on      TEXT NOT NULL,
    price_cents  INTEGER NOT NULL CHECK (price_cents >= 0),
    delivered    INTEGER CHECK (delivered >= 0),
    credit_cents INTEGER CHECK (credit_cents >= 0),
    UNIQUE (member_id, cycle_index),
    CHECK (starts_on < ends_on),
    CHECK (credit_cents IS NULL OR credit_cents <= price_cents)
);

-- ---------------------------------------------------------------- events (FR-LM-40)

CREATE TABLE events (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    venue_id  INTEGER NOT NULL REFERENCES venues (id),
    starts_at TEXT NOT NULL
);

CREATE TABLE event_attendance (
    event_id  INTEGER NOT NULL REFERENCES events (id),
    member_id INTEGER NOT NULL REFERENCES members (id),
    PRIMARY KEY (event_id, member_id)
);

-- ---------------------------------------------------------------- bookings (FR-LM-37)

-- For list dates, member_a is the one who chose. States and sources match
-- src/lastfirstdate/booking.py (a test checks they stay the same).
CREATE TABLE bookings (
    id           INTEGER PRIMARY KEY,
    member_a     INTEGER NOT NULL REFERENCES members (id),
    member_b     INTEGER NOT NULL REFERENCES members (id),
    source       TEXT NOT NULL CHECK (source IN ('list', 'event')),
    event_id     INTEGER REFERENCES events (id),
    date_on      TEXT NOT NULL,
    part         TEXT NOT NULL CHECK (part IN ('lunch', 'evening', 'late')),
    state        TEXT NOT NULL CHECK (state IN ('proposed', 'held', 'confirmed', 'completed',
                                                'cancelled', 'no_show', 'expired')),
    proposed_at  TEXT NOT NULL,
    held_at      TEXT,
    confirmed_at TEXT,
    venue_id     INTEGER REFERENCES venues (id),
    cancelled_by INTEGER REFERENCES members (id),
    cancelled_at TEXT,
    showed_a     INTEGER CHECK (showed_a IN (0, 1)),
    showed_b     INTEGER CHECK (showed_b IN (0, 1)),
    CHECK (member_a <> member_b),
    CHECK (cancelled_by IS NULL OR cancelled_by IN (member_a, member_b)),
    CHECK (state NOT IN ('confirmed', 'completed', 'no_show') OR venue_id IS NOT NULL),
    CHECK ((source = 'event') = (event_id IS NOT NULL))
);

CREATE INDEX bookings_a_date ON bookings (member_a, date_on);
CREATE INDEX bookings_b_date ON bookings (member_b, date_on);

-- Each person's $10 date fee and any refund.
CREATE TABLE payments (
    id             INTEGER PRIMARY KEY,
    booking_id     INTEGER NOT NULL REFERENCES bookings (id),
    member_id      INTEGER NOT NULL REFERENCES members (id),
    amount_cents   INTEGER NOT NULL CHECK (amount_cents > 0),
    paid_at        TEXT NOT NULL,
    refunded_cents INTEGER NOT NULL DEFAULT 0
                   CHECK (refunded_cents >= 0 AND refunded_cents <= amount_cents),
    UNIQUE (booking_id, member_id)
);

-- ---------------------------------------------------------------- ratings and trust (FR-LM-39)

-- Private: never shown to the person rated. Trust reads respect and accuracy only;
-- conversation and would_meet_again feed learning (Stage 8).
CREATE TABLE ratings (
    booking_id       INTEGER NOT NULL REFERENCES bookings (id),
    rater_id         INTEGER NOT NULL REFERENCES members (id),
    rated_id         INTEGER NOT NULL REFERENCES members (id),
    respect          INTEGER NOT NULL CHECK (respect BETWEEN 1 AND 5),
    accuracy         INTEGER NOT NULL CHECK (accuracy BETWEEN 1 AND 5),
    conversation     INTEGER NOT NULL CHECK (conversation BETWEEN 1 AND 5),
    would_meet_again INTEGER NOT NULL CHECK (would_meet_again IN (0, 1)),
    rated_at         TEXT NOT NULL,
    PRIMARY KEY (booking_id, rater_id),
    CHECK (rater_id <> rated_id)
);

CREATE INDEX ratings_rated ON ratings (rated_id);

-- Members flagged by the trust score, and what a person decided.
CREATE TABLE trust_reviews (
    id         INTEGER PRIMARY KEY,
    member_id  INTEGER NOT NULL REFERENCES members (id),
    flagged_at TEXT NOT NULL,
    conduct    REAL NOT NULL CHECK (conduct BETWEEN 0 AND 1),
    n_ratings  INTEGER NOT NULL CHECK (n_ratings >= 0),
    outcome    TEXT NOT NULL CHECK (outcome IN ('open', 'cleared', 'removed'))
);
