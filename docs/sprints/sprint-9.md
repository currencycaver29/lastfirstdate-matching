# Sprint 9: the handoff to the app

**In one paragraph.** You're building the algorithm; someone else will build the app. Until
now the engine got its inputs from the test harness: synthetic people answered the photo exercise,
and synthetic months produced the bookings. A real app has none of that, only its database.
Sprint 9 makes the engine run **from the app's database alone**. It adds `app.py`, which reads
the tables designed in Sprint 7 and returns everything the app needs. It also writes
`docs/handoff/README.md`, the guide an app developer follows: which call happens when, what
it reads, what it returns, and what the app does next. Every setting is classified by what
changing it requires. The tests fill a database with a fake Phase 1 zone and check that the
engine on those rows gives the same answers as the engine on the original profiles, down to
the last decimal.

**Methods used.** None new: this sprint is engineering. It puts the earlier sprints' research
into a form an app can call.

---

## 1. What got built

| File | What it does |
|---|---|
| `src/lastfirstdate/app.py` | Engine side, no test harness: loads the market from rows, refits taste models from stored answers, nightly scores, daily lists with date windows, bookings, trust, weekly alerts, cycle-end credits, the gate before a sale |
| `src/lastfirstdate/app_fill.py` | Test side: writes a fake zone into the database the way the app would, including each member's photo exercise |
| `docs/handoff/README.md` | The guide for the app developer: call sequence, data contract, settings, assumptions, what's not covered yet |
| `db/schema.sql` | One new table, `list_shows` (who was shown to whom, and whether they were picked) |
| `config.py` | Every setting classified by what changing it requires (`setting_impact`, `settings_table`) |

---

## 2. How the app uses the engine, in one picture

```
sign-up ── nearest_neighbourhood_id ── photo exercise (looks.calibrate) ──▶ calibration_answers
every night ── load_market ▸ fit_tastes ▸ nightly_scores ──▶ the day's scores, in memory
every morning ── day_lists (all women at once) ──▶ each woman's list, each profile with a date
she picks ── propose ▸ Booking.hold ▸ Booking.pay ×2 ──▶ bookings + payments; venue revealed
after a date ── ratings ▸ trust_from_db ──▶ flags for review, lower ranking for low trust
every week ── weekly_alerts ──▶ men likely to miss their 3 dates
cycle end ── close_cycle ──▶ delivered dates and the credit
before a sale ── gate_for_sale ──▶ how many waitlisted men to admit
```

**Design choices:**

- **The engine only reads.** It runs plain `SELECT * FROM <table>` queries, so the same code
  works on PostgreSQL (production) and SQLite (tests). The app does all writing; the engine
  hands back objects and `booking_row` turns a booking into column values.
- **Days are counted from a Monday** (`Calendar`), because weekly free slots are numbered from
  Monday.
- **Neighbourhoods, not addresses:** every member is placed at their neighbourhood's centre.
- **It refuses what the database can't support.** If the settings ask for Big Five scores or
  interest tags, which Phase 1 doesn't collect, `load_market` stops with a clear error instead
  of quietly guessing.

---

## 3. Proving it gives the same answers

The test fills SQLite with the seed-7 Phase 1 zone: 60 men subscribed, the other men on the
waitlist, and everyone's photo exercise. Then it loads it all back:

| Check | Result |
|---|---|
| Profiles | identical, except location (neighbourhood centre) and paying (subscribers only), both by design |
| Eligibility and personality | identical to computing them on the same profiles directly |
| Taste models refitted from the stored answers | identical to the ones the photo exercise built (largest difference 7 × 10⁻¹⁶) |
| The gate | identical decision to `gate_decision` on arrays |
| A day of lists | every profile eligible, admitted, with a window both can make |
| A booking written back | passes the schema's checks; next time her week is full and the pair is closed |

The taste-model check means the nightly job can be rebuilt from the database at any time.
Nothing is lost if a server restarts.

---

## 4. Speed

| Call | Phase 1 zone |
|---|---:|
| Nightly: load, refit 383 taste models, all scores | under 1 second |
| Every morning: all 300 women's lists | 0.16 seconds |
| Weekly alerts, the gate | 2 ms each |

---

## 5. Two things found on the way

1. **Fair lists must be built together.** My first draft let the app build one woman's list
   whenever she opened the app. But the exposure cap shares attention across all of a day's
   lists. Built one at a time, a popular man could land on every list again, undoing Sprint 4.
   The handoff now says: one call each morning for every chooser.
2. **A signup before day 0.** I used −1 to mean "no subscription", but a man who signed up
   before the calendar's start also has a negative day, so he vanished from the gate and the
   alerts. A proper "no signup" marker fixed it, caught by the first end-to-end run.

---

## 6. Settings: what changing each one requires (FR-LM-48)

| Class | Examples | What to do before changing it |
|---|---|---|
| Business terms | price, date fee, refund rules, trial length, the gate's target and floor, trust flag line, venues | change any time, with legal review where money is involved |
| Test first | list size, reshow days, the 50/50 mix, personality weights, calibration length, trust weights | rerun the reports and check fulfilment |
| Refit the forecast | date caps, dates per week, booking lead and horizon, the payment hold, cycle length | `forecast_fit --delivered`, then rerun the test runs |
| Test runs only | `[test_runs]`, `[cycle_runs]`, `[population]` | nothing: the app never reads them |
| Engineering only | intent questions, Big Five, text encoder, model penalties | changes what the app collects or how models are fitted |

The full list of 119 settings is in the handoff, generated from the code; a test keeps the two
identical, so a new setting can't be added without being classified.

---

## 7. Honest limits

- **The photo exercise runs through callbacks.** `looks.calibrate` asks for each answer with a
  callback, so the app must wrap it in a session. A step-by-step API (ask for the next
  question, send an answer) would be cleaner.
- **Each call reloads the tables it needs.** That's instant for Phase 1; a city launch would
  cache the market and scores between calls.
- **The gate runs once per sale** on the current market. Re-running it every cycle as men
  leave and women join is the next step.
- **Fairness audits need group data** the schema doesn't hold (on purpose). If you want audits
  of exposure by group, collect consented group data separately.

---

## 8. What to study

1. **Reading versus writing:** why the engine only reads, and why that makes it safe to run
   anywhere.
2. **Rebuilding state from data:** taste models refitted from stored answers come out
   identical, because the fit has one optimum and the answers fully define it.
3. **Batch versus per-request:** fair sharing is a property of the whole day's lists, so it
   needs one batch call.
4. **Sentinels:** why a "missing" marker must be a value that real data can never take.

## Try it yourself

```bash
.venv/bin/python -m pytest tests/test_sprint9.py -q
```

Then read `docs/handoff/README.md` as if you were the app developer.

---

## Questions for you

1. **Who builds the app,** and when? The handoff is ready to send them.
2. **Should the photo exercise get a step-by-step API** before the app starts? It's a small
   sprint.
3. **Sprint 10:** (a) the step-by-step calibration API plus caching for scale; (b) re-running
   the gate every cycle with real data; (c) pause the engine work until the app exists and real
   Phase 1 data arrives, then refit everything on real cycles. My recommendation is (a) if the
   app is starting soon; otherwise (c).
