# Handoff: plugging the matching engine into the app

For whoever builds the lastfirstdate app. The engine is a Python library (`lastfirstdate`, with
a compiled Rust core) that reads the app's database and returns decisions. The app owns the
database, the screens, payments and notifications; the engine never writes to the database
and never talks to members.

Everything below is tested on a database filled with fake members
(`tests/test_sprint9.py`, FR-LM-46 to FR-LM-48).

## Setting up

- **Install** the package (see the main README); the app imports `lastfirstdate.app`.
- **Database:** create the tables in [`db/schema.sql`](../../db/schema.sql) (PostgreSQL in
  production; the tests use SQLite). The engine only runs `SELECT * FROM <table>`, so any
  DB-API connection works.
- **Settings:** `cfg = lastfirstdate.config.load("config/phase1.toml")`. Which settings may be
  changed, and what a change requires, is listed under [Settings](#settings).
- **Calendar:** `cal = app.Calendar(epoch)`, where `epoch` is any Monday before launch. Every
  day number is counted from it, because weekly free slots are numbered from Monday.
- **Times (D19):** store timestamps (the `_at` columns) in UTC, best with an offset, for
  example `2026-11-05T23:30:00+00:00`; a timestamp without one is read as UTC. Dates and date
  windows are New York time (`[booking] timezone`). The engine converts before comparing, so a
  cancellation 26 hours ahead counts as early whatever the clocks do.

## The call sequence

| When | Call | Reads (tables) | Returns | The app then |
|---|---|---|---|---|
| **Sign-up** | `app.nearest_neighbourhood_id` | neighbourhoods | a neighbourhood id | stores only the neighbourhood, never the address |
| **Photo exercise** | `app.start_calibration` (or `app.resume_calibration` after an interruption), then `CalibrationSession.next_question` and `CalibrationSession.answer`, one question at a time | calibration_answers (to resume) | the next photo or pair (`app.question_photo_ids`), until done | shows it; saves each answer with `app.answer_row` |
| **Every night** | `MarketCache.refresh`: it calls `app.load_market`, `app.calibration_answers`, `app.fit_tastes` and `app.nightly_scores`, refitting only members whose answers changed | members, free_slots, intent_answers, prompts, neighbourhoods, calibration_answers | today's `NightlyScores` (also kept in the cache) | passes the decrypted photo embeddings; once a week passes `refit_start=True` |
| **Every morning** | `app.day_lists`, once, for all the day's choosers | bookings, payments, list_shows, subscriptions, ratings, waitlist, venues | each woman's list: up to 30 `ListEntry` (profile id, date, part of day) | when she opens the app, shows her list; inserts a `list_shows` row for each profile with `app.list_show_row` (it records the position and match value) |
| **She picks** | `app.recheck_pick`, then `app.propose` with the entry it returns, in one transaction | bookings, payments, subscriptions, blocks, venues | the window (unchanged, or the next one both can make, or None), then a `Booking` and the fair venue's id | if None, tells her he is no longer available; if the window moved, shows the new one; saves the booking with `app.booking_row`; sets `list_shows.picked = 1`; asks him |
| **He accepts** | `app.recheck_pick` with the booking's id, then `Booking.hold`, then `Booking.pay` for each $10 payment, in one transaction | bookings, payments, subscriptions, blocks | the window still free (or not), then confirmed, or expired after 15 minutes | saves `app.booking_row` and the `payments`; reveals the venue |
| **A cancellation or the date itself** | `Booking.cancel`, `Booking.finish`, `billing.fee_refunds` | (in memory) | the new state; refunds per person | updates the booking and payments |
| **After a date** | `app.trust_from_db`, then `TrustBook.flagged` and `TrustBook.ranking_factor` | ratings, bookings | who to review | sets `paused_for_review`; adds a `trust_reviews` row |
| **Every week** | `app.weekly_alerts` | bookings, subscriptions | men likely to miss their 3 dates, and every man's chance | gives them priority, an event invite or a message |
| **Cycle end** | `app.close_cycle` | bookings, subscriptions | delivered dates and the credit in dollars | writes `cycles.delivered` and `credit_cents`; credits him |
| **Before a men's sale** | `app.gate_for_sale` | waitlist, subscriptions | men to admit (in waitlist order), deferrals, the forecast and why it stopped | admits them; logs `gate_decisions` |
| **An event connection** | `app.recheck_pick`, then `app.propose` with `source="event"` | bookings, blocks, venues | a `Booking` | saves it with the `event_id`; same hold and payments |
| **A member blocks someone** | `app.upcoming_between` | bookings | bookings still ahead between them | inserts a `blocks` row; cancels those bookings. The block counts at once: lists, the photo exercise and `recheck_pick` read it immediately |
| **A member deletes their account** | `app.forget_statements` | (none) | the SQL that forgets them (decision D18) | deletes their photo files from storage; runs the statements in one transaction; from then on passes a zero embedding for them each night |
| **Every month, once real cycles exist** | `refit.propose_refit` | members, subscriptions, bookings | whether to switch to a forecast recalibrated on the app's own cycles, and the new model | if it says switch, saves the model to `config/forecast_delivered.toml` |

### The photo exercise in practice

```python
session = app.start_calibration(market, features, member_id, cache.start, cfg)
q = session.next_question()                 # the same question until it is answered
show(app.question_photo_ids(market, q))     # one photo, or two for "who are you more drawn to?"
session.answer(tapped, question=q)          # liked it, or chose the first photo
save(app.answer_row(market, member_id, position, q, tapped, now))
```

Repeat until `session.done`. If the member leaves halfway, `app.resume_calibration` rebuilds
the session from the stored answers and asks exactly the question they would have seen next.
Passing `question=q` refuses an answer to a question already answered, for example after a
double tap. The nightly refit rebuilds the same taste models from the stored rows; the tests
check they match to the last digit.

### The nightly cache

`cache = app.MarketCache(cfg, calendar)` lives in the app's server process.
`cache.refresh(con, today, embeddings)` each night:
- refits only the members whose answers changed;
- recomputes the scores;
- if nothing changed at all, returns the cached scores in a fraction of a second.

Once a week, `refresh(..., refit_start=True)` refits the photo compression and the starting
taste, and with them every taste model.

| Zone size | Full rebuild | One member's answers changed | Nothing changed | Scores in memory |
|---|---:|---:|---:|---:|
| 400 members (Phase 1) | 0.5 s | 0.23 s | 0.07 s | 5 MB |
| 2,000 members | 4.9 s | 3.1 s | 0.5 s | 132 MB |

### The daily list

`app.day_lists(con, market, scores, choosers, cfg)` builds the day's lists **in one call for
everyone**. Fair sharing caps how many lists one man appears on that day, so it needs every
chooser at once; building one woman's list at a time would let a popular man fill every list.
Pass the database ids of all women who can choose today, in the order they should take turns,
and shuffle that order every day so nobody always goes first. For Phase 1 the call takes about
0.2 seconds. A profile is shown only if:
- both people are eligible (orientation, age, distance, dealbreakers, verification);
- he isn't paused or still on the waitlist;
- they haven't already picked or dated each other;
- neither has blocked the other;
- he wasn't shown to her in the last 7 days;
- they share a free window, which is the date shown with the profile.

### Picks, blocks and leaving

- **Check a pick again before writing it.** Several women can be shown the same man for the
  same evening on one day. `app.recheck_pick` reads the bookings at that moment: it returns the
  window if it is still free for both and both are under their caps, else the earliest window
  both can still make, else None. A pending pick holds its day for both people. Run the check
  and the insert in **one transaction** (on PostgreSQL, `SERIALIZABLE` or a row lock on the
  two members), so two picks at the same instant can't both win. Check again, with the
  booking's id, when he accepts.
- **Blocks** take effect at once. Insert the `blocks` row, then cancel what
  `app.upcoming_between` returns.
- **Leaving.** `app.forget_statements` deletes what describes the member and keeps an
  anonymous record (decision D18; the lawyer should confirm it). Because it also deletes other
  members' answers about their photo, number new photo-exercise answers from the highest
  stored `position` plus one, not from a count. Those members' taste models refit that night.
- **List logs.** Save each shown profile with `app.list_show_row`: its position and match value let
  a later version learn without bias and be tested on past lists. They can't be added later
  for lists already shown.

## Refitting on real cycles

The guarantee forecast was fitted on simulated markets. Once Phase 1 has finished some
cycles, `refit.propose_refit(con, market, scores.eligible, gate_model(cfg), cfg)` decides
whether a forecast recalibrated on the app's own records is better:

- **Too little data:** with fewer than `[refit] min_man_cycles` (150) completed paid cycles,
  it says no.
- **Otherwise:** it recalibrates on the earlier cycles, tests on the latest 30%, and
  recommends switching only if the error (Brier score) drops by at least
  `[refit] min_improvement`.
- **Run it monthly.** Fulfilment drifts as a zone ages (people pass on the same profiles), so a
  recalibration goes stale. Testing on the latest cycles before switching guards against
  that.
- **What recalibration changes:** two numbers, a level and a steepness, folded into the
  coefficients. One zone can't teach the forecast how much women per man matters, but it can
  correct how optimistic it is. The model file records what was done and on how many cycles.

In a simulated test where people showed up less often than the forecast assumed (80% instead
of 92%), the forecast said 83% while 76% got their dates; the recalibrated forecast said 76%.

## The data contract

| Table | Read by | Written by the app when |
|---|---|---|
| `neighbourhoods`, `venues` | `load_market`, `propose` | the zone is set up |
| `members`, `free_slots`, `intent_answers`, `prompts` | `load_market` | sign-up and profile edits; `forget_statements` when a member deletes their account |
| `photo_embeddings` | (the app decrypts them and passes the matrix to `photo_feature_matrix`) | photos are added |
| `calibration_answers` | `calibration_answers` | the photo exercise |
| `list_shows` | `day_lists` | lists are shown (`list_show_row`) and picks are made |
| `blocks` | `load_market`, `day_lists`, `recheck_pick` | a member blocks someone |
| `bookings`, `payments` | `day_lists`, `weekly_alerts`, `close_cycle`, `trust_from_db` | picks, payments, cancellations, dates |
| `ratings`, `trust_reviews` | `trust_from_db` | after each date; reviews |
| `subscriptions`, `cycles` | `day_lists`, `weekly_alerts`, `close_cycle`, `gate_for_sale` | sales; cycle ends |
| `waitlist`, `gate_decisions` | `gate_for_sale`, `day_lists` | applications; gate runs |

## Settings

What changing each setting in production requires (FR-LM-48). This table is generated from
`config.settings_table()`, and a test keeps them identical.

<!-- settings:start -->
**Business terms: change any time (with legal review where money is involved)**

`calibration.group_share_tolerance`, `fairness.min_rate_ratio`, `gate.target`, `gate.min_women_per_man`, `gate.min_pool`, `gate.at_risk_threshold`, `booking.part_starts`, `billing.price_per_cycle`, `billing.trial_days`, `billing.date_fee`, `billing.full_refund_hours`, `billing.late_refund_share`, `billing.own_absence_counts`, `trust.flag_below`, `trust.flag_min_ratings`, `geo.neighbourhoods`, `geo.venues`

**Simulate first: rerun the reports and check fulfilment before changing**

`filter.require_verified`, `filter.active_within_days`, `filter.min_shared_slots`, `personality.w_text`, `personality.w_values`, `mix.looks_weight`, `mix.chooser_weight`, `calibration.n_profiles`, `calibration.n_variety`, `calibration.n_pairs`, `lists.daily_size`, `lists.reshow_after_days`, `lists.policy`, `lists.exposure_multiple`, `lists.guarantee_priority`, `lists.hide_booked`, `lists.nsw_epsilon`, `lists.holdback_yeses`, `lists.exploration_share`, `learning.months`, `learning.taste_date_weight`, `learning.lambda_date_weight`, `learning.lambda_choice_weight`, `learning.lambda_prior_slope`, `learning.lambda_prior_strength`, `learning.lambda_min`, `learning.lambda_max`, `trust.prior_mean`, `trust.prior_strength`, `trust.reliability_weight`, `trust.ranking_weight`

**Refit the forecast: python -m lastfirstdate.forecast_fit --delivered, then simulate**

`dates.guaranteed_per_month_paying`, `dates.cap_per_month`, `dates.cap_per_month_free`, `dates.free_cap_genders`, `dates.max_per_week`, `booking.lead_days`, `booking.horizon_days`, `booking.hold_minutes`, `billing.cycle_days`

**Simulation only: the app never reads it**

`simulation.seed`, `simulation.guarantee_target`, `simulation.min_pool_for_guarantee`, `simulation.open_app_rate`, `simulation.picks_per_day`, `simulation.personality_signal`, `simulation.like_threshold`, `simulation.accept_threshold`, `simulation.again_threshold`, `simulation.choice_temperature`, `simulation.show_up_rate`, `simulation.meeting_noise`, `simulation.rating_cutoffs`, `simulation.repeat_pick_factor`, `simulation.swipe_threshold`, `simulation.training_swipes`, `population.n_members`, `population.share_men`, `population.orientation`, `population.paying_share`, `population.verified_share`, `population.active_share`, `population.age_min`, `population.age_max`, `population.max_km_min`, `population.max_km_max`, `population.embedding_dim`, `population.n_look_types`, `population.n_audit_groups`, `population.interests_min`, `population.interests_max`, `population.slots_min`, `population.slots_max`, `population.dealbreaker_rate`, `population.area_jitter_deg`, `population.prompt_style`, `population.exact_counts`, `population.areas`, `cycle_simulation.cycles`, `cycle_simulation.stagger`, `cycle_simulation.pay_in_time_chooser`, `cycle_simulation.pay_in_time_chosen`, `cycle_simulation.cancel_rate`, `cycle_simulation.late_cancel_share`, `cycle_simulation.event_rate`, `cycle_simulation.new_women_per_cycle`, `cycle_simulation.bad_actor_share`, `cycle_simulation.bad_actor_conduct`, `cycle_simulation.rating_center`, `cycle_simulation.conduct_effect`, `cycle_simulation.rating_noise`, `cycle_simulation.rating_halo`

**Engineering only: changes the data the app collects or how a model is fitted**

`personality.w_big5`, `personality.text_encoder`, `calibration.pca_dim`, `calibration.l2_penalty`, `calibration.w0_l2_penalty`, `gate.forecast_l2`, `gate.forecast`, `dates.weeks_per_month`, `dates.date_value`, `dates.late_month_bonus`, `values`, `booking.timezone`, `refit.min_man_cycles`, `refit.holdout_share`, `refit.min_improvement`
<!-- settings:end -->

## Assumptions the app must keep

- **Rules R1 to R7** (`docs/SPEC.md`). In particular, nothing anywhere stores or shows an
  attractiveness score, and photo embeddings stay encrypted and separate.
- **Locations:** members are placed at their neighbourhood's centre.
- **Data Phase 1 doesn't collect:**
  - no Big Five scores (`personality.w_big5` must stay 0);
  - prompts rather than interest tags;
  - one answer per intent question.

  `load_market` refuses settings that need data the database doesn't hold.
- **Billing:** subscriptions follow `[billing]` (a 30-day trial, then 28-day cycles).
  `signup_days` refuses rows that differ.
- **Open decisions:** his own late cancellation or no-show counts as delivered (D17, pending
  the lawyer); declined picks; the gate's target. See `docs/risk-plan.md`.

## Not covered yet

- **Re-running the gate every cycle** with real data as men leave and women join.
- **City scale:** the scores are five members-by-members grids, so memory grows with the
  square of a zone's size (about 3.3 GB at 10,000 members). Run a city as separate zones of a
  few thousand members, each with its own `MarketCache`.
