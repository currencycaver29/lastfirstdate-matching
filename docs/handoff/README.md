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

## The call sequence

| When | Call | Reads (tables) | Returns | The app then |
|---|---|---|---|---|
| **Sign-up** | `app.nearest_neighbourhood_id` | neighbourhoods | a neighbourhood id | stores only the neighbourhood, never the address |
| **Photo exercise** | `looks.calibrate`, with the questions chosen by `looks.select_varied`, `looks.select_pair` and `PersonalTaste.information` | (in memory) | the next photo or pair, until done | stores each answer in `calibration_answers` |
| **Every night** | `app.load_market`, `app.photo_feature_matrix`, `app.calibration_answers`, `app.fit_tastes`, `app.nightly_scores` | members, free_slots, intent_answers, prompts, neighbourhoods, calibration_answers | a `Market` and today's `NightlyScores` | keeps both in memory for the day |
| **Every morning** | `app.day_lists`, once, for all the day's choosers | bookings, payments, list_shows, subscriptions, ratings, waitlist, venues | each woman's list: up to 30 `ListEntry` (profile id, date, part of day) | when she opens the app, shows her list; inserts a `list_shows` row for each profile |
| **She picks** | `app.propose` | venues | a `Booking` and the fair venue's id | sets `list_shows.picked = 1`; asks him |
| **He accepts** | `Booking.hold`, then `Booking.pay` for each $10 payment | (in memory) | confirmed, or expired after 15 minutes | saves `app.booking_row` and the `payments`; reveals the venue |
| **A cancellation or the date itself** | `Booking.cancel`, `Booking.finish`, `billing.fee_refunds` | (in memory) | the new state; refunds per person | updates the booking and payments |
| **After a date** | `app.trust_from_db`, then `TrustBook.flagged` and `TrustBook.ranking_factor` | ratings, bookings | who to review | sets `paused_for_review`; adds a `trust_reviews` row |
| **Every week** | `app.weekly_alerts` | bookings, subscriptions | men likely to miss their 3 dates, and every man's chance | gives them priority, an event invite or a message |
| **Cycle end** | `app.close_cycle` | bookings, subscriptions | delivered dates and the credit in dollars | writes `cycles.delivered` and `credit_cents`; credits him |
| **Before a men's sale** | `app.gate_for_sale` | waitlist, subscriptions | men to admit (in waitlist order), deferrals, the forecast and why it stopped | admits them; logs `gate_decisions` |
| **An event connection** | `app.propose` with `source="event"` | venues | a `Booking` | saves it with the `event_id`; same hold and payments |

### The photo exercise in practice

`looks.calibrate(taste, pool, features, genders, swipe, cfg, pair=pair)` asks its questions one
at a time through the two callbacks: `swipe(j)` must return whether the member liked photo
`j`, and `pair(a, b)` whether they chose `a`. In the app, run it in a session whose callbacks
wait for the member's tap. Store every answer in `calibration_answers`, in order (`shown_b` is
empty for a single photo). The nightly `app.fit_tastes` rebuilds the same taste models from
those rows: the tests check they match to the last digit.

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
- he wasn't shown to her in the last 7 days;
- they share a free window, which is the date shown with the profile.

## The data contract

| Table | Read by | Written by the app when |
|---|---|---|
| `neighbourhoods`, `venues` | `load_market`, `propose` | the zone is set up |
| `members`, `free_slots`, `intent_answers`, `prompts` | `load_market` | sign-up and profile edits |
| `photo_embeddings` | (the app decrypts them and passes the matrix to `photo_feature_matrix`) | photos are added |
| `calibration_answers` | `calibration_answers` | the photo exercise |
| `list_shows` | `day_lists` | lists are shown and picks are made |
| `bookings`, `payments` | `day_lists`, `weekly_alerts`, `close_cycle`, `trust_from_db` | picks, payments, cancellations, dates |
| `ratings`, `trust_reviews` | `trust_from_db` | after each date; reviews |
| `subscriptions`, `cycles` | `day_lists`, `weekly_alerts`, `close_cycle`, `gate_for_sale` | sales; cycle ends |
| `waitlist`, `gate_decisions` | `gate_for_sale`, `day_lists` | applications; gate runs |

## Settings

What changing each setting in production requires (FR-LM-48). This table is generated from
`config.settings_table()`, and a test keeps them identical.

<!-- settings:start -->
**Business terms: change any time (with legal review where money is involved)**

`calibration.group_share_tolerance`, `fairness.min_rate_ratio`, `gate.target`, `gate.min_women_per_man`, `gate.min_pool`, `gate.at_risk_threshold`, `billing.price_per_cycle`, `billing.trial_days`, `billing.date_fee`, `billing.full_refund_hours`, `billing.late_refund_share`, `billing.own_absence_counts`, `trust.flag_below`, `trust.flag_min_ratings`, `geo.neighbourhoods`, `geo.venues`

**Simulate first: rerun the reports and check fulfilment before changing**

`filter.require_verified`, `filter.active_within_days`, `filter.min_shared_slots`, `personality.w_text`, `personality.w_values`, `mix.looks_weight`, `mix.chooser_weight`, `calibration.n_profiles`, `calibration.n_variety`, `calibration.n_pairs`, `lists.daily_size`, `lists.reshow_after_days`, `lists.policy`, `lists.exposure_multiple`, `lists.guarantee_priority`, `lists.hide_booked`, `lists.nsw_epsilon`, `lists.holdback_yeses`, `lists.exploration_share`, `learning.months`, `learning.taste_date_weight`, `learning.lambda_date_weight`, `learning.lambda_choice_weight`, `learning.lambda_prior_slope`, `learning.lambda_prior_strength`, `learning.lambda_min`, `learning.lambda_max`, `trust.prior_mean`, `trust.prior_strength`, `trust.reliability_weight`, `trust.ranking_weight`

**Refit the forecast: python -m lastfirstdate.forecast_fit --delivered, then simulate**

`dates.guaranteed_per_month_paying`, `dates.cap_per_month`, `dates.cap_per_month_free`, `dates.free_cap_genders`, `dates.max_per_week`, `booking.lead_days`, `booking.horizon_days`, `booking.hold_minutes`, `billing.cycle_days`

**Simulation only: the app never reads it**

`simulation.seed`, `simulation.guarantee_target`, `simulation.min_pool_for_guarantee`, `simulation.open_app_rate`, `simulation.picks_per_day`, `simulation.personality_signal`, `simulation.like_threshold`, `simulation.accept_threshold`, `simulation.again_threshold`, `simulation.choice_temperature`, `simulation.show_up_rate`, `simulation.meeting_noise`, `simulation.rating_cutoffs`, `simulation.repeat_pick_factor`, `simulation.swipe_threshold`, `simulation.training_swipes`, `population.n_members`, `population.share_men`, `population.orientation`, `population.paying_share`, `population.verified_share`, `population.active_share`, `population.age_min`, `population.age_max`, `population.max_km_min`, `population.max_km_max`, `population.embedding_dim`, `population.n_look_types`, `population.n_audit_groups`, `population.interests_min`, `population.interests_max`, `population.slots_min`, `population.slots_max`, `population.dealbreaker_rate`, `population.area_jitter_deg`, `population.prompt_style`, `population.exact_counts`, `population.areas`, `cycle_simulation.cycles`, `cycle_simulation.stagger`, `cycle_simulation.pay_in_time_chooser`, `cycle_simulation.pay_in_time_chosen`, `cycle_simulation.cancel_rate`, `cycle_simulation.late_cancel_share`, `cycle_simulation.event_rate`, `cycle_simulation.new_women_per_cycle`, `cycle_simulation.bad_actor_share`, `cycle_simulation.bad_actor_conduct`, `cycle_simulation.rating_center`, `cycle_simulation.conduct_effect`, `cycle_simulation.rating_noise`, `cycle_simulation.rating_halo`

**Engineering only: changes the data the app collects or how a model is fitted**

`personality.w_big5`, `personality.text_encoder`, `calibration.pca_dim`, `calibration.l2_penalty`, `calibration.w0_l2_penalty`, `gate.forecast_l2`, `gate.forecast`, `dates.weeks_per_month`, `dates.date_value`, `dates.late_month_bonus`, `values`
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

- **A step-by-step calibration API:** today the app wraps `looks.calibrate` with callbacks.
- **Re-running the gate every cycle** with real data as men leave and women join.
- **Speed at city scale.** Each call reloads the tables it needs; that is instant for Phase 1
  (hundreds of members). A city launch would cache the market and scores between calls.
