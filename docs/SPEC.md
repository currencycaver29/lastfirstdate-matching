# lastfirstdate matching engine: spec and sprint plan

Current sprint: 13

## Goal

After a member swipes on about 50 profiles, the engine searches the member database and
presents matches scored 50% on how they look **to that member** and 50% on personality and
interests. Scores are made mutual, daily lists are spread fairly so every paying member gets
dates, and dates are scheduled into time both people have free.

## The eight stages

| Stage | What it does | Papers (see `docs/research.md`) | Sprint |
|---|---|---|---|
| 1. Filter | Removes anyone who fails a hard rule | none, plain engineering | 1 |
| 2. Calibration profiles | Chooses the ~50 profiles a new member swipes on | 1, 2, 3 | 2 |
| 3. Photo taste | Learns how much member i likes how member j looks: V_i(j) | 3, 4, 5, 6, 7 | 2 |
| 4. Personality score | How well i and j fit on interests, values and personality: P_i(j) | 8, 9 | 1 |
| 5. Mix and make mutual | 50/50 mix S(i to j), then mutual score M(i, j) | 6, 10 | 3 |
| 6. Daily list | Up to 30 profiles a day, spread fairly | 11, 12, 13, 14, 16 | 1 (basic), 4 (fair) |
| 7. Scheduling | Weekly assignment of dates into shared free slots | 15, 16 | 1 |
| 8. Learning | Retunes everything from how real dates go | 17 | 5 |

## Rules (never broken)

- **R1 Mutual orientation lock.** j is shown to i only if j's gender is one i seeks and i's
  gender is one j seeks.
- **R2 No global attractiveness score.** Looks are only ever scored from one member's point
  of view, V_i(j). Nothing stores, ranks, logs or shows anyone by overall attractiveness.
  The starting taste w_0 (Stage 3) is a starting point under a member's own swipes, never a
  ranking on its own.
- **R3 Synthetic data only in this repo.** No real photos, profiles or embeddings. Photo
  embeddings count as biometric data: in production they are encrypted, stored apart from
  the profile and deleted when a member leaves.
- **R4 Tests run offline.** No model downloads or network calls in tests. Real models
  (CLIP, Sentence-BERT) sit behind interfaces with offline stand-ins.
- **R5 Every tunable number lives in `config/lastfirstdate.toml`.**
- **R6 Dealbreakers are hard filters**, never soft score penalties.
- **R7 Same seed, same result.** All randomness goes through a seeded generator.

## Decisions (defaults in use; change any of them)

| # | Decision | Default | Why |
|---|---|---|---|
| D1 | Who chooses | Women pick from their daily list; men accept or decline. Same-gender pairs: both get lists, whoever picks first, the other accepts. | Paper 15: letting the short side initiate can raise welfare |
| D2 | Looks vs personality | 50 / 50 (`mix.looks_weight`) | Your brief. Sprint 5 can learn a personal split. |
| D3 | Daily list size | Up to 30. Someone passed over can reappear after 7 days (`lists.reshow_after_days`). If fewer eligible people are available, show fewer; never pad with people who failed the filter. | Your brief; paper 16 on caps; Sprint 1 test run |
| D4 | Dates per month | 3 guaranteed for paying members; nobody gets more than 4 | Your brief |
| D5 | Calibration set | 50 profiles: 20 picked for variety, 30 for uncertainty | Papers 1 and 2 |
| D6 | Personality split | 0.4 interests text, 0.3 values, 0.3 Big Five | Starting guess; paper 9 says retune from outcomes |
| D7 | Exposure cap | Automatic: no one on more lists per day than a fair share (`lists.exposure_multiple` = 1.0) | Paper 12; Sprint 4 found 1x spreads attention most at no clear cost |
| D8 | Guarantee target | 95% of paying members with at least 10 possible matches reach 3 dates in the synthetic month. Checked from Sprint 4 (FR-LM-21). | Measurable version of the promise |
| D9 | Distance | Straight-line km within both people's max distance. Replaced by the original SQL query's route rule once that query is added to `db/`. | The SQL query is not in this repo yet |
| D10 | Exploration | 10% of list slots go to profiles the model is unsure about | Paper 17 |
| D11 | Test population | 1,000 synthetic NYC members, seed 7. Settings are tuned on a different fake city (seed 101) so tuning cannot flatter the results. | Fast and big enough to see effects |
| D13 | Chooser weight in M | 0.5: the plain harmonic mean, both sides equal (`mix.chooser_weight`) | Paper 10 and the research doc. Sprint 3's test run found a weight near 0.8 gives about 1 point more mutual yeses because choosers are the pickier side; kept at 0.5 until real data says otherwise (Sprint 5 can learn it) |
| D14 | Settings profiles | `config/lastfirstdate.toml` holds the research defaults the Sprint 1 to 5 results use; `config/phase1.toml` holds lastfirstdate's real Phase 1 rules from its website (every man pays, women free, 3-date cap, man-woman only, one zone, 100 men and 300 women, 30-photo calibration with pairs, single-answer intent questions, free-text prompts) | Keeps earlier results reproducible while matching the real app |
| D15 | Calibration length | Count answers, not photos: a pair question teaches about as much as one single swipe but shows two photos (Sprint 6) | Sprint 6 test run |
| D17 | His own absence | His own late cancellation (under 24 hours) or no-show counts as one of his 3 delivered dates: no credit, no refill. Her cancellation or no-show never counts against him and his slot is refilled. In either case the person who wasn't at fault gets the $10 fee back (`billing.own_absence_counts`) | The site says "no-shows and cancellations don't count"; counting his own absences as delivered stops credits being gamed. To confirm with the lawyer (risk plan) |
| D18 | Forgetting a member | When a member deletes their account, delete what describes them (photos, photo embeddings, prompts, intent answers, free slots, their photo-exercise answers and others' answers about their photo) and keep only the birth year; keep bookings, payments, ratings, blocks and list history under the same anonymous id, for accounting, other members' trust scores and safety. The member row stays, marked forgotten, and never matches again | Chosen by the owner (Sprint 12); the lawyer should confirm |
| D19 | Times | Timestamps are stored in UTC, with an offset (`Z` or `+00:00`) or without one (then read as UTC); a value with no time of day is a date in the zone's own time. Date windows (`date_on` and the part of the day) are the zone's local time (`booking.timezone`). The engine converts before comparing | One rule removes the 4 to 5 hour error behind issue #9 |
| D20 | Production models | Photos: Google SigLIP base (`google/siglip-base-patch16-224`, Apache 2.0), from a pinned ONNX export; text: `sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0, English), ONNX. Both run with ONNX Runtime on a normal server's CPU, with every file pinned by revision and sha256 (`[models]`). Tests keep offline stand-ins (R4) | Chosen by the owner (Sprint 13). Licences allow commercial use (OpenAI CLIP's model card puts deployed use out of scope); small installs, no graphics card |
| D16 | Languages | Python for the engine, the test harness and the tests; Rust (`rust/`, called through PyO3 as `lastfirstdate._core`) for the hot loops: the logistic fit, calibration's uncertainty and pair search, the fair daily lists and the ratio gate. Each Rust function has a Python twin with the same answers; `LASTFIRSTDATE_BACKEND=python` runs Python only. The scheduler stays on OR-Tools (already C++), the test harness in Python | Profiling: these loops were Python-overhead bound; NumPy and OR-Tools parts were already compiled. The owner asked for Rust where it fits |
| D12 | Calibration pool | The ~50 calibration profiles come from matchable members of a gender the member seeks and inside their age range; distance and the other filters are ignored | The goal is to learn taste, and many members' own match pools are under 50 |

## Requirements by sprint

Each requirement has an id. Every test names the id it proves, and `scripts/trace.py` fails
the checks if a requirement in a finished or current sprint has no test.

### Sprint 0: foundation

#### FR-LM-01 · Config
Sprint: 0
All weights, caps and thresholds load from `config/lastfirstdate.toml` into typed settings.
Out-of-range or misspelled values fail loudly.
Test: default config loads with the agreed defaults; bad values, missing sections and
unknown keys are rejected.

#### FR-LM-02 · Member schema
Sprint: 0
A member holds only real profile fields: gender, who they seek, age and age range, location
and max distance, verified, last active, paying, free weekly slots, interests, bio, values
answers and accepted answers, dealbreakers, Big Five, and an audit group used only for
fairness audits. No attractiveness field (R2); no photo or embedding field (R3).
Test: field names contain no score or rank words; slots round-trip.

#### FR-LM-03 · Synthetic population
Sprint: 0
Generates fake NYC members from the config, deterministic by seed (R7). Also generates
hidden ground truth (true visual taste, true looks weight, pickiness) kept apart from the
profiles, so tests can check whether the engine learns what is really there. Photo
embeddings cluster by look type; fake demographic groups lean toward look types so the
fairness audit has a bias to catch.
Test: determinism; shares match config; every field valid; clusters exist; truth is not on
profiles.

### Sprint 1: a working matcher on personality only

#### FR-LM-04 · Hard filter (Stage 1)
Sprint: 1
A pair (i, j) passes only if: both verified (when required); both active within
`filter.active_within_days`; mutual orientation lock (R1); each age inside the other's
range; distance within both people's max distance (D9); each passes the other's
dealbreakers (R6); at least `filter.min_shared_slots` shared free slots; and they have not
already dated or blocked each other. The filter is symmetric.
Test: symmetric; each rule removes its counterexample; matches a brute-force reference.

#### FR-LM-05 · Interest similarity t (Stage 4)
Sprint: 1
Cosine similarity of interest tags, weighted so rare shared interests count more than
common ones. Behind a `TextEncoder` interface so Sentence-BERT (paper 8) can replace it
later without changing callers; tests use the tag encoder only (R4).
Test: identical interests give 1, no overlap gives 0, always between 0 and 1.

#### FR-LM-06 · Values compatibility A (Stage 4)
Sprint: 1
A_ij is the share of values questions where each person's answer is one the other accepts.
Test: hand-built pairs give the expected share; symmetric.

#### FR-LM-07 · Big Five similarity p (Stage 4)
Sprint: 1
p_ij = 1 - distance between the two Big Five vectors / sqrt(5), so it runs from 0 to 1.
Test: identical gives 1, opposite corners give 0.

#### FR-LM-08 · Personality score P (Stage 4)
Sprint: 1
P_i(j) = w_text * t + w_values * A + w_big5 * p, weights from config (D6).
Test: weights applied; result between 0 and 1; symmetric.

#### FR-LM-09 · Basic daily list (Stage 6)
Sprint: 1
Each chooser (D1) gets up to `lists.daily_size` filtered candidates, ranked by score, with
fully booked members removed (D3, D4). Someone shown but not picked can reappear after
`lists.reshow_after_days` days; pairs already picked, declined or dated never reappear.
Test: size limit; best first; no filtered-out, closed or booked profiles; passed profiles
return only after the rest period.

#### FR-LM-10 · Weekly date scheduler (Stage 7)
Sprint: 1
Each week, picks which picked-and-accepted pairs get a date and in which shared slot, as a
b-matching solved by min-cost flow with OR-Tools. Each person's capacity is their remaining
dates this month; members still below their guarantee get `dates.late_month_bonus` added
late in the month.
Test: capacities respected; only allowed pairs and shared slots used; result matches brute
force on small cases.

#### FR-LM-11 · Behavior test harness
Sprint: 1
Fake members swipe, pick, accept and rate dates using only the hidden ground truth, never
the engine's scores.
Test: deterministic; members like profiles they truly prefer more often.

#### FR-LM-12 · Month test run report
Sprint: 1
Runs a synthetic month (daily lists, picks, weekly scheduling) and reports dates per paying
member, the share reaching the guarantee, candidate pool sizes, list sizes by week, and
everyone who missed with the reason.
Test (acceptance): the month runs within the caps; week 4 lists keep at least 80% of their
week 1 size; every paying member who missed is listed with a reason; the guarantee share
stays at or above the Sprint 1 baseline floor (75%). The 95% target moved to FR-LM-21.

### Sprint 2: the looks model

#### FR-LM-13 · Photo embeddings (Stage 3)
Sprint: 2
A `PhotoEncoder` interface: production uses a pretrained image model such as CLIP; tests use
the synthetic embeddings (R4). Embeddings are shrunk to `calibration.pca_dim` numbers with
PCA fitted on the population. Papers 4 to 7 all start from embeddings like these.
Test: output shape; PCA keeps most of the variance on synthetic data.

#### FR-LM-14 · Starting taste w_0 (Stage 3)
Sprint: 2
One starting taste per orientation group, fitted by logistic regression on pooled swipes.
Used only as the starting point for a member's own taste (R2). Papers 4 and 5.
Test: never exposed as a ranking; improves on a coin flip for a brand-new member.

#### FR-LM-15 · Personal taste V_i(j) (Stage 3)
Sprint: 2
V_i(j) = sigmoid((w_0 + delta_i) . x_j + b_i), where delta_i is member i's personal
correction, learned from their swipes with an L2 penalty that keeps it small until they
have swiped enough. Also reports how sure it is, for Stage 2. Papers 3, 4, 5.
Test: recovers a synthetic member's hidden taste better than w_0 alone.

#### FR-LM-16 · Calibration profiles (Stage 2)
Sprint: 2
Profiles 1 to 20: the most varied set by greedy selection for a determinantal point process
(paper 2): each next profile is the one least explained by those already chosen. Profiles
21 to 50: each next one is the profile with the most expected information,
log(1 + p(1 - p) s^2), where p is the predicted like chance and s^2 the model's
uncertainty (papers 1 and 3). The pool follows D12.
Test (acceptance): on synthetic members, taste learned from these 50 predicts held-out
likes better than taste learned from 50 random profiles (paired, by more than two standard
errors); the first 20 cover more look types than 20 random profiles.

#### FR-LM-17 · Looks report
Sprint: 2
Reports how well the looks model predicts held-out swipes, by number of swipes, and checks
it by audit group so a photo taste turning into a group filter shows up.
Test: report runs and includes the per-group breakdown.

### Sprint 3: the 50/50 mutual score

#### FR-LM-18 · Percentile scaling (Stage 5)
Sprint: 3
V and P are converted to percentiles within member i's filtered pool, so neither half
dominates just because its numbers spread wider.
Test: percentiles are uniform within a pool.

#### FR-LM-19 · 50/50 directional score S (Stage 5)
Sprint: 3
S(i to j) = looks_weight * pct(V_i(j)) + (1 - looks_weight) * pct(P_i(j)).
Test: weight applied from config.

#### FR-LM-20 · Mutual score M (Stage 5)
Sprint: 3
M(i, j) = harmonic mean of S(i to j) and S(j to i): high only when both are high. Papers 6
and 10. `mix.chooser_weight` generalizes it to a weighted harmonic mean (D13); at 0.5 it is
the plain harmonic mean and M is symmetric.
Test: symmetric at 0.5; zero if either side is zero; between the two sides. Acceptance, judged
by the test harness's hidden truth on the top 10 of each chooser's list: candidates ranked by
M are likelier to accept the chooser than candidates ranked by the chooser's one-sided S;
M gets more mutual yeses than looks alone or personality alone, and is within 1.5 points of
one-sided S.

### Sprint 4: fair daily lists

#### FR-LM-21 · Fair greedy lists (Stage 6)
Sprint: 4
Lists filled round-robin by M (plus a bonus for paying members still below their guarantee)
under an exposure cap per profile per day (D7); members with no dates left are removed.
Papers 12 and 16. The scheduler's urgency bonus also rises to 1.0 (`dates.late_month_bonus`).
Optional switches, off by default after testing: hide members whose waiting yeses fill their
month (`lists.hide_booked`), and a monthly cap for free members (`dates.cap_per_month_free`).
Test (acceptance): no one exceeds the cap; on the default city the fair policy beats Sprint 3
on the guarantee share and on exposure spread; at least `test_runs.guarantee_target` of
paying members with at least `test_runs.min_pool_for_guarantee` possible matches reach
their guarantee (D8). The last check is not met (about 82%) and is tracked as a strict
expected failure in `tests/test_fairness.py`.

#### FR-LM-22 · Nash social welfare allocation (Stage 6)
Sprint: 4
Alternative allocation (`lists.policy = "nsw"`) that greedily maximizes the sum of
log(epsilon + expected matches) over choosers and candidates (paper 11). Paper 13's point,
that a candidate's capacity limits the value of showing them, is covered by the guarantee
bonus and tested via `lists.hide_booked`.
Test: on a day with a few very popular candidates, exposure is spread more evenly than top-k
lists while keeping at least 85% of the total score; the fast lazy greedy matches a plain
greedy.

#### FR-LM-23 · Exposure and fairness report
Sprint: 4
Reports how often each profile is shown, spread measures (Gini, minimum, median), and
shown and dated rates by audit group and age band. The shown rate is adjusted for audience
(times shown per chooser allowed to see the member), so members' own settings are not
blamed on the engine. A group below `fairness.min_rate_ratio` (0.8) of the overall rate is
flagged. Gender is not audited this way, because D1 gives the sexes different roles.
Test: report flags a deliberately biased allocation and not a fair one.

#### FR-LM-24 · Hold-back experiment (Stage 6)
Sprint: 4
Optional experiment from paper 14 (`lists.holdback_yeses`): hide members who received many
mutual yeses in the last 7 days. Report only; switched off by default.
Test: experiment runs and reports the difference.

### Sprint 5: learning from dates

#### FR-LM-25 · Learn from date outcomes (Stage 8)
Sprint: 5
Records show-up, "would meet again" and a private 1 to 5 rating, and adds each held date to
both people's looks models as a weighted label (`learning.taste_date_weight`).
Test: at the chosen weight, looks accuracy does not get worse (within 0.005 AUC). The original
claim, that accuracy improves, was **not met** in test runs and is a strict expected failure:
"would meet again" also depends on personality and chemistry, and only already-liked people
get dated, so a heavy weight (3.0, as first planned) made accuracy worse. The weight is 0.5.

#### FR-LM-26 · Personal looks weight (Stage 8)
Sprint: 5
lambda_i = beta_V / (beta_V + beta_P) from a two-feature logistic regression on member i's
date outcomes and list picks and passes (with a prior that starts it at 0.5), kept between
`learning.lambda_min` and `learning.lambda_max`. List choices were added because a member has
at most about 12 dates in three months, too few to learn from alone.
Test: learned lambda correlates with synthetic members' hidden looks weight.

#### FR-LM-27 · Exploration (Stage 8)
Sprint: 5
`lists.exploration_share` of the profiles actually shown are reserved, before the rest of the
list is filled by score, for the candidates the member's looks model is least sure about,
under the same daily exposure cap. Paper 17.
Test: share respected (within 3 points) with no cap breaches. The claim that exploration
raises second dates was **not met** in test runs and is a strict expected failure.

#### FR-LM-28 · Three-month test run
Sprint: 5
Three synthetic months with learning against the same months without it: second-date rate,
guarantee share, match quality and exposure spread. Pairs who dated never meet again.
Test: month 1 is identical with and without learning; pairs never repeat; the report shows
every measure. Acceptance, "learning raises the second-date rate" (by 2 points in months 2
and 3, over 3 replicates), was **not met** and is a strict expected failure. Every requirement
has a test.

### Sprint 6: lastfirstdate's real rules, the forecast and the ratio gate

Sources: lastfirstdate's public website (home, product demo, business, investors, privacy),
read 4 November 2026; see `docs/fit-report.md`.

#### FR-LM-29 · Phase 1 settings profile
Sprint: 6
`config/phase1.toml` holds lastfirstdate's Phase 1 rules (D14). Areas, their spread, prompt
style and exact head counts move into the settings file; with the default profile the fake
population and every earlier result are unchanged.
Test: the profile loads; the population has exactly 100 men and 300 women in one zone,
man-woman only, every man paying and no woman paying, everyone verified; the default
population is unchanged.

#### FR-LM-30 · Single-answer intent questions
Sprint: 6
A values question can use mode "closeness": one answer per member, options in order,
compatibility 1 - |rank_i - rank_j| / (options - 1), and listed answer pairs that rule a pair
out for everyone (for example "Want them" with "Not for me").
Test: closeness scores as defined; incompatible pairs removed in both the fast and the
one-pair filter; "accepts" questions unchanged.

#### FR-LM-31 · Free-text prompts
Sprint: 6
`WordEncoder` turns free-text prompts into TF-IDF word vectors offline (R4), selected by
`personality.text_encoder`; Sentence-BERT replaces it in production through the same
interface.
Test: shared words raise similarity, stop words are ignored, rows have length 1; the setting
selects the encoder.

#### FR-LM-32 · Pair questions in calibration
Sprint: 6
"Who are you more drawn to?" pairs are learned as Bradley-Terry comparisons,
P(A over B) = sigmoid((w_0 + delta) . (x_A - x_B)), in the same taste model as single swipes;
each next pair is the most informative among the most uncertain profiles.
`calibration.n_pairs` sets how many.
Test: a pair adds one example on the photo difference; the test harness's pair answers follow
true looks; per answer, pairs teach about as much as single swipes (within 0.01 AUC).

#### FR-LM-33 · Guarantee forecast and at-risk alerts
Sprint: 6
Each paying man's chance of reaching his guarantee, from women per man, a shortfall term
log(min(women per man, 1)), his possible matches and his share of women's attention (start of
cycle), plus dates booked, yeses waiting and weeks left (mid-cycle at-risk alerts). Logistic
regressions fitted on synthetic markets from 0.5 to 5 women per man (`forecast_fit.py`, saved
to `config/forecast_model.toml` with that range) and refitted on real cycles later.
Test: features as defined; the model saves and loads; the committed model has sensible signs,
and on markets it never saw it ranks men well (AUC above 0.85) and forecasts the share
reaching 3 dates within 7 points.

#### FR-LM-34 · Ratio gate
Sprint: 6
Admits waitlisted men in order while women per man stays at or above `gate.min_women_per_man`
and inside the forecast's fitted range, and forecast fulfilment stays at or above
`gate.target`; defers applicants whose pool is below `gate.min_pool`.
Test: the gate stops at the ratio limit, at the end of the fitted range and at the forecast
limit, defers small pools, keeps waitlist order; in test runs, with a waitlist longer than
the women, the men it admits reach 3 dates within 5 points of the target and of the forecast,
while admitting everyone fails most of them.

#### FR-LM-35 · Rust core
Sprint: 6
The hot loops run in Rust (D16): `rust/lastfirstdate-core` holds the algorithms with no
Python dependency (so a future Rust service can reuse them), and `rust/lastfirstdate-py`
exposes them to Python. The engine uses Rust whenever the compiled module is present.
Test: on random problems the Rust and Python versions agree (logistic fit weights and
covariance to 1e-9, uncertainty, pair choice, daily lists and exposure cap, gate decisions
for every stopping rule, date windows: free windows, the first shared window and blocked
days); members see the same calibration photos, a synthetic month books the same dates, and a
day of the app's lists and a billing cycle of test runs come out the same with either backend;
the Rust crate passes its own tests and Clippy.

### Sprint 7: booking, billing, trust and geography

Sources: lastfirstdate's public website (see `docs/fit-report.md`, sections 1 and 4). No app
repo or database exists yet, so this sprint also defines the data the app must keep
(FR-LM-36) and builds the engine side of each rule against it.

#### FR-LM-36 · App data model
Sprint: 7
`db/schema.sql` holds the tables the app needs for matching, booking, billing, ratings,
events and geography, in SQL that runs on PostgreSQL and SQLite. Photo embeddings sit in
their own table so they can be encrypted and deleted alone (R3).
Test: the schema loads; it accepts valid rows and its checks reject invalid ones (a rating
outside 1 to 5, an unknown booking state, a member rating themselves); its booking states
and sources match the engine's.

#### FR-LM-37 · Proposed date windows and the payment hold
Sprint: 7
Each profile on her list comes with a proposed window: the earliest evening slot, from
`booking.lead_days` to `booking.horizon_days` ahead, that is free for both, not on a day
either already has a date, inside both people's weekly and monthly caps. Members with no
common window are not shown that day. A booking moves through proposed, held (both $10
payments must arrive within `booking.hold_minutes`), confirmed (venue revealed), then
completed, cancelled, no-show or expired; any other move is refused.
Test: the earliest common window is found, blocked days and lead time respected, none when
there is no overlap; legal state moves work, illegal ones and late payments are refused, an
unpaid hold expires and frees the slot.

#### FR-LM-38 · Billing cycles, delivered dates, refills and credits
Sprint: 7
Each paying man has his own cycle of `billing.cycle_days` starting when his trial ends. A
date counts toward his 3 only if both confirmed and paid, it is scheduled inside his cycle,
and it took place. Her cancellation or no-show does not count and his missing count goes back
up, so the list bonus refills the slot. His own late cancellation or no-show counts as
delivered when `billing.own_absence_counts` is on (D17). At cycle end he is credited
`price * missing / 3`. Date fees are refunded in full for cancellations at least
`billing.full_refund_hours` ahead, `billing.late_refund_share` later, nothing for a no-show.
Test: cycle boundaries; each counting rule; refill raises his bonus; credits and refunds as
defined.

#### FR-LM-39 · Ratings and the trust score
Sprint: 7
After a date each person privately rates the other on respectfulness, profile accuracy and
conversation (1 to 5), plus "would meet again". The trust score uses behaviour only:
respectfulness, profile accuracy and reliability (showing up, not cancelling late), each
shrunk toward `trust.prior_mean` so a few ratings cannot sink anyone. Conversation and
"would meet again" are preferences, not conduct, and go to learning (Stage 8), never to trust
(R2). Low trust lowers a member's list scores; trust below `trust.flag_below` with at least
`trust.flag_min_ratings` ratings flags the member for review and pauses their profile.
Members with no ratings get the prior, never a penalty.
Test: shrinkage and reliability as defined; conversation and "again" do not move trust; the
ranking factor is 1 at or above the prior and lower below it; flags need enough ratings.

#### FR-LM-40 · Event dates
Sprint: 7
A connection made at a members-only event can be booked as a date (`source = "event"`). It
follows the same confirm-and-pay rules, counts toward his guarantee and both people's caps,
and the pair is then never shown to each other on lists.
Test: an event date counts like a list date, respects caps and closes the pair.

#### FR-LM-41 · Neighbourhoods and venue midpoints
Sprint: 7
Members give a neighbourhood, not an address; distances use neighbourhood centres. The venue
for a date is the one that keeps the longer of the two trips shortest, ties broken by the
total trip. Neighbourhoods and venues live in the settings file.
Test: nearest neighbourhood; the venue minimises the longer trip; ties go to the shorter
total; one-neighbourhood couples get a venue in or next to it.

#### FR-LM-42 · Phase 1 cycle test run
Sprint: 7
A day-by-day test run of the real flow on the Phase 1 market: daily lists with proposed
windows, picks, payment holds, cancellations, no-shows, refills, event dates, ratings, trust
and credits, over `cycle_runs.cycles` billing cycles per man, with cycle starts all on
launch day or spread out.
Test: the books balance (every past booking ends in a final state; nobody passes a cap;
delivered dates never exceed 3 a cycle; credits match missing dates; refunds never exceed
fees); with no friction at least 5 points more men reach 3 delivered dates (10 before the
Sprint 8 correction); trust flags at
least 80% of synthetic bad actors with 6 or more ratings, at least 80% of flags are bad
actors, and trust does not track looks. Reaching `gate.target` in the real flow is tracked as
a strict expected failure (Sprint 7 finding).

### Sprint 8: the forecast and the gate on delivered dates

Sprint 6 fitted the guarantee forecast on the batch test run's *booked* dates; Sprint 7
showed the real booking flow delivers fewer. This sprint refits the forecast and the
mid-cycle alerts on the real flow, and lets the gate use them.

#### FR-LM-43 · Forecast on delivered dates
Sprint: 8
At launch, each paying man's chance of 3 delivered dates in a cycle, averaged over his first
`cycle_runs.cycles` cycles, from the same features as FR-LM-33 (women per man, his
possible matches, his share of women's attention, the shortfall below one woman per man).
Fitted on the real flow (`cycle_runs.py`) across market sizes and women per man, under the
settings' frictions and rate of new women, and saved with its fitted range and those
assumptions in `config/forecast_delivered.toml`.
Test: the model file loads and records what it assumes; on a market it never saw, the
forecast is within 7 points of the share of man-cycles that reached 3 delivered dates.

#### FR-LM-44 · Mid-cycle alerts on delivered dates
Sprint: 8
Each week of a man's cycle, his chance of ending it with 3 delivered dates, from the start
features plus dates delivered so far, confirmed dates still to come and days left in his own
cycle. Men below `gate.at_risk_threshold` are flagged.
Test: on a market it never saw, the weekly chances rank men who end the cycle with 3
delivered dates above those who don't (AUC above 0.85), and delivered dates so far raise the
chance.

#### FR-LM-45 · The gate on delivered dates
Sprint: 8
`gate.forecast` chooses the forecast the gate uses: `booked` (Sprint 6) or `delivered`
(FR-LM-43). The Phase 1 settings use `delivered`; the gate's rules and code are unchanged.
Test: the setting picks the model file and is validated. In the real flow with a long
waitlist: in a 300-woman zone, the men the booked forecast admits fall more than 5 points
short of `gate.target` and the delivered forecast admits fewer men (or none); in a 500-woman
zone, the men the delivered forecast admits reach 3 delivered dates within 5 points of the
target and of the forecast.

### Sprint 9: handoff to the app

The owner builds the algorithm; someone else builds the app. This sprint makes the engine run
from the app's database alone, not from test harness objects, and writes down exactly when the
app calls what.

#### FR-LM-46 · Call sequence
Sprint: 9
`docs/handoff/README.md` says when the app calls each engine function, what it passes and what
it gets back:
- at sign-up, calibration;
- every night, scores and taste models;
- when a woman opens the app, her list with date windows;
- when she picks, the booking and payment hold;
- after a date, ratings and trust;
- every week, delivered-date alerts;
- at cycle end, delivered dates and credits;
- before each men's sale, the gate.

Every function it names exists in `src/lastfirstdate/app.py` or the engine modules.
Test: every function the handoff names can be imported.

#### FR-LM-47 · The engine from database rows
Sprint: 9
`app.py` (engine side, no test run imports) loads members, calibration answers, list
history, bookings, ratings, subscriptions and the waitlist from the tables in `db/schema.sql`,
and returns lists with windows, bookings, trust, alerts, credits and gate decisions. Members'
locations are their neighbourhood centres. The schema gains a `list_shows` table (who was shown
to whom, and picked).
Test:
- members and taste models loaded from rows match those built directly from the same
  profiles and answers;
- a full synthetic day runs from rows alone: every listed man is eligible and has a window both
  can make, and bookings written back pass the schema's checks and block those days next time;
- the gate on rows matches the gate on arrays.

#### FR-LM-48 · Operator settings
Sprint: 9
Every setting is classified by what changing it requires: business terms (change any time),
test first, refit the forecast, test runs only, or engineering only. The handoff lists
them.
Test: every setting in `Config` is classified, and the handoff's table matches the
classification.

### Sprint 10: step-by-step calibration and caching

Two things the handoff (Sprint 9) listed as not covered yet.

#### FR-LM-49 · Step-by-step calibration
Sprint: 10
A `CalibrationSession` asks the photo exercise one question at a time: `next_question()`
returns the next single photo or pair (the same question until it is answered), `answer()`
records the member's answer, and `resume()` rebuilds a session from stored answers after an
interruption. `looks.calibrate` runs on top of it, so both always ask the same questions.
`app.start_calibration` and `app.resume_calibration` speak in database ids.
Test: with the same answers, a session asks exactly the questions `calibrate` asked before
this sprint and ends with the same taste model; a session interrupted and resumed asks the same
remaining questions; answering a finished session or answering twice is refused; the default
city's scores are unchanged.

#### FR-LM-50 · A cache for the nightly job
Sprint: 10
`app.MarketCache` keeps the market, photo features, taste models and scores between calls.
`refresh()` reloads the members, refits only the taste models of members whose answers
changed, using the cached starting taste (refitted only when asked), and recomputes the
scores.
Test: after new answers arrive, a refresh refits only those members and gives exactly the
result of a full rebuild with the same starting taste; an unchanged refresh refits nobody.

### Sprint 11: ready for real data

There is no real data yet, and this repo never holds any (R3). This sprint builds and tests
what turns Phase 1's first real cycles into better forecasts, so refitting is one tested call
when they arrive. Tests use synthetic histories written into the database as if they were
real.

#### FR-LM-51 · Outcomes from the app's records
Sprint: 11
`refit.py` rebuilds, from the app's tables, every completed paid cycle: the market at its start
(women who had joined, men who had subscribed), whether it reached 3 delivered dates, and each
week's progress (delivered so far, confirmed dates to come, days left).
Test: on a synthetic history written as app records, every cycle's outcome and every weekly
count equals what the test run recorded.

#### FR-LM-52 · Recalibrating the forecast on real cycles
Sprint: 11
With one zone, real data cannot re-learn how women per man matters, but it can correct the
forecast's level and steepness: logit(p) = a + b · logit(p_model), fitted on real cycles and
folded into the coefficients (the gate and its Rust twin need no change). The model file
records the recalibration and how many cycles it used.
Test: a = 0, b = 1 leaves the model unchanged; folding gives exactly sigmoid(a + b · logit);
when the "real" world shows up less often than the model assumed, recalibration cuts the
forecast's error on held-out cycles.

#### FR-LM-53 · When to switch to the refitted forecast
Sprint: 11
`refit.propose_refit` refuses with fewer than `refit.min_man_cycles` completed cycles; otherwise
it recalibrates on earlier cycles, compares the candidate and the current model on the latest
ones (Brier score), and recommends switching only if the candidate is better by at least
`refit.min_improvement`.
Test: too little data is refused; a worse-than-assumed world leads to a switch, recalibrated
on all the cycles.

### Sprint 12: bug fixes before Phase 1

The full review after Sprint 11 found four problems a real app would hit on day one (issues
#9 to #12) and one piece of data that can't be added after launch. This sprint fixes them;
earlier results must not move.

#### FR-LM-54 · Times in the zone's own clock
Sprint: 12
Every timestamp is read under D19 and converted to the zone's time (`booking.timezone`) before
the engine compares it with a date window or takes its date. The hour each part of the day
begins moves into the settings (`booking.part_starts`, R5). Hours before a date are measured
between true instants, so they stay right across a daylight-saving change.
Test: a cancellation 26 hours ahead, stored in UTC, counts as early and one 23 hours ahead as
late; `Z`, `+00:00` and no offset all parse; the hours are right in the week the clocks change;
a member's last-active and join dates use New York's date, not UTC's.

#### FR-LM-55 · A pick is checked again before it's booked
Sprint: 12
`app.recheck_pick` reads the current bookings when she picks, and again when he accepts. If
the window shown is still free for both and both are under their caps, it returns it; if not,
it returns the earliest window both can still make, or nothing. A pending pick holds its day
for both people. The app runs the check and the insert in one transaction.
Test: after one woman books a man's window, a second woman's pick of the same window is moved
or refused; a man at his cycle cap is refused; the booking being accepted doesn't block itself.

#### FR-LM-56 · Blocks
Sprint: 12
A `blocks` table records who blocked whom. A blocked pair, in either direction, never appears
on each other's lists, never sees each other's photo in the photo exercise, and can't be
booked together; the block takes effect at once, before the next nightly refresh.
`app.upcoming_between` lists bookings between them for the app to cancel.
Test: a blocked pair is never eligible, never listed and never in each other's photo pool;
`recheck_pick` refuses them; a block changes the cached scores; with no blocks, every earlier
result is unchanged.

#### FR-LM-57 · Forgetting a member
Sprint: 12
`app.forget_statements` returns the portable SQL that applies D18. The engine then loads a
forgotten member as permanently inactive, so past bookings still resolve, and fits the photo
compression only on members who still have a photo.
Test: a member with history can be forgotten; their profile data and others' answers about
their photo are gone, the birth year remains, bookings and payments remain; the market still
loads, and the forgotten member is never eligible, listed or shown in the photo exercise.

#### FR-LM-58 · What each list showed, where and why
Sprint: 12
Each list entry carries its position on the list and its match value (the mutual score times
the trust factor, which ordered it), and
`app.list_show_row` gives the `list_shows` row with the engine's version. Without this, Phase
1's data couldn't be used later to learn without bias or to test a new version offline.
Test: positions run 1, 2, 3 in list order with match values from the ranking; the row fits the
schema and carries the version.

### Sprint 13: the real photo and text models

The owner chose the production models (D20). This sprint plugs them in behind the interfaces
the engine already had, keeps the tests offline (R4), and checks the real models once outside
the tests. No real photos or member text enter the repo (R3).

#### FR-LM-59 · Pinned, checked model files
Sprint: 13
`[models]` names each model, its exact revision and each file's sha256.
`python -m lastfirstdate.encoders download` fetches them into `models.cache_dir` and refuses
any file whose fingerprint differs; files already in place are kept.
Test: a file with the wrong fingerprint is refused and not kept; a file already in place is
not fetched again (with a stand-in for the network).

#### FR-LM-60 · Photo embeddings from SigLIP
Sprint: 13
`encoders.PhotoModel.embed` turns one photo into numbers the way SigLIP expects: resize to
224 by 224 (bicubic), scale each colour from 0..255 to -1..1, run the vision model, and scale
its pooled output to length 1. The app stores the vector, encrypted, with
`encoders.photo_model_id` in `photo_embeddings.model`.
Test (offline, a stand-in model): the pixels follow the recipe, the vector has length 1, and
the same photo gives the same vector.

#### FR-LM-61 · Prompts read by MiniLM
Sprint: 13
With `personality.text_encoder = "sentence"`, prompts go through `encoders.SentenceEncoder`:
split into word pieces (at most `models.text_max_tokens`), run the model, average the token
vectors over the real tokens only, and scale to length 1; an empty text gives zeros, as the
word encoder does. The engine on the app's database accepts it.
Test (offline, stand-ins): padding never changes a text's vector, rows have length 1, and the
personality score uses the encoder; with "words" every earlier result is unchanged.

#### FR-LM-62 · One photo model at a time
Sprint: 13
The nightly refresh refuses to run when a stored photo embedding was made by a different model
than the settings name, so a model change means re-embedding every photo first (and refitting
the starting taste).
Test: a mismatched `photo_embeddings.model` stops the refresh with a clear error; a matching
one passes.

## Changes to this plan

- **Sprint 1.** The test run showed two problems with the plan as first written. (1) "Never
  show anyone twice in a month" emptied every list by week 4, because the typical member has
  only about 62 possible matches; passed profiles now return after 7 days (D3, FR-LM-09).
  (2) A personality-only matcher reaches about 80% on the guarantee, not 95%: most misses are
  paying members who get too few mutual yeses, which is what Sprint 4's fair lists address.
  The 95% acceptance check moved from FR-LM-12 to FR-LM-21 and is tracked as an expected
  failure until then. Details: `docs/sprints/sprint-1.md`.
- **Sprint 2.** The looks model follows the plan. Two refinements: Stage 2's first 20
  profiles use greedy DPP selection itself rather than the farthest-point stand-in, and
  profiles 21 to 50 use expected information (which accounts for the model's uncertainty)
  rather than "closest to 50%". Both choices were compared on the tuning city (seed 101);
  the differences between reasonable methods were small. Finding: choosing the 50 well
  beats choosing them at random by only about 0.004 to 0.009 AUC; most of the gain comes
  from having a personal model at all (about +0.09 AUC over w_0). Details:
  `docs/sprints/sprint-2.md`.
- **Sprint 3.** The mutual score behaves as designed (people at the top of a list are
  likelier to accept), but the original acceptance test, "more mutual likes than one side
  only", did not hold: in the test run choosers are pickier than the people they pick, so
  weighing both sides equally costs about as many of her picks as it gains in his
  acceptances (24.2% versus 25.7% mutual yeses in the top 10). The test now checks what M
  reliably does, and a chooser weight (D13) was added, kept at 0.5. A second finding: in a
  1,000-member city, members see almost everyone in their pool each month, so the ranking
  method barely changes monthly outcomes; it matters with shorter lists or bigger pools.
  Details: `docs/sprints/sprint-3.md`.
- **Sprint 4.** Fair lists spread attention much more evenly (exposure Gini 0.39 to 0.31) and,
  with a stronger scheduler urgency, raise the guarantee from about 78% to 82%, but not to 95%.
  Nearly every remaining miss is a paying member who gets fewer than 3 mutual yeses in a
  month; in a 1,000-member city members already see their whole pool, so no list rule can
  create those yeses. Two intuitive rules made things worse and are off: hiding members whose
  waiting yeses fill their month, and capping free members' dates (most free members are the
  women whose dates paying men need). Offering the guarantee only to members with at least 50
  possible matches would reach about 96%. That is a business decision, so the 95% check stays a
  strict expected failure. Details: `docs/sprints/sprint-4.md`.
- **Sprint 5.** Learning works mechanically but does not measurably raise second dates in a
  1,000-member city over three months. Each member's own looks weight is learned well
  (correlation about +0.3 to +0.4 with the hidden truth), but better ranking has little
  leverage when members see nearly their whole pool, and second dates fall about 10 points
  from month 1 to month 3 in every arm as the best pairs are used up. Date outcomes do not
  improve the looks model (a heavy weight made it worse), and exploration gave no consistent
  gain. These three claims are strict expected failures. Details: `docs/sprints/sprint-5.md`.
- **Sprint 6.** Added after reading lastfirstdate's website (`docs/fit-report.md`): a second
  settings profile with the real Phase 1 rules, and the forecast and ratio gate. Three
  findings changed the plan. (1) At a fixed 30 photos, swapping 18 single swipes for 9 pairs
  loses 0.019 AUC, but per answer pairs teach as much as singles (difference 0.000 at 21
  answers), so calibration length is counted in answers (D15). (2) The first forecast, fitted
  only on 1 to 5 women per man, said 83% where 7% of men actually reached 3 dates (0.64 women
  per man): with fewer women than men their date caps run out. A shortfall term and training
  down to 0.5 women per man fixed it on held-out markets (mean error 3.3 points), and the model
  now records its fitted range so the gate never relies on it outside that range. It is still
  too optimistic far below one woman per man in markets larger than any it was fitted on (a
  strict expected failure); this does not affect the gate, which stops long before.
  (3) The gate works: with a waitlist longer than the women, admitted men reach 3 dates 96%
  of the time (gate as set) or 92% (forecast alone) against 12.5% for admitting everyone. In a
  zone of about 290 women the 3:1 floor is cautious: the forecast alone admits about 1.8 times
  as many men and still reaches 92%. Booking, billing, trust and geography items wait for the
  app's database (Sprint 7). Details: `docs/sprints/sprint-6.md`.
- **Rust core (after Sprint 6).** At the owner's request, the loops that profiling showed
  were slowed by Python's per-call overhead moved to Rust (D16, FR-LM-35). Building every
  member's looks taste in the 1,000-member city went from 89 to 25 seconds, the daily lists
  became 10 times faster and the gate 28 to 81 times faster, with identical scores and
  bookings. Writing the twin found a weakness shared by both versions: plain Newton steps can
  overshoot and diverge on hard fitting problems (never in the test run's 9,444
  calibration fits). Both now halve a step that makes the fit worse. Details:
  `docs/sprints/rust-core.md`.
- **Sprint 7.** No app exists yet, so the sprint defined the database the app should keep
  (`db/schema.sql`) and built proposed date windows, the 15-minute payment hold, per-member
  billing cycles with refills and credits, a behaviour-only trust score, event dates and venue
  midpoints, then synthetic the real flow day by day over three cycles. The headline changed
  the business picture: in three synthetic Phase 1 markets, **about 68% of men get 3 delivered
  dates a cycle**, not the 87% to 96% of the earlier batch test runs. The earlier figure
  counted booked dates, ignored payment holds, cancellations and no-shows (about one booked
  date in three falls through), and forgot passes each month. Without new women joining it is
  58% and falls every cycle, because each woman sees every man in her pool about weekly and
  passes pile up. Show-ups are the biggest single lever (97% instead of 92%: +8 points);
  showing men less often or allowing two dates a week did not help. Credits cost about 17% of
  subscription revenue. The original acceptance check, "near the batch test run", failed and
  is kept as a strict expected failure. Writing the books-balance test also caught a
  test run bug: his own late cancellation was counted as delivered and also gave his slot
  back. Details: `docs/sprints/sprint-7.md`.
- **Sprint 8.** A second Sprint 7 bug turned up first: a no-show was never refilled (the man
  never got his slot back). Fixed; the real flow is about **82%**, not 68%, and credits about
  12%. The forecast and the mid-cycle alerts were refitted on delivered dates in the real flow
  (held-out error 3.8 points; alerts AUC 0.97 to 1.00), and the Phase 1 gate now uses them.
  The headline: **the size of the zone matters far more than women per man.** With 500 women,
  about 90% of men get 3 delivered dates even at 3:1; with 300, the 90% promise covers only a
  handful of men (the Sprint 6 gate would sell 95 there at about 75%). In large zones the gate
  is sensitive, because fulfilment is flat in the number of men, so a few points of forecast
  error move the stopping point by about 100 men. FR-LM-45's first test wording ("better than
  the booked forecast everywhere") was wrong for large zones; it now checks protection in a
  small zone and calibration in a large one. Details: `docs/sprints/sprint-8.md`.
- **Sprint 9.** The engine now runs from the app's database alone (`app.py`), and
  `docs/handoff/README.md` tells the app developer which call happens when. Loaded from rows,
  profiles, scores, taste models and gate decisions match the originals exactly; the nightly
  job takes under a second and a day's lists 0.16 seconds for Phase 1. Two findings: fair
  lists must be built in one call for all of a day's choosers (the exposure cap is shared
  across lists), and a signup before the calendar's day 0 needed a proper "no signup" marker.
  The schema gained `list_shows`; all 119 settings are classified by what a change requires.
  Details: `docs/sprints/sprint-9.md`.
- **Sprint 10.** The photo exercise now runs one question at a time (`CalibrationSession`), and
  `looks.calibrate` runs on top of it, so the app and the test runs share one
  implementation. It asks exactly what the old code asked: the default city's scores are
  byte-identical. Interrupted exercises resume where they stopped. `MarketCache` keeps the
  nightly work, refits only members whose answers changed (equal to a full rebuild with the
  same frozen photo compression and starting taste), and returns cached scores when nothing
  changed: 0.07 s instead of 0.5 s for Phase 1. Memory grows with the square of a zone's size
  (132 MB at 2,000 members), so a city should run as zones of a few thousand. Details:
  `docs/sprints/sprint-10.md`.
- **Sprint 11.** Option (b), "wait for real data", built as readiness: `refit.py` rebuilds every
  completed cycle and its weekly progress from the app's records (identical to the test run
  on all 291 cycles and 838 snapshots), recalibrates the forecast with two numbers folded into
  its coefficients (Platt scaling; one zone can't re-learn the ratio effect), and switches only
  after testing on the latest cycles. In a synthetic world with 80% show-ups instead of 92%,
  the forecast said 83% against 76% actual; recalibrated, 76%. Finding: fulfilment drifts as a
  zone ages, so a recalibration on first cycles alone did worse on later ones; recalibrate
  monthly on recent cycles. Details: `docs/sprints/sprint-11.md`.
- **Sprint 12.** Bug fixes found by the full review (issues #9 to #12), plus list logging.
  Times now follow one rule (D19): a cancellation stored in UTC no longer looks 4 to 5 hours
  later than it was, which had made every cancellation 24 to 28 hours ahead count as late,
  and a `Z` no longer crashes the engine. Picks are re-checked before booking: when every
  woman in a test zone picked her top profile at once, 160 picks would have landed on a taken
  window; with the check none double-booked. Blocks, forgetting a member (D18, chosen by the
  owner) and each list entry's position and match value were added to the schema and the engine.
  The test history written for Sprint 11 now uses UTC timestamps and still rebuilds every
  cycle identically. Details: `docs/sprints/sprint-12.md`.
- **Sprint 13.** Option (a): the owner chose Google SigLIP for photos and MiniLM for prompts
  (D20), both Apache 2.0, run with ONNX Runtime on a CPU and pinned by revision and sha256.
  The tests stay offline with stand-ins; the real models were checked once on made-up texts
  and generated images. MiniLM scores "rock climbing" and "bouldering" at 0.67 where word
  counts scored 0, and changes about 2 of each member's top 10. On the synthetic prompts it
  separates shared lifestyles only slightly better (AUC 0.888 against 0.869), because those
  prompts reuse a small vocabulary. The photo settings, tuned on synthetic embeddings, still
  need a consented pilot. Details: `docs/sprints/sprint-13.md`.
- **Rust, round 2 (v0.13.1).** The date windows got Rust twins: each member's free windows,
  the first window two members share (one instruction per pair) and the blocked-day loop in
  `app.free_windows`. Identical answers on both backends; the app's daily lists run 1.6× to
  2× faster and test runs about 1.3× faster. Details: `docs/sprints/rust-core.md`, section 10.
