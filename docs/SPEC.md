# lastfirstdate matching engine: spec and sprint plan

Current sprint: 6

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
| D3 | Daily list size | Up to 30. Someone passed over can reappear after 7 days (`lists.reshow_after_days`). If fewer eligible people are available, show fewer; never pad with people who failed the filter. | Your brief; paper 16 on caps; Sprint 1 simulation |
| D4 | Dates per month | 3 guaranteed for paying members; nobody gets more than 4 | Your brief |
| D5 | Calibration set | 50 profiles: 20 picked for variety, 30 for uncertainty | Papers 1 and 2 |
| D6 | Personality split | 0.4 interests text, 0.3 values, 0.3 Big Five | Starting guess; paper 9 says retune from outcomes |
| D7 | Exposure cap | Automatic: no one on more lists per day than a fair share (`lists.exposure_multiple` = 1.0) | Paper 12; Sprint 4 found 1x spreads attention most at no clear cost |
| D8 | Guarantee target | 95% of paying members with at least 10 possible matches reach 3 dates in the simulated month. Checked from Sprint 4 (FR-LM-21). | Measurable version of the promise |
| D9 | Distance | Straight-line km within both people's max distance. Replaced by the original SQL query's route rule once that query is added to `db/`. | The SQL query is not in this repo yet |
| D10 | Exploration | 10% of list slots go to profiles the model is unsure about | Paper 17 |
| D11 | Test population | 1,000 synthetic NYC members, seed 7. Settings are tuned on a different fake city (seed 101) so tuning cannot flatter the results. | Fast and big enough to see effects |
| D13 | Chooser weight in M | 0.5: the plain harmonic mean, both sides equal (`mix.chooser_weight`) | Paper 10 and the research doc. Sprint 3's simulation found a weight near 0.8 gives about 1 point more mutual yeses because choosers are the pickier side; kept at 0.5 until real data says otherwise (Sprint 5 can learn it) |
| D14 | Settings profiles | `config/lastfirstdate.toml` holds the research defaults the Sprint 1 to 5 results use; `config/phase1.toml` holds lastfirstdate's real Phase 1 rules from its website (every man pays, women free, 3-date cap, man-woman only, one zone, 100 men and 300 women, 30-photo calibration with pairs, single-answer intent questions, free-text prompts) | Keeps earlier results reproducible while matching the real app |
| D15 | Calibration length | Count answers, not photos: a pair question teaches about as much as one single swipe but shows two photos (Sprint 6) | Sprint 6 simulation |
| D16 | Languages | Python for the engine, the simulator and the tests; Rust (`rust/`, called through PyO3 as `lastfirstdate._core`) for the hot loops: the logistic fit, calibration's uncertainty and pair search, the fair daily lists and the ratio gate. Each Rust function has a Python twin with the same answers; `LASTFIRSTDATE_BACKEND=python` runs Python only. The scheduler stays on OR-Tools (already C++), the simulator in Python | Profiling: these loops were Python-overhead bound; NumPy and OR-Tools parts were already compiled. The owner asked for Rust where it fits |
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

#### FR-LM-11 · Behavior simulator
Sprint: 1
Fake members swipe, pick, accept and rate dates using only the hidden ground truth, never
the engine's scores.
Test: deterministic; members like profiles they truly prefer more often.

#### FR-LM-12 · Month simulation report
Sprint: 1
Runs a simulated month (daily lists, picks, weekly scheduling) and reports dates per paying
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
by the simulator's hidden truth on the top 10 of each chooser's list: candidates ranked by
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
on the guarantee share and on exposure spread; at least `simulation.guarantee_target` of
paying members with at least `simulation.min_pool_for_guarantee` possible matches reach
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
claim, that accuracy improves, was **not met** in simulation and is a strict expected failure:
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
raises second dates was **not met** in simulation and is a strict expected failure.

#### FR-LM-28 · Three-month simulation
Sprint: 5
Three simulated months with learning against the same months without it: second-date rate,
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
Test: a pair adds one example on the photo difference; the simulator's pair answers follow
true looks; per answer, pairs teach about as much as single swipes (within 0.01 AUC).

#### FR-LM-33 · Guarantee forecast and at-risk alerts
Sprint: 6
Each paying man's chance of reaching his guarantee, from women per man, a shortfall term
log(min(women per man, 1)), his possible matches and his share of women's attention (start of
cycle), plus dates booked, yeses waiting and weeks left (mid-cycle at-risk alerts). Logistic
regressions fitted on simulated markets from 0.5 to 5 women per man (`forecast_fit.py`, saved
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
limit, defers small pools, keeps waitlist order; in simulation, with a waitlist longer than
the women, the men it admits reach 3 dates within 5 points of the target and of the forecast,
while admitting everyone fails most of them.

#### FR-LM-35 · Rust core
Sprint: 6
The hot loops run in Rust (D16): `rust/lastfirstdate-core` holds the algorithms with no
Python dependency (so a future Rust service can reuse them), and `rust/lastfirstdate-py`
exposes them to Python. The engine uses Rust whenever the compiled module is present.
Test: on random problems the Rust and Python versions agree (logistic fit weights and
covariance to 1e-9, uncertainty, pair choice, daily lists and exposure cap, gate decisions
for every stopping rule); members see the same calibration photos and a simulated month
books the same dates with either backend; the Rust crate passes its own tests and Clippy.

#### Deferred to Sprint 7 (needs the app's database)
A proposed date window shown with each profile and a 15-minute payment hold; per-member
billing cycles, counting only confirmed and paid dates, refills after cancellations; the
three rating dimensions and a behavior-only trust score in ranking; dates booked at events;
neighborhood geography with venue midpoints.

## Changes to this plan

- **Sprint 1.** The simulation showed two problems with the plan as first written. (1) "Never
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
  only", did not hold: in the simulation choosers are pickier than the people they pick, so
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
  overshoot and diverge on hard fitting problems (never in the simulation's 9,444
  calibration fits). Both now halve a step that makes the fit worse. Details:
  `docs/sprints/rust-core.md`.
