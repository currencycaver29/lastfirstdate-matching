# Sprint 4: fair daily lists

**In one paragraph.** Sprint 4 decides **who appears on whose list each day** so attention
and dates are shared more fairly. Lists are now filled round-robin under a daily cap on how
often one person can be shown, paying members who are behind on their 3 dates get a bonus,
and the scheduler pushes harder for them near the end of the month. On the main city this
spreads attention much more evenly (inequality of exposure falls from 0.39 to 0.31 on the
Gini scale) and raises the guarantee from **78% to 82%**. It does **not** reach the 95%
target. The diagnosis is clear and important: almost every remaining miss is a paying member
whom too few people in their pool say yes to, and in a 1,000-member city no list rule can
create yeses. 95% is reachable with a business rule (offer the guarantee to members with at
least 50 possible matches gives about 96%), so that decision is yours.

**Papers used**

| Paper | What we took from it | Where in the code |
|---|---|---|
| 12. Singh and Joachims, *Fairness of Exposure in Rankings* | Exposure as the thing to share fairly; a cap so no one takes far more than a fair share | `fairness.fair_day_lists`, `fairness.exposure_cap` |
| 16. Arnosti, Johari and Kanoria, *Managing Congestion in Matching Markets* | Limits per person reduce congestion around popular members | the exposure cap and the guarantee bonus |
| 11. Tomita and Yokoyama, *Fair Reciprocal Recommendation in Matching Markets* | Nash social welfare: maximize the sum of logs of everyone's expected matches | `fairness.nsw_day_lists` |
| 13. Su, Bayoumi and Joachims, *Optimizing Rankings in Matching Markets* | Capacity matters: showing someone whose month is full wastes a pick. Tested as `hide_booked`; it backfired here (section 6) | `fairness.effectively_booked` |
| 14. Rios, Saban and Zheng, *Assortment Optimization in Dating Markets* | Hold back members who just got many matches. Tested as an experiment; off by default | `fairness.held_back`, `fairness_report.holdback_experiment` |

---

## 1. What got built

| File | What it does |
|---|---|
| `fairness.py` | Engine: guarantee bonus, "effectively booked" rule, exposure cap, round-robin fair lists, Nash social welfare lists, hold-back rule |
| `fairness_report.py` | Test side: exposure spread (Gini), audience-adjusted shown and date rates by fake group and age band, the hold-back experiment |
| `month_runs.py` | The month now builds each day's lists with the configured policy (`basic`, `fair` or `nsw`), supports a separate cap for free members, and counts any cap breach |
| `config/lastfirstdate.toml` | New `[lists]` switches, a new `[fairness]` section, and `late_month_bonus` raised from 0.2 to 1.0 |
| `tests/test_fairness.py` | 16 tests for FR-LM-21 to FR-LM-24, including the 95% check as an expected failure |

## 2. First, a diagnosis

Before writing rules, I checked **why** paying members missed under the Sprint 3 matcher (main
city, one month):

| | Paying members who reached 3 | Paying members who missed |
|---|---|---|
| How many | 243 | 66 (51 men, 15 women) |
| Median possible matches | **85** | **22** |
| Chance a chooser picks them | 23.3% | 17.9% |
| Times shown (median) | | 61.5 |
| Times picked (median) | | 2 |

And on the supply side: **227 of 349 women** hit the 4-date cap, and **185 paying members got a
4th date** while others got none.

So the misses come from three things: small pools, being picked less often, and women's
limited dates going to men who already had 3. Notice that the members who missed were
**shown** about 60 times. Showing them more is not the problem.

## 3. The fair-list rules

### Rule 1: a bonus for paying members who are behind (papers 12, 16)

```
bonus_j = guarantee_priority * missing dates_j / weeks left
```

With `guarantee_priority = 1.0`, a paying member with 0 dates gets +0.75 in week 1 and +3.0 in
week 4, added to the mutual score M (which runs from 0 to 1). It moves them up every list they
could appear on.

### Rule 2: an exposure cap, filled round-robin (paper 12, decision D7)

```
cap = ceil(exposure_multiple * list slots today / candidates today)
```

If today there are 7,200 list slots and 430 possible candidates, a fair share is about 17
appearances, and with `exposure_multiple = 1.0` nobody appears on more than 17 lists today.

Lists are filled **round-robin**: everyone's first slot, then everyone's second slot, and so
on, in a fresh random order of choosers each day. Each chooser gets her best available
candidate who hasn't hit the cap. Without round-robin, the first chooser in line would
always get the popular people and the last would get leftovers.

**Exposure** is simply how often a person appears on someone's list. Paper 12's idea is that
exposure is a resource, like money, and a ranking hands it out. A top-k list gives almost all
of it to the people everyone ranks highly.

### Rule 3: urgency in the scheduler

The scheduler's urgency bonus (Sprint 1) rose from 0.2 to 1.0 per missing date per week left.
When a woman has yeses from several men, the one still short of his guarantee now wins more
often.

### Option: Nash social welfare lists (paper 11, FR-LM-22)

Instead of round-robin, fill lists one slot at a time, each time adding the (chooser, candidate)
pair that most increases

```
sum over all people of  log(epsilon + their expected matches today)
```

where "expected matches" is the sum of their mutual scores on the lists so far. Why logs? log
grows fast at first and slowly later: going from 0 to 1 expected match adds a lot, going from
10 to 11 adds almost nothing. So the objective prefers giving a first chance to someone with
nothing over giving a tenth chance to someone popular. Nash proposed this sum-of-logs as a
fair way to divide a resource.

It's computed with a **lazy greedy**: because gains only shrink as lists fill (the objective
is concave), a stale gain can be recomputed only when it reaches the top of a priority queue.
A test checks that the fast version picks exactly what recomputing every gain each step
would pick.

## 4. Measuring fairness (FR-LM-23)

**The Gini coefficient** summarizes how unequal exposure is. Sort everyone by how often they
were shown; Gini is 0 if all are equal, and close to 1 if one person gets everything. With 4
people and one getting all the exposure, Gini = 3/4.

**Audience-adjusted shown rate.** Some members are shown less only because fewer people are
allowed to see them (their own narrow age range or short distance). So:

```
shown rate_j = times shown_j / number of choosers allowed to see j
```

Each group's average is compared with the overall average, and the same is done for dates per
member. A group below **80%** of the overall level is flagged. That threshold is borrowed from
the "four-fifths rule" used in US hiring audits.

**Gender is left out** of this audit on purpose. Women mostly choose and men are mostly chosen
(decision D1), so their exposure differs by design.

## 5. What the experiments showed

All rule choices were made on the **tuning city (seed 101)**, averaging 5 synthetic months each,
then confirmed on the main city (seed 7).

### List rules alone barely move the guarantee (tuning city)

| Setting | Guarantee |
|---|---|
| basic lists, urgency 0.2 (Sprint 3) | 82.5% ± 0.8 |
| basic lists, urgency 1.0 | **84.3% ± 0.7** |
| fair lists (cap 1x) plus bonus, urgency 0.2 | 82.3% ± 0.7 |
| fair lists (cap 1x) plus bonus, urgency 1.0 | 83.0% ± 0.9 |
| Nash welfare lists plus bonus, urgency 1.0 | 81.8% ± 0.7 |
| basic lists, urgency 3.0 or 10.0 | 84.5% ± 0.6 |

Raising urgency past 1.0 does nothing more. The list policy changes the guarantee by about a
point either way, which is within the noise.

### But fair lists spread attention much more evenly (tuning city)

| Policy | Exposure Gini | Least-shown member (times) | Guarantee |
|---|---|---|---|
| basic | 0.436 | 1.3 | 84.1% |
| fair, cap 2x | 0.422 | 1.0 | 80.4% |
| **fair, cap 1x** | **0.347** | **2.7** | 82.2% |
| Nash welfare | 0.409 | 1.3 | 81.2% |

So the chosen defaults are fair lists with a 1x cap (the most even spread) and urgency 1.0.

On the 600-member test city (not used for tuning), one month each:

| Policy | Exposure Gini | Dates | Guarantee |
|---|---|---|---|
| basic | 0.385 | 612 | 77.3% |
| fair (default) | **0.334** | **636** | **78.4%** |
| Nash welfare | 0.372 | 629 | 76.1% |

Nash welfare does spread attention at no cost in dates, as paper 11 suggests, but the simpler
round-robin cap does it better here.

### Confirmed on the main city (5 months each)

| | Sprint 3 | Sprint 4 |
|---|---|---|
| Guarantee | 78.3% ± 0.3 | **82.4% ± 0.7** |
| Exposure Gini | 0.393 | **0.309** |
| Dates booked | 1,160 | 1,176 |
| Second-date rate | 38% | 37% |
| Match quality (true appeal) | +0.84 | +0.82 |

A clear gain on both fairness and the guarantee, at a very small cost in match quality: some
attention moves from members' very best matches to people who are behind.

### The audit on the main city

The five fake groups are all within a few percent of the overall rate under both policies, so
none is flagged. One band is flagged under both: **age 41+**. They are shown at a normal rate
(1.11 of overall) but get fewer dates (0.65). People pick them less, which reflects members'
own tastes and age ranges in the fake data. There are only 10 of them, so the number is noisy,
but this is exactly the kind of outcome a real audit should surface for a human to look at.

## 6. Two intuitive rules that backfired

**Hiding members who are "effectively booked"** (their dates plus waiting yeses already fill
the month). The idea, from paper 13, is that a pick on someone who can't take another date is
wasted. In practice the guarantee fell to **72.6%** and lists shrank to 3 profiles a day. A
member can collect 4 waiting yeses within days, but at one date a week most of those are
**options**, and the scheduler needs options to fit everyone. Off by default.

**Capping dates for members who don't pay** (your question from Sprint 1):

| Free-member cap | Guarantee (tuning city) |
|---|---|
| 4 (no extra cap) | 84.3% |
| 3 | 81.4% |
| 2 | **69.3%** |
| 2, free **men** only | 82.9% |

It backfires because **70% of women don't pay**, and they're the ones whose dates paying men
need. Capping only free men (who compete with paying men) doesn't help either. So the setting
exists (`cap_per_month_free`, `free_cap_genders`) but stays at no extra cap.

**The hold-back rule (paper 14)** cost 40 dates on the main city (1,184 to 1,144). Its benefit in
the paper comes from members getting pickier after many matches, and our test harness doesn't
model that. So this test can show the cost but not the benefit. Off by default.

## 7. Why 95% isn't reached, and what would reach it

On the tuning city, with the best settings, about **47 of 49 misses** are "fewer than 3 mutual
yeses all month" and only about 2 are scheduling problems. In a 1,000-member city each member
already sees nearly everyone in their pool (Sprint 3). A paying member with 22 possible matches,
few of whom like them back, can't be given yeses by any list order.

What does move it is **who the guarantee is offered to** (main city, Sprint 4 settings, 5
months):

| Offer the guarantee to paying members with at least | Paying members eligible | Reach 3 dates |
|---|---|---|
| 10 possible matches (today's rule) | 309 | 82.4% |
| 20 | 275 | 88.6% |
| 30 | 240 | 92.8% |
| 40 | 214 | 94.5% |
| **50** | **194** | **95.8%** |

Pool size depends only on the member's own settings and location, so it is fair to use. It
would **not** be acceptable to decide eligibility by how often a member is picked: that would
be a desirability score, which your homepage promises not to have (rule R2).

Other ways to raise it: a bigger city (Sprint 1 showed 2,000 members reaching about 88% even
without these rules), prompting members with small pools to widen their age or distance
settings, or a refund or credit policy when a month falls short.

The 95% check stays in the tests as a **strict expected failure**. If the target is ever met,
the test suite turns red, so the marker can't be forgotten.

## 8. Honest limits

- The test harness decides who says yes. If real women are less picky, or real men more picky,
  the numbers move.
- Single-month differences of a point or two are noise. Every comparison above that matters
  uses 5 months.
- Paper 14's mechanism (members get pickier after matching) isn't in the test harness.
- Paper 13's full method (optimizing everyone's rankings together with explicit capacities) is
  not implemented. Its key idea was tested through `hide_booked`.

## 9. What changed in the plan

- **D7:** the exposure cap is 1x a fair share, not 2x.
- **Scheduler urgency** rose from 0.2 to 1.0.
- **New optional switches, off after testing:** `hide_booked`, `cap_per_month_free`,
  `free_cap_genders`, `holdback_yeses`, and the `nsw` policy.
- **The 95% check** moved from the month tests to `tests/test_fairness.py`, using the full
  engine, and stays a strict expected failure.
- **Gender removed** from the fairness audit groups.

All written up in `docs/SPEC.md`.

## 10. Math to study

| Topic | Where it shows up | Where to learn it |
|---|---|---|
| Logarithms and concave functions | Nash social welfare | Any calculus text; *Mathematics for Machine Learning*, section 7.3 (convexity) |
| Nash social welfare / proportional fairness | `nsw_day_lists` | Paper 11, introduction; "Nash bargaining solution" in any game theory intro |
| Greedy algorithms for concave (diminishing-returns) objectives, lazy greedy | `nsw_day_lists` | Krause and Golovin, *Submodular Function Maximization* (free survey) |
| Gini coefficient, Lorenz curve | Exposure report | Any intro economics text on inequality |
| Exposure in rankings | The cap and the audit | Paper 12, sections 1 to 3 |
| Disparate impact, four-fifths rule | The audit threshold | US EEOC Uniform Guidelines (search "four-fifths rule") |

## 11. Try it yourself

```bash
.venv/bin/python -m lastfirstdate.fairness_report            # about 1 minute
.venv/bin/python -m lastfirstdate.fairness_report --holdback 5
```

Then in `config/lastfirstdate.toml` (change them back afterwards):

- `policy = "nsw"`: Nash welfare lists. Compare the Gini.
- `exposure_multiple = 2.0`: a looser cap. Watch the Gini rise.
- `hide_booked = true`: watch the lists shrink and the guarantee fall.
- `min_pool_for_guarantee = 50` (in `[test_runs]`): the guarantee share measured the way a
  50-match rule would sell it.

## 12. Read before Sprint 5 (learning from dates)

| Paper | What to look for |
|---|---|
| 17. Liu, Mania and Jordan, *Competing Bandits in Matching Markets* | Learning preferences from noisy outcomes while matching; why you must keep exploring a little |
| 9. Joel, Eastwick and Finkel (again) | Why real date outcomes, not profiles, should retune the personality weights |

Math: the explore-exploit trade-off and multi-armed bandits (Lattimore and Szepesvári,
*Bandit Algorithms*, chapters 1 and 6, free online), and weighted logistic regression.

## 13. Questions for you

1. **The guarantee rule.** Keep it for every paying member with 10+ possible matches (about 82%
   reach it), or offer it only at 40 to 50+ possible matches (about 95%)? Or keep it broad and
   add a credit when a month falls short? This is the biggest decision from this sprint.
2. **Fairness versus best matches.** Fair lists cost a little match quality (+0.84 to +0.82) for
   much more even attention. I recommend keeping them on.
3. **The age 41+ flag.** It reflects members' own tastes in fake data, but it's the kind of result
   your lawyer should see from the real audit.
4. Still open: chooser weight (0.5 or 0.8), 50 onboarding swipes, who owns the audit tolerance,
   expected NYC member numbers, and the original pre-filter SQL query.
