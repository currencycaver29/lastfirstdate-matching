# Sprint 5: learning from dates

**In one paragraph.** Sprint 5 closes the loop: after each month, the engine looks at what
actually happened (who picked whom, which dates went well) and updates itself. It learns each
member's own balance between looks and personality, adds date outcomes to their looks model,
and reserves 10% of every list for profiles it is unsure about so it keeps learning. Over three
synthetic months, the personal balance is learned well. But **learning does not measurably raise
second dates** in a 1,000-member city, and two parts of the research doc's plan didn't work as
expected: date outcomes don't improve the looks model (counting them heavily made it worse), and
exploration gave no consistent gain. The biggest effect in the test run is something else
entirely: second dates fall about 10 points from month 1 to month 3 in every version, because
the best pairs get used up.

**Ideas used**

| Idea | What we took from it | Where in the code |
|---|---|---|
| Keep exploring | Explore a little while matching, because you only learn about options you try | `learning.explore_picks` |
| Attraction is hard to predict from profiles | Learn the weights from real outcomes instead of fixing them | `learning.looks_weights` |

---

## 1. What got built

| File | What it does |
|---|---|
| `learning.py` | Engine: date records, adding date outcomes to looks models, collecting evidence, learning each member's looks weight, personal 50/50 scores, uncertainty, exploration slots |
| `learning_report.py` | Test side: three months, three versions, replicates, and what the engine learned |
| `month_runs.py` | The month can now continue with the same people (`month`), skip pairs who already dated (`excluded`), reserve exploration slots (`explore`), and record every list choice and date outcome (`record`). With none of these, it is identical to Sprint 4, checked line by line |
| `member_behavior.py` | New: after a date, each person says whether they'd meet again and gives a private 1 to 5 rating |
| `looks.py` | Labels can now carry weights, so a date outcome can count differently from a swipe |
| `fairness.py` | Fair lists accept reserved slots, so exploration counts toward the daily cap |
| `tests/test_learning.py` | 22 tests for FR-LM-25 to FR-LM-28, including 3 strict expected failures |

## 2. How three months work

```
month 1: Sprint 4 engine; record every list choice and every date outcome
  learn: update looks models from dates; learn each member's looks weight
month 2: rebuild scores with what was learned; pairs who dated never meet again
  learn again
month 3: the same
```

Three versions run over the same city, the same people and the same day-to-day randomness:

1. **no learning:** the Sprint 4 engine all along;
2. **learning;**
3. **learning plus exploration.**

Because learning only starts after month 1, versions 1 and 2 must produce **identical** first
months. A test checks this, and it holds.

## 3. What happens after a date

The test harness decides each person's reaction from their true appeal plus a random shift for
what you learn by meeting:

```
appeal after meeting = U_ij + noise
would meet again?    = yes with chance sigmoid((appeal - 0.5) / 0.5)
rating (1 to 5)      = steps up at -1.0, -0.25, 0.5 and 1.25 on that same appeal
```

The engine records it as a `DateRecord`: who, which week, whether both showed up, each
person's "would meet again" and each person's rating.

## 4. Learning part 1: each member's looks weight (FR-LM-26)

Until now everyone used S = 0.5 x looks + 0.5 x personality. But some people care more about
looks and some less. In the fake data, each member's hidden true looks weight varies around
0.5. The engine now learns its own estimate, lambda_i, from what each member actually did:

```
logit P(yes) = b0 + b_V * (pct V - 0.5) + b_P * (pct P - 0.5)
lambda_i     = b_V / (b_V + b_P)
```

pct V and pct P are the looks and personality percentiles the engine showed her for that
person. If her yeses track pct V much more than pct P, then b_V is large and lambda_i rises
above 0.5.

- **The evidence:**
  - every pick or pass on her daily list, at weight 0.2 each, because there are thousands of these;
  - every "would meet again" after a date, at weight 3.
- **Why list choices too:** the original plan used dates only, but a member has at most about
  12 dates in three months, which is far too few to estimate two slopes from.
- **The prior:** b_V = b_P = 2, so with no evidence lambda_i is exactly 0.5, and it moves only
  as evidence builds. It is also kept between 0.2 and 0.8.
- **This is the same Bayesian logistic regression as Sprint 2,** just with two features
  instead of 65.

**Result:** after three months, lambda_i has moved about 0.12 from 0.5 on average, and it
correlates **+0.33 to +0.39** with members' hidden true looks weights (main city). The engine
genuinely learns who cares more about looks.

## 5. Learning part 2: date outcomes in the looks model (FR-LM-25)

The research doc said to count "would meet again" as a **stronger** label than a swipe. Each
held date adds one example to each person's looks model: the other person's photo, labeled
with their answer.

**Tuning city, 2 replicates:**

| How much a date counts in the looks model | Looks AUC after 3 months |
|---|---|
| 0 (not used) | 0.861 (before: 0.861) |
| 0.5 | 0.861 |
| 1 | 0.859 |
| 3 (the plan) | **0.841, worse** |

**Why it backfires:**
1. "Would meet again" isn't about looks alone: it also depends on personality and on chemistry
   no profile can predict.
2. **Selection bias.** Only people she already found attractive ever get dated. When a date
   with an attractive man goes badly for personality reasons, the looks model wrongly learns
   that she dislikes his kind of looks.
3. There are few outcomes, and each one counts heavily.

So the weight is set to 0.5, which is harmless. The claim "date outcomes improve looks
accuracy" stays in the tests as a strict expected failure.

## 6. Learning part 3: exploration (FR-LM-27)

The lesson: if you only ever show what you already believe is best, you never learn
whether you were wrong. So about 10% of the profiles actually shown are reserved for the
candidates each member's looks model is **least sure about** (the largest s^2 from Sprint 2).
These are picked round-robin under the same daily cap, and only then is the rest of the list
filled by score.

Getting this right took two fixes, both caught by tests:

1. **Placed last, it barely happened (0.5% of slots).** In a small city the main list already
   uses up nearly everyone available, so nothing was left to explore with.
2. **Reserved as a fixed 3 slots, it was too much (18 to 23%).** Many lists are shorter than
   30 (small pools plus the daily cap), so 3 slots were a big share. It now scales with each
   list's real length, and lands at **9.6 to 9.9%**.

**Result:** no consistent effect on second dates. Months 2 to 3 came out +1.7 points on the test
city and -1.4 on the main city. That's noise around zero. The claim stays a strict expected
failure.

## 7. The three-month results

**Main city, 3 replicates each:**

| Second-date rate | Month 1 | Month 2 | Month 3 |
|---|---|---|---|
| no learning | 37.8% | 33.8% | 28.2% |
| learning | 37.8% | 32.1% | 29.1% |
| learning plus exploration | 38.6% | 31.5% | 27.7% |

| Guarantee share | Month 1 | Month 2 | Month 3 |
|---|---|---|---|
| no learning | 82.3% | 76.2% | 74.0% |
| learning | 82.3% | 76.0% | 73.7% |
| learning plus exploration | 83.9% | 75.6% | 71.3% |

Match quality and exposure spread barely differ between versions (quality +0.83 to +0.60 over
three months in all of them; Gini 0.31 to 0.33).

Other runs agree: the tuning city with 10-profile lists (the setting where ranking matters
more) gave month 3 rates of 27.1%, 25.8% and 28.6% (that run used the earlier version of
exploration), and the 600-member test city gave 25.7%, 25.6% and 27.3%. The differences between versions go in both directions and are within noise.

### Why learning doesn't help here

1. **Ranking has little leverage in a small city.** We've seen this in every sprint since
   Sprint 3: members see nearly their whole pool each month, so better ordering changes little
   about who they end up picking.
2. **The pool runs out.** Every version loses about 10 points of second dates between month 1
   and month 3, and about 8 points of guarantee. Pairs who dated never meet again, and the best
   pairs are used first. In month 3 the engine is choosing among weaker leftovers, whatever it
   has learned. In a real app, new members joining each month would refill pools. The
   test run doesn't model new sign-ups, so this decline is a worst case.
3. **The personal weights are modest.** True looks weights vary around 0.5 with a spread of
   about 0.15, and learning moves lambda_i about 0.12. That's real, but too small to change
   many decisions.

## 8. What changed in the plan

- **FR-LM-25:** the date weight in the looks model is 0.5, not "stronger than a swipe", and the
  original claim is a strict expected failure.
- **FR-LM-26:** looks weights learn from list picks and passes as well as dates.
- **FR-LM-27:** exploration slots are reserved first, scale with each list's real length, and
  respect the daily cap. The gain claim is a strict expected failure.
- **FR-LM-28:** the acceptance check is a strict expected failure, judged over 3 replicates.

All written up in `docs/SPEC.md`.

## 9. A lesson about testing noisy claims

One "expected failure" first **passed**: on a single run, exploration happened to beat
learning-only by more than 2 points. Over 3 replicates the gain was 1.7 points on that city and
-1.4 on the main city. A single synthetic month swings by a couple of points by chance, so the
three-month tests now average 3 replicates and compare months 2 to 3. That made the suite
slower (about 3 minutes in all) but its verdicts trustworthy. When you see one good result in real data
later, the same caution applies.

## 10. Honest limits

- **No new members join during the test run.** Real apps refill pools, so the month-on-month
  decline is exaggerated, and learning might matter more with fresh people to rank.
- **Three months is short.** The research doc suggests a dedicated second-date model once there
  are a few hundred completed dates. With about 1,000 dates a month, that point would come
  quickly, but it isn't built.
- **All of this is fake data.** The test harness's thresholds and noise levels shape every result.

## 11. Math to study

| Topic | Where it shows up | Where to learn it |
|---|---|---|
| Explore-exploit trade-off, multi-armed bandits | Exploration slots | Any bandit-algorithms textbook, the opening chapters |
| Weighted logistic regression, priors | Looks weights and weighted labels | Jurafsky and Martin, chapter 5; Bishop, section 4.5 |
| Selection bias | Why date outcomes mislead the looks model | Any causal-inference introduction (search "selection bias" or "missing not at random") |
| Replicates and standard error | Judging noisy results | Any statistics introduction, "standard error of the mean" |

## 12. Try it yourself

```bash
.venv/bin/python -m lastfirstdate.learning_report                  # about 2 minutes
.venv/bin/python -m lastfirstdate.learning_report --replicates 3   # about 6 minutes
```

In `config/lastfirstdate.toml` (change them back afterwards):

- `taste_date_weight = 3.0`: watch looks accuracy drop after three months.
- `lambda_choice_weight = 0.0`: learn looks weights from dates only. How much worse is the
  recovery?
- `exploration_share = 0.3`: much more exploration. Does anything improve?

---

## Where the project stands after five sprints

### What exists

A complete matching engine for lastfirstdate, built and tested on fake New York members:

| Stage | What it does | Built in |
|---|---|---|
| 1. Filter | Verified, active, mutual orientation, age, distance, dealbreakers, shared free time | Sprint 1 |
| 2. Calibration | 20 varied plus 30 informative profiles for each new member | Sprint 2 |
| 3. Looks taste | Each member's own photo taste from their swipes | Sprint 2 |
| 4. Personality | Interests, values and Big Five | Sprint 1 |
| 5. Mutual 50/50 score | Percentiles, the 50/50 mix, the harmonic mean | Sprint 3 |
| 6. Fair daily lists | Round-robin under an exposure cap, a boost for members behind | Sprint 4 |
| 7. Scheduling | Exact weekly date booking (min-cost flow or integer program) | Sprint 1 |
| 8. Learning | Personal looks weights, date outcomes, exploration | Sprint 5 |

There are 28 requirements, each with tests, plus 4 strict expected failures that record claims
the test run did not support. Every finding is written up in `docs/sprints/` and every plan
change in `docs/SPEC.md`.

### The findings that matter most for the business

1. **Your 50/50 idea holds up.** Mixing looks and personality beats either alone (Sprint 3).
2. **City size changes everything.** With about 1,000 members, members see nearly their whole
   pool, so the scoring and learning barely change outcomes. They start to matter as the city
   grows (Sprints 3 and 5).
3. **The 3-date guarantee is the hard part.** The engine reaches about 82% of paying members,
   not 95%. Offering the guarantee only to members with 50 or more possible matches reaches
   about 96% (Sprint 4).
4. **Some intuitive rules backfire:** capping free members' dates, hiding members with many
   pending yeses, and counting date outcomes heavily in the looks model (Sprints 4 and 5).
5. **Fairness is measurable.** The audit adjusts for each member's own settings and flagged one
   group (age 41+) for a human to review (Sprint 4).

### What it would take to go live

1. **Real models behind the two interfaces:** an image model such as CLIP for photos, and
   a sentence model for interests. Both plug in without changing the engine.
2. **Connect to your database:** read members, write daily lists and dates back, run the jobs
   (daily lists overnight, weekly scheduling, learning monthly). The hosting choice depends on
   what your website already uses.
3. **Your original pre-filter SQL query** in `db/`, to replace straight-line distance.
4. **Privacy and legal:** encrypt photo embeddings and delete them when a member leaves; disclose
   their use; have a New York lawyer review looks-based matching and the audit.
5. **Decisions only you can make:**
   - the guarantee rule;
   - the chooser weight;
   - the number of onboarding swipes;
   - who owns the audit.
6. **A launch experiment:** real data will differ from the test harness. Measure what the test harness
   could only guess, such as how picky each side really is, and A/B test the choices this
   project couldn't settle.

### Questions still open

1. The guarantee rule: everyone with 10 or more possible matches (about 82%), or 50 or more
   (about 96%), or a credit when a month falls short?
2. Chooser weight: 0.5 (equal) or about 0.8?
3. 50 onboarding swipes, or fewer (30 loses about 0.016 AUC)?
4. Who owns the fairness audit and its threshold?
5. How many NYC members do you expect at launch and after 6 months?
6. The original pre-filter SQL query.
7. What your website runs on, for the live connection.
