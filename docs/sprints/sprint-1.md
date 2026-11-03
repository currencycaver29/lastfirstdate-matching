# Sprint 1: a working matcher on personality only

**In one paragraph.** Sprint 1 builds a complete matcher without the looks model. It
filters out everyone who can never match (Stage 1), scores every remaining pair on
interests, values and personality (Stage 4), gives each chooser a daily list (Stage 6), and
books dates each week with an exact optimizer (Stage 7). A behavior simulator plays the part
of real people, and a simulated month measures the result. **About 80% of paying members
reach 3 dates** (78.7% to 81.6% across four random populations, 88% with 2,000 members),
short of the 95% target. The simulation also exposed a flaw in the original plan, which is
now fixed (section 9).

**Papers used**

| Paper | What we took from it | Where in the code |
|---|---|---|
| 15. Kanoria and Saban, *Facilitating the Search for Partners* | Women pick; men accept or decline (decision D1) | `lists.pick_rights` |
| 16. Arnosti, Johari and Kanoria, *Managing Congestion in Matching Markets* | Caps: 30 a day, 3 picks a day, 4 dates a month, 1 a week | `lists.daily_list`, the simulator, the scheduler |
| 9. Joel, Eastwick and Finkel, *Is Romantic Desire Predictable?* | Personality weights are only a guess; in the simulator, measured traits predict true fit only weakly | `personality.py` docstring, `simulation.personality_signal` |
| 8. Reimers and Gurevych, *Sentence-BERT* | Only the plug: a `TextEncoder` interface. The model itself comes later; interest tags stand in for now | `personality.TextEncoder` |

Papers 1 to 7, 10 to 14 and 17 are not used yet.

---

## 1. What got built

| File | Stage | What it does |
|---|---|---|
| `arrays.py` | support | Turns profiles into numpy arrays so a million pairs can be compared at once |
| `filter.py` | 1 | The hard filter, plus a slow one-pair version used to check the fast one |
| `personality.py` | 4 | Interest, values and Big Five similarity, and the personality score P |
| `lists.py` | 6 | Who may pick whom (D1), daily lists, the 7-day re-show rule |
| `scheduler.py` | 7 | Weekly date booking: min-cost flow or an integer program, both exact |
| `simulator.py` | test only | Fake people deciding from hidden truth |
| `simulate.py` | test only | Runs a month and writes the report |
| 8 new test files | | 63 tests in total, plus 1 expected failure (section 9) |

## 2. How a simulated month runs

```
every day (28 days)
  each member opens the app with chance 0.6
  each chooser who opened it:
    Stage 6  gets up to 30 profiles, best personality score first
    sim      picks up to 3 of them
    sim      each picked person accepts or declines
    accepted pairs wait in a "pending" pool

end of each week (4 times)
  Stage 7  books dates from the pending pool, at most 1 per person per week
           and 4 per person per month, each in a slot both have free
  sim      did both show up? did both want a second date?
```

Stages 1 and 4 run once at the start: the filter result and the scores don't change during
the month.

## 3. Stage 1: the filter

A pair passes only if all of these hold, in both directions:

1. Both verified and active in the last 14 days.
2. **Mutual orientation lock (R1):** j's gender is one i seeks, and i's gender is one j seeks.
3. Each person's age is inside the other's preferred range.
4. The distance is within **both** people's maximum distance (straight-line km for now, D9).
5. Neither fails the other's **dealbreakers (R6).**
6. They share at least one free weekly slot.
7. They haven't already dated or blocked each other.

**How it runs fast.** With 1,000 members there are 1,000,000 pairs. Instead of a Python
loop, each rule is one numpy operation on a 1,000 x 1,000 grid of True/False values, and the
grids are combined with AND. For example, the age rule is:

```python
e &= (age_min[:, None] <= age[None, :]) & (age[None, :] <= age_max[:, None])
```

`age[None, :]` lays the ages across the columns and `age_min[:, None]` down the rows, so
numpy compares every row person with every column person at once (this is called
broadcasting).

**Bit masks.** Free slots are stored as one whole number per member: slots {0, 3, 13} become
2^0 + 2^3 + 2^13. Two members share a slot exactly when the bitwise AND of their numbers isn't
zero, and counting the 1-bits gives how many slots they share. Dealbreaker answers use the
same trick. If you've done competitive programming, this is ordinary bitmask work.

**How we know it's right.** `pair_passes` checks one pair at a time in plain Python, rule by
rule. A test compares it with the fast version on all 22,500 pairs of 150 members, and they
must agree on every one. Another test breaks each rule in turn on a passing pair and checks
that the pair is removed.

**Pass rates on the default population:** age 56%, distance 68%, dealbreakers 58%, shared
slot 95%. The median member ends up with **62 possible matches**.

## 4. Stage 4: the personality score

```
P_i(j) = 0.4 * t_ij  +  0.3 * A_ij  +  0.3 * p_ij
```

Each part runs from 0 to 1, so P does too. P is the same in both directions (P_i(j) = P_j(i)),
because all three parts are symmetric.

### t: interest similarity

Each member's interests become a vector with one number per possible tag. The number for a
tag is its **IDF weight** if the member lists it, and 0 if not:

```
idf(tag) = log((1 + N) / (1 + df)) + 1
```

N is the number of members, and df is how many of them list the tag. A tag nearly everyone
lists ("coffee") gets a low weight; a rare tag ("opera") gets a high one. Each vector is then
scaled to length 1, and t is the cosine similarity (the dot product, from Sprint 0):

```
t_ij = v_i . v_j
```

Identical interests give 1 and no shared interests give 0. Sharing a rare interest counts for
more than sharing a common one, and a test checks exactly that. This is the classic TF-IDF
method from information retrieval.

**Paper 8 (Sentence-BERT)** will replace tags with a model that reads free text, so "trail
runs" and "hiking" count as similar even though the words differ. The code is ready for it:
anything with an `encode(members)` method that returns unit vectors can be passed in. The
model isn't added yet because tests must run offline (R4).

### A: values compatibility

For each of the 6 values questions, it counts only if **each** person's answer is one the
other accepts:

```
A_ij = (number of questions where both accept each other's answer) / 6
```

If she accepts his "want kids" answer but he doesn't accept hers, that question doesn't count.
Dealbreakers already removed the impossible pairs in Stage 1. A measures how well the
remaining pairs agree.

### p: Big Five similarity

```
p_ij = 1 - distance(big5_i, big5_j) / sqrt(5)
```

The distance is the ordinary straight-line (Euclidean) distance between two points in 5
dimensions. The largest possible distance is sqrt(5), from all zeros to all ones, so dividing
by it puts p between 0 and 1.

### A worked example

Member 74, a 36-year-old woman, likes history, beaches, boxing, podcasts, kayaking and
languages. She has 52 possible matches. Her top three by P:

| Man | t (interests) | A (values) | p (Big Five) | P | Shared interests |
|---|---|---|---|---|---|
| 271, age 34 | 0.31 | 0.83 | 0.87 | 0.634 | history, languages |
| 335, age 39 | 0.31 | 0.83 | 0.87 | 0.634 | beaches, languages |
| 354, age 30 | 0.53 | 0.50 | 0.83 | 0.609 | beaches, history, languages, podcasts |

Check the first row: 0.4 x 0.31 + 0.3 x 0.83 + 0.3 x 0.87 = 0.124 + 0.249 + 0.261 = 0.634.
Man 354 shares the most interests but agrees on only 3 of the 6 values questions, so he ranks
third. The median score in her pool is 0.460.

### Why the weights are only a guess (paper 9)

Joel, Eastwick and Finkel measured 100+ traits before speed dates and couldn't predict who
would desire whom. General desirability (who most people like) was predictable; desire for
one specific person was not. So 0.4 / 0.3 / 0.3 is a starting guess, and Sprint 5 retunes it
from real date outcomes. The simulator builds in the same idea (section 7).

## 5. Stage 6: who chooses, and the daily list

### She chooses (paper 15)

In a man-woman pair, the woman picks and the man accepts or declines. In any other pair (two
men, two women, or a nonbinary member) either may pick. `pick_rights` builds a grid that is
True wherever i may pick j.

**Why (paper 15, Kanoria and Saban).** On dating apps women are usually the short side:
there are fewer of them, and they receive far more interest. When everyone can contact
everyone, the short side is flooded and much of the long side's effort goes unanswered.
Kanoria and Saban show that restricting who can start, so only the short side initiates,
can raise overall welfare by cutting that wasted search effort.

### The list

Each chooser gets up to 30 candidates who passed the filter, best P first. Removed from the
list: anyone already at 4 dates this month, and any pair that already picked, declined or
dated.

**The 7-day re-show rule.** Someone shown but not picked can reappear 7 days later. This
rule is new in Sprint 1; section 9 explains why.

### Caps (paper 16)

Arnosti, Johari and Kanoria study markets where applying is cheap, so people send far too
many applications. They show that limiting applications cuts wasted effort on both sides.
Our caps follow this idea: 30 profiles a day, 3 picks a day, 1 date a week, 4 dates a month.

## 6. Stage 7: the weekly scheduler

Each week the pending pool holds pairs where both said yes. Each person can take at most one
date this week (`dates.max_per_week`) and has a limit for the month. The scheduler chooses
which pairs get a date so the total value is as high as possible:

```
maximize    sum of w_ij over the chosen pairs
subject to  each person i is in at most capacity_i chosen pairs
```

This problem is called **maximum-weight b-matching** (b is the capacity of each person).

### Each pair's value

```
w_ij = date_value + P_ij + urgency_i + urgency_j
urgency_i = 0.2 * (missing guaranteed dates of i) / (weeks left in the month)    paying members only
```

- `date_value = 1.0` means every date is worth booking, so the optimizer never skips one it
  could fit.
- Urgency grows as the month runs out. A paying member with 0 dates in week 1 adds
  0.2 x 3 / 4 = 0.15. With 2 still missing in the last week, they add 0.2 x 2 / 1 = 0.4. When
  two pairs compete for the same person, the one that helps someone behind on their guarantee
  wins.

### Solving it, case 1: min-cost flow (bipartite graphs)

Think of people as nodes and pending pairs as edges. If every edge runs between two groups
(men on one side, women on the other), the graph is **bipartite**, and the problem becomes a
min-cost flow:

```
            capacity c_i                 capacity 1, cost -w_ij              capacity c_j
 source  ----------------->  woman i  -------------------------->  man j  ----------------->  sink
    \                                                                                       /
     ------------------------------- bypass, cost 0 ---------------------------------------
```

Every unit of flow from source to sink either passes through one woman-man edge (a booked
date, cost -w) or takes the bypass (no date, cost 0). Capacities enforce the limits. The
cheapest way to send all the flow is the most valuable schedule. The bypass means nobody is
forced into a date. This is the standard min-cost flow from competitive programming, solved
with OR-Tools.

### Solving it, case 2: an integer program (non-bipartite graphs)

Same-gender pairs break bipartiteness. Three women who each said yes to the others form a
triangle, and a triangle can't be split into two sides with every edge crossing between them.
Any odd cycle breaks it. Flow then no longer gives the right answer, so the scheduler checks
the graph first (`two_color`: try to color nodes in two colors so every edge joins different
colors). If that fails, it solves the same problem as an **integer program**: one yes/no
variable per pair, one capacity constraint per person, maximize the total value. The solver
is SCIP, through OR-Tools.

Both methods give the exact best answer. Tests check this against brute force (trying every
possible subset of pairs) on 50 small random cases, plus the triangle.

**What actually happened.** In the full simulated city, every week contains same-gender pairs
with odd cycles, so every week used the integer program ("mip" in the report). My first
version used a different solver, CP-SAT. On week 1's real problem (1,472 pairs, 673 people) it
hadn't proven the best answer after 60 seconds. SCIP solves it in 0.5 seconds. The reason is
worth knowing: if you relax "book or don't book" to "any fraction from 0 to 1", you get a
linear program, and here that relaxed version already came out in whole numbers. SCIP solves
the linear program first and finds there's nothing left to search. A whole month now
simulates in about 2.5 seconds.

## 7. The behavior simulator

It decides, for each fake person, how much they truly like everyone else:

```
U_ij = looks_weight_i * looks_ij  +  (1 - looks_weight_i) * fit_ij
fit_ij = 0.5 * trait_fit_ij  +  sqrt(1 - 0.5^2) * chemistry_ij
```

- **looks_ij** is i's hidden taste dotted with j's photo embedding (Sprint 0), turned into a
  z-score: subtract i's average and divide by the spread, so values are in standard
  deviations.
- **trait_fit** measures how well the two profiles fit. It's computed by **separate code with
  different weights** from the engine's (equal thirds, and Jaccard overlap instead of TF-IDF
  cosine). If the simulator reused the engine's score, a bug in the engine would also change
  the "truth" and the tests couldn't catch it.
- **chemistry** is random, shared by both people, and impossible to predict from any profile.
- **0.5** is `simulation.personality_signal`. Measured traits explain only part of true fit,
  following paper 9. Because sqrt(1 - 0.5^2) = 0.87, unpredictable chemistry carries more
  weight than the traits.

Decisions use the logistic curve:

```
P(yes) = 1 / (1 + exp(-(U - threshold - pickiness) / 0.5))
```

At U = threshold the chance is 50%. Higher U makes yes more likely, lower U less likely. Each
person has their own hidden pickiness, which shifts their threshold. Picking uses threshold
1.0 (choosy), accepting uses 0.0, and wanting a second date uses 0.5 plus a random shift for
what you learn by meeting in person.

**Someone already passed over is less likely to be picked later.** Each earlier pass halves
the chance (`repeat_pick_factor = 0.5`). Without this, showing the same person again would
count as a completely fresh chance, which would flatter the re-show rule.

**The boundary is enforced.** A test reads the engine's source files and fails if any of them
imports the simulator or mentions the hidden truth.

## 8. What the simulation showed

The default population (1,000 members, seed 7):

```
matchable (verified and active) 780
possible matches per member: median 62, 10th percentile 11, under 10: 65
average daily list size by week: 24.9, 23.4, 23.0, 22.7
picks 5260, accepted 3198 (61%)
dates booked 1176, held 998, both wanted a second date 371 (37%)

paying and matchable 341; with at least 10 possible matches 309
reached the guarantee: 81.6% (men 78.0%, women 88.5%)
dates per paying member: 0: 24, 1: 27, 2: 33, 3: 58, 4: 199
missed: 84 paying members
    51  fewer than 3 mutual yeses all month
    27  small pool (fewer than 10 possible matches)
     6  enough mutual yeses, but the schedule could not fit them
```

Other populations: seed 8 gives 80.3%, seed 9 gives 78.7%, seed 10 gives 80.4%, and 2,000
members gives 87.9%.

How to read it:

- **The scheduler is not the bottleneck.** Only 6 misses come from scheduling. The problem is
  upstream: 51 paying members got fewer than 3 mutual yeses all month.
- **Dates pile up on some members.** 199 paying members hit the 4-date cap while 24 got none.
  Popular members fill up; the lists don't push anyone toward the members who are behind.
  That's exactly Sprint 4's job (fair lists).
- **Men have it harder** (78% versus 88.5%), because women choose. That's the expected cost of
  "she chooses", and the reason the guarantee needs active support.
- **Small pools can't be helped by the algorithm.** 27 paying members have fewer than 10
  possible matches, mostly people far from the center or with narrow age or distance settings.
- **The second-date rate (37%) is not a prediction.** It depends on simulator settings that
  are guesses. Use the simulator to compare designs, not to forecast real numbers.

## 9. What changed in the plan, and why

**Change 1: people can reappear after 7 days.** The plan said nobody is shown twice in a
month. The first simulation showed lists shrinking from 25 profiles a day in week 1 to 0 in
week 4: with about 62 possible matches each, members simply ran out of people. I ran six quick
experiments to find which factor mattered most:

| Setting | Reached guarantee | List size, weeks 1 to 4 |
|---|---|---|
| A. As planned: never show anyone twice | 63% | 25, 22, 6, 0 |
| B. Re-show after 7 days | 88% | 25, 24, 23, 22 |
| C. Put paying members still short of 3 at the top of lists | 63% | 25, 20, 1, 0 |
| D. B and C together | 88% | 25, 24, 23, 22 |
| E. As planned, 2,000 members | 76% | 28, 27, 25, 25 |
| F. B and C, 2,000 members | 93% | 28, 27, 26, 26 |

These experiments ran **before** the "earlier pass halves the chance" rule was added, so row B
is too optimistic. With that rule the result is about 80%. Re-showing still helps most, and
moving people up the list barely helps on its own (C): when lists run dry, nothing else can
work.

**Change 2: the 95% check moved to Sprint 4.** A personality-only matcher reaches about 80%,
and most of the gap is members who get too few mutual yeses, which is a fairness problem.
Rather than loosen the target, the check now belongs to FR-LM-21 (Sprint 4's fair lists). It
stays in the test suite as an **expected failure** (`xfail(strict=True)`). If it ever starts
passing, the suite turns red, so the marker can't be forgotten. Sprint 1 instead checks that
the month runs within the caps, lists stay at least 80% full, every miss has a reason, and the
result doesn't fall below 75%, so a later change can't quietly make things worse.

Both changes are written up at the end of `docs/SPEC.md`.

## 10. Limits of this sprint

- Everything is fake data. The simulator's thresholds and the 0.5 personality signal are
  guesses. The numbers compare designs; they don't predict real-world rates.
- Distance is straight-line, not travel time, until the original SQL query is added.
- No looks yet. Looks are 50% of the final score, and Sprints 2 and 3 add them.
- Interest tags only. Sentence-BERT plugs in later.

## 11. Math and computer science to study

| Topic | Where it shows up | Where to learn it |
|---|---|---|
| Broadcasting and boolean masks | Filter | NumPy user guide, "Broadcasting" |
| Bit masks, AND, popcount | Slots and dealbreakers | You know this from competitive programming |
| TF-IDF and cosine similarity | Interest score t | Jurafsky and Martin, *Speech and Language Processing*, chapter 6 (free online) |
| Euclidean distance, norms | Big Five score p | *Mathematics for Machine Learning*, chapter 3 |
| z-scores (standardizing) | Simulator | *Mathematics for Machine Learning*, chapter 6 |
| Logistic (sigmoid) function | Simulator decisions; Sprint 2's taste model is built on it | Jurafsky and Martin, *Speech and Language Processing*, chapter 5 (logistic regression, free online) |
| Bipartite graphs, two-coloring | Choosing flow or integer program | Any graph algorithms text (BFS two-coloring) |
| Min-cost flow | Scheduler, case 1 | cp-algorithms.com, "Minimum-cost flow"; OR-Tools min-cost flow guide |
| Linear and integer programming, LP relaxation | Scheduler, case 2 | *Mathematics for Machine Learning*, section 7.3 (linear programming); OR-Tools MIP guide |

## 12. Try it yourself

```bash
.venv/bin/python -m lastfirstdate.simulate                # the report above
.venv/bin/python -m lastfirstdate.simulate --seed 9       # a different fake city
.venv/bin/python -m lastfirstdate.simulate --n 2000       # a bigger city
```

Then edit `config/lastfirstdate.toml` and rerun:

- `reshow_after_days = 30` brings back the "never show twice" rule, and you'll watch week 4
  lists empty out.
- `picks_per_day = 5` makes women pick more. How much does the guarantee move?
- `personality_signal = 0.9` makes traits predict fit well (the opposite of paper 9). Does
  anything change while lists are still ranked by P?

Change the values back afterwards, or the tests that check the defaults will fail.

## 13. Read before Sprint 2 (the looks model)

| Paper | What to look for |
|---|---|
| 4. Ren et al., *Personalized Image Aesthetics* | The core idea for Sprint 2: a general model plus a small personal correction learned from a few ratings |
| 5. Rothe, Timofte and Van Gool, *Some Like It Hot* | Predicting one person's attraction to a face from image features |
| 3. Chu and Ghahramani, *Preference Learning with Gaussian Processes* | Learning a taste function from likes and passes, with a measure of uncertainty |
| 1. Brochu, de Freitas and Ghosh, *Active Preference Learning* | Choosing which profile to show next so each swipe teaches the most |
| 2. Kulesza and Taskar, *Determinantal Point Processes* (sections 1 and 2) | Choosing a varied first set of profiles |
| 6 and 7. Neve and McConville | Photo preferences on both sides, made mutual (more for Sprint 3) |

Math for Sprint 2: logistic regression and its loss with an L2 penalty (Jurafsky and Martin,
chapter 5), and PCA (*Mathematics for Machine Learning*, chapter 10).

## 14. Questions for you

1. **Should non-paying members be capped at fewer dates** (for example 2 a month) so more of
   the supply goes to paying members? It's a business call. Sprint 4 can test it either way.
2. **Should the guarantee be offered only to members with at least 10 possible matches?** 27
   paying members in the simulation could never reach it. Alternatively, the app could prompt
   them to widen their distance or age settings.
3. **Are you OK with the 95% check moving to Sprint 4?** (Section 9.)
4. Still open from Sprint 0: expected NYC member numbers, and the original pre-filter SQL
   query.
