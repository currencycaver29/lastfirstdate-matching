# Sprint 3: the 50/50 mutual score

**In one paragraph.** Sprint 3 combines the two halves into the score your brief describes:
50% looks, 50% personality and interests, made mutual so a pair ranks high only when both
people would like each other. Every fake member now calibrates on 50 photos, so the engine
has a looks score for everyone. The results are more interesting than "it works". **Mixing
looks and personality clearly beats either alone.** The mutual score does its job: people
at the top of a list are about 5 points more likely to accept. But in our simulated market,
choosers are the pickier side, so weighing both sides equally costs about as many of her
picks as it gains in his acceptances. And in a city of 1,000 members, **the ranking barely
changes the month**, because members see almost everyone in their pool anyway. The
algorithm's quality starts to matter with bigger pools or shorter lists.

**Papers used**

| Paper | What we took from it | Where in the code |
|---|---|---|
| 10. Pizzato et al., *Reciprocal Recommender System for Online Dating* (RECON) | Score both directions, then combine them with the harmonic mean | `mutual.mutual_score` |
| 6. Neve and McConville, *ImRec* | Both people's photo preferences feed one mutual score: his looks score for her and hers for him both enter M | `mutual.looks_scores`, `mutual.directional_score` |
| 7. Neve and McConville, *Photos Are All You Need* | Supporting evidence that photos alone carry much of the signal; our looks-only ranking beats personality-only on mutual yeses | comparison in `mutual_report.py` |

---

## 1. What got built

| File | What it does |
|---|---|
| `mutual.py` | Stage 5: percentiles inside each pool, the 50/50 directional score S, the mutual score M, and V for every calibrated member |
| `mutual_report.py` | Simulation side: calibrates every member, builds all the scores, measures the top of each list against the hidden truth, and runs a month with each of four rankings |
| `simulate.py` | New: match-quality measures from the hidden truth, one shared definition of a month's simulated people (`month_people`), and date weights that work for one-sided scores too |
| `looks.py` | Calibration now fits once after the 20 variety profiles instead of after each one (identical result, about twice as fast), and marks a calibration as complete |
| `tests/test_mutual.py` | 10 tests for FR-LM-18 to FR-LM-20 |

## 2. Step 1: percentiles (FR-LM-18)

V (looks) and P (personality) both run from 0 to 1, but they spread very differently. For
one member, V might run from 0.01 to 0.99 while P only runs from 0.35 to 0.65. Add them raw
and V decides almost everything, because its differences are bigger, not because it matters
more.

So each score is replaced by its **percentile inside the member's own filtered pool**:

```
pct_i(j) = (rank of j among i's k candidates - 0.5) / k
```

With 4 candidates, the percentiles are 0.125, 0.375, 0.625 and 0.875: evenly spaced, with
an average of exactly 0.5. A test checks that a wide score (0.01 to 0.99) and a narrow score
(0.45 to 0.55) in the same order get **identical** percentiles. Ties share a rank.

## 3. Step 2: the 50/50 directional score (FR-LM-19)

```
S(i -> j) = 0.5 * pct_i(V_i(j)) + 0.5 * pct_i(P_i(j))
```

Example: if j is her 90th-percentile candidate on looks and 40th on personality, then
S = 0.5 x 0.9 + 0.5 x 0.4 = 0.65. The 0.5 is `mix.looks_weight`, your 50/50 split. S is
**one-directional**: how much i would like j. S(j -> i) uses j's taste and j's pool, so it is
usually different.

## 4. Step 3: the mutual score (FR-LM-20)

A date needs two yeses. Paper 10 (RECON) scores both directions and combines them with the
**harmonic mean**:

```
M(i, j) = 2 * S(i -> j) * S(j -> i) / (S(i -> j) + S(j -> i))
```

The harmonic mean is pulled toward the smaller value:

| S(i to j) | S(j to i) | Ordinary average | Geometric mean | Harmonic mean (M) |
|---|---|---|---|---|
| 0.9 | 0.9 | 0.90 | 0.90 | 0.90 |
| 0.9 | 0.5 | 0.70 | 0.67 | 0.64 |
| 0.9 | 0.1 | 0.50 | 0.30 | **0.18** |

A one-sided crush (0.9 versus 0.1) scores 0.18, not 0.5. In general, harmonic ≤ geometric ≤
ordinary average, with equality only when the two sides are equal.

**The chooser weight (new, decision D13).** A weighted harmonic mean lets you trust the
chooser's side more:

```
M(i -> j) = 1 / ( a / S(i -> j)  +  (1 - a) / S(j -> i) )
```

At a = 0.5 (the default, `mix.chooser_weight`) this is exactly the formula above. At a = 0.8,
the 0.9 versus 0.1 pair gets 0.35 from the chooser's side instead of 0.18. Section 7 explains
why this might matter.

## 5. Calibrating every member

To score M, the engine needs V for both people in every pair, so all 780 matchable fake
members calibrate:

1. The starting taste w_0 is fitted on "history" swipes: 50 random swipes from each member.
2. Each member calibrates on their 50 profiles (Sprint 2).
3. V_i(j) is computed for everyone in i's Stage 1 pool. `list_scores` refuses any member whose
   calibration hasn't finished, so w_0 alone can never reach a list (rule R2).

This takes about 45 seconds for the whole city.

## 6. How the rankings were compared

Four ways of ranking a chooser's list:

| Ranking | What it sorts by |
|---|---|
| personality | P only (the Sprint 1 matcher) |
| looks only | the chooser's percentile of V |
| one-sided | S(chooser -> candidate): her own 50/50 view |
| mutual | M: both views combined |

Two measurements:

1. **The top of the list.** For each chooser's top 10, the hidden truth gives the chance she'd
   pick each candidate and the chance he'd accept her. Multiplying them gives the chance of a
   **mutual yes**. This judges the ranking itself, with no dependence on how a month happens
   to unfold.
2. **A whole simulated month** with each ranking: same people, same random choices. New
   measures from the hidden truth: **appeal** (how much the two people in each booked date
   truly like each other, in standard deviations) and **both keen** (the share of dates where
   each likes the other above their own average).

## 7. Results

### The top of the list (main city, seed 7)

| Ranking | She picks | He accepts | Both (mutual yes) |
|---|---|---|---|
| personality | 29.5% | 63.0% | 20.7% |
| looks only | 41.2% | 52.3% | 24.0% |
| one-sided | 39.4% | 59.7% | **25.7%** |
| mutual | 34.7% | **64.3%** | 24.2% |

On the tuning city (seed 101), I also tried other combinations:

| Combination | She picks | He accepts | Both |
|---|---|---|---|
| one-sided S | 41.9% | 57.8% | 26.6% |
| harmonic, equal (M) | 37.4% | 63.7% | 25.7% |
| harmonic, chooser weight 0.65 | 39.0% | 62.1% | 26.3% |
| harmonic, chooser weight 0.8 | 40.4% | 60.5% | **26.7%** |
| ordinary average | 37.9% | 63.9% | 26.0% |
| geometric mean | 37.7% | 63.8% | 25.9% |

### What this means

1. **Mixing looks and personality wins clearly.** One-sided S (25.7%) beats looks only (24.0%)
   and personality only (20.7%). Your 50/50 idea is supported.
2. **The mutual score does what it promises.** Men at the top of her list are 4.6 points more
   likely to accept (64.3% versus 59.7%).
3. **But it costs some of her picks** (34.7% versus 39.4%). In this simulated market, women
   pick about 35–40% of the top of their lists, while men accept about 60%. The pick is the scarce
   step, so the chooser's view matters more. Weighing both sides equally lands slightly below
   her own view (24.2% versus 25.7%). Weighting the chooser at 0.8 lands slightly above it.
4. **Is the market really like this?** Women being more selective on dating apps is commonly
   reported, so the direction is plausible. But the size of the gap in our simulator is my
   guess (`like_threshold` 1.0 versus `accept_threshold` 0.0). That's why the chooser weight
   stays at 0.5 for now, and it's your call (section 12).

### A whole month (main city, 1,000 members, 30 profiles a day)

| Ranking | Mutual yeses | Dates | Second date | Appeal | Both keen | Guarantee |
|---|---|---|---|---|---|---|
| personality | 3,198 | 1,176 | 37% | +0.83 | 78% | 81.6% |
| looks only | 3,172 | 1,152 | 40% | +0.87 | 80% | 78.3% |
| one-sided | 3,102 | 1,174 | 38% | +0.85 | 78% | 80.6% |
| mutual | 3,135 | 1,152 | 39% | +0.85 | 80% | 78.6% |

**Almost no difference.** The typical member has about 63 possible matches and sees up to 30
profiles a day, with re-showing after 7 days. Over a month she sees nearly everyone in her
pool several times, so she ends up picking the people she truly likes whatever order the list
was in. The ranking only decides who she sees *first*.

### When the ranking does matter

**Shorter lists** (same city):

| Profiles a day | Ranking | Acceptance rate | Dates | Appeal | Guarantee |
|---|---|---|---|---|---|
| 5 | one-sided | 61% | 1,147 | +0.78 | 75.7% |
| 5 | mutual | **65%** | **1,166** | **+0.81** | **79.0%** |
| 10 | one-sided | 61% | 1,192 | +0.81 | 80.3% |
| 10 | mutual | **63%** | 1,181 | **+0.82** | **81.6%** |

**A bigger city** (3,000 members, median pool 188, 30 a day):

| Ranking | Appeal | Both keen | Second date | Guarantee |
|---|---|---|---|---|
| personality | +0.96 | 84% | 42% | 88.9% |
| looks only | +1.00 | 83% | 44% | 87.1% |
| one-sided | **+1.05** | 85% | **46%** | 87.1% |
| mutual | +1.03 | 85% | 45% | 87.2% |

When members can't see everyone, adding looks raises match quality: second-date rates go from
42% to 45–46%. The mutual score also helps the guarantee when lists are short (79.0% versus
75.7%): it stops everyone being shown the same few popular people.

**Looks-based ranking slightly lowers the guarantee in every setting** (for example 88.9% to
87.2%). Shared taste means many women's top candidates overlap, so attention piles onto fewer
men. Fixing that is Sprint 4's job: fair lists with exposure caps.

## 8. What changed in the plan

- **The FR-LM-20 acceptance test was rewritten.** It originally said ranking by M gives "more
  mutual likes than one side only". That didn't hold, for the reason in section 7. The test now
  checks what M reliably does:
  - the top 10 by M are likelier to accept than the top 10 by one-sided S;
  - M beats looks alone and personality alone on mutual yeses;
  - M stays within 1.5 points of one-sided S.
- **New setting `mix.chooser_weight`** (decision D13), kept at 0.5.
- **Faster calibration.** It fits once after the variety profiles, with mathematically the same
  result.

All written up in `docs/SPEC.md`.

## 9. Honest limits

- **The simulator's pickiness gap is a guess**, and it decides whether equal weighting or
  chooser weighting wins.
- **Percentiles throw away confidence.** If her top candidate has V = 0.99 or V = 0.60, both
  become her top percentile. A model with calibrated probabilities could use more information.
  That's a possible later improvement.
- **The month results depend on city size.** The 3,000-member run took about 6 minutes, so
  only one big-city run was done.

## 10. Math to study

| Topic | Where it shows up | Where to learn it |
|---|---|---|
| Ranks and percentiles | Step 1 | Any intro statistics text, "order statistics" and percentiles |
| Harmonic, geometric and arithmetic means, and HM ≤ GM ≤ AM | Step 3 | Any text on means and inequalities; the AM-GM inequality |
| Weighted harmonic mean | The chooser weight | Same; it's the reciprocal of a weighted average of reciprocals |
| Independent events: P(A and B) = P(A) x P(B) | The mutual-yes measure | Any probability intro; *Mathematics for Machine Learning*, chapter 6 |
| Reciprocal recommenders | The whole sprint | Paper 10, then paper 6 |

## 11. Try it yourself

```bash
.venv/bin/python -m lastfirstdate.mutual_report                      # about 1 minute
.venv/bin/python -m lastfirstdate.mutual_report --looks-weight 0.7   # 70% looks
.venv/bin/python -m lastfirstdate.mutual_report --seed 9
```

Then in `config/lastfirstdate.toml` (change them back afterwards):

- `chooser_weight = 0.8`: does the mutual row's "both" beat one-sided?
- `daily_size = 5`: watch the rankings pull apart in the month table.
- `like_threshold = 0.0`: women become as easy-going as men. Does equal weighting now win?

## 12. Read before Sprint 4 (fair daily lists)

| Paper | What to look for |
|---|---|
| 12. Singh and Joachims, *Fairness of Exposure in Rankings* | Exposure: how much attention each position in a list gets, and how to share it fairly |
| 11. Tomita and Yokoyama, *Fair Reciprocal Recommendation in Matching Markets* | Nash social welfare (sum of logs) as a way to spread chances while keeping matches high |
| 13. Su, Bayoumi and Joachims, *Optimizing Rankings for Recommendation in Matching Markets* | Optimizing everyone's lists together, counting capacity on both sides |
| 14. Rios, Saban and Zheng, *Assortment Optimization in Dating Markets* | Holding back likely matches from members who just matched a lot |
| 16. Arnosti, Johari and Kanoria (again) | Why caps per profile reduce congestion |

Math: logarithms and why maximizing a sum of logs spreads value out; the Gini coefficient.

## 13. Questions for you

1. **Chooser weight: keep 0.5 (equal) or move to about 0.8?** In simulation, 0.8 gives about 1
   point more mutual yeses, but only if women really are much pickier than men on your app.
   My recommendation: keep 0.5 until real data exists, then let Sprint 5's learning set it.
2. **What this means for launch.** With about 1,000 members, the ranking barely matters,
   because members see nearly their whole pool. Early on, member count, the filter and the
   fair scheduling matter more than the 50/50 scoring. The scoring pays off as the city grows.
   How many members you expect matters here (still open from Sprint 0).
3. Still open:
   - capping dates for members who don't pay;
   - offering the guarantee only to members with at least 10 possible matches;
   - 50 onboarding swipes or fewer;
   - who owns the audit tolerance;
   - the original pre-filter SQL query.
