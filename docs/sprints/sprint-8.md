# Sprint 8: the forecast and the gate on delivered dates

**In one paragraph.** The ratio gate from Sprint 6 decides how many men to sell subscriptions
to. Its forecast was fitted on the old batch test run, which counts dates *booked*. Sprint 8
refits it on the real booking flow from Sprint 7, which counts dates *delivered*: confirmed,
paid, inside his cycle, and actually happened. On the way, a bug in Sprint 7's test run
turned up: a no-show was never refilled. Fixing it raises the real flow from 68% to **82%**.
The refit changes what the gate believes. **The size of the zone matters far more than the
ratio of women to men.** With 500 committed women, about 90% of men get their 3 delivered
dates even at 3 women per man. With 300, only a handful of men can be promised 90%; the old
gate would sell 95 subscriptions there that deliver about 75%. The new gate refuses. Your site
plans 300 to 500 women before 100 subscriptions: this sprint says **aim for the top of that
range**.

**Methods used.** Like Sprint 6, this is logistic regression applied to your business rule.
Market balance and congestion are known to matter; this sprint measures what actually limits
fulfilment in your flow.

---

## 1. What got built

| File | What it does |
|---|---|
| `forecast.py` | A model now records what it forecasts (`kind`: booked or delivered) and what it assumes. New: delivered-date progress features, `at_risk_delivered`, and `gate_model(cfg)` to pick the gate's forecast |
| `forecast_fit.py --delivered` | Fits the forecast on the real flow across 60 synthetic markets and writes `config/forecast_delivered.toml` |
| `cycle_runs.py` | Records forecast rows (each man's market at launch; his progress every week); fixed the refill after a no-show |
| `gate_report.py` | The gate in the real flow: two zone sizes, three versions, three markets each |
| `config/*.toml` | `[gate] forecast = "booked"` (research) or `"delivered"` (Phase 1) |

The gate's rules and code, including its Rust twin, are unchanged; only the coefficients it
reads changed.

---

## 2. The bug, and what it changed

Your site says no-shows "don't count and the slot is refilled". Sprint 7's test run counted
correctly (her no-show didn't count for him), but it never gave him his weekly slot and cap
slot back. With a cap of 3 dates a cycle, one no-show by her could leave him at most 2. The
fix (`release_slot`): whoever was stood up gets their slot back, and whoever didn't come
loses theirs, consistent with D17.

| Same three markets | Sprint 7 | Corrected |
|---|---:|---:|
| Real flow | 67.7% | **81.9% ± 2.8** |
| No new women | 58.2% | 69.2% ± 3.2 |
| Show-up 97% | 75.5% | 81.7% ± 2.3 |
| No friction | 89.5% | 89.5% ± 3.1 |
| Credits, real flow | 17.2% | 12.2% |

Sprint 7's conclusions that change:
- **Friction** costs about 8 points, not 22.
- **Show-ups** are no longer the biggest lever: refills absorb most no-shows.
- **New women joining every cycle** are now the biggest lever (+13 points).

`docs/sprints/sprint-7.md` has a correction note at the top. One Sprint 7 test claimed
friction costs at least 10 points; it now checks at least 5, with the reason written in its
docstring. That isn't a weakened test: the original claim came from the bug.

---

## 3. The forecast on delivered dates (FR-LM-43)

### The model

Same form and features as Sprint 6:

    P_j = sigmoid(b0 + b1·log r + b2·log(1 + pool_j) + b3·log(1 + share_j) + b4·log(min(r, 1)))

- **The label changed:** for each paying man, the share of his first three paid cycles that
  reached 3 *delivered* dates. A share between 0 and 1 is fine for logistic regression: its
  gradient is just (p − y), whatever y is.
- **The rows changed:** each man's market at launch, in 60 synthetic real-flow markets (60,
  100 or 150 men; 1 to 7 women per man; four seeds, because markets differ by about ±5
  points).
- **The file says what it assumes:** 15% new women per cycle, 92% show-ups, 5% cancellations,
  the payment rates, three 28-day cycles. If those change, refit.

### What the coefficients say

| | log r | log(1 + pool) | log(1 + share) |
|---|---:|---:|---:|
| Sprint 6 (booked) | 0.28 | 1.44 | 1.89 |
| Sprint 8 (delivered) | 0.10 | 1.98 | 0.41 |

With delivered dates, **a man's pool**, how many women nearby could be matched with him,
carries most of the weight. Women per man barely matters on its own.

### How well it forecasts

On four held-out markets it is off by 3.8 points on average (7.6 at most), leaning cautious.
The held-out markets share one seed, and markets vary by about ±5 points in ways no feature
captures, so part of that is luck.

---

## 4. Mid-cycle alerts on delivered dates (FR-LM-44)

Each week of a man's cycle, the engine works out his chance of ending the cycle with 3
delivered dates. It uses the start features plus:

- **delivered so far ÷ 3;**
- **log(1 + confirmed dates still to come);**
- **days left ÷ 28.**

| After week | 1 | 2 | 3 |
|---|---:|---|---:|
| AUC on held-out markets | 0.971 | 0.989 | 0.996 |

By week 2 it knows almost exactly who is in trouble, with two weeks left to act: extra list
priority, an event invitation, a concierge message. `at_risk_delivered` refuses a model fitted
on booked dates, so the two kinds can't be mixed up.

---

## 5. The gate on delivered dates (FR-LM-45)

`gate.forecast` picks the file; Phase 1 now uses `delivered`. The real-flow gate trial, three
markets per zone with a waitlist of 450 men:

| Zone | Gate | Men admitted | Women per man | Actual | Credits |
|---|---|---:|---:|---:|---:|
| **300 women** | booked forecast (Sprint 6) | 95 | 3.0 | 75.5% | 16.4% |
| | **delivered forecast** | **6** | 32 | **91.3%** | 6.5% |
| | everyone | 422 | 0.7 | 43.8% | 34.4% |
| **500 women** | booked forecast (Sprint 6) | 158 | 3.0 | 89.6% | 7.0% |
| | delivered forecast | 56 | 9.1 | 88.3% | 7.5% |
| | everyone | 426 | 1.1 | 81.3% | 11.2% |

The forecast against the number of men, in one market of each zone:

| Men admitted | 10 | 50 | 100 | 150 | 300 |
|---|---:|---:|---:|---:|---:|
| 280-woman zone | 90.1% | 77.5% | 72.1% | 71.4% | 67.3% |
| 475-woman zone | 95.6% | 91.9% | 89.4% | 87.6% | 85.4% |

### Why the zone matters more than the ratio

In the test run, a man's dates depend on how many women near him *like* him and pick him.
Other men barely compete until there are so many that women's 30-profile lists overflow
(around 1 woman per man). A bigger zone means a bigger pool for every man; a few more or fewer
men changes little.

This is how the fake women behave: they don't become less picky when men are scarce. Real
women may, which would make the ratio matter more. Real Phase 1 data will tell.

### Two consequences

1. **Small zones:** the delivered gate protects them. With 300 women it sells almost nothing
   at a 90% promise, which is the honest answer. The old gate would sell 95 subscriptions there
   and credit back about a sixth of the revenue.
2. **Large zones:** the gate is sensitive. Near 90% the curve is flat, so a forecast error of
   2 to 3 points moves the stopping point by about 100 men. In the 500-woman zone it stopped
   at 56 men although 158 would have done about as well. A slightly lower target, or a
   forecast refitted on real cycles, would admit many more.

---

## 6. What changed in the plan

- The Phase 1 gate uses the delivered forecast (`[gate] forecast`).
- Sprint 7's numbers are corrected everywhere they appear: README, fit report, risk plan, and
  the note at the top of the Sprint 7 notes.
- My first FR-LM-45 test said the delivered gate's men would beat the booked gate's men in
  every zone. In the 500-woman zone they didn't, because 3:1 was already enough. The test now
  checks what the gate is for: protecting small zones, and staying calibrated where it admits
  men.

---

## 7. Honest limits

- **People don't adapt** in the test run: scarcity doesn't make anyone less picky. This
  drives the "zone size over ratio" result.
- **The forecast assumes today's friction and recruitment.** If show-ups, cancellations or the
  rate of new women change, refit (the file lists what it assumed).
- **Markets differ by about ±5 points** in ways no feature captures. Expect that much
  uncertainty in any single zone.
- **The gate decides once, at launch.** Re-running it every cycle with real data, as men leave
  and women join, is the natural next step.
- **Above the fitted range** (more than 7 women per man), the forecast extrapolates, mostly
  through pool size.

---

## 8. Maths to study

1. **Logistic regression with share labels:** why the gradient Σ(p − y)x works for y between 0
   and 1.
2. **Calibration versus discrimination:** the alerts rank men almost perfectly (AUC 0.99),
   while the start forecast's *level* is uncertain by a few points. These are different
   qualities.
3. **Sensitivity of a threshold rule:** where a curve is flat, a small error in the forecast
   moves the crossing point a lot (Δmen ≈ Δforecast ÷ slope).
4. **Confounding:** with 100 men fixed, a higher ratio also means more women. Section 5's zone
   comparison separates the two effects.

## Try it yourself

```bash
.venv/bin/python -m lastfirstdate.gate_report             # the gate in both zones (~8 min)
.venv/bin/python -m lastfirstdate.gate_report --ratios    # plus the women-per-man table
.venv/bin/python -m lastfirstdate.forecast_fit --delivered --no-save   # refit without saving
.venv/bin/python -m pytest tests/test_sprint8.py -q
```

---

## Questions for you

1. **Zone size:** can Phase 1 recruit about 500 committed women, the top of your plan, before
   selling subscriptions? With 300, the honest 90% promise covers only a handful of men.
2. **The target:** keep 90%, or use 85% to 88% in large zones, where the gate is very sensitive?
   At 300 women even 80% would admit only a few dozen men.
3. **Selling below 90%:** if you sell more subscriptions than the 90% gate allows, do you
   accept the credit cost? About 16% of revenue at 300 women and 3:1.
4. **Sprint 9:** (a) re-run the gate every cycle with real data and the recruitment rate as an
   input; (b) the handoff to the app developer: which engine call happens when; (c) a service
   the app can call. My recommendation is (b), so whoever builds the app can start.
