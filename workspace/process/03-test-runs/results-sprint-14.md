# Sprint 14 results: abuse testing

Tests: `python -m pytest tests/test_sprint14.py -q` (26 tests, about a minute).
The attack report: `python -m lastfirstdate.attacks` (about 6 minutes on this machine). Phase 1
settings, zones of 300 women and 100 men at launch, seeds 2026, 2027 and 2028, which no setting
was tuned on. Means over the three seeds. Synthetic members only (R3).

## 1. Stuffed prompts (FR-LM-64, D21)

Ten men per zone rewrite their prompts. Their mean percentile in eligible women's personality
rankings (0.5 is the middle):

| Prompt | Centring off | Centring on |
|---|---:|---:|
| honest | 0.491 | 0.480 |
| popular words (a list of the interests women mention most) | 0.592 | 0.457 |
| popular, as a prompt (the top twelve in the app's prompt sentences) | 0.517 | 0.475 |
| everyone's words (the words nearly every prompt contains, repeated) | **0.674** | 0.450 |

The same attack read by MiniLM, the production text model
(`python -m lastfirstdate.attacks --minilm MODELS_DIR`, model files already downloaded):

| Prompt | Centring off | Centring on |
|---|---:|---:|
| honest | 0.494 | 0.484 |
| popular words | 0.228 | 0.429 |
| popular, as a prompt | **0.587** | 0.475 |
| everyone's words | 0.183 | 0.416 |

The two encoders reward different stuffing. Word counts reward repeating shared words; MiniLM
punishes a bare list (it lands far from real prompts) but rewards popular interests written
as a prompt. Centring removes every gain for both.

What it buys in the real flow (everyone's words, three cycles), attackers against the other
men:

| | In the top 10 of | Shown per cycle | Picked per cycle |
|---|---:|---:|---:|
| no attack | 9% vs 10% of eligible women | 161.7 vs 168.9 | 11.7 vs 12.5 |
| stuffed, centring off | **12%** vs 10% | 158.3 vs 175.3 | 10.8 vs 12.4 |
| stuffed, centring on | 8% vs 10% | 152.2 vs 176.3 | 11.3 vs 12.2 |

Stuffing moves the attackers up women's rankings but doesn't get them shown or picked more:
the exposure cap limits how often anyone is shown, and in the test runs women pick on true
appeal, which no prompt changes. Real people have a position bias (they pick from the top more
often); the test runs don't model it, so they understate the harm of a distorted ranking.

**Honest matching with centring.** On the tuning seeds (Phase 1 settings, seeds 101 to 103),
chance both say yes in the top 10 moved 0.1681 → 0.1668, 0.1936 → 0.1960, 0.2076 → 0.2072,
and men with 3 delivered dates moved 73.8% → 72.1%, 77.0% → 80.0%, 71.5% → 72.6%: within
noise. Code:

```python
from lastfirstdate.attacks import centered
from lastfirstdate.config import load
from lastfirstdate.cycle_runs import CycleRun, join_days, market
from lastfirstdate.mutual_report import build_scores, top_quality_all
cfg = load("config/phase1.toml")
rate = cfg.cycle_runs.new_women_per_cycle
for seed in (101, 102, 103):
    for on in (False, True):
        c = centered(cfg, on)
        pop = market(c, seed, rate)
        join = join_days(pop, c, rate)
        sc = build_scores(pop, c, seed)
        top = top_quality_all(pop, c, seed, sc)["mutual"].mutual
        rep = CycleRun(pop, c, seed, sc.mutual, join).run("")
        print(seed, on, round(top, 4), round(rep.guarantee_share, 3))
```

The NYC demo (`python -m lastfirstdate.nyc_report`, held-out seeds 2026 to 2030), before
(v0.13.1, centring off) and after:

| | 300 women, before | after | 500 women, before | after |
|---|---:|---:|---:|---:|
| Men with 3 delivered dates, all cycles | 75.5% | 77.2% | 90.2% | 90.6% |
| Credits, share of subscriptions | 16.1% | 14.8% | 6.6% | 6.0% |
| Top 10, chance both say yes (mutual) | 20.4% | 20.4% | 23.3% | 23.3% |
| Bad actors caught after 6+ ratings | 41 of 44 | 40 of 40 | 64 of 66 | 57 of 60 |
| Members wrongly flagged | 0 | 0 | 0 | 1 |

Moving any pair's score changes who dates whom, so the counts of well-rated bad actors and the
one wrongly flagged member move with it; they're not an effect of centring on trust.

## 2-3. The starting taste (FR-LM-65, D23)

The target man's mean percentile in honest women's looks rankings, and honest women's looks
AUC against their hidden taste:

| Attack | Target | Honest AUC |
|---|---:|---:|
| no attack | 0.473 | 0.936 |
| 1 account floods 20 exercises, no defence | **0.517** | 0.934 |
| 1 account floods 20 exercises, one exercise per account | 0.473 | 0.936 |
| 15 verified fakes (5% of women) | 0.510 | 0.934 |
| 30 verified fakes (10%) | **0.537** | 0.932 |
| 30 unverified fakes, verified-only fit | 0.471 | 0.936 |
| 30 verified fakes, robust fit (trimmed mean of each member's fit) | 0.488 | 0.917 |
| no attack, robust fit | 0.461 | 0.920 |

The robust fit halves the gain from verified fakes (+0.064 → +0.027) but costs every honest
member 0.016 AUC, so it is not used: identity checks are the defence.

## 4. Fake women on the gate (FR-LM-66, D22)

The gate at launch with 300 honest women and 450 men waiting; the fakes are counted but never
date:

| | Women counted | Places sold | Forecast | Actual 3-date share |
|---|---:|---:|---:|---:|
| no fakes | 284 | 3.0 | 94.1% | 94.4% |
| 50 fakes | 331 | **6.7** | 95.1% | **82.6%** |

After four weeks of the real flow, with the fakes opening the app every day and never picking,
the silent rule (14 list days, no pick ever) stops counting **83% of the fakes and 0.6% of
honest women**. At launch, before any lists, only identity checks stop fakes.

## 5. Rating attacks (FR-LM-69)

Three cycles of the real flow per seed.

| Seed | Honest men wrongly flagged: no attack | with 15 harsh raters | Targets paused: ring of 3 | ring of 6 |
|---|---:|---:|---:|---:|
| 2026 | 0 | 0 | 2 of 8 | 2 of 8 |
| 2027 | 0 | 0 | 3 of 8 | 3 of 8 |
| 2028 | 0 | 1 | 0 of 8 | 3 of 8 |

Rings are women eligible for the target who pick him whenever he's on their list, whom he
always accepts, and who rate him 1: an upper bound, since a real man can decline. A ring
succeeds only when it gets 3 dates with him before his honest ratings build up; he has at most
3 dates a cycle, so most rings never get them.

## 6. Spies (FR-LM-70)

20 accounts per zone that never pick read their lists every day for two weeks and rank men by
when, then where, each first appears, pooled over the accounts. Spearman correlation with how
much the zone's honest women like each man's looks (hidden truth):

| | Spearman |
|---|---:|
| exposure cap on (the engine) | **0.042** |
| exposure cap off | 0.135 |

## Malformed rows (FR-LM-67, FR-LM-68)

Checked by tests, not measured: ratings on cancelled bookings, from strangers or naming the
wrong person change no trust; seven kinds of malformed booking row are refused by the schema
and, without the schema's checks, named by `app.load_bookings`; payments from strangers are
ignored; `app.check_rows` lists all of these and is empty on a clean database.
