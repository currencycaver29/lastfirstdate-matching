# Fit report: the matching engine and lastfirstdate

**As of 4 November 2026.** This report checks the engine in this repo against what
lastfirstdate actually promises and plans, as published on its website. It covers which
rules match, what the engine does differently, what the onboarding must collect, and how the
real business rules perform in simulation.

**Sources (public pages, read 4 November 2026):**
[home](https://lastfirstdate.vercel.app/), [product demo](https://lastfirstdate.vercel.app/product/),
[business](https://lastfirstdate.vercel.app/business/),
[investors](https://lastfirstdate.vercel.app/investors/),
[privacy](https://lastfirstdate.vercel.app/privacy/). The product demo is labeled "Illustrative ·
Sample data", and the privacy page says the site is a prototype whose forms do not save
anything. **No app or member database exists yet**, so the onboarding can still be designed
around what the engine needs.

**All simulation numbers come from fake members.** They compare designs and expose
arithmetic; they are not predictions of real behavior.

## Summary

1. **The core design fits.** AI ranking before her daily list, "she chooses", two-way
   matches, a photo exercise that learns each member's type, no attractiveness scores, and
   learning from private post-date ratings are all built and tested.
2. **The Phase 1 plan works in simulation.** In one NYC zone with 100 paying men, about **87%**
   of men reach 3 dates with 300 women (3:1) and about **94%** with 500 women (5:1), close to and
   above the 90% north star.
3. **The women's promise needs rewording.** At 3:1 the average woman can get at most about 1 date
   a month (100 men × 3 dates = 300 dates for 300 women), and in simulation 34% get none. "Commit
   to three dates a month" should read "up to three".
4. **The gender ratio is the biggest business risk.** With the site's rules but a city-wide mix of
   55% men, only about 52% of men reach 3 dates.
5. **Several app details differ from the engine** (section 4). The most important: 30 photos
   including pairs, single-answer intent questions, free-text prompts, a proposed date window
   shown with each profile, per-member billing cycles, and a private trust score.
6. **Two items on the site need fixing before launch:** the phrase "attractive women" on the
   business page, and the placeholder privacy policy.

## 1. What lastfirstdate promises and plans

| Topic | What the site says |
|---|---|
| Men | $200 a month after a 30-day free trial; $200k+ earned income verified through Plaid; **3 guaranteed dates per cycle**, pro-rated credit if fewer |
| Women | Free; up to 30 verified profiles a day; choose who they meet; "commit to three dates a month" |
| Every date | $10 per person at booking; full refund 24h+ before, 50% within 24h, none for no-shows |
| Delivered date | Both confirmed a venue and time, both paid, scheduled before his cycle ends; no-shows and cancellations don't count and the slot is refilled |
| The promise to men | "Three dates with committed women who chose and accepted them, not three women who show up" |
| Verification | ID, liveness and face match for everyone |
| Matching | "AI ranks candidates on shared interests, personality fit from the intent questionnaire, mutual attraction and location midpoint"; "Matches are two-way"; "No attractiveness scores, ever" |
| Flow | His profile appears in her up-to-30 daily profiles **with a proposed date window**; she chooses (or suggests another time); both $10 payments must go through within 15 minutes; then the venue is revealed and chat opens |
| Photo exercise | 30 photos in the demo, single photos ("Interested?") and pairs ("Who are you more drawn to?") |
| Intent questions | Single answers, e.g. "When do you want to be in a committed relationship? Now / Within a year / No timeline"; "Exclusivity after a few great dates? Yes / Open to discuss / Not yet"; "Kids someday? Want them / Open to it / Not for me". "Scored transparently, and you can always appeal" |
| Profile | Name, age, neighborhood, job, employer, photos, free-text prompts ("A perfect Sunday looks like…", "I'm looking for…") |
| After the date | Private ratings on respectfulness, accuracy of profile and conversation, plus "would meet again"; a private trust score feeds ranking and safety; men whose dates consistently rate poorly are flagged for review |
| Events | Free members-only events; a connection made there can become one of the 3 dates |
| Launch plan | Phase 1: one NYC zone; recruit 300 to 500 committed women before selling the first 100 subscriptions; "if the ratio isn't there, male sales pause". North star: 90%+ guarantee fulfilment in month one |

## 2. Rule by rule

| The site says | The engine | Fit |
|---|---|---|
| Every profile ranked by AI before her daily list; up to 30 a day | Stages 1 to 6; `lists.daily_size = 30` | Yes |
| She chooses; he can't force a date | Women pick (decision D1); men can only be picked | Yes |
| Two-way matches | The mutual score M (Stage 5) | Yes |
| The photo exercise teaches your type; no attractiveness scores | Calibration plus personal taste (Stages 2 and 3); rule R2 | Yes, but see pairs and 30 photos in section 4 |
| Intent questions: values, lifestyle, timelines, dealbreakers | Values questions with accepted answers and dealbreakers | Partly: the site uses single answers |
| Shared interests from prompts | Interest tags; Sentence-BERT plugs in later | Partly: prompts are free text |
| Learns from ratings and "would meet again" | Stage 8 (Sprint 5) | Built, but no measurable gain in simulation yet |
| Guarantee for every subscriber, at-risk alerts trigger priority matching | Guarantee bonus on lists and urgency in the scheduler | Yes, but the earlier simulations assumed only half of men pay |
| Women commit to 3 dates | Not modeled | Gap |
| Proposed date window shown with each profile | A weekly batch schedule after picks | Gap |
| Delivered = confirmed and paid before his cycle ends; refills after cancellations | Booked dates per calendar month | Gap |
| Private trust score feeds ranking; poorly rated men flagged | One 1 to 5 rating; no trust score | Gap |
| Event dates count toward the 3 | Not modeled | Gap |
| Man-woman only | Supports all orientations | Simpler configuration |

## 3. The real business rules in simulation

### Phase 1 as planned

One dense NYC zone, 100 paying men, man-woman only, nobody over 3 dates, 30 calibration photos.
Five simulated months each:

| | 300 women (3:1) | 500 women (5:1) |
|---|---|---|
| Median possible matches per man | 70 | 133 |
| **Men reaching 3 dates** | **87.3% ± 1.3** | **93.9% ± 1.1** |
| Average dates per woman | 1.10 | 0.64 |
| Women who did 3 dates | 11.3% | 2.4% |
| Women with no date that month | **34.3%** | **53.0%** |

(The fake population marks about 8% of members unverified and 15% inactive, so about 85 to 91
of the 100 men were matchable. In a real beta everyone is verified.)

### The same rules across a city, by share of men

1,000 fake members spread across NYC, every man paying, women free, nobody over 3 dates. Three
simulated months each:

| Share of men | Men reaching 3 dates | Women completing 3 | Missing dates per man |
|---|---|---|---|
| 55% | 51.8% | 75.9% | 0.80 |
| 50% | 75.3% | 62.4% | 0.51 |
| 45% | 79.3% | 49.8% | 0.45 |
| 40% | 83.0% | 36.9% | 0.38 |

If each missed date is credited at a third of $200, missing dates cost about $25 to $53 per man
per month, or 13% to 27% of subscription revenue.

### What this means

- **The ratio plan on the business page is the right instinct**, and the engine supports it.
- **Every date needs one man and one woman.** With men capped at 3 dates, the average woman's
  dates are at most 3 × men ÷ women. Women's "commitment to three" can only be a willingness, not
  something most women will reach.
- **Many women will have empty months.** This is a retention risk that free events can help
  with.
- **The simulation does not model the commitment's effect.** Women who must do dates may pick more
  widely than the fake women, which would help men's numbers.

## 4. Where the app differs from the engine, and the change needed

| The app (from the demo) | The engine today | Change |
|---|---|---|
| 30 photos, single and **pairs** | 50 single like/pass swipes | Support pair comparisons (P(A over B) = sigmoid(w · (x_A − x_B))), which fits papers 1 and 3 even better, and a 30-photo calibration |
| Intent questions with one answer each | Members also list accepted answers | Compatibility from closeness of answers; some combinations as dealbreakers (for example "Want them" with "Not for me") |
| Free-text prompts | Interest tags | A text encoder: Sentence-BERT in production, a word-level offline encoder for tests |
| No Big Five in the demo | Big Five is 30% of P | Add personality questions, or reweight to text plus intent |
| Neighborhood, venue at the location midpoint | Exact coordinates and straight-line distance | Neighborhood centers; distance matters little in a one-zone beta |
| A proposed date window with each profile; 15-minute payment hold | Weekly batch scheduling | Propose a window per candidate at list time; needs members' available evenings |
| Per-member billing cycles; refills after cancellations | Calendar months; booked dates | Track cycles, count only confirmed and paid dates, refill |
| Three rating dimensions plus trust score in ranking | One rating plus "would meet again" | Store the dimensions; a behavior-only trust score as a ranking factor and safety flag |
| Event connections count as dates | Not modeled | Accept dates booked outside the engine |
| Man-woman only | All orientations | Configuration |

## 5. What the onboarding must collect

| Data | Why the engine needs it | Seen on the site? |
|---|---|---|
| Gender and who they want to meet | Mutual orientation lock (R1) | Implied (man-woman) |
| Age, and preferred age range | Stage 1 filter | Age yes, range not seen |
| Neighborhood, and how far they'll travel | Stage 1 filter, venue midpoint | Neighborhood yes, distance not seen |
| **Available evenings each week** | Proposing a date window | Not seen |
| 30 calibration photos, singles and pairs | Learning their type (Stages 2 and 3) | Yes |
| Intent answers | Stage 4 and dealbreakers | Yes (single answers) |
| Free-text prompts | Interest similarity | Yes |
| Personality questions | Stage 4 | "Personality questions" mentioned; content not seen |
| Their own photos | Being scored by others' tastes | Yes |
| Verification result only | Filter | Yes |

## 6. Fix on the site before launch

1. **"250,000 TAM — attractive women"** on the business page contradicts "No attractiveness scores,
   ever" and could create legal and reputational risk. Something like "eligible single women"
   says the same thing safely.
2. **The privacy policy is a placeholder.** It must cover photo embeddings and face matching as
   biometric data, and income verification data, before any real data is collected.

## 7. Recommended next steps

1. Design the onboarding around section 5.
2. Build the ratio gate and fulfilment forecast into the engine, so "male sales pause" happens
   automatically (Sprint 6).
3. Adapt the engine to the app's real rules in section 4: the data-model items in Sprint 6, the
   booking and billing items once the app has a database.
4. Brief the lawyer with `docs/risk-plan.md`.
5. When the app has a repo, check its real fields against section 5.
