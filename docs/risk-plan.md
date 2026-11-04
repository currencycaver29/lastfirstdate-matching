# Risk plan

**As of 4 November 2026.** No app can take risk to zero. This plan shrinks each known risk to a
small, measured size and catches problems early. It is grouped by when each action is needed,
and every action has a suggested owner. Tick items off as they are done.

Background for the business numbers is in `docs/fit-report.md`. **This is not legal advice.**
Section 2 is a briefing for a New York lawyer.

Owners: **Founder** (business decisions), **Engineering** (whoever builds the app),
**Engine** (this repo), **Lawyer**, **Security tester** (outside firm).

## 1. Before the Phase 1 beta

### Business

- [ ] **Enforce the gender ratio in software.** In each zone, men's sign-ups open only while active
  committed women are at least 3 per paying man and the forecast fulfilment is at least 90%.
  *Owner: Engine (Sprint 6) and Engineering.* Why: at 55% men only about 52% of men reach 3 dates.
- [ ] **Reword the women's commitment** to "up to three dates a month". At 3:1 the average woman
  can have at most about 1 date a month. *Owner: Founder.*
- [ ] **Budget for credits** from day one (13% to 27% of revenue in the city-wide simulations) and
  track fulfilment weekly. *Owner: Founder.*
- [ ] **Decide who the guarantee is offered to.** In simulation, offering it only to men with 50 or
  more possible matches reaches about 96%. Pool size depends only on a member's own settings and
  location, so it is fair to use. Never use how often someone is picked: that would be a
  desirability score (rule R2). *Owner: Founder.*
- [ ] **Plan for women's empty months** (34% to 53% of women had no date in a simulated Phase 1
  month): events, honest expectations, and a measure of it on the dashboard. *Owner: Founder.*

### Security and privacy

- [ ] **Server-side access checks on every request.** A member may only see profiles on their own
  list; contact details stay hidden on the server, not just in the screens; use per-row access
  rules in the database if available. *Owner: Engineering.*
- [ ] **Use a proven login provider**; never store passwords yourself. *Owner: Engineering.*
- [ ] **Two-factor sign-in** on GitHub, Vercel, Stripe, Plaid, the identity vendor and the database.
  *Owner: Founder.*
- [ ] **Payments through Stripe only**; never handle card numbers. *Owner: Engineering.*
- [ ] **Store only verification results** ("verified", "$200k+"), never bank transactions or ID
  images, as the site already promises. *Owner: Engineering.*
- [ ] **Treat photo embeddings and face matching as biometric data:** a separate encrypted store,
  explicit consent, deletion when a member leaves, and a written retention policy.
  *Owner: Engineering and Lawyer.*
- [ ] **Secrets only in environment settings** (Vercel and the engine's host), never in code;
  separate test and live keys. *Owner: Engineering.*
- [ ] **Daily database backups**, with a tested restore. *Owner: Engineering.*
- [ ] **A breach plan:** who decides, who is notified, and how fast. *Owner: Founder and Lawyer.*

### Legal and public wording

- [ ] **Brief a New York lawyer** with section 2 and get answers before launch. *Owner: Founder.*
- [ ] **Real privacy policy and terms**, replacing the placeholders. *Owner: Lawyer.*
- [ ] **Remove "attractive women"** from the business page. *Owner: Founder.*

### Member safety

- [ ] **Report and block**, with a ban list that stops banned people from re-applying.
  *Owner: Engineering.*
- [ ] **Human review** of reports and of members whose trust scores or ratings are flagged, with
  someone reachable during date hours. *Owner: Founder.*
- [ ] **Photo moderation** for new uploads. *Owner: Engineering.*

### Engineering

- [x] **Automatic checks on every push** to the matching repo (GitHub Actions running lint, format,
  tests and traceability). *Owner: Engine.*
- [ ] **A staging copy** of the app with fake data, separate from the live app. *Owner: Engineering.*

## 2. Questions for a New York lawyer (ready to send)

> **Subject: Legal review before launch: lastfirstdate (NYC dating service)**
>
> Hello,
>
> lastfirstdate is a verified, AI-matched dating service launching first in one New York City
> neighborhood (closed beta), then citywide. Before launch we would like your advice on the
> questions below. Summary of the model: men pay $200 a month (after a 30-day free trial) for three
> guaranteed dates per billing cycle, with pro-rated credit if fewer are delivered, and must verify
> $200,000+ earned income through Plaid. Women join free, see up to 30 verified profiles a day,
> choose who they meet, and commit to dates. Each date costs $10 per person at booking. Everyone
> passes ID, liveness and face-match checks. Matching uses each member's photo preferences (learned
> from a short photo exercise), interests, and an intent questionnaire. After each date both people
> rate each other privately; the ratings feed a private trust score used in ranking and safety. We
> also host free members-only events.
>
> 1. **Different terms for men and women.** Men pay and must meet an income threshold; women join
>    free. Is this permitted under the New York State Human Rights Law and the New York City Human
>    Rights Law (including any rules on gender-based pricing in public accommodations)? If not, what
>    structure would be?
> 2. **Income verification.** We use Plaid payroll or bank data to confirm $200k+ earned income and
>    keep only the pass or fail result and a tier badge. Do consumer-reporting or financial-privacy
>    rules apply (for example the Fair Credit Reporting Act), and what consent, retention and
>    adverse-action steps do we need?
> 3. **Biometric data.** We run face matching during verification (through a vendor) and turn
>    member photos into numerical embeddings for matching. What do New York's SHIELD Act and any New
>    York City biometric rules require of us (consent, security, retention, deletion, breach
>    notification)? Do other states' biometric laws (for example Illinois) matter if members travel
>    or move? What should our vendor contracts say?
> 4. **Looks-based matching and fairness.** Photo preferences can correlate with race, ethnicity and
>    age. We run a monthly audit of who is shown and who gets dates, by group and age band. What
>    documentation and audit practices reduce discrimination risk, and is there anything we should
>    not do?
> 5. **The trust score.** Ratings on respectfulness, profile accuracy and conversation feed a
>    private score used in ranking and in flagging members for review. What notice, appeal and
>    data-access rights should members have?
> 6. **The guarantee.** Our terms will say the promise is "three dates with committed women who chose
>    and accepted them, not three women who show up", with pro-rated credit for shortfalls and
>    refills for cancellations. Is that wording adequate, and how should disputes be handled?
> 7. **Free trial and renewal.** The 30-day free trial converts to $200 a month. What do New York's
>    automatic-renewal rules require (disclosures, consent, cancellation)?
> 8. **Date fees and refunds.** $10 per person per date: full refund 24 hours or more before, 50%
>    within 24 hours, none for no-shows. Any issues with this policy, chargebacks or holding funds
>    for 15 minutes before capture?
> 9. **The women's commitment.** We ask women to commit to dates each month. Can we enforce it, and
>    how should it be worded so it is not misleading if a woman cannot reach the full number?
> 10. **Marketing claims.** Words like "guaranteed", "AI-matched" and "verified", illustrative sample
>     stories, and how we describe our market: what do FTC and New York consumer-protection rules
>     require?
> 11. **Safety obligations.** Are we required to run background checks, or to show safety notices if
>     we do not? What is our exposure for incidents at dates, at partner venues, and at our events
>     (including alcohol and ID checks at the door)?
> 12. **Privacy policy and terms.** Please draft or review both, including members' rights to access
>     and delete their data and our breach-notification duties.
>
> Thank you. We can share product screens, data-flow diagrams and the matching engine's
> documentation.

## 3. Before the citywide launch

- [ ] **An outside security test** (penetration test) of the app and its database.
  *Owner: Security tester.*
- [ ] **Fulfilment at or above the target for at least two cycles in the beta zone** before
  opening the next zone. *Owner: Founder.*
- [ ] **Real data replaces the simulator's guesses:** measure how picky each side is, how often
  people show up, and the share of women with empty months; rerun the reports with those numbers.
  *Owner: Engine.*

## 4. Ongoing

- [ ] **Monthly fairness audit** on real data, with a person reviewing every flag (the fake-data
  audit already flagged one age band). *Owner: Engine and Founder.*
- [ ] **Rule R2 stays enforced:** no global attractiveness score anywhere; the trust score measures
  behavior only. *Owner: Engine.*
- [ ] **Small A/B tests** to settle what the simulation could only guess: 30 versus 50 calibration
  photos, the chooser weight, exploration. *Owner: Engine.*
- [ ] **Explain decisions for appeals** (the intent check promises "you can always appeal").
  *Owner: Engineering.*
- [ ] **Keep dependencies updated** and review security alerts weekly. *Owner: Engineering.*
- [ ] **Watch the dashboard:** guarantee fulfilment, credit rate, women's commitment rate, women with
  empty months, no-show rate, month-1 churn. *Owner: Founder.*
