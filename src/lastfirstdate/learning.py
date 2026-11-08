"""Stage 8: learning from dates and list choices (FR-LM-25 to FR-LM-27).

After each month the engine looks at what actually happened and updates two things.

1. Looks taste (FR-LM-25). Every held date adds one labeled example to each person's looks
   model: "would meet again" (yes or no) about the other person's photo features, counted
   `learning.taste_date_weight` times as much as a calibration swipe.

2. Looks weight (FR-LM-26). Each member gets their own split between looks and personality,
   lambda_i, instead of everyone using 0.5. A small logistic regression per member predicts
   their yes or no from the two percentiles the engine showed them:

       logit P(yes) = b0 + b_V * (pct V - 0.5) + b_P * (pct P - 0.5)
       lambda_i     = b_V / (b_V + b_P)

   Examples: "would meet again" after dates (weight `lambda_date_weight`) and picks and
   passes on daily lists (weight `lambda_choice_weight`). The prior says b_V = b_P =
   `lambda_prior_slope`, so with no data lambda_i = 0.5, and it moves only as evidence builds.
   lambda_i is kept between `lambda_min` and `lambda_max`.

3. Exploration (FR-LM-27). Part of each list (`lists.exploration_share`) is
   reserved, before the rest is filled by score, for the candidates the looks model is least
   sure about, so the model keeps learning instead of only showing what it already believes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from lastfirstdate.config import Config
from lastfirstdate.looks import PersonalTaste, fit_logistic
from lastfirstdate.mutual import percentile_rows


@dataclass(frozen=True)
class DateRecord:
    """What happened on one booked date."""

    a: int
    b: int
    week: int
    held: bool  # both showed up
    again_a: bool  # a would meet b again
    again_b: bool
    rating_a: int  # a's private 1 to 5 rating; 0 if the date was not held
    rating_b: int


@dataclass
class Evidence:
    """Everything one member's yes or no told us, as rows of (pct V, pct P, label, weight)."""

    rows: list[tuple[float, float, float, float]]


def update_tastes(
    tastes: Mapping[int, PersonalTaste],
    records: Sequence[DateRecord],
    features: np.ndarray,
    genders: np.ndarray,
    weight: float,
) -> int:
    """Add held dates to both people's looks models and refit them. Returns dates used."""
    touched: set[int] = set()
    used = 0
    for r in records:
        if not r.held:
            continue
        used += 1
        for me, other, again in ((r.a, r.b, r.again_a), (r.b, r.a, r.again_b)):
            if me in tastes:
                tastes[me].add_label(features[other], genders[other], again, weight, refit=False)
                touched.add(me)
    for me in touched:
        tastes[me].refit()
    return used


def collect_evidence(
    evidence: dict[int, Evidence],
    pct_v: np.ndarray,
    pct_p: np.ndarray,
    choices: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    records: Sequence[DateRecord],
    cfg: Config,
) -> None:
    """Append this month's list choices and date outcomes to each member's evidence."""
    w_choice = cfg.learning.lambda_choice_weight
    w_date = cfg.learning.lambda_date_weight
    if choices is not None and w_choice > 0:
        who, whom, picked = choices
        for i, j, y in zip(who.tolist(), whom.tolist(), picked.tolist(), strict=True):
            evidence.setdefault(i, Evidence([])).rows.append(
                (pct_v[i, j], pct_p[i, j], float(y), w_choice)
            )
    for r in records:
        if not r.held:
            continue
        for me, other, again in ((r.a, r.b, r.again_a), (r.b, r.a, r.again_b)):
            evidence.setdefault(me, Evidence([])).rows.append(
                (pct_v[me, other], pct_p[me, other], float(again), w_date)
            )


def looks_weights(evidence: Mapping[int, Evidence], n: int, cfg: Config) -> np.ndarray:
    """(n,) lambda_i for every member; 0.5 for members with no evidence."""
    learn = cfg.learning
    lam = np.full(n, 0.5)
    prior = np.array([0.0, learn.lambda_prior_slope, learn.lambda_prior_slope])
    for i, ev in evidence.items():
        if not ev.rows:
            continue
        rows = np.array(ev.rows)
        x = np.column_stack([np.ones(len(rows)), rows[:, 0] - 0.5, rows[:, 1] - 0.5])
        w, _ = fit_logistic(
            x, rows[:, 2], learn.lambda_prior_strength, prior_mean=prior, sample_weight=rows[:, 3]
        )
        b_v, b_p = w[1], w[2]
        if b_v + b_p <= 0:
            continue  # no usable signal: keep 0.5
        lam[i] = float(np.clip(b_v / (b_v + b_p), learn.lambda_min, learn.lambda_max))
    return lam


def personal_directional_score(
    looks: np.ndarray, personality: np.ndarray, pool: np.ndarray, lam: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S(i -> j) with each member's own looks weight. Also returns pct V and pct P."""
    pct_v = percentile_rows(looks, pool)
    pct_p = percentile_rows(personality, pool)
    s = lam[:, None] * pct_v + (1 - lam[:, None]) * pct_p
    return np.where(pool, s, 0.0), pct_v, pct_p


def uncertainty_matrix(
    tastes: Mapping[int, PersonalTaste], features: np.ndarray, pool: np.ndarray
) -> np.ndarray:
    """(n, n) how unsure each member's looks model is about each candidate in their pool."""
    u = np.zeros(pool.shape)
    for i, taste in tastes.items():
        idx = np.flatnonzero(pool[i])
        if idx.size:
            u[i, idx] = taste.uncertainty(features[idx])
    return u


def exploration_slots(size: int, share: float) -> int:
    """How many slots of a list of this length go to exploration."""
    return int(round(size * share))


def explore_picks(
    order: Sequence[int],
    available: Mapping[int, np.ndarray],
    uncertainty: np.ndarray,
    k: int | Mapping[int, int],
    cap: int | None = None,
) -> dict[int, list[int]]:
    """Each chooser's exploration slots: the available candidates their model is least sure of.

    `k` is the number of slots, for everyone or per chooser. Filled round-robin (everyone's
    first exploration slot, then everyone's second), so with a `cap` (the fair policy's daily
    exposure cap) no chooser always gets first pick. These slots are reserved before the rest
    of the list is filled by score.
    """
    picks: dict[int, list[int]] = {i: [] for i in order}
    k_of = {i: (k[i] if isinstance(k, Mapping) else k) for i in order}
    rounds = max(k_of.values(), default=0)
    if rounds <= 0:
        return picks
    ranked = {}
    for i in order:
        cand = np.flatnonzero(available[i])
        ranked[i] = cand[np.lexsort((cand, -uncertainty[i, cand]))].tolist()
    exposure: dict[int, int] = {}
    pointer = dict.fromkeys(order, 0)
    for e in range(rounds):
        for i in order:
            if e >= k_of[i]:
                continue
            r, p = ranked[i], pointer[i]
            while p < len(r) and cap is not None and exposure.get(r[p], 0) >= cap:
                p += 1
            if p < len(r):
                picks[i].append(r[p])
                exposure[r[p]] = exposure.get(r[p], 0) + 1
                p += 1
            pointer[i] = p
    return picks
