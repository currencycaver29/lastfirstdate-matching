"""Stage 5: mix looks and personality 50/50, then make the score mutual (FR-LM-18 to FR-LM-20).

1. Percentiles (FR-LM-18). V (looks, 0 to 1) and P (personality, 0 to 1) spread very
   differently: one member's V might run from 0.01 to 0.99 while their P runs from 0.35 to
   0.65. Mixed raw, the wider one would decide almost everything. So each is replaced by
   its percentile inside member i's own filtered pool: the best candidate gets close to 1,
   the median 0.5, the worst close to 0.

       pct_i(j) = (rank of j among i's k candidates - 0.5) / k      (ties share a rank)

2. Directional score (FR-LM-19). How much i would like j:

       S(i -> j) = looks_weight * pct_i(V_i(j)) + (1 - looks_weight) * pct_i(P_i(j))

3. Mutual score (FR-LM-20, papers 6 and 10). A date needs both people to say yes, so
   combine the two directions with the harmonic mean:

       M(i, j) = 2 * S(i -> j) * S(j -> i) / (S(i -> j) + S(j -> i))

   The harmonic mean is dragged toward the smaller of the two: 0.9 and 0.1 give 0.18,
   while the ordinary average would give 0.5. So M is high only when both are high.

   `mix.chooser_weight` (a) generalizes it to a weighted harmonic mean,
   M(i -> j) = 1 / (a / S(i -> j) + (1 - a) / S(j -> i)). At the default a = 0.5 this is
   exactly the formula above and M is symmetric. A larger a trusts the chooser's own view
   more, which helps when choosers are the picky side (Sprint 3 study notes).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata

from lastfirstdate.looks import PersonalTaste


def percentile_rows(x: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """(n, n) each row's values replaced by their percentile inside that row's pool.

    Entries outside the pool are 0. A pool of k candidates gets the evenly spaced values
    0.5/k, 1.5/k, ..., (k - 0.5)/k, so their average is exactly 0.5.
    """
    out = np.zeros(x.shape, dtype=float)
    for i in range(x.shape[0]):
        idx = np.flatnonzero(pool[i])
        if idx.size:
            out[i, idx] = (rankdata(x[i, idx]) - 0.5) / idx.size
    return out


def directional_score(
    looks: np.ndarray, personality: np.ndarray, pool: np.ndarray, looks_weight: float
) -> np.ndarray:
    """(n, n) S(i -> j), 0 outside i's pool."""
    if not 0.0 <= looks_weight <= 1.0:
        raise ValueError("looks_weight must be between 0 and 1")
    s = looks_weight * percentile_rows(looks, pool) + (1 - looks_weight) * percentile_rows(
        personality, pool
    )
    return np.where(pool, s, 0.0)


def mutual_score(directional: np.ndarray, chooser_weight: float = 0.5) -> np.ndarray:
    """(n, n) M(i -> j): weighted harmonic mean of S(i -> j) and S(j -> i).

    0 when either side is 0. With chooser_weight 0.5 it is the plain harmonic mean,
    2ab / (a + b), and symmetric.
    """
    if not 0.0 < chooser_weight < 1.0:
        raise ValueError("chooser_weight must be strictly between 0 and 1")
    a, b = directional, directional.T
    both = (a > 0) & (b > 0)
    denom = chooser_weight * b + (1 - chooser_weight) * a  # a*b / M, rearranged
    return np.divide(a * b, denom, out=np.zeros_like(a, dtype=float), where=both)


def looks_scores(
    tastes: Mapping[int, PersonalTaste],
    features: np.ndarray,
    genders: np.ndarray,
    pool: np.ndarray,
) -> np.ndarray:
    """(n, n) V_i(j) for every member with a finished calibration, inside their pool.

    Members without a taste model (not matchable) get a row of zeros. `list_scores`
    refuses unfinished calibrations, so w_0 alone can never reach a list (R2).
    """
    v = np.zeros(pool.shape, dtype=float)
    for i, taste in tastes.items():
        idx = np.flatnonzero(pool[i])
        if idx.size:
            v[i, idx] = taste.list_scores(features[idx], genders[idx])
    return v
