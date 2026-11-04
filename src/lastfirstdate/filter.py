"""Stage 1: the hard filter (FR-LM-04).

A pair passes only if every rule holds in both directions. Anyone who fails is never
shown, scored or scheduled, however high they would score.

Rules: verified (when required), active recently, mutual orientation lock (R1), each age
inside the other's range, distance within both people's limit (D9), each passes the
other's dealbreakers (R6), enough shared free slots, and not already dated or blocked.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lastfirstdate.arrays import MemberArrays, distance_matrix, haversine_km, shared_slots
from lastfirstdate.config import Config, ValuesQuestion
from lastfirstdate.models import Member


def member_ok(arr: MemberArrays, cfg: Config) -> np.ndarray:
    """(n,) members who can be matched at all: verified (if required) and active."""
    ok = arr.last_active <= cfg.filter.active_within_days
    if cfg.filter.require_verified:
        ok = ok & arr.verified
    return ok


def dealbreakers_pass(
    arr: MemberArrays, questions: Sequence[ValuesQuestion] | None = None
) -> np.ndarray:
    """(n, n) True where no dealbreaker rules the pair out, in either direction.

    "accepts" questions: j's answer must be one i accepts on each of i's dealbreakers.
    "closeness" questions: the two answers must not be a listed incompatible pair.
    """
    n = arr.n
    one_way = np.ones((n, n), dtype=bool)
    for q in range(arr.answer.shape[1]):
        question = questions[q] if questions is not None else None
        if question is not None and question.mode == "closeness":
            if question.incompatible:
                k = len(question.options)
                bad = np.zeros((k, k), dtype=bool)
                for x, y in question.incompatible:
                    a, b = question.options.index(x), question.options.index(y)
                    bad[a, b] = bad[b, a] = True
                one_way &= ~bad[arr.answer[:, q][:, None], arr.answer[:, q][None, :]]
            continue
        # bit answer[j] of accepts[i] says whether i accepts j's answer to question q
        accepted = ((arr.accepts[:, q][:, None] >> arr.answer[:, q][None, :]) & 1).astype(bool)
        one_way &= ~arr.dealbreaker[:, q][:, None] | accepted
    return one_way & one_way.T


def eligibility_matrix(
    arr: MemberArrays, cfg: Config, excluded: np.ndarray | None = None
) -> np.ndarray:
    """(n, n) boolean: True where members i and j may be matched.

    `excluded` marks pairs that already dated or blocked each other. The result is
    symmetric and the diagonal is False.
    """
    ok = member_ok(arr, cfg)
    e = ok[:, None] & ok[None, :]
    # R1 mutual orientation lock
    e &= (arr.seeking[:, None] & arr.gender_bit[None, :]) != 0
    e &= (arr.seeking[None, :] & arr.gender_bit[:, None]) != 0
    # each age inside the other's range
    e &= (arr.age_min[:, None] <= arr.age[None, :]) & (arr.age[None, :] <= arr.age_max[:, None])
    e &= (arr.age_min[None, :] <= arr.age[:, None]) & (arr.age[:, None] <= arr.age_max[None, :])
    # D9 distance within both limits
    e &= distance_matrix(arr) <= np.minimum(arr.max_km[:, None], arr.max_km[None, :])
    # R6 dealbreakers are hard filters
    e &= dealbreakers_pass(arr, cfg.values)
    # a date needs a shared free slot
    e &= shared_slots(arr) >= max(1, cfg.filter.min_shared_slots)
    if excluded is not None:
        e &= ~excluded
    np.fill_diagonal(e, False)
    return e


def pair_passes(a: Member, b: Member, cfg: Config) -> bool:
    """Plain-Python version of the filter for one pair. Used as the reference in tests."""
    f = cfg.filter
    for m in (a, b):
        if f.require_verified and not m.verified:
            return False
        if m.last_active_days > f.active_within_days:
            return False
    if a.id == b.id:
        return False
    if b.gender not in a.seeking or a.gender not in b.seeking:
        return False
    if not (
        a.age_min_pref <= b.age <= a.age_max_pref and b.age_min_pref <= a.age <= b.age_max_pref
    ):
        return False
    if haversine_km(a.lat, a.lon, b.lat, b.lon) > min(a.max_km, b.max_km):
        return False
    modes = {q.key: q for q in cfg.values}
    for x, y in ((a, b), (b, a)):
        for q in x.dealbreakers:
            if modes[q].mode == "accepts" and y.values[q] not in x.accepts[q]:
                return False
    for q in cfg.values:
        if q.mode == "closeness":
            pair = {a.values[q.key], b.values[q.key]}
            if any(pair == {x, y} for x, y in q.incompatible):
                return False
    return len(a.free_slots & b.free_slots) >= max(1, f.min_shared_slots)
