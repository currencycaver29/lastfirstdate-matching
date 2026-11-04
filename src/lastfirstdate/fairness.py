"""Stage 6, fair version: who appears on whose list each day (FR-LM-21, FR-LM-22, FR-LM-24).

The basic list (Sprint 1) lets every chooser see her own top 30. Two problems follow:
popular members appear on everyone's list and fill up, and paying members who are behind on
their 3 dates sit lower down. These rules share attention out more fairly. Each one is a
switch in `config/lastfirstdate.toml`.

1. Guarantee priority (papers 12 and 16). A paying member still short of 3 dates gets a bonus
   on every list they could appear on:
       bonus_j = guarantee_priority * missing dates_j / weeks left in the month
2. No wasted picks (paper 13's point about capacity). A member whose booked dates plus
   accepted-but-unscheduled pairs already fill their month is hidden: a pick on them could
   never become a date.
3. Exposure cap (paper 12, decision D7). No one appears on more than
       cap = ceil(exposure_multiple * list slots today / candidates today)
   lists in a day. Lists are filled round-robin (everyone's first slot, then everyone's
   second, ...) in a random order of choosers, so no chooser gets first claim every day.
4. Nash social welfare (paper 11, FR-LM-22). An alternative fill that adds, one slot at a time,
   the (chooser, candidate) pair that most raises
       sum over people of log(epsilon + their expected matches today)
   where "expected matches" is the sum of their mutual scores M on the lists so far. The log
   makes a person's first expected match worth far more than their tenth, so attention
   spreads out instead of piling onto a few.
5. Hold-back (paper 14, FR-LM-24, off by default). A member who received at least
   `holdback_yeses` mutual yeses in the last 7 days is hidden until that count drops.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence

import numpy as np

POLICIES = ("basic", "fair", "nsw")


def guarantee_bonus(
    dates: np.ndarray, paying: np.ndarray, guarantee: int, weeks_left: int, weight: float
) -> np.ndarray:
    """(n,) list bonus for paying members still short of their guarantee."""
    missing = np.where(paying, np.maximum(0, guarantee - dates), 0)
    return weight * missing / max(1, weeks_left)


def effectively_booked(dates: np.ndarray, pending: np.ndarray, cap) -> np.ndarray:
    """(n,) members whose dates plus waiting yeses already fill their month."""
    return dates + pending >= cap


def exposure_cap(n_slots: int, n_candidates: int, multiple: float) -> int:
    """Most lists one person may appear on today."""
    if n_candidates == 0:
        return 0
    return max(1, math.ceil(multiple * n_slots / n_candidates))


def _ranked(available: np.ndarray, scores: np.ndarray, bonus: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(available)
    return idx[np.lexsort((idx, -(scores[idx] + bonus[idx])))]


def day_exposure_cap(
    order: Sequence[int], available: Mapping[int, np.ndarray], size: int, multiple: float
) -> int:
    """Today's exposure cap, from today's list slots and candidates."""
    candidates: set[int] = set()
    n_slots = 0
    for i in order:
        idx = np.flatnonzero(available[i])
        candidates.update(idx.tolist())
        n_slots += min(size, idx.size)
    return exposure_cap(n_slots, len(candidates), multiple)


def fair_day_lists(
    order: Sequence[int],
    available: Mapping[int, np.ndarray],
    scores: np.ndarray,
    bonus: np.ndarray,
    size: int,
    multiple: float,
    reserved: Mapping[int, Sequence[int]] | None = None,
) -> tuple[dict[int, np.ndarray], int]:
    """Round-robin lists under an exposure cap. Returns the lists and the cap used.

    `reserved` holds slots already given out today (exploration, Sprint 5): they come first in
    each list and count toward the cap.
    """
    cap = day_exposure_cap(order, available, size, multiple)
    ranked = {i: _ranked(available[i], scores[i], bonus) for i in order}
    lists: dict[int, list[int]] = {i: list(reserved.get(i, [])) if reserved else [] for i in order}
    exposure: dict[int, int] = {}
    for lst in lists.values():
        for j in lst:
            exposure[j] = exposure.get(j, 0) + 1
    pointer = dict.fromkeys(order, 0)
    for _ in range(size):
        progressed = False
        for i in order:
            if len(lists[i]) >= size:
                continue
            r, p = ranked[i], pointer[i]
            while p < len(r) and (exposure.get(int(r[p]), 0) >= cap or int(r[p]) in lists[i]):
                p += 1
            if p < len(r):
                j = int(r[p])
                lists[i].append(j)
                exposure[j] = exposure.get(j, 0) + 1
                p += 1
                progressed = True
            pointer[i] = p
        if not progressed:
            break
    return {i: np.array(v, dtype=int) for i, v in lists.items()}, cap


def nsw_day_lists(
    order: Sequence[int],
    available: Mapping[int, np.ndarray],
    scores: np.ndarray,
    bonus: np.ndarray,
    size: int,
    epsilon: float,
) -> dict[int, np.ndarray]:
    """Greedy Nash social welfare fill (paper 11).

    Each step adds the pair with the largest gain in sum(log(epsilon + E)) over both the
    chooser and the candidate, where E is a person's sum of scores on today's lists. The
    objective is concave, so gains only shrink as lists fill: a stale gain on the heap is
    recomputed and pushed back ("lazy greedy"). Scores include the guarantee bonus.
    """
    e: dict[int, float] = {}
    lists: dict[int, list[int]] = {i: [] for i in order}

    def gain(i: int, j: int) -> float:
        s = float(scores[i, j] + bonus[j])
        ei, ej = e.get(i, 0.0), e.get(j, 0.0)
        return (
            math.log(epsilon + ei + s)
            - math.log(epsilon + ei)
            + math.log(epsilon + ej + s)
            - math.log(epsilon + ej)
        )

    heap = []
    for i in order:
        for j in np.flatnonzero(available[i]):
            heap.append((-gain(i, int(j)), i, int(j)))
    heapq.heapify(heap)
    while heap:
        neg, i, j = heapq.heappop(heap)
        if len(lists[i]) >= size:
            continue
        g = gain(i, j)
        if heap and g < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-g, i, j))
            continue
        lists[i].append(j)
        s = float(scores[i, j] + bonus[j])
        e[i] = e.get(i, 0.0) + s
        e[j] = e.get(j, 0.0) + s
    # show each list best first by the chooser's own score
    out = {}
    for i, v in lists.items():
        arr = np.array(v, dtype=int)
        out[i] = arr[np.lexsort((arr, -(scores[i, arr] + bonus[arr])))] if arr.size else arr
    return out


def held_back(recent_yeses: np.ndarray, threshold: int) -> np.ndarray:
    """(n,) members hidden by the hold-back rule. threshold 0 switches it off."""
    if threshold <= 0:
        return np.zeros(recent_yeses.shape, dtype=bool)
    return recent_yeses >= threshold
