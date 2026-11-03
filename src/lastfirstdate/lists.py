"""Stage 6, basic version: who chooses, and each chooser's daily list (FR-LM-09).

Decision D1 (paper 15): in a man-woman pair the woman picks and the man accepts or
declines. In any other pair (two men, two women, or a nonbinary member) both may pick.

The basic list is the top `lists.daily_size` candidates by score that the chooser may pick
and who still have room for dates. Someone shown but not picked can reappear after
`lists.reshow_after_days` days; pairs already picked, declined or dated never reappear. If
fewer candidates are available, the list is shorter (D3): it is never padded with people
who failed the filter. Sprint 4 replaces this with fair lists that cap exposure.
"""

from __future__ import annotations

import numpy as np

from lastfirstdate.arrays import GENDERS, MemberArrays

MAN, WOMAN = GENDERS.index("man"), GENDERS.index("woman")
NEVER_SHOWN = -(10**6)  # value of last_shown for pairs never shown


def pick_rights(arr: MemberArrays) -> np.ndarray:
    """(n, n) True where i is allowed to pick j: everything except a man picking a woman."""
    man = arr.gender == MAN
    woman = arr.gender == WOMAN
    return ~(man[:, None] & woman[None, :])


def daily_list(scores: np.ndarray, available: np.ndarray, size: int) -> np.ndarray:
    """Ids of up to `size` available candidates, best score first (ties: lower id first)."""
    idx = np.flatnonzero(available)
    if idx.size == 0:
        return idx
    order = np.lexsort((idx, -scores[idx]))
    return idx[order[:size]]


def available_row(
    i: int,
    eligible: np.ndarray,
    rights: np.ndarray,
    last_shown: np.ndarray,
    today: int,
    reshow_after_days: int,
    full: np.ndarray,
    closed: np.ndarray,
) -> np.ndarray:
    """(n,) True where j may appear on chooser i's list today.

    eligible: Stage 1 result. rights: pick_rights. last_shown: (n, n) day j was last on i's
    list, NEVER_SHOWN if never. full: (n,) members with no dates left this month. closed:
    (n, n) pairs already picked, declined or dated.
    """
    if full[i]:
        return np.zeros(eligible.shape[1], dtype=bool)
    rested = today - last_shown[i] >= reshow_after_days
    return eligible[i] & rights[i] & rested & ~closed[i] & ~full
