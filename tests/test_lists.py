"""FR-LM-09: who chooses (D1) and the basic daily list."""

import numpy as np

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.lists import NEVER_SHOWN, available_row, daily_list, pick_rights


def test_women_pick_men_and_same_gender_pairs_both_pick(cfg, small_pop):
    """FR-LM-09 / D1: a woman may pick a man, a man may not pick a woman, same gender both may."""
    m = small_pop.members
    arr = MemberArrays.from_members(m, cfg.values)
    rights = pick_rights(arr)
    man = next(x.id for x in m if x.gender == "man")
    man2 = next(x.id for x in m if x.gender == "man" and x.id != man)
    woman = next(x.id for x in m if x.gender == "woman")
    assert rights[woman, man]
    assert not rights[man, woman]
    assert rights[man, man2] and rights[man2, man]


def test_daily_list_takes_the_best_up_to_the_size():
    """FR-LM-09: best score first, ties broken by lower id, never more than the size."""
    scores = np.array([0.1, 0.9, 0.5, 0.9, 0.7])
    avail = np.array([True, True, True, True, False])
    assert daily_list(scores, avail, 3).tolist() == [1, 3, 2]
    assert daily_list(scores, avail, 30).tolist() == [1, 3, 2, 0]
    assert daily_list(scores, np.zeros(5, dtype=bool), 30).size == 0


def _setup(n=4):
    eligible = np.ones((n, n), dtype=bool)
    np.fill_diagonal(eligible, False)
    rights = np.ones((n, n), dtype=bool)
    last_shown = np.full((n, n), NEVER_SHOWN)
    full = np.zeros(n, dtype=bool)
    closed = np.zeros((n, n), dtype=bool)
    return eligible, rights, last_shown, full, closed


def test_list_excludes_filtered_closed_and_full_members():
    """FR-LM-09: never shows a filtered-out person, a closed pair or a fully booked member."""
    eligible, rights, last_shown, full, closed = _setup()
    eligible[0, 1] = False
    closed[0, 2] = True
    full[3] = True
    assert not available_row(0, eligible, rights, last_shown, 0, 7, full, closed).any()
    full[0] = True
    full[3] = False
    assert not available_row(0, eligible, rights, last_shown, 0, 7, full, closed).any()


def test_passed_profiles_reappear_only_after_the_rest_period():
    """FR-LM-09: someone shown but not picked comes back after reshow_after_days days."""
    eligible, rights, last_shown, full, closed = _setup()
    last_shown[0, 1] = 10
    row = available_row(0, eligible, rights, last_shown, 16, 7, full, closed)
    assert not row[1] and row[2] and row[3]
    assert available_row(0, eligible, rights, last_shown, 17, 7, full, closed)[1]
