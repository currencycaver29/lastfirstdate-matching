"""FR-LM-10: weekly date scheduler (max-weight b-matching)."""

import itertools

import numpy as np
import pytest

from lastfirstdate.scheduler import (
    WEIGHT_SCALE,
    schedule_week,
    solve_b_matching,
    two_color,
    urgency,
)


def _brute_force(edges, weights, capacity):
    best = 0
    for r in range(len(edges) + 1):
        for subset in itertools.combinations(range(len(edges)), r):
            load: dict[int, int] = {}
            for k in subset:
                for v in edges[k]:
                    load[v] = load.get(v, 0) + 1
            if all(load[v] <= capacity[v] for v in load):
                best = max(best, sum(round(weights[k] * WEIGHT_SCALE) for k in subset))
    return best


def _random_case(rng, bipartite):
    n_left, n_right = 4, 4
    edges = set()
    while len(edges) < 9:
        if bipartite:
            edges.add((int(rng.integers(n_left)), n_left + int(rng.integers(n_right))))
        else:
            a, b = sorted(map(int, rng.choice(8, 2, replace=False)))
            edges.add((a, b))
    edges = sorted(edges)
    weights = [float(w) for w in rng.uniform(0.5, 3.0, len(edges))]
    capacity = {v: int(rng.integers(1, 3)) for v in range(8)}
    return edges, weights, capacity


@pytest.mark.parametrize("bipartite", [True, False])
def test_schedule_is_optimal_against_brute_force(bipartite):
    """FR-LM-10: on small cases the scheduler finds the best possible total value."""
    rng = np.random.default_rng(0)
    methods = set()
    for _ in range(25):
        edges, weights, capacity = _random_case(rng, bipartite)
        chosen, method = solve_b_matching(edges, weights, capacity)
        methods.add(method)
        got = sum(round(weights[k] * WEIGHT_SCALE) for k in chosen)
        assert got == _brute_force(edges, weights, capacity)
        load: dict[int, int] = {}
        for k in chosen:
            for v in edges[k]:
                load[v] = load.get(v, 0) + 1
        assert all(load[v] <= capacity[v] for v in load)
    if bipartite:
        assert methods == {"flow"}
    else:
        assert "mip" in methods


def test_odd_cycle_uses_integer_program():
    """FR-LM-10: a triangle is not bipartite, so flow cannot be used; the answer is still exact."""
    edges = [(0, 1), (1, 2), (0, 2)]
    assert two_color(edges) is None
    chosen, method = solve_b_matching(edges, [1.0, 2.0, 1.5], {0: 1, 1: 1, 2: 1})
    assert method == "mip"
    assert chosen == [1]


def test_bookings_use_a_shared_slot_and_respect_capacity():
    """FR-LM-10: each date lands in a slot both people have free; nobody exceeds capacity."""
    pairs = [(0, 2), (0, 3), (1, 3)]
    masks = [0b0110, 0b1000, 0b0001]
    capacity = {0: 1, 1: 1, 2: 1, 3: 1}
    sched = schedule_week(pairs, [1.0, 5.0, 4.5], masks, capacity, np.random.default_rng(1))
    booked = {(b.a, b.b) for b in sched.bookings}
    assert booked == {(0, 2), (1, 3)}  # 1.0 + 4.5 beats 5.0 alone
    assert np.isclose(sched.total_weight, 5.5)
    for b in sched.bookings:
        k = pairs.index((b.a, b.b))
        assert masks[k] >> b.slot & 1
    with pytest.raises(ValueError):
        schedule_week([(0, 1)], [1.0], [0], {0: 1, 1: 1})


def test_people_without_capacity_get_no_dates():
    """FR-LM-10: a member with no dates left is never booked."""
    chosen, method = solve_b_matching([(0, 1), (1, 2)], [1.0, 1.0], {0: 0, 1: 1, 2: 1})
    assert chosen == [1]
    assert solve_b_matching([(0, 1)], [1.0], {0: 0, 1: 1}) == ([], "empty")


def test_duplicate_pairs_are_rejected():
    """FR-LM-10: each pair may appear once."""
    with pytest.raises(ValueError):
        solve_b_matching([(0, 1), (1, 0)], [1.0, 1.0], {0: 1, 1: 1})


def test_urgency_grows_near_month_end_for_members_short_of_guarantee():
    """FR-LM-10: paying members further from 3 dates, later in the month, weigh more."""
    dates = np.array([0, 2, 3, 0])
    paying = np.array([True, True, True, False])
    early = urgency(dates, paying, guarantee=3, weeks_left=4, bonus=0.2)
    late = urgency(dates, paying, guarantee=3, weeks_left=1, bonus=0.2)
    assert early[0] > early[1] > early[2] == 0.0
    assert early[3] == 0.0
    assert np.all(late >= early)
    assert np.isclose(late[0], 0.6)
