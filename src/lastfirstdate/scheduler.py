"""Stage 7: the weekly date scheduler (FR-LM-10).

Each week we hold a set of candidate pairs: pairs where one picked the other and the other
accepted. Each person can take at most `capacity[i]` dates this week (their remaining
dates this month, and never more than one a week). We choose which pairs get a date so the
total value is as high as possible:

    maximize   sum over chosen pairs (i, j) of  w_ij
    subject to each person i is in at most capacity[i] chosen pairs

This is called maximum-weight b-matching. How it is solved depends on the shape of the graph
(people are nodes, candidate pairs are edges):

- If the graph is bipartite (people split into two sides with every pair crossing between
  them, as with man-woman pairs), it is a min-cost flow problem:
      source -> each left person (capacity c_i)
      left person -> right person (capacity 1, cost -w_ij)
      right person -> sink (capacity c_j)
      source -> sink bypass (cost 0), so no one is forced into a bad date
  The cheapest flow is the best schedule. Solved exactly with OR-Tools min-cost flow.
- If it is not bipartite (an odd cycle, which same-gender pairs can create), flow no
  longer applies, so we solve the same problem exactly as an integer program (one yes/no
  variable per pair) with the SCIP solver in OR-Tools. Its relaxed version, where a pair
  may be "half booked", usually already comes out in whole numbers, so this is fast.

Each pair's value w_ij = date_value + score_ij + urgency_i + urgency_j, where urgency is
`late_month_bonus * missing guaranteed dates / weeks left` for paying members. The closer
the month end and the further someone is from 3 dates, the more their pairs are worth.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ortools.graph.python import min_cost_flow
from ortools.linear_solver import pywraplp

WEIGHT_SCALE = 10_000  # solvers need whole numbers: weights are kept to 4 decimals


@dataclass(frozen=True)
class Booking:
    a: int
    b: int
    slot: int
    weight: float


@dataclass(frozen=True)
class Schedule:
    bookings: list[Booking]
    method: str  # "flow", "mip" or "empty"
    total_weight: float


def urgency(
    dates_so_far: np.ndarray,
    paying: np.ndarray,
    guarantee: int,
    weeks_left: int,
    bonus: float,
) -> np.ndarray:
    """(n,) extra value per pair for paying members still short of their guarantee."""
    missing = np.where(paying, np.maximum(0, guarantee - dates_so_far), 0)
    return bonus * missing / max(1, weeks_left)


def two_color(n_edges_nodes: Sequence[tuple[int, int]]) -> dict[int, int] | None:
    """Split the graph's nodes into two sides with every edge crossing, or None if impossible."""
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in n_edges_nodes:
        adj[a].append(b)
        adj[b].append(a)
    color: dict[int, int] = {}
    for start in sorted(adj):
        if start in color:
            continue
        color[start] = 0
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    stack.append(v)
                elif color[v] == color[u]:
                    return None
    return color


def _solve_flow(
    edges: Sequence[tuple[int, int]],
    weights: Sequence[int],
    capacity: dict[int, int],
    color: dict[int, int],
) -> list[int]:
    nodes = sorted(color)
    node_id = {v: k + 2 for k, v in enumerate(nodes)}  # 0 = source, 1 = sink
    source, sink = 0, 1
    smcf = min_cost_flow.SimpleMinCostFlow()
    total = sum(capacity[v] for v in nodes if color[v] == 0)
    for v in nodes:
        if color[v] == 0:
            smcf.add_arc_with_capacity_and_unit_cost(source, node_id[v], capacity[v], 0)
        else:
            smcf.add_arc_with_capacity_and_unit_cost(node_id[v], sink, capacity[v], 0)
    smcf.add_arc_with_capacity_and_unit_cost(source, sink, total, 0)
    pair_arcs = []
    for (a, b), w in zip(edges, weights, strict=True):
        left, right = (a, b) if color[a] == 0 else (b, a)
        pair_arcs.append(
            smcf.add_arc_with_capacity_and_unit_cost(node_id[left], node_id[right], 1, -w)
        )
    smcf.set_node_supply(source, total)
    smcf.set_node_supply(sink, -total)
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"min-cost flow failed with status {status}")
    return [k for k, arc in enumerate(pair_arcs) if smcf.flow(arc) > 0]


def _solve_mip(
    edges: Sequence[tuple[int, int]],
    weights: Sequence[int],
    capacity: dict[int, int],
) -> list[int]:
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools was built without the SCIP solver")
    x = [solver.BoolVar(f"x{k}") for k in range(len(edges))]
    incident: dict[int, list[int]] = defaultdict(list)
    for k, (a, b) in enumerate(edges):
        incident[a].append(k)
        incident[b].append(k)
    for v, ks in incident.items():
        solver.Add(sum(x[k] for k in ks) <= capacity[v])
    solver.Maximize(sum(w * x[k] for k, w in enumerate(weights)))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"SCIP did not prove an optimal schedule (status {status})")
    return [k for k in range(len(edges)) if x[k].solution_value() > 0.5]


def solve_b_matching(
    edges: Sequence[tuple[int, int]],
    weights: Sequence[float],
    capacity: dict[int, int],
) -> tuple[list[int], str]:
    """Indices of the chosen edges and the method used. Exact in both cases."""
    if len({frozenset(e) for e in edges}) != len(edges) or any(a == b for a, b in edges):
        raise ValueError("edges must be distinct pairs of two different people")
    usable = [
        k for k, (a, b) in enumerate(edges) if capacity.get(a, 0) > 0 and capacity.get(b, 0) > 0
    ]
    if not usable:
        return [], "empty"
    sub_edges = [edges[k] for k in usable]
    int_w = [round(weights[k] * WEIGHT_SCALE) for k in usable]
    color = two_color(sub_edges)
    if color is not None:
        chosen, method = _solve_flow(sub_edges, int_w, capacity, color), "flow"
    else:
        chosen, method = _solve_mip(sub_edges, int_w, capacity), "mip"
    return [usable[k] for k in chosen], method


def schedule_week(
    pairs: Sequence[tuple[int, int]],
    weights: Sequence[float],
    shared_slot_masks: Sequence[int],
    capacity: dict[int, int],
    rng: np.random.Generator | None = None,
) -> Schedule:
    """Choose this week's dates and give each one a shared free slot.

    With at most one date per person per week (dates.max_per_week = 1), any shared slot
    works, so the slot is drawn at random among them (or the earliest if no rng is given).
    """
    for mask in shared_slot_masks:
        if mask == 0:
            raise ValueError("every candidate pair needs at least one shared free slot")
    chosen, method = solve_b_matching(pairs, weights, capacity)
    bookings = []
    for k in chosen:
        slots = [s for s in range(64) if shared_slot_masks[k] >> s & 1]
        slot = int(rng.choice(slots)) if rng is not None else slots[0]
        a, b = pairs[k]
        bookings.append(Booking(a=a, b=b, slot=slot, weight=float(weights[k])))
    return Schedule(bookings=bookings, method=method, total_weight=sum(b.weight for b in bookings))
