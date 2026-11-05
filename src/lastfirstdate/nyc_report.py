"""NYC demo: the whole engine on fresh synthetic New York zones, start to finish.

Usage: python -m lastfirstdate.nyc_report [--json]

Ten zones in the Phase 1 neighbourhoods: five with 300 women and 100 paying men, five with 500
women and 167 men (always 3 women per man), on seeds 2026 to 2030, which no setting was tuned
on. Each runs three billing cycles of the real flow (daily lists, picks, the payment hold,
cancellations, no-shows, ratings, credits). Members have hidden "true" tastes the engine never
reads; the report grades the engine against them.
"""

from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config, load
from lastfirstdate.cycle_runs import CycleRun, join_days, market
from lastfirstdate.filter import member_ok
from lastfirstdate.forecast import forecast_fulfilment, gate_model, market_features
from lastfirstdate.looks_report import auc
from lastfirstdate.member_behavior import Behavior
from lastfirstdate.mutual_report import build_scores, top_quality_all

SEEDS = (2026, 2027, 2028, 2029, 2030)
ZONES = (300, 500)  # women; men = women / 3


def zone_config(n_women: int) -> Config:
    base = load("config/phase1.toml")
    n_men = round(n_women / 3)
    n = n_women + n_men
    pop = dataclasses.replace(base.population, n_members=n, share_men=n_men / n)
    return dataclasses.replace(base, population=pop)


def zone_result(seed: int, n_women: int) -> dict:
    """Every stage's outcome for one zone, graded against the hidden truth."""
    cfg = zone_config(n_women)
    rate = cfg.cycle_runs.new_women_per_cycle
    pop = market(cfg, seed, rate)
    join = join_days(pop, cfg, rate)
    sc = build_scores(pop, cfg, seed)
    arr = MemberArrays.from_members(pop.members, cfg.values)
    ok = member_ok(arr, cfg)
    is_man = np.array([m.gender == "man" for m in pop.members])
    men = np.flatnonzero(ok & is_man & arr.paying & (join == 0))
    women = np.flatnonzero(ok & ~is_man & (join == 0))
    behavior = Behavior(pop, arr, cfg, np.random.default_rng([seed, 1]))
    aucs = []
    for v in sc.tastes or {}:
        cand = np.flatnonzero(sc.eligible[v])
        if cand.size >= 20:
            truth = behavior.looks[v, cand]
            aucs.append(auc(sc.looks[v, cand], truth > np.median(truth)))
    top = {k: q.mutual for k, q in top_quality_all(pop, cfg, seed, sc).items()}
    shuffled = dataclasses.replace(
        sc, mutual=np.random.default_rng([seed, 2]).random(sc.mutual.shape)
    )
    top["random order"] = top_quality_all(pop, cfg, seed, shuffled)["mutual"].mutual
    run = CycleRun(pop, cfg, seed, sc.mutual, join).run("NYC demo")
    states = {k: int(v) for k, v in run.states.items()}
    return {
        "seed": seed,
        "women": n_women,
        "men": round(n_women / 3),
        "forecast": float(
            forecast_fulfilment(gate_model(cfg), market_features(sc.eligible, women, men))
        ),
        "per_cycle": run.per_cycle(),
        "guarantee": run.guarantee_share,
        "credit_share": run.credits / max(run.revenue, 1.0),
        "looks_auc": float(np.mean(aucs)),
        "top10_both_yes": top,
        "states": states,
        "women_no_date": float(np.mean(run.women_dates == 0)),
        "bad_well_rated": run.bad_well_rated,
        "caught_well_rated": run.caught_well_rated,
        "wrongly_flagged": run.flagged - run.flagged_bad,
    }


def _mean(zone: list[dict], f) -> float:
    return float(np.mean([f(r) for r in zone]))


def summary(rows: list[dict]) -> dict:
    """Averages per zone size."""
    out = {}
    for w in ZONES:
        z = [r for r in rows if r["women"] == w]
        names = sorted({k for r in z for k in r["states"]})
        total = {k: sum(r["states"].get(k, 0) for r in z) for k in names}
        n_booked = sum(total.values())
        orderings = list(z[0]["top10_both_yes"])
        out[w] = {
            "per_cycle": [_mean(z, lambda r, k=k: r["per_cycle"][k]) for k in range(3)],
            "guarantee": _mean(z, lambda r: r["guarantee"]),
            "guarantee_range": [min(r["guarantee"] for r in z), max(r["guarantee"] for r in z)],
            "forecast": _mean(z, lambda r: r["forecast"]),
            "credit_share": _mean(z, lambda r: r["credit_share"]),
            "looks_auc": _mean(z, lambda r: r["looks_auc"]),
            "top10_both_yes": {
                k: _mean(z, lambda r, k=k: r["top10_both_yes"][k]) for k in orderings
            },
            "outcomes": {k: v / n_booked for k, v in total.items()},
            "women_no_date": _mean(z, lambda r: r["women_no_date"]),
            "bad_caught": [
                sum(r["caught_well_rated"] for r in z),
                sum(r["bad_well_rated"] for r in z),
            ],
            "wrongly_flagged": sum(r["wrongly_flagged"] for r in z),
        }
    return out


def main() -> None:
    rows = [zone_result(seed, w) for w in ZONES for seed in SEEDS]
    s = summary(rows)
    if "--json" in sys.argv:
        print(json.dumps({"zones": rows, "summary": s}))
        return
    for w, z in s.items():
        print(f"{w} women, {round(w / 3)} paying men, seeds {SEEDS[0]} to {SEEDS[-1]}")
        print(
            "  men with 3 dates: "
            + ", ".join(f"cycle {k + 1} {x:.0%}" for k, x in enumerate(z["per_cycle"]))
            + f"; all cycles {z['guarantee']:.1%} "
            f"({z['guarantee_range'][0]:.0%} to {z['guarantee_range'][1]:.0%}); "
            f"forecast {z['forecast']:.1%}"
        )
        print(f"  credits {z['credit_share']:.1%} of subscriptions; looks AUC {z['looks_auc']:.3f}")
        print(
            "  top 10, chance both say yes: "
            + ", ".join(f"{k} {v:.1%}" for k, v in z["top10_both_yes"].items())
        )
        print(
            "  booked dates: "
            + ", ".join(f"{k} {v:.0%}" for k, v in z["outcomes"].items())
            + f"; women with no date in 28 days {z['women_no_date']:.0%}"
        )
        print(
            f"  trust: caught {z['bad_caught'][0]} of {z['bad_caught'][1]} bad actors with 6+ "
            f"ratings; wrongly flagged {z['wrongly_flagged']}"
        )


if __name__ == "__main__":
    main()
