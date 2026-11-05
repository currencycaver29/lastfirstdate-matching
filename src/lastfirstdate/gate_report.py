"""The ratio gate in the real booking flow (FR-LM-45, Sprint 8).

Test runs only. Sprint 6 tested the gate with the weekly batch test run, which counts
booked dates. Here the men it admits live through the real flow (`cycle_runs.py`): proposed
windows, payment holds, cancellations, no-shows, refills, new women arriving, three billing
cycles. Three versions on the same markets:

- the gate with Sprint 6's forecast (fitted on booked dates);
- the gate with Sprint 8's forecast (fitted on delivered dates, FR-LM-43);
- admitting the whole waitlist.

Two zones, the ends of the site's plan of 300 to 500 committed women, each with a waitlist of
450 men. `--ratios` adds the real flow's delivered fulfilment by women per man.

Run `python -m lastfirstdate.gate_report` (about 8 minutes; `--ratios` adds about 8).
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass

import numpy as np

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config, load
from lastfirstdate.cycle_runs import NEVER, join_days, market, run_cycles
from lastfirstdate.filter import member_ok
from lastfirstdate.forecast import (
    DEFAULT_MODEL_PATH,
    DELIVERED_MODEL_PATH,
    ForecastModel,
    forecast_fulfilment,
    gate_decision,
    market_features,
)
from lastfirstdate.mutual_report import build_scores

PHASE1 = "config/phase1.toml"
VERSIONS = ("booked forecast", "delivered forecast", "everyone")
ZONES = {
    # name: (committed women invited, men on the waitlist)
    "300 women (the site's minimum)": (300, 450),
    "500 women (the site's maximum)": (500, 450),
}


@dataclass
class RealGateRow:
    seed: int
    version: str
    women: int
    waitlist: int
    admitted: int
    women_per_man: float
    forecast: float  # the delivered forecast for the men admitted
    actual: float  # share of their man-cycles with 3 delivered dates
    credits: float  # credits as a share of their subscriptions
    stopped_because: str


def _with_market(cfg: Config, n_women: int, n_men: int) -> Config:
    n = n_women + n_men
    pop = dataclasses.replace(cfg.population, n_members=n, share_men=n_men / n)
    return dataclasses.replace(cfg, population=pop)


def real_gate_trial(
    cfg: Config,
    booked: ForecastModel,
    delivered: ForecastModel,
    n_women: int = 300,
    n_waitlist: int = 450,
    seeds: tuple[int, ...] = (401, 402, 403),
    versions: tuple[str, ...] = VERSIONS,
) -> list[RealGateRow]:
    """Let each version of the gate admit men, then run the real flow with only them."""
    rate = cfg.cycle_runs.new_women_per_cycle
    c = _with_market(cfg, n_women, n_waitlist)
    rows = []
    for seed in seeds:
        pop = market(c, seed, rate)
        sc = build_scores(pop, c, seed)
        arr = MemberArrays.from_members(pop.members, c.values)
        ok = member_ok(arr, c)
        men = np.array([m.gender == "man" for m in pop.members])
        join = join_days(pop, c, rate)
        women = np.flatnonzero(ok & ~men & (join == 0))
        waitlist = np.flatnonzero(ok & men & arr.paying)  # in application order
        for version in versions:
            if version == "everyone":
                admitted = waitlist
                mf = market_features(sc.eligible, women, admitted)
                ratio, why = mf.women_per_man, "admits everyone"
            else:
                model = booked if version == "booked forecast" else delivered
                d = gate_decision(model, sc.eligible, women, [], waitlist, c)
                admitted = np.array(d.admit, dtype=int)
                ratio, why = d.women_per_man, d.stopped_because
            if admitted.size == 0:  # the gate sold nothing: there is nobody to run
                forecast = actual = credits = float("nan")
            else:
                mf = market_features(sc.eligible, women, admitted)
                forecast = forecast_fulfilment(delivered, mf)
                j = join.copy()
                j[np.setdiff1d(np.flatnonzero(men), admitted)] = NEVER
                report = run_cycles(pop, c, seed, sc.mutual, version, j)
                actual = report.guarantee_share
                credits = report.credits / max(report.revenue, 1.0)
            rows.append(
                RealGateRow(
                    seed,
                    version,
                    int(women.size),
                    int(waitlist.size),
                    int(admitted.size),
                    float(ratio) if admitted.size else float("nan"),
                    forecast,
                    actual,
                    credits,
                    why,
                )
            )
    return rows


def mean_by_version(rows: list[RealGateRow], field: str) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r.version, []).append(getattr(r, field))
    return {
        k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan") for k, v in out.items()
    }


def _pct(x: float, width: int) -> str:
    return f"{'-':>{width}s}" if np.isnan(x) else f"{x:{width}.1%}"


def trial_text(rows: list[RealGateRow], target: float, title: str = "") -> str:
    lines = [
        f"Ratio gate in the real booking flow{': ' + title if title else ''} (target "
        f"{target:.0%} of admitted men with 3 delivered dates a cycle)",
        f"  {'seed':>4s} {'version':19s} {'women':>5s} {'waitlist':>8s} {'admitted':>8s} "
        f"{'women/man':>9s} {'forecast':>8s} {'actual':>7s} {'credits':>8s}",
    ]
    for r in rows:
        ratio = f"{'-':>9s}" if np.isnan(r.women_per_man) else f"{r.women_per_man:9.2f}"
        lines.append(
            f"  {r.seed:4d} {r.version:19s} {r.women:5d} {r.waitlist:8d} {r.admitted:8d} "
            f"{ratio} {_pct(r.forecast, 8)} {_pct(r.actual, 7)} {_pct(r.credits, 8)}"
        )
    lines.append("  averages:")
    admitted = mean_by_version(rows, "admitted")
    ratio = mean_by_version(rows, "women_per_man")
    forecast = mean_by_version(rows, "forecast")
    actual = mean_by_version(rows, "actual")
    credits = mean_by_version(rows, "credits")
    for v in admitted:
        if admitted[v] == 0:
            lines.append(f"    {v:19s} admitted nobody: no number of men reaches the target")
            continue
        lines.append(
            f"    {v:19s} admitted {admitted[v]:5.1f} at {ratio[v]:.2f} women per man; "
            f"forecast {forecast[v]:.1%}, actual {actual[v]:.1%}, credits {credits[v]:.1%}"
        )
    return "\n".join(lines)


def ratio_table(
    cfg: Config,
    ratios: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 8.0),
    seeds: tuple[int, ...] = (7, 8),
    n_men: int = 100,
) -> list[tuple[float, float]]:
    """(women per man, share of man-cycles with 3 delivered dates) in the real flow."""
    rate = cfg.cycle_runs.new_women_per_cycle
    out = []
    for ratio in ratios:
        shares = []
        for seed in seeds:
            c = _with_market(cfg, int(round(n_men * ratio)), n_men)
            pop = market(c, seed, rate)
            sc = build_scores(pop, c, seed)
            shares.append(
                run_cycles(
                    pop, c, seed, sc.mutual, join_day=join_days(pop, c, rate)
                ).guarantee_share
            )
        out.append((ratio, float(np.mean(shares))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="The ratio gate in the real booking flow.")
    parser.add_argument("--ratios", action="store_true", help="also tabulate by women per man")
    args = parser.parse_args()
    cfg = load(PHASE1)
    booked = ForecastModel.load(DEFAULT_MODEL_PATH)
    delivered = ForecastModel.load(DELIVERED_MODEL_PATH)
    for title, (n_women, n_waitlist) in ZONES.items():
        rows = real_gate_trial(cfg, booked, delivered, n_women, n_waitlist)
        print(trial_text(rows, cfg.gate.target, title))
        print()
    if args.ratios:
        print()
        print("Real flow: men with 3 delivered dates a cycle, by women per man (100 men, 2 seeds)")
        for ratio, share in ratio_table(cfg):
            print(f"  {ratio:4.1f} women per man: {share:.1%}")


if __name__ == "__main__":
    main()
