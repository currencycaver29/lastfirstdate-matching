"""Fit the guarantee forecast on synthetic markets and test it on others (FR-LM-33).

Test runs only. Before launch there are no real cycles, so the forecast's starting
coefficients come from synthetic Phase 1-style markets (lastfirstdate's rules from
`config/phase1.toml`, varying the number of men and women per man, including markets with
fewer women than men so the forecast sees the cliff there). They are checked on separate
markets with different seeds and sizes, then written to `config/forecast_model.toml` with the
range of women per man they cover. Once real cycles exist, `fit_forecast` is rerun on real rows
the same way.

Run `python -m lastfirstdate.forecast_fit` (about ten minutes).

Sprint 8 (FR-LM-43, FR-LM-44): `python -m lastfirstdate.forecast_fit --delivered` fits the same
forecast on *delivered* dates in the real booking flow (`cycle_runs.py`), under the settings'
frictions and rate of new women, and writes `config/forecast_delivered.toml` (about 20 minutes).
Start rows: each paying man's market at launch, labelled with the share of his paid cycles that
reached 3 delivered dates. Progress rows: his progress each week of each cycle.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass

import numpy as np

from lastfirstdate.config import Config, load
from lastfirstdate.forecast import (
    DEFAULT_MODEL_PATH,
    DELIVERED_MODEL_PATH,
    ForecastModel,
    fit_forecast,
    market_features,
    progress_matrix,
    start_matrix,
)
from lastfirstdate.looks_report import auc
from lastfirstdate.month_runs import run_month
from lastfirstdate.mutual_report import build_scores
from lastfirstdate.synthetic import Population, generate

PHASE1 = "config/phase1.toml"
TRAIN = {
    "men": (60, 100, 150),
    "ratios": (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
    "seeds": (201, 202),
}
HOLDOUT = {"men": (100, 150), "ratios": (0.6, 1.25, 2.5, 4.0), "seeds": (301,)}
MONTHS = 2
DELIVERED_TRAIN = {
    "men": (60, 100, 150),
    "ratios": (1.0, 2.0, 3.0, 4.5, 7.0),
    "seeds": (201, 202, 203, 204),  # markets differ by about 5 points: average several
}
DELIVERED_HOLDOUT = {"men": (80, 120), "ratios": (2.5, 4.0), "seeds": (301,)}


def market(cfg: Config, n_men: int, ratio: float, seed: int) -> Population:
    """A one-zone market with exactly n_men men and ratio women per man."""
    n = int(round(n_men * (1 + ratio)))
    pop_cfg = dataclasses.replace(
        cfg.population, n_members=n, share_men=n_men / n, exact_counts=True
    )
    return generate(dataclasses.replace(cfg, population=pop_cfg), seed=seed)


@dataclass
class Rows:
    x_start: list[np.ndarray]
    y_start: list[np.ndarray]
    x_progress: list[np.ndarray]
    y_progress: list[np.ndarray]
    markets: list[dict]

    def ratio_range(self) -> tuple[float, float]:
        r = [mk["mf"].women_per_man if "mf" in mk else mk["ratio"] for mk in self.markets]
        return min(r), max(r)

    def stacked(self):
        return (
            np.vstack(self.x_start),
            np.concatenate(self.y_start),
            np.vstack(self.x_progress),
            np.concatenate(self.y_progress),
        )


def market_rows(pop: Population, cfg: Config, seed: int, rows: Rows, label: dict) -> None:
    """Run MONTHS cycles of one synthetic market and add its forecast rows."""
    sc = build_scores(pop, cfg, seed)
    men_mask = np.array([m.gender == "man" for m in pop.members])
    for k in range(MONTHS):
        m = run_month(pop, cfg, seed, sc.mutual, month=k + 1, record=True)
        women = np.flatnonzero(m.matchable & ~men_mask)
        men = np.flatnonzero(m.paying & men_mask)
        mf = market_features(sc.eligible, women, men)
        reached = m.dates[men] >= cfg.dates.guaranteed_per_month_paying
        rows.x_start.append(start_matrix(mf))
        rows.y_start.append(reached)
        weeks = cfg.dates.weeks_per_month
        for w in range(weeks - 1):  # snapshots after weeks 1 to 3
            x = progress_matrix(mf, m.weekly_dates[w], m.weekly_pending[w], weeks - w - 1, cfg)
            rows.x_progress.append(x)
            rows.y_progress.append(reached)
        rows.markets.append({**label, "month": k + 1, "actual": float(reached.mean()), "mf": mf})


def collect(cfg: Config, plan: dict) -> Rows:
    rows = Rows([], [], [], [], [])
    for seed in plan["seeds"]:
        for n_men in plan["men"]:
            for ratio in plan["ratios"]:
                pop = market(cfg, n_men, ratio, seed)
                market_rows(pop, cfg, seed, rows, {"men": n_men, "ratio": ratio, "seed": seed})
    return rows


def evaluate(model: ForecastModel, rows: Rows) -> str:
    xs, ys, xp, yp = rows.stacked()
    lines = ["Held-out markets: forecast versus synthetic fulfilment"]
    lines.append(f"  {'men':>4s} {'women/man':>9s} {'month':>5s} {'forecast':>9s} {'actual':>7s}")
    errors = []
    for mk in rows.markets:
        f = float(model.p_start(start_matrix(mk["mf"])).mean())
        errors.append(abs(f - mk["actual"]))
        lines.append(
            f"  {mk['men']:4d} {mk['mf'].women_per_man:9.2f} {mk['month']:5d} {f:9.1%} "
            f"{mk['actual']:7.1%}"
        )
    lines.append(
        f"  market forecast error: mean {np.mean(errors):.1%}, largest {np.max(errors):.1%}"
    )
    lines.append(f"  per-man AUC at the start of a cycle: {auc(model.p_start(xs), ys):.3f}")
    weeks = len(xp) // max(len(xs), 1)
    per_month = [len(x) for x in rows.x_start]
    for w in range(weeks):
        # each month added its weeks in order, n rows per week
        mask = np.zeros(len(xp), dtype=bool)
        start = 0
        for n in per_month:
            mask[start + w * n : start + (w + 1) * n] = True
            start += weeks * n
        a = auc(model.p_progress(xp[mask]), yp[mask])
        lines.append(f"  per-man AUC after week {w + 1} (at-risk model): {a:.3f}")
    return "\n".join(lines)


def delivered_market(cfg: Config, n_men: int, ratio: float, seed: int):
    """Run the real flow on one synthetic market (with new women arriving) and record rows."""
    from lastfirstdate.cycle_runs import CycleRun, join_days
    from lastfirstdate.cycle_runs import market as cycle_market

    rate = cfg.cycle_runs.new_women_per_cycle
    n = int(round(n_men * (1 + ratio)))
    pop_cfg = dataclasses.replace(cfg.population, n_members=n, share_men=n_men / n)
    c = dataclasses.replace(cfg, population=pop_cfg)
    pop = cycle_market(c, seed, rate)
    sc = build_scores(pop, c, seed)
    sim = CycleRun(pop, c, seed, sc.mutual, join_days(pop, c, rate), record=True)
    report = sim.run(f"{n_men} men, {ratio:g} women per man, seed {seed}")
    return sim.forecast_rows(report), report


def collect_delivered(cfg: Config, plan: dict) -> Rows:
    rows = Rows([], [], [], [], [])
    for seed in plan["seeds"]:
        for n_men in plan["men"]:
            for ratio in plan["ratios"]:
                (xs, ys, xp, yp), report = delivered_market(cfg, n_men, ratio, seed)
                rows.x_start.append(xs)
                rows.y_start.append(ys)
                rows.x_progress.append(xp)
                rows.y_progress.append(yp)
                rows.markets.append(
                    {
                        "men": n_men,
                        "ratio": float(np.exp(xs[0, 1])),
                        "seed": seed,
                        "actual": report.guarantee_share,
                        "x": xs,
                        "xp": xp,
                        "yp": yp,
                    }
                )
                print(f"  {report.label}: {report.guarantee_share:.1%}", flush=True)
    return rows


def evaluate_delivered(model: ForecastModel, rows: Rows) -> str:
    lines = ["Held-out markets: forecast versus share of man-cycles with 3 delivered dates"]
    lines.append(f"  {'men':>4s} {'women/man':>9s} {'seed':>5s} {'forecast':>9s} {'actual':>7s}")
    errors = []
    for mk in rows.markets:
        f = float(model.p_start(mk["x"]).mean())
        errors.append(abs(f - mk["actual"]))
        lines.append(
            f"  {mk['men']:4d} {mk['ratio']:9.2f} {mk['seed']:5d} {f:9.1%} {mk['actual']:7.1%}"
        )
    lines.append(
        f"  market forecast error: mean {np.mean(errors):.1%}, largest {np.max(errors):.1%}"
    )
    xp = np.vstack([mk["xp"] for mk in rows.markets])
    yp = np.concatenate([mk["yp"] for mk in rows.markets])
    for week, left in ((1, 0.75), (2, 0.5), (3, 0.25)):
        mask = np.isclose(xp[:, -1], left)
        if mask.any() and 0 < yp[mask].mean() < 1:
            a = auc(model.p_progress(xp[mask]), yp[mask])
            lines.append(f"  mid-cycle alerts, after week {week}: AUC {a:.3f}")
    return "\n".join(lines)


def fit_delivered(cfg: Config, save: bool) -> ForecastModel:
    cs = cfg.cycle_runs
    train = collect_delivered(cfg, DELIVERED_TRAIN)
    model = fit_forecast(
        *train.stacked(),
        cfg.gate.forecast_l2,
        note=f"real booking flow (cycle_runs): men {DELIVERED_TRAIN['men']}, women per man "
        f"{DELIVERED_TRAIN['ratios']}, seeds {DELIVERED_TRAIN['seeds']}",
        fitted_women_per_man=train.ratio_range(),
        kind="delivered",
        assumptions={
            "new_women_per_cycle": cs.new_women_per_cycle,
            "show_up_rate": cfg.test_runs.show_up_rate,
            "pay_in_time_chosen": cs.pay_in_time_chosen,
            "pay_in_time_chooser": cs.pay_in_time_chooser,
            "cancel_rate": cs.cancel_rate,
            "cycles": cs.cycles,
            "cycle_days": cfg.billing.cycle_days,
            "stagger": cs.stagger,
        },
    )
    print("Coefficients (start):", np.round(model.start, 3))
    print("Coefficients (progress):", np.round(model.progress, 3))
    print(evaluate_delivered(model, collect_delivered(cfg, DELIVERED_HOLDOUT)))
    if save:
        model.save(DELIVERED_MODEL_PATH)
        print(f"saved {DELIVERED_MODEL_PATH}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and test the guarantee forecast.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--delivered", action="store_true", help="fit on the real booking flow")
    args = parser.parse_args()
    cfg = load(PHASE1)
    if args.delivered:
        fit_delivered(cfg, save=not args.no_save)
        return
    train = collect(cfg, TRAIN)
    model = fit_forecast(
        *train.stacked(),
        cfg.gate.forecast_l2,
        note=f"synthetic Phase 1 markets: men {TRAIN['men']}, women per man {TRAIN['ratios']}, "
        f"seeds {TRAIN['seeds']}, {MONTHS} cycles each",
        fitted_women_per_man=train.ratio_range(),
    )
    print("Coefficients (start):", np.round(model.start, 3))
    print("Coefficients (progress):", np.round(model.progress, 3))
    print(evaluate(model, collect(cfg, HOLDOUT)))
    if not args.no_save:
        model.save(DEFAULT_MODEL_PATH)
        print(f"saved {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
