"""Exposure and fairness report (FR-LM-23) and the hold-back experiment (FR-LM-24).

Test runs only.

Exposure is how often a member appears on someone's list. Raw exposure partly reflects each
member's own settings (a narrow age range or a short distance means fewer people can see
them), so the report adjusts for audience size:

    shown rate_j = times shown_j / number of choosers allowed to see j

and compares each group's average with the overall average. A fake demographic group or
age band whose shown rate or dates per member falls below `fairness.min_rate_ratio`
(default 0.8, after the "four-fifths" rule of thumb used in US hiring audits) of the
overall level is flagged.

Spread of exposure is summarized by the Gini coefficient: 0 if everyone is shown equally
often, near 1 if a few members get all the attention.

Run `python -m lastfirstdate.fairness_report` to print it.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass

import numpy as np

from lastfirstdate.config import Config, load
from lastfirstdate.month_runs import MonthReport, run_month
from lastfirstdate.mutual_report import build_scores
from lastfirstdate.synthetic import Population, generate

AGE_BANDS = ((21, 25), (26, 30), (31, 35), (36, 40), (41, 99))


def gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative values: 0 = perfectly equal."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    k = np.arange(1, x.size + 1)
    return float((2 * np.sum(k * x) / (x.size * x.sum())) - (x.size + 1) / x.size)


@dataclass(frozen=True)
class GroupRow:
    name: str
    members: int
    shown_ratio: float  # group's shown rate / overall shown rate
    dates_ratio: float  # group's dates per member / overall dates per member
    flagged: bool


@dataclass
class ExposureReport:
    gini: float
    min_shown: int
    median_shown: float
    rows: list[GroupRow]
    threshold: float

    @property
    def flagged(self) -> list[str]:
        return [r.name for r in self.rows if r.flagged]

    def text(self) -> str:
        lines = [
            f"Exposure: Gini {self.gini:.3f} (0 = equal), least shown {self.min_shown}, "
            f"median {self.median_shown:.0f} appearances",
            f"  {'group':10s} {'members':>7s} {'shown rate':>11s} {'dates':>7s}   (1.00 = overall)",
        ]
        for r in self.rows:
            mark = "  FLAG" if r.flagged else ""
            lines.append(
                f"  {r.name:10s} {r.members:7d} {r.shown_ratio:11.2f} {r.dates_ratio:7.2f}{mark}"
            )
        lines.append(
            f"  flagged (below {self.threshold:.0%} of overall): "
            + (", ".join(self.flagged) if self.flagged else "none")
        )
        return "\n".join(lines)


def exposure_report(
    month: MonthReport, groups: dict[str, np.ndarray], threshold: float
) -> ExposureReport:
    """Exposure spread and per-group rates. `groups` maps a name to a member mask."""
    seen = month.matchable & (month.audience > 0)
    rate = np.where(seen, month.times_shown / np.maximum(month.audience, 1), 0.0)
    overall_rate = rate[seen].mean()
    overall_dates = month.dates[seen].mean()
    rows = []
    for name, mask in groups.items():
        g = mask & seen
        if not g.any():
            continue
        sr = rate[g].mean() / overall_rate if overall_rate > 0 else 1.0
        dr = month.dates[g].mean() / overall_dates if overall_dates > 0 else 1.0
        rows.append(
            GroupRow(name, int(g.sum()), float(sr), float(dr), sr < threshold or dr < threshold)
        )
    shown = month.times_shown[seen]
    return ExposureReport(
        gini=gini(shown),
        min_shown=int(shown.min()) if shown.size else 0,
        median_shown=float(np.median(shown)) if shown.size else 0.0,
        rows=rows,
        threshold=threshold,
    )


def member_groups(pop: Population) -> dict[str, np.ndarray]:
    """Fake demographic groups and age bands as member masks."""
    out = {}
    names = sorted({m.audit_group for m in pop.members})
    for g in names:
        out[g] = np.array([m.audit_group == g for m in pop.members])
    ages = np.array([m.age for m in pop.members])
    for lo, hi in AGE_BANDS:
        label = f"age {lo}-{hi}" if hi < 99 else f"age {lo}+"
        out[label] = (ages >= lo) & (ages <= hi)
    # Gender is left out on purpose: women mostly choose and men are mostly chosen (D1),
    # so their exposure differs by design, not by unfairness.
    return out


@dataclass(frozen=True)
class HoldbackResult:
    threshold: int
    off: MonthReport
    on: MonthReport

    def text(self) -> str:
        def line(name, m):
            return (
                f"  {name:9s} mutual yeses {m.accepts:5d}  dates {m.booked:5d}  "
                f"guarantee {m.guarantee_share():6.1%}"
            )

        return "\n".join(
            [
                f"Hold-back experiment: hide members after {self.threshold} yeses in 7 days",
                line("off", self.off),
                line("on", self.on),
                f"  difference in dates: {self.on.booked - self.off.booked:+d}",
            ]
        )


def holdback_experiment(
    pop: Population, cfg: Config, seed: int, scores: np.ndarray, threshold: int
) -> HoldbackResult:
    """The same month with the hold-back rule off and on."""
    off_cfg = dataclasses.replace(cfg, lists=dataclasses.replace(cfg.lists, holdback_yeses=0))
    on_cfg = dataclasses.replace(
        cfg, lists=dataclasses.replace(cfg.lists, holdback_yeses=threshold)
    )
    return HoldbackResult(
        threshold, run_month(pop, off_cfg, seed, scores), run_month(pop, on_cfg, seed, scores)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exposure and fairness report.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--holdback", type=int, default=3, help="yeses in 7 days before hiding")
    args = parser.parse_args()
    cfg = load()
    seed = cfg.test_runs.seed if args.seed is None else args.seed
    pop = generate(cfg, seed=seed)
    scores = build_scores(pop, cfg, seed).mutual
    groups = member_groups(pop)
    basic = dataclasses.replace(cfg, lists=dataclasses.replace(cfg.lists, policy="basic"))
    for label, c in (
        ("basic lists (Sprint 3)", basic),
        (f"{cfg.lists.policy} lists (configured)", cfg),
    ):
        m = run_month(pop, c, seed, scores)
        print(f"== {label}: guarantee {m.guarantee_share():.1%}, dates {m.booked}")
        print(exposure_report(m, groups, cfg.fairness.min_rate_ratio).text())
        print()
    print(holdback_experiment(pop, cfg, seed, scores, args.holdback).text())


if __name__ == "__main__":
    main()
