"""Three months with and without learning (FR-LM-25 to FR-LM-28).

Test runs only. The same city and the same synthetic people go through
`learning.months` months three times:

- no learning: the Sprint 4 engine, unchanged all along
- learning: after each month, looks tastes are updated from date outcomes and each member
  gets their own looks weight lambda_i (Stage 8)
- learning + exploration: as above, and `lists.exploration_share` of each list goes to the
  candidates the member's looks model is least sure about

Pairs who dated are never matched again. Replicate r uses month numbers 100r + 1 to
100r + 3: the same people, different day-to-day randomness.

Run `python -m lastfirstdate.learning_report` to print it.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field

import numpy as np

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config, load
from lastfirstdate.fairness_report import exposure_report, member_groups
from lastfirstdate.learning import (
    Evidence,
    collect_evidence,
    looks_weights,
    personal_directional_score,
    uncertainty_matrix,
    update_tastes,
)
from lastfirstdate.looks_report import auc, rank_corr
from lastfirstdate.month_runs import MonthReport, month_people, run_month
from lastfirstdate.mutual import looks_scores, mutual_score
from lastfirstdate.mutual_report import Scores, build_scores
from lastfirstdate.synthetic import Population, generate

ARMS = ("no learning", "learning", "learning + exploration")


@dataclass
class ArmResult:
    arm: str
    months: list[list[MonthReport]] = field(default_factory=list)  # [replicate][month]
    gini: list[list[float]] = field(default_factory=list)
    lam: list[np.ndarray] = field(default_factory=list)  # final looks weights per replicate
    looks_auc: list[tuple[float, float]] = field(default_factory=list)  # (before, after)
    looks_corr: list[tuple[float, float]] = field(default_factory=list)

    def per_month(self, fn) -> list[float]:
        return [
            float(np.mean([fn(rep[k]) for rep in self.months])) for k in range(len(self.months[0]))
        ]


def second_date_rate(m: MonthReport) -> float:
    return m.second_dates / max(m.held, 1)


def looks_accuracy(
    after: dict,
    before: dict,
    features: np.ndarray,
    pop: Population,
    arr: MemberArrays,
    cfg: Config,
    seed: int,
    pool: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Looks-model accuracy on each member's pool, before and after learning.

    AUC against synthetic photo swipes (fixed in advance), and rank correlation with the
    hidden true looks appeal. Same members and same candidates for both.
    """
    behavior, _ = month_people(pop, arr, cfg, seed)
    aucs, corrs = [], []
    for models in (before, after):
        a, c = [], []
        for i, taste in models.items():
            idx = np.flatnonzero(pool[i])
            if idx.size < 10:
                continue
            v = taste.like_chance(features[idx], arr.gender[idx])
            a.append(auc(v, behavior.swipe_outcomes(i, seed)[idx]))
            c.append(rank_corr(v, behavior.looks[i, idx]))
        aucs.append(float(np.nanmean(a)))
        corrs.append(float(np.nanmean(c)))
    return (aucs[0], aucs[1]), (corrs[0], corrs[1])


def run_arm(
    pop: Population, cfg: Config, seed: int, base: Scores, arm: str, replicate: int
) -> tuple[list[MonthReport], list[float], np.ndarray, dict]:
    """One arm, one replicate: `learning.months` months. Returns months, Ginis, lambda, tastes."""
    arr = MemberArrays.from_members(pop.members, cfg.values)
    n = len(pop)
    tastes = copy.deepcopy(base.tastes)
    feats = base.features
    groups = member_groups(pop)
    lam = np.full(n, 0.5)
    evidence: dict[int, Evidence] = {}
    dated = np.zeros((n, n), dtype=bool)
    learns = arm != "no learning"
    explores = arm == "learning + exploration"
    months, ginis = [], []
    for k in range(cfg.learning.months):
        pool = base.eligible & ~dated
        v = looks_scores(tastes, feats, arr.gender, pool)
        s, pct_v, pct_p = personal_directional_score(v, base.personality, pool, lam)
        m_scores = mutual_score(s, cfg.mix.chooser_weight)
        unc = uncertainty_matrix(tastes, feats, pool) if explores else None
        month = 100 * replicate + k + 1
        m = run_month(
            pop, cfg, seed, m_scores, month=month, excluded=dated, explore=unc, record=True
        )
        months.append(m)
        ginis.append(exposure_report(m, groups, cfg.fairness.min_rate_ratio).gini)
        for r in m.date_records:
            dated[r.a, r.b] = dated[r.b, r.a] = True
        if learns:
            update_tastes(tastes, m.date_records, feats, arr.gender, cfg.learning.taste_date_weight)
            collect_evidence(evidence, pct_v, pct_p, m.choices, m.date_records, cfg)
            lam = looks_weights(evidence, n, cfg)
    return months, ginis, lam, tastes


@dataclass
class LearningReport:
    seed: int
    arms: dict[str, ArmResult]
    true_looks_weight: np.ndarray
    matchable: np.ndarray

    def lam_recovery(self, arm: str) -> float:
        """Correlation between learned lambda_i and each member's hidden true looks weight."""
        vals = []
        for lam in self.arms[arm].lam:
            moved = self.matchable & (lam != 0.5)
            if moved.sum() > 10:
                vals.append(float(np.corrcoef(lam[moved], self.true_looks_weight[moved])[0, 1]))
        return float(np.mean(vals)) if vals else float("nan")

    def text(self) -> str:
        n_months = len(self.arms[ARMS[0]].months[0])
        reps = len(self.arms[ARMS[0]].months)
        lines = [
            f"Learning over {n_months} months, seed {self.seed}, {reps} replicate(s)",
            "",
            "Second-date rate (both want to meet again) by month",
            f"  {'arm':24s} " + " ".join(f"{'month ' + str(k + 1):>9s}" for k in range(n_months)),
        ]
        for a in ARMS:
            r = self.arms[a].per_month(second_date_rate)
            lines.append(f"  {a:24s} " + " ".join(f"{x:9.1%}" for x in r))
        lines += ["", "Guarantee share by month"]
        for a in ARMS:
            r = self.arms[a].per_month(lambda m: m.guarantee_share())
            lines.append(f"  {a:24s} " + " ".join(f"{x:9.1%}" for x in r))
        lines += ["", "Match quality (true appeal inside booked pairs) by month"]
        for a in ARMS:
            r = self.arms[a].per_month(lambda m: m.pair_appeal)
            lines.append(f"  {a:24s} " + " ".join(f"{x:+9.2f}" for x in r))
        lines += ["", "Exposure Gini by month"]
        for a in ARMS:
            g = [float(np.mean([rep[k] for rep in self.arms[a].gini])) for k in range(n_months)]
            lines.append(f"  {a:24s} " + " ".join(f"{x:9.3f}" for x in g))
        lines += ["", "What the engine learned (after the last month)"]
        for a in ARMS[1:]:
            res = self.arms[a]
            lam = np.mean([np.mean(np.abs(x[self.matchable] - 0.5)) for x in res.lam])
            b, af = np.mean([x[0] for x in res.looks_auc]), np.mean([x[1] for x in res.looks_auc])
            cb, ca = (
                np.mean([x[0] for x in res.looks_corr]),
                np.mean([x[1] for x in res.looks_corr]),
            )
            share = np.mean(
                [m.explore_shown / max(m.total_shown, 1) for rep in res.months for m in rep]
            )
            lines.append(
                f"  {a:24s} looks AUC {b:.3f} -> {af:.3f}, "
                f"corr. with true looks {cb:.3f} -> {ca:.3f}"
            )
            lines.append(
                f"  {'':24s} looks weight moved {lam:.3f} from 0.5 on average; "
                f"corr. with true looks weight {self.lam_recovery(a):+.2f}; "
                f"exploration share of list slots {share:.1%}"
            )
        return "\n".join(lines)


def run_learning_report(
    pop: Population | None = None,
    cfg: Config | None = None,
    seed: int | None = None,
    replicates: int = 1,
    base: Scores | None = None,
) -> LearningReport:
    cfg = cfg or load()
    seed = cfg.test_runs.seed if seed is None else seed
    pop = pop or generate(cfg, seed=seed)
    base = base or build_scores(pop, cfg, seed)
    arr = MemberArrays.from_members(pop.members, cfg.values)
    results = {a: ArmResult(a) for a in ARMS}
    for rep in range(replicates):
        for a in ARMS:
            months, ginis, lam, tastes = run_arm(pop, cfg, seed, base, a, rep)
            res = results[a]
            res.months.append(months)
            res.gini.append(ginis)
            res.lam.append(lam)
            if a != "no learning":
                acc, corr = looks_accuracy(
                    tastes, base.tastes, base.features, pop, arr, cfg, seed, base.eligible
                )
                res.looks_auc.append(acc)
                res.looks_corr.append(corr)
    matchable = np.array([i in base.tastes for i in range(len(pop))])
    return LearningReport(seed, results, pop.truth.looks_weight, matchable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three months with and without learning.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=1)
    args = parser.parse_args()
    cfg = load()
    seed = cfg.test_runs.seed if args.seed is None else args.seed
    print(run_learning_report(generate(cfg, seed=seed), cfg, seed, args.replicates).text())


if __name__ == "__main__":
    main()
