"""Compare ways of ranking lists in a synthetic month (FR-LM-18 to FR-LM-20).

Test runs only. Every matchable fake member first calibrates (Stages 2 and 3) so the
engine has a looks score V for them. Then the same month (same members, same random
choices) is run four times, ranking daily lists and weighting dates by:

- personality: P only (Sprint 1)
- looks only: the chooser's percentile of V (looks_weight = 1)
- one-sided: S(chooser -> candidate), the chooser's own 50/50 view
- mutual: M, the harmonic mean of both directions (the Sprint 3 engine)

Run `python -m lastfirstdate.mutual_report` to print the comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass

import numpy as np

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config, load
from lastfirstdate.filter import eligibility_matrix, member_ok
from lastfirstdate.lists import pick_rights
from lastfirstdate.looks import (
    ArrayPhotoEncoder,
    PersonalTaste,
    StartingTaste,
    Swipe,
    calibrate,
    calibration_pool,
    photo_features,
)
from lastfirstdate.member_behavior import Behavior
from lastfirstdate.month_runs import MonthReport, month_people, run_month
from lastfirstdate.mutual import directional_score, looks_scores, mutual_score
from lastfirstdate.personality import personality_matrix
from lastfirstdate.synthetic import Population, generate

RANKINGS = ("personality", "looks only", "one-sided", "mutual")
TOP_K = 10


@dataclass
class Scores:
    eligible: np.ndarray
    personality: np.ndarray  # P
    looks: np.ndarray  # V
    directional: np.ndarray  # S
    mutual: np.ndarray  # M
    n_calibrated: int
    tastes: dict[int, PersonalTaste] | None = None  # each member's looks model (Sprint 5 learns)
    features: np.ndarray | None = None  # every member's photo features

    def ranking(self, name: str, cfg: Config) -> np.ndarray:
        if name == "personality":
            return self.personality
        if name == "looks only":
            return directional_score(self.looks, self.personality, self.eligible, 1.0)
        if name == "one-sided":
            return self.directional
        if name == "mutual":
            return self.mutual
        raise ValueError(f"unknown ranking {name!r}")


def learn_tastes(
    pop: Population, arr: MemberArrays, cfg: Config, seed: int
) -> tuple[dict[int, PersonalTaste], np.ndarray]:
    """Calibrate every matchable member; return their taste models and photo features.

    The starting taste w_0 is fitted first on earlier "history" swipes: each matchable
    member's swipes on `test_runs.training_swipes` random profiles from their pool.
    """
    behavior = Behavior(pop, arr, cfg, np.random.default_rng(seed + 3_000_017))
    _, feats = photo_features(ArrayPhotoEncoder(pop.photo_embeddings), len(pop), cfg)
    rng = np.random.default_rng(seed + 4_000_037)
    members = np.flatnonzero(member_ok(arr, cfg))
    pools = {int(v): calibration_pool(arr, cfg, int(v)) for v in members}
    likes = {v: behavior.swipe_outcomes(v, seed) for v in pools}
    history = []
    for v, pool in pools.items():
        for j in rng.choice(pool, min(cfg.test_runs.training_swipes, pool.size), replace=False):
            history.append(
                Swipe(int(arr.gender[v]), int(arr.gender[j]), feats[j], bool(likes[v][j]))
            )
    start = StartingTaste.fit(history, cfg.calibration.w0_l2_penalty)
    tastes: dict[int, PersonalTaste] = {}
    for v, pool in pools.items():
        if pool.size == 0:
            continue
        taste = PersonalTaste(
            int(arr.gender[v]), start, cfg.calibration.l2_penalty, cfg.calibration.n_profiles
        )
        calibrate(taste, pool, feats, arr.gender, lambda j, lk=likes[v]: bool(lk[j]), cfg)
        tastes[v] = taste
    return tastes, feats


def build_scores(pop: Population, cfg: Config, seed: int) -> Scores:
    """Every Stage 1 to 5 matrix the month needs."""
    arr = MemberArrays.from_members(pop.members, cfg.values)
    eligible = eligibility_matrix(arr, cfg)
    p = personality_matrix(pop.members, arr, cfg)
    tastes, feats = learn_tastes(pop, arr, cfg, seed)
    v = looks_scores(tastes, feats, arr.gender, eligible)
    s = directional_score(v, p, eligible, cfg.mix.looks_weight)
    m = mutual_score(s, cfg.mix.chooser_weight)
    return Scores(eligible, p, v, s, m, len(tastes), tastes=tastes, features=feats)


@dataclass(frozen=True)
class TopQuality:
    """Expected outcomes for the top of each chooser's list, from the hidden truth."""

    pick: float  # chance the chooser would pick the candidate
    accept: float  # chance the candidate would accept the chooser
    mutual: float  # chance of both: pick and accept


def top_quality(
    ranking: np.ndarray,
    eligible: np.ndarray,
    rights: np.ndarray,
    behavior: Behavior,
    cfg: Config,
    k: int = TOP_K,
) -> TopQuality:
    """Average pick, accept and mutual-yes chances over each chooser's top k by `ranking`.

    Uses the test harness's true appeal, so it measures how good the ranking is with no
    dependence on how a month happens to unfold. Choosers with fewer than k candidates
    are left out.
    """
    sim = cfg.test_runs
    u = behavior.utility
    t = sim.choice_temperature
    pick = 1 / (1 + np.exp(-(u - sim.like_threshold - behavior.pickiness[:, None]) / t))
    accept = 1 / (1 + np.exp(-(u.T - sim.accept_threshold - behavior.pickiness[None, :]) / t))
    rows_pick, rows_acc, rows_mut = [], [], []
    for i in range(u.shape[0]):
        cand = np.flatnonzero(eligible[i] & rights[i])
        if cand.size < k:
            continue
        top = cand[np.lexsort((cand, -ranking[i, cand]))[:k]]
        rows_pick.append(pick[i, top].mean())
        rows_acc.append(accept[i, top].mean())
        rows_mut.append((pick[i, top] * accept[i, top]).mean())
    return TopQuality(float(np.mean(rows_pick)), float(np.mean(rows_acc)), float(np.mean(rows_mut)))


@dataclass
class MutualReport:
    seed: int
    n_calibrated: int
    months: dict[str, MonthReport]
    top: dict[str, TopQuality]

    def text(self) -> str:
        lines = [
            f"Ranking comparison, seed {self.seed} "
            f"({self.n_calibrated} members calibrated on 50 swipes each)",
            "",
            f"Top {TOP_K} of each chooser's list (chances from the hidden truth)",
            f"  {'ranking':12s} {'she picks':>10s} {'he accepts':>11s} {'both':>6s}",
        ]
        for name in RANKINGS:
            q = self.top[name]
            lines.append(f"  {name:12s} {q.pick:10.1%} {q.accept:11.1%} {q.mutual:6.1%}")
        lines += ["", "A synthetic month with each ranking"]
        head = (
            f"  {'ranking':12s} {'picks':>6s} {'accepted':>9s} {'rate':>5s} {'dates':>6s} "
            f"{'2nd date':>8s} {'appeal':>7s} {'both keen':>9s} {'guarantee':>9s}"
        )
        lines.append(head)
        for name in RANKINGS:
            m = self.months[name]
            lines.append(
                f"  {name:12s} {m.picks:6d} {m.accepts:9d} {m.accepts / max(m.picks, 1):5.0%} "
                f"{m.booked:6d} {m.second_dates / max(m.held, 1):8.0%} {m.pair_appeal:+7.2f} "
                f"{m.both_keen:9.0%} {m.guarantee_share():9.1%}"
            )
        lines += [
            "",
            "  accepted = picks the other person accepted (mutual yeses)",
            "  appeal = average true appeal inside booked pairs, in standard deviations",
            "  both keen = booked pairs where each truly likes the other above their average",
            "  guarantee = paying members with 10+ possible matches who reached 3 dates",
        ]
        return "\n".join(lines)


def compare_rankings(
    pop: Population | None = None, cfg: Config | None = None, seed: int | None = None
) -> MutualReport:
    cfg = cfg or load()
    seed = cfg.test_runs.seed if seed is None else seed
    pop = pop or generate(cfg, seed=seed)
    scores = build_scores(pop, cfg, seed)
    top = top_quality_all(pop, cfg, seed, scores)
    months = {name: run_month(pop, cfg, seed, scores.ranking(name, cfg)) for name in RANKINGS}
    return MutualReport(seed=seed, n_calibrated=scores.n_calibrated, months=months, top=top)


def top_quality_all(
    pop: Population, cfg: Config, seed: int, scores: Scores
) -> dict[str, TopQuality]:
    """top_quality for every ranking, judged by the same synthetic people as run_month."""
    arr = MemberArrays.from_members(pop.members, cfg.values)
    behavior, _ = month_people(pop, arr, cfg, seed)
    rights = pick_rights(arr)
    return {
        name: top_quality(scores.ranking(name, cfg), scores.eligible, rights, behavior, cfg)
        for name in RANKINGS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare list rankings in a synthetic month.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--looks-weight", type=float, default=None)
    args = parser.parse_args()
    cfg = load()
    if args.looks_weight is not None:
        cfg = dataclasses.replace(
            cfg, mix=dataclasses.replace(cfg.mix, looks_weight=args.looks_weight)
        )
    seed = cfg.test_runs.seed if args.seed is None else args.seed
    pop = generate(cfg, seed=seed, n=args.n)
    print(compare_rankings(pop, cfg, seed).text())


if __name__ == "__main__":
    main()
