"""Looks report: how well the looks model learns each member's taste (FR-LM-17).

Test runs only (reads ground truth through the behavior test harness).

1. Matchable members are split in half at random. The first half each swipe on
   `test_runs.training_swipes` random profiles; their pooled swipes fit the starting
   taste w_0. The second half are evaluated; they are different people.
2. Each evaluated member calibrates twice from scratch: once with the calibration method
   (20 varied, then 30 most informative) and once with 50 random profiles. Common random
   numbers: every swipe outcome is fixed in advance, so if both methods show the same
   profile the member answers the same way. Differences come only from which profiles
   each method chose.
3. Both models then predict likes on held-out profiles neither method showed. Scores:
   AUC against the member's synthetic held-out swipes (0.5 = coin flip, 1 = perfect), and
   the rank correlation with their hidden true looks appeal.
4. Audit: for each fake demographic group, the share of each member's top 30 by the
   model, by w_0 alone, and by true taste. A group whose model share differs from its
   true-taste share by more than `calibration.group_share_tolerance` is flagged.

Run `python -m lastfirstdate.looks_report` to print it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config, load
from lastfirstdate.filter import member_ok
from lastfirstdate.looks import (
    ArrayPhotoEncoder,
    PersonalTaste,
    StartingTaste,
    Swipe,
    calibrate,
    calibration_pool,
    photo_features,
    sigmoid,
)
from lastfirstdate.member_behavior import Behavior
from lastfirstdate.synthetic import Population, generate

CURVE = (10, 20, 30, 40, 50)
STRATEGIES = ("start", "random", "calibrated", "truth")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Chance a random liked profile scores above a random passed one (Mann-Whitney)."""
    labels = np.asarray(labels, dtype=bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation."""
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def group_shares(ids: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(groups[ids], minlength=n_groups) / max(len(ids), 1)


def flag_groups(model_share: np.ndarray, true_share: np.ndarray, tolerance: float) -> list[int]:
    """Groups whose top-slot share under the model is off from true taste by more than tolerance."""
    return [int(g) for g in np.flatnonzero(np.abs(model_share - true_share) > tolerance)]


@dataclass
class ViewerResult:
    viewer: int
    pool: int
    heldout: int
    auc: dict[str, float]
    corr: dict[str, float]
    curve: dict[str, dict[int, float]]
    top_share: dict[str, np.ndarray]
    coverage: dict[str, int]  # look types seen in the first n_variety profiles


@dataclass
class LooksReport:
    seed: int
    pca_explained: float
    n_train: int
    n_train_swipes: int
    n_groups: int
    tolerance: float
    list_size: int
    n_variety: int
    n_look_types: int
    results: list[ViewerResult] = field(default_factory=list)
    skipped: int = 0

    def mean(self, metric: str, strategy: str) -> float:
        vals = [getattr(r, metric)[strategy] for r in self.results]
        return float(np.nanmean(vals))

    def paired(self, metric: str, a: str, b: str) -> tuple[float, float]:
        """Mean of (a - b) across members, and its standard error."""
        d = np.array([getattr(r, metric)[a] - getattr(r, metric)[b] for r in self.results])
        d = d[~np.isnan(d)]
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))

    def curve(self, strategy: str) -> dict[int, float]:
        return {k: float(np.nanmean([r.curve[strategy][k] for r in self.results])) for k in CURVE}

    def coverage(self, strategy: str) -> float:
        return float(np.mean([r.coverage[strategy] for r in self.results]))

    def shares(self, kind: str) -> np.ndarray:
        return np.mean([r.top_share[kind] for r in self.results], axis=0)

    def flagged(self, kind: str = "calibrated") -> list[int]:
        return flag_groups(self.shares(kind), self.shares("truth"), self.tolerance)

    def text(self) -> str:
        d_rand, se_rand = self.paired("auc", "calibrated", "random")
        d_start, se_start = self.paired("auc", "calibrated", "start")
        lines = [
            f"Looks report, seed {self.seed}",
            f"  photo features: PCA keeps {self.pca_explained:.1%} of photo variance",
            f"  starting taste w_0 fitted on {self.n_train_swipes} swipes from "
            f"{self.n_train} other members",
            f"  members evaluated {len(self.results)} "
            f"(skipped {self.skipped} with too small a pool)",
            "",
            "Predicting held-out swipes (AUC: 0.5 = coin flip, 1.0 = perfect)",
            f"  {'model':34s} {'AUC':>6s} {'corr. with true taste':>22s}",
        ]
        names = {
            "start": "w_0 alone (no personal swipes)",
            "random": "personal, 50 random profiles",
            "calibrated": "personal, 50 calibration profiles",
            "truth": "ceiling: knows true taste",
        }
        for s in STRATEGIES:
            lines.append(
                f"  {names[s]:34s} {self.mean('auc', s):6.3f} {self.mean('corr', s):22.3f}"
            )
        lines += [
            f"  calibration minus random:  {d_rand:+.3f} AUC (standard error {se_rand:.3f})",
            f"  calibration minus w_0:     {d_start:+.3f} AUC (standard error {se_start:.3f})",
            "",
            "AUC by number of swipes",
            "  swipes      " + "  ".join(f"{k:6d}" for k in CURVE),
        ]
        for s in ("random", "calibrated"):
            c = self.curve(s)
            lines.append(f"  {s:10s}  " + "  ".join(f"{c[k]:6.3f}" for k in CURVE))
        lines += [
            "",
            f"Look types seen in the first {self.n_variety} profiles (of {self.n_look_types}): "
            f"calibration {self.coverage('calibrated'):.2f}, random {self.coverage('random'):.2f}",
            "",
            f"Audit: share of each member's top {self.list_size} by fake group",
            "  group   pool   true taste   model   w_0 alone",
        ]
        pool, truth, model, start = (
            self.shares(k) for k in ("pool", "truth", "calibrated", "start")
        )
        for g in range(self.n_groups):
            lines.append(
                f"  G{g + 1}    {pool[g]:5.1%}      {truth[g]:5.1%}    "
                f"{model[g]:5.1%}      {start[g]:5.1%}"
            )
        flagged = self.flagged()
        lines.append(
            f"  flagged (model off from true taste by more than {self.tolerance:.0%}): "
            + (", ".join(f"G{g + 1}" for g in flagged) if flagged else "none")
        )
        return "\n".join(lines)


def _taste_from(swipes, viewer_gender, start, cfg, features, genders, likes) -> PersonalTaste:
    t = PersonalTaste(viewer_gender, start, cfg.calibration.l2_penalty, cfg.calibration.n_profiles)
    if swipes:
        t.features = [features[j] for j in swipes]
        t.candidate_genders = [int(genders[j]) for j in swipes]
        t.liked = [bool(likes[j]) for j in swipes]
        t.refit()
    return t


def run_looks_report(
    pop: Population | None = None,
    cfg: Config | None = None,
    seed: int | None = None,
    max_viewers: int | None = None,
) -> LooksReport:
    cfg = cfg or load()
    seed = cfg.test_runs.seed if seed is None else seed
    pop = pop or generate(cfg, seed=seed)
    n = len(pop)
    rng = np.random.default_rng(seed + 2_000_003)
    arr = MemberArrays.from_members(pop.members, cfg.values)
    behavior = Behavior(pop, arr, cfg, rng)
    pca, feats = photo_features(ArrayPhotoEncoder(pop.photo_embeddings), n, cfg)
    genders = arr.gender
    groups = np.array([int(m.audit_group[1:]) - 1 for m in pop.members])
    n_groups = cfg.population.n_audit_groups
    k_cal = cfg.calibration.n_profiles
    list_size = cfg.lists.daily_size

    viewers = rng.permutation(np.flatnonzero(member_ok(arr, cfg)))
    train, evaluate = viewers[: len(viewers) // 2], viewers[len(viewers) // 2 :]

    def outcomes(v: int) -> np.ndarray:
        return behavior.swipe_outcomes(v, seed)

    swipes: list[Swipe] = []
    for v in train:
        pool = calibration_pool(arr, cfg, int(v))
        likes = outcomes(int(v))
        for j in rng.choice(pool, min(cfg.test_runs.training_swipes, pool.size), replace=False):
            swipes.append(Swipe(int(genders[v]), int(genders[j]), feats[j], bool(likes[j])))
    start = StartingTaste.fit(swipes, cfg.calibration.w0_l2_penalty)

    report = LooksReport(
        seed=seed,
        pca_explained=pca.explained_total,
        n_train=len(train),
        n_train_swipes=len(swipes),
        n_groups=n_groups,
        tolerance=cfg.calibration.group_share_tolerance,
        list_size=list_size,
        n_variety=cfg.calibration.n_variety,
        n_look_types=cfg.population.n_look_types,
    )
    for v in evaluate[:max_viewers] if max_viewers else evaluate:
        v = int(v)
        pool = calibration_pool(arr, cfg, v)
        if pool.size < k_cal + list_size:
            report.skipped += 1
            continue
        likes = outcomes(v)
        g_v = int(genders[v])
        shown: dict[str, list[int]] = {}
        for strategy in ("calibrated", "random"):
            taste = PersonalTaste(g_v, start, cfg.calibration.l2_penalty, k_cal)
            shown[strategy] = calibrate(
                taste,
                pool,
                feats,
                genders,
                lambda j, likes=likes: bool(likes[j]),
                cfg,
                strategy=strategy,
                rng=np.random.default_rng([seed, v, 1]),
            )
        held = np.setdiff1d(pool, shown["calibrated"] + shown["random"])
        x, g, y = feats[held], genders[held], likes[held]
        true = behavior.looks[v, held]
        models = {
            s: _taste_from(shown[s], g_v, start, cfg, feats, genders, likes)
            for s in ("calibrated", "random")
        }
        scores = {
            "start": sigmoid(start.offsets(g_v, x, g)),
            "random": models["random"].list_scores(x, g),
            "calibrated": models["calibrated"].list_scores(x, g),
            "truth": true,  # the ceiling: swipes are noisy, so even this is below 1
        }
        curve = {
            s: {
                k: auc(
                    _taste_from(shown[s][:k], g_v, start, cfg, feats, genders, likes).like_chance(
                        x, g
                    ),
                    y,
                )
                for k in CURVE
            }
            for s in ("random", "calibrated")
        }

        def top(sc: np.ndarray, held: np.ndarray = held) -> np.ndarray:
            return held[np.argsort(-sc, kind="stable")[:list_size]]

        report.results.append(
            ViewerResult(
                viewer=v,
                pool=int(pool.size),
                heldout=int(held.size),
                auc={s: auc(sc, y) for s, sc in scores.items()},
                corr={s: rank_corr(sc, true) for s, sc in scores.items()},
                curve=curve,
                coverage={
                    s: len(set(pop.truth.look_type[shown[s][: cfg.calibration.n_variety]].tolist()))
                    for s in ("calibrated", "random")
                },
                top_share={
                    "pool": group_shares(held, groups, n_groups),
                    "truth": group_shares(top(true), groups, n_groups),
                    "calibrated": group_shares(top(scores["calibrated"]), groups, n_groups),
                    "start": group_shares(top(scores["start"]), groups, n_groups),
                },
            )
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report how well the looks model learns taste.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--viewers", type=int, default=None, help="evaluate at most this many")
    args = parser.parse_args()
    cfg = load()
    seed = cfg.test_runs.seed if args.seed is None else args.seed
    pop = generate(cfg, seed=seed, n=args.n)
    print(run_looks_report(pop, cfg, seed=seed, max_viewers=args.viewers).text())


if __name__ == "__main__":
    main()
