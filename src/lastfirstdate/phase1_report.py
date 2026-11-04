"""Phase 1 report: lastfirstdate's real rules in simulation (Sprint 6).

Simulation only. Two parts:

1. Calibration (FR-LM-32): does a 30-photo exercise with pair questions ("Who are you more
   drawn to?") learn a member's type as well as single swipes? Every version shows the same
   member the same questions and gets the same answers (common random numbers); accuracy is
   AUC on photos none of the versions showed.
2. Gate (FR-LM-34): starting from committed women and a men's waitlist, admit the men the gate
   allows, then simulate the admitted market and compare its fulfilment with the forecast and
   the target. Three versions: the gate as set, the forecast alone (no women-per-man floor),
   and admitting everyone. (The forecast's own held-out test is in `forecast_fit.py`.)

Run `python -m lastfirstdate.phase1_report` to print it.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field

import numpy as np

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config, load
from lastfirstdate.filter import member_ok
from lastfirstdate.forecast import (
    ForecastModel,
    forecast_fulfilment,
    gate_decision,
    market_features,
)
from lastfirstdate.looks import (
    ArrayPhotoEncoder,
    PersonalTaste,
    StartingTaste,
    Swipe,
    calibrate,
    calibration_pool,
    photo_features,
)
from lastfirstdate.looks_report import auc, rank_corr
from lastfirstdate.mutual_report import build_scores
from lastfirstdate.simulate import run_month
from lastfirstdate.simulator import Behavior
from lastfirstdate.synthetic import Population, generate

PHASE1 = "config/phase1.toml"
CALIBRATION_ARMS = {
    # name: (photos shown, varied singles, pair questions)
    "50 single photos": (50, 20, 0),
    "30 single photos": (30, 12, 0),
    "30 photos with 9 pairs": (30, 12, 9),
}


@dataclass
class CalibrationComparison:
    seed: int
    aucs: dict[str, list[float]] = field(default_factory=dict)
    corrs: dict[str, list[float]] = field(default_factory=dict)
    skipped: int = 0

    def mean(self, arm: str, what: str = "auc") -> float:
        return float(np.nanmean((self.aucs if what == "auc" else self.corrs)[arm]))

    def paired(self, a: str, b: str) -> tuple[float, float]:
        d = np.array(self.aucs[a]) - np.array(self.aucs[b])
        d = d[~np.isnan(d)]
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))

    def text(self) -> str:
        n = len(next(iter(self.aucs.values())))
        lines = [
            f"Calibration: single photos versus pair questions, seed {self.seed}, {n} members",
            f"  {'version':26s} {'AUC':>6s} {'corr. with true taste':>22s}",
        ]
        for arm in CALIBRATION_ARMS:
            lines.append(f"  {arm:26s} {self.mean(arm):6.3f} {self.mean(arm, 'corr'):22.3f}")
        d, se = self.paired("30 photos with 9 pairs", "30 single photos")
        lines.append(f"  pairs minus singles (30 photos): {d:+.3f} AUC (standard error {se:.3f})")
        d, se = self.paired("30 photos with 9 pairs", "50 single photos")
        lines.append(f"  30 with pairs minus 50 singles: {d:+.3f} AUC (standard error {se:.3f})")
        return "\n".join(lines)


def calibration_comparison(
    pop: Population, cfg: Config, seed: int, max_viewers: int | None = None
) -> CalibrationComparison:
    """Compare the calibration versions in CALIBRATION_ARMS on the same members."""
    n = len(pop)
    rng = np.random.default_rng(seed + 5_000_011)
    arr = MemberArrays.from_members(pop.members, cfg.values)
    behavior = Behavior(pop, arr, cfg, rng)
    _, feats = photo_features(ArrayPhotoEncoder(pop.photo_embeddings), n, cfg)
    viewers = rng.permutation(np.flatnonzero(member_ok(arr, cfg)))
    train, evaluate = viewers[: len(viewers) // 2], viewers[len(viewers) // 2 :]
    swipes = []
    for v in train:
        pool = calibration_pool(arr, cfg, int(v))
        likes = behavior.swipe_outcomes(int(v), seed)
        for j in rng.choice(pool, min(cfg.simulation.training_swipes, pool.size), replace=False):
            swipes.append(Swipe(int(arr.gender[v]), int(arr.gender[j]), feats[j], bool(likes[j])))
    start = StartingTaste.fit(swipes, cfg.calibration.w0_l2_penalty)
    out = CalibrationComparison(
        seed, {a: [] for a in CALIBRATION_ARMS}, {a: [] for a in CALIBRATION_ARMS}
    )
    held_min = cfg.lists.daily_size
    for v in evaluate[:max_viewers] if max_viewers else evaluate:
        v = int(v)
        pool = calibration_pool(arr, cfg, v)
        if pool.size < 50 + held_min:
            out.skipped += 1
            continue
        likes = behavior.swipe_outcomes(v, seed)
        tastes, shown_all = {}, set()
        for arm, (k, n_var, n_pairs) in CALIBRATION_ARMS.items():
            c = dataclasses.replace(
                cfg,
                calibration=dataclasses.replace(
                    cfg.calibration, n_profiles=k, n_variety=n_var, n_pairs=n_pairs
                ),
            )
            taste = PersonalTaste(int(arr.gender[v]), start, cfg.calibration.l2_penalty, k)
            shown = calibrate(
                taste,
                pool,
                feats,
                arr.gender,
                lambda j, lk=likes: bool(lk[j]),
                c,
                pair=lambda a, b, v=v: behavior.pair_outcome(v, a, b, seed),
            )
            tastes[arm] = taste
            shown_all |= set(shown)
        held = np.setdiff1d(pool, sorted(shown_all))
        for arm, taste in tastes.items():
            score = taste.like_chance(feats[held], arr.gender[held])
            out.aucs[arm].append(auc(score, likes[held]))
            out.corrs[arm].append(rank_corr(score, behavior.looks[v, held]))
    return out


GATE_ARMS = ("gate as set", "forecast only", "everyone")
GATE_SCENARIOS = {
    # name: (women invited, men on the waitlist); about 6% are inactive and drop out
    "waitlist smaller than the women": (300, 200),
    "waitlist larger than the women": (300, 450),
}


@dataclass
class GateRow:
    seed: int
    arm: str
    women: int
    waitlist: int
    admitted: int
    women_per_man: float
    forecast: float
    actual: list[float]  # share of admitted men reaching the guarantee, per month
    stopped_because: str


@dataclass
class GateTrial:
    rows: list[GateRow]
    target: float
    title: str = ""

    def mean_actual(self, arm: str) -> float:
        return float(np.mean([np.mean(r.actual) for r in self.rows if r.arm == arm]))

    def mean_forecast(self, arm: str) -> float:
        return float(np.mean([r.forecast for r in self.rows if r.arm == arm]))

    def text(self) -> str:
        n = len(self.rows[0].actual)
        lines = [
            f"Gate trial{': ' + self.title if self.title else ''} (target {self.target:.0%} "
            f"of admitted men reaching 3 dates, months 1 to {n})",
            f"  {'seed':>4s} {'version':14s} {'women':>5s} {'waitlist':>8s} {'admitted':>8s} "
            f"{'women/man':>9s} {'forecast':>8s}  actual by month",
        ]
        for r in self.rows:
            lines.append(
                f"  {r.seed:4d} {r.arm:14s} {r.women:5d} {r.waitlist:8d} {r.admitted:8d} "
                f"{r.women_per_man:9.2f} {r.forecast:8.1%}  "
                + ", ".join(f"{x:.1%}" for x in r.actual)
            )
        lines.append("  averages:")
        for arm in dict.fromkeys(r.arm for r in self.rows):
            admitted = np.mean([r.admitted for r in self.rows if r.arm == arm])
            lines.append(
                f"    {arm:14s} admitted {admitted:5.1f}, forecast "
                f"{self.mean_forecast(arm):.1%}, actual {self.mean_actual(arm):.1%}"
            )
        return "\n".join(lines)


def gate_trial(
    cfg: Config,
    model: ForecastModel,
    n_women: int = 300,
    n_waitlist: int = 200,
    seeds: tuple[int, ...] = (401, 402, 403),
    months: int = 2,
    arms: tuple[str, ...] = GATE_ARMS,
    title: str = "",
) -> GateTrial:
    """Let the gate admit men from a waitlist, then simulate the admitted market.

    Versions: "gate as set" uses `[gate]` as configured; "forecast only" drops the
    women-per-man floor so only the forecast decides; "everyone" admits the whole waitlist.
    Men not admitted are kept out of the simulated month entirely.
    """
    n = n_women + n_waitlist
    pop_cfg = dataclasses.replace(
        cfg.population, n_members=n, share_men=n_waitlist / n, exact_counts=True
    )
    guarantee = cfg.dates.guaranteed_per_month_paying
    rows = []
    for seed in seeds:
        pop = generate(dataclasses.replace(cfg, population=pop_cfg), seed=seed)
        sc = build_scores(pop, cfg, seed)
        arr = MemberArrays.from_members(pop.members, cfg.values)
        ok = member_ok(arr, cfg)
        men_mask = np.array([m.gender == "man" for m in pop.members])
        women = np.flatnonzero(ok & ~men_mask)
        waitlist = np.flatnonzero(ok & men_mask & arr.paying)  # in application order
        for arm in arms:
            if arm == "everyone":
                mf = market_features(sc.eligible, women, waitlist)
                admitted = waitlist
                ratio, forecast = mf.women_per_man, forecast_fulfilment(model, mf)
                why = "admits everyone"
            else:
                c = cfg
                if arm == "forecast only":
                    c = dataclasses.replace(
                        cfg, gate=dataclasses.replace(cfg.gate, min_women_per_man=0.0)
                    )
                d = gate_decision(model, sc.eligible, women, [], waitlist, c)
                admitted = np.array(d.admit, dtype=int)
                ratio, forecast, why = d.women_per_man, d.forecast, d.stopped_because
            others = np.setdiff1d(np.flatnonzero(men_mask), admitted)
            excluded = np.zeros(sc.eligible.shape, dtype=bool)
            excluded[others, :] = True
            excluded[:, others] = True
            actual = []
            for k in range(months):
                m = run_month(pop, cfg, seed, sc.mutual, month=k + 1, excluded=excluded)
                actual.append(
                    float((m.dates[admitted] >= guarantee).mean()) if admitted.size else 1.0
                )
            rows.append(
                GateRow(
                    seed,
                    arm,
                    int(women.size),
                    int(waitlist.size),
                    int(admitted.size),
                    float(ratio),
                    float(forecast),
                    actual,
                    why,
                )
            )
    return GateTrial(rows, cfg.gate.target, title)


def main() -> None:
    parser = argparse.ArgumentParser(description="lastfirstdate's real rules in simulation.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--viewers", type=int, default=150)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()
    cfg = load()
    seed = cfg.simulation.seed if args.seed is None else args.seed
    if not args.skip_calibration:
        print(calibration_comparison(generate(cfg, seed=seed), cfg, seed, args.viewers).text())
        print()
    phase1, model = load(PHASE1), ForecastModel.load()
    for title, (n_women, n_waitlist) in GATE_SCENARIOS.items():
        print(gate_trial(phase1, model, n_women, n_waitlist, title=title).text())
        print()


if __name__ == "__main__":
    main()
