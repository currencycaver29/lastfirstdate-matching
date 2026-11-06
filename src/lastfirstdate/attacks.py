"""Abuse testing: synthetic attackers against the engine, and what each one gains (FR-LM-63).

Test runs only. The attackers are made-up members of a made-up Phase 1 zone, and the engine
never imports this module. Each attack is measured as what the attacker gains, with the
engine's defence off and on, on seeds no setting was tuned on (2026 to 2028):

1. Stuffed prompts (FR-LM-64, D21). Ten men rewrite their prompts. "Popular words" lists the
   interests women mention most; "popular, as a prompt" puts the top twelve into the app's
   prompt sentences; "everyone's words" repeats the words nearly every prompt contains (the
   text version of the "average attack" on recommenders: look like everyone).
   Gain: their mean percentile in women's personality rankings (0.5 is the middle) and the
   share of them reaching 3 delivered dates a cycle.
2. Answer flood (FR-LM-65, D23). One fake woman writes twenty exercises' worth of answers,
   liking photos that look like a target man and passing the rest. Gain: the target's mean
   percentile in honest women's looks rankings.
3. Fake accounts steering the starting taste (D23). 15 or 30 fake women (5% or 10%) each
   answer one exercise the same way. Defence: identity checks, because only verified members'
   answers fit w_0. The "robust fit" row shows what a statistical defence would cost.
4. Fake women on the gate (FR-LM-66, D22). 50 fake women who never date join a 300-woman zone
   before launch. Harm: the places the gate sells, and the 3-date share of the men who buy
   them. Then, after four weeks of lists, the share of fakes and of honest women the silent
   rule stops counting.
5. Rating attacks (FR-LM-69). Harsh raters: 15 women who rate every man 1. Rings: 3 or 6
   women eligible for the same man who pick him whenever he's on their list, whom he always
   accepts, and who rate him 1. Gain: honest men wrongly flagged; targets paused.
6. Spies (FR-LM-70). 20 women who never pick read their lists every day for two weeks and
   rank men by when and where they first appear. Gain: the Spearman correlation of that
   ranking with how much women in the zone like each man's looks (the hidden truth), with the
   exposure cap on and off.

Run `python -m lastfirstdate.attacks` (about 15 minutes). `--minilm MODELS_DIR` runs attack 1
against the production text model instead (the model files must already be downloaded).
"""

from __future__ import annotations

import argparse
import dataclasses
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata, spearmanr

from lastfirstdate.arrays import GENDERS, MemberArrays
from lastfirstdate.config import Config, load
from lastfirstdate.cycle_runs import NEVER, CycleRun, join_days, market
from lastfirstdate.filter import eligibility_matrix, member_ok
from lastfirstdate.forecast import DELIVERED_MODEL_PATH, ForecastModel, gate_decision
from lastfirstdate.looks import (
    ArrayPhotoEncoder,
    PersonalTaste,
    StartingTaste,
    Swipe,
    calibrate,
    calibration_pool,
    fit_logistic,
    photo_features,
)
from lastfirstdate.looks_report import auc
from lastfirstdate.member_behavior import Behavior
from lastfirstdate.mutual import looks_scores, percentile_rows
from lastfirstdate.mutual_report import build_scores
from lastfirstdate.personality import WordEncoder, personality_matrix
from lastfirstdate.synthetic import Population
from lastfirstdate.trust import Rating

PHASE1 = "config/phase1.toml"
SEEDS = (2026, 2027, 2028)
MAN, WOMAN = GENDERS.index("man"), GENDERS.index("woman")
PROMPT_LIMIT = 500  # characters, as db/schema.sql allows
EVERYONE_SHARE = 0.9  # a word in at least this share of women's prompts is "everyone's"
LOOKALIKE_QUANTILE = 0.7  # fakes like photos more like the target than 70% of men's


def with_market(cfg: Config, n_women: int, n_men: int) -> Config:
    """The settings with a zone of `n_women` women and `n_men` men at launch."""
    n = n_women + n_men
    pop = dataclasses.replace(cfg.population, n_members=n, share_men=n_men / n)
    return dataclasses.replace(cfg, population=pop)


def centered(cfg: Config, on: bool) -> Config:
    return dataclasses.replace(
        cfg, personality=dataclasses.replace(cfg.personality, center_text=on)
    )


def launch_women(pop: Population, cfg: Config, join: np.ndarray) -> np.ndarray:
    arr = MemberArrays.from_members(pop.members, cfg.values)
    return np.flatnonzero(member_ok(arr, cfg) & (arr.gender == WOMAN) & (join == 0))


def men_of(pop: Population) -> np.ndarray:
    return np.flatnonzero([m.gender == "man" for m in pop.members])


# ---------------------------------------------------------------- 1. stuffed prompts


def stuffed_texts(pop: Population, women: Sequence[int]) -> dict[str, str]:
    """The two stuffed prompts, built from what the zone's women write."""
    df: Counter[str] = Counter()
    for w in women:
        df.update(set(WordEncoder.tokens(pop.members[w].bio)))
    cut = EVERYONE_SHARE * len(women)
    everyone = [x for x, c in df.most_common() if c >= cut]
    popular = [x for x, c in df.most_common() if c < cut]
    repeated = " ".join(everyone)
    return {
        "popular words": ("I'm into " + ", ".join(popular))[:PROMPT_LIMIT],
        "popular, as a prompt": (
            f"A perfect Sunday looks like… {', '.join(popular[:6])}. "
            f"I'm looking for… someone into {', '.join(popular[6:12])}."
        ),
        "everyone's words": " ".join([repeated] * (PROMPT_LIMIT // (len(repeated) + 1)))
        if everyone
        else "",
    }


def with_prompts(pop: Population, members: Sequence[int], text: str) -> Population:
    people = list(pop.members)
    for m in members:
        people[m] = dataclasses.replace(people[m], bio=text)
    return dataclasses.replace(pop, members=people)


def personality_rank(pop: Population, cfg: Config, attackers: Sequence[int]) -> float:
    """The attackers' mean percentile in eligible women's personality rankings."""
    arr = MemberArrays.from_members(pop.members, cfg.values)
    eligible = eligibility_matrix(arr, cfg)
    pct = percentile_rows(personality_matrix(pop.members, arr, cfg), eligible)
    women = np.flatnonzero(arr.gender == WOMAN)
    return float(np.mean([pct[w, a] for a in attackers for w in women if eligible[w, a]]))


@dataclass(frozen=True)
class Exposure:
    """What a group of men gets from women's lists: (attackers, other men)."""

    top10: tuple[float, float]  # share of eligible women with him in their top 10 by M
    shown: tuple[float, float]  # times on a woman's list per cycle
    picked: tuple[float, float]  # times picked per cycle


def stuffed_exposure(
    pop: Population, cfg: Config, seed: int, join: np.ndarray, attackers: Sequence[int]
) -> Exposure:
    """Attackers' place in women's rankings and what three cycles of the real flow give them,
    against the other men. Women still pick on true appeal, which no prompt changes."""
    sc = build_scores(pop, cfg, seed)
    arr = MemberArrays.from_members(pop.members, cfg.values)
    women = np.flatnonzero(arr.gender == WOMAN)
    men = np.flatnonzero(arr.gender == MAN)
    top = np.zeros(len(pop))
    rows = 0
    for w in women:
        cand = np.flatnonzero(sc.eligible[w])
        if cand.size > 10:
            rows += 1
            top[cand[np.argsort(-sc.mutual[w, cand], kind="stable")[:10]]] += 1
    run = AttackedRun(pop, cfg, seed, sc.mutual, join)
    run.run("stuffed prompts")
    mine = np.isin(men, attackers)
    cycles = cfg.cycle_runs.cycles

    def split(x: np.ndarray) -> tuple[float, float]:
        return float(x[men][mine].mean()), float(x[men][~mine].mean())

    return Exposure(
        split(top / max(rows, 1)), split(run.times_shown / cycles), split(run.times_picked / cycles)
    )


# ---------------------------------------------------------------- 2, 3. the starting taste


@dataclass(frozen=True)
class TasteAttack:
    fakes: int  # fake accounts
    answers_each: int  # answers each writes, as a multiple of one exercise
    one_exercise: bool = True  # defence: one exercise per account (FR-LM-65)
    verified: bool = True  # do the fakes pass identity checks? Only verified answers fit w_0
    robust: bool = False  # w_0 as a trimmed mean of each member's own fit, not one pooled fit


ROBUST_TRIM = 0.25  # share cut from each end of every weight in the robust fit
ROBUST_L2 = 1.0  # penalty on each member's own fit in the robust fit


def robust_start(swipes: list[Swipe], owners: np.ndarray, l2: float) -> StartingTaste:
    """w_0 as a trimmed mean of each member's own fit: a fake account's answers move it only
    as one member among many, but each member's fit is noisy (the cost shows in accuracy)."""
    pooled = StartingTaste.fit(swipes, l2)
    x = np.array([s.features for s in swipes])
    y = np.array([s.liked for s in swipes], dtype=float)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, s in enumerate(swipes):
        groups[s.viewer_gender, s.candidate_gender].append(k)
    by_group = {}
    for g, idx in groups.items():
        idx = np.array(idx)
        fits = np.array(
            [fit_logistic(x[r], y[r], ROBUST_L2)[0] for r in _by_owner(idx, owners[idx])]
        )
        fits.sort(axis=0)
        k = int(ROBUST_TRIM * len(fits))
        by_group[g] = fits[k : len(fits) - k].mean(axis=0)
    return StartingTaste(by_group=by_group, pooled=pooled.pooled)


def _by_owner(idx: np.ndarray, owners: np.ndarray) -> list[np.ndarray]:
    return [idx[owners == v] for v in np.unique(owners)]


def taste_attack(
    pop: Population, cfg: Config, seed: int, target: int, attack: TasteAttack
) -> tuple[float, float]:
    """(target's mean percentile in honest women's looks rankings, their looks AUC).

    As in the test runs' `learn_tastes`, w_0 is fitted on each member's earlier swipes
    (`test_runs.training_swipes` of them, one "exercise"), then every honest woman calibrates
    from it. Fake women answer about photos from their own pool, liking those that look like
    the target. "One exercise" keeps each account to one exercise's worth of photos, each once.
    """
    arr = MemberArrays.from_members(pop.members, cfg.values)
    behavior = Behavior(pop, arr, cfg, np.random.default_rng(seed + 3_000_017))
    _, feats = photo_features(ArrayPhotoEncoder(pop.photo_embeddings), len(pop), cfg)
    emb = pop.photo_embeddings / np.linalg.norm(pop.photo_embeddings, axis=1, keepdims=True)
    men = np.flatnonzero(arr.gender == MAN)
    looks_like = emb @ emb[target] > np.quantile(emb[men] @ emb[target], LOOKALIKE_QUANTILE)
    members = np.flatnonzero(member_ok(arr, cfg))
    women = np.flatnonzero(member_ok(arr, cfg) & (arr.gender == WOMAN))
    fakes = set(np.random.default_rng([seed, 5]).choice(women, attack.fakes, replace=False))
    per = cfg.test_runs.training_swipes
    pools = {int(v): calibration_pool(arr, cfg, int(v)) for v in members}
    likes = {v: behavior.swipe_outcomes(v, seed) for v in pools}
    swipes, owners = [], []
    for v, pool in pools.items():
        rng = np.random.default_rng([seed, v, 41])  # each member's own: the same honest swipes
        if attack.answers_each == 1 or v not in fakes:
            shown = rng.choice(pool, min(per, pool.size), replace=False)
        else:  # a flood: many answers, photos repeating
            shown = rng.choice(pool, per * attack.answers_each, replace=True)
        if v in fakes:
            if attack.one_exercise:
                shown = np.array(list(dict.fromkeys(shown.tolist()))[:per])
            if not attack.verified:
                continue  # only verified members' answers fit w_0 (D23)
            answers = [(int(j), bool(looks_like[j])) for j in shown]
        else:
            answers = [(int(j), bool(likes[v][j])) for j in shown]
        for j, liked in answers:
            swipes.append(Swipe(int(arr.gender[v]), int(arr.gender[j]), feats[j], liked))
            owners.append(v)
    l2 = cfg.calibration.w0_l2_penalty
    start = (
        robust_start(swipes, np.array(owners), l2)
        if attack.robust
        else StartingTaste.fit(swipes, l2)
    )
    honest = [int(w) for w in women if w not in fakes]
    eligible = eligibility_matrix(arr, cfg)
    tastes = {}
    for w in honest:
        taste = PersonalTaste(WOMAN, start, cfg.calibration.l2_penalty, cfg.calibration.n_profiles)
        calibrate(taste, pools[w], feats, arr.gender, lambda j, lk=likes[w]: bool(lk[j]), cfg)
        tastes[w] = taste
    v = looks_scores(tastes, feats, arr.gender, eligible)
    pct = percentile_rows(v, eligible)
    rank = float(np.mean([pct[w, target] for w in honest if eligible[w, target]]))
    aucs = []
    for w in honest:
        cand = np.flatnonzero(eligible[w])
        if cand.size >= 20:
            truth = behavior.looks[w, cand]
            aucs.append(auc(v[w, cand], truth > np.median(truth)))
    return rank, float(np.mean(aucs))


def popular_target(pop: Population, cfg: Config, seed: int) -> int:
    """A man eligible for many women, so his rank is measured over many lists."""
    arr = MemberArrays.from_members(pop.members, cfg.values)
    eligible = eligibility_matrix(arr, cfg)
    men = np.flatnonzero(arr.gender == MAN)
    counts = eligible[np.ix_(arr.gender == WOMAN, men)].sum(axis=0)
    middle = men[(counts >= np.median(counts))]
    return int(np.random.default_rng([seed, 3]).choice(middle))


# ---------------------------------------------------------------- 4. fake women on the gate


@dataclass(frozen=True)
class GateTrial:
    counted: int  # women the gate counted
    admitted: int  # men it sold places to
    forecast: float
    actual: float  # share of their paid cycles with 3 delivered dates


def gate_with_fakes(cfg: Config, seed: int, n_women: int, n_fake: int) -> GateTrial:
    """The gate at launch with `n_fake` fake women counted, then three cycles of the real flow
    in which the fakes never date (they never join)."""
    model = ForecastModel.load(DELIVERED_MODEL_PATH)
    c = with_market(cfg, n_women + n_fake, 450)
    rate = c.cycle_runs.new_women_per_cycle
    pop = market(c, seed, rate)
    sc = build_scores(pop, c, seed)
    arr = MemberArrays.from_members(pop.members, c.values)
    join = join_days(pop, c, rate)
    women = launch_women(pop, c, join)
    fakes = np.random.default_rng([seed, 77]).choice(women, n_fake, replace=False)
    men = arr.gender == MAN
    waitlist = np.flatnonzero(member_ok(arr, c) & men & arr.paying)
    d = gate_decision(model, sc.eligible, women, [], waitlist, c)
    admitted = np.array(d.admit, dtype=int)
    if admitted.size == 0:
        return GateTrial(int(women.size), 0, d.forecast, float("nan"))
    j = join.copy()
    j[np.setdiff1d(np.flatnonzero(men), admitted)] = NEVER
    j[fakes] = NEVER
    report = CycleRun(pop, c, seed, sc.mutual, j).run("fake women")
    return GateTrial(int(women.size), int(admitted.size), d.forecast, report.guarantee_share)


# ---------------------------------------------------------------- the run with attackers


class AttackedRun(CycleRun):
    """A cycle run with attackers, through the run's hooks (test runs only).

    silent: members who open the app every day and never pick (spies, fake women); the first
    day and position each profile appears on their lists is kept. ring and target: women who
    pick the target whenever he is on their list, whom he always accepts, and who rate him 1.
    trolls: members who rate everyone they meet 1.
    """

    def __init__(
        self,
        pop: Population,
        cfg: Config,
        seed: int,
        scores: np.ndarray,
        join_day: np.ndarray | None = None,
        *,
        silent: Sequence[int] = (),
        ring: Sequence[int] = (),
        target: int | None = None,
        trolls: Sequence[int] = (),
        days: int | None = None,
    ):
        super().__init__(pop, cfg, seed, scores, join_day)
        self.silent = np.zeros(self.n, dtype=bool)
        self.silent[np.asarray(silent, dtype=int)] = True
        self.ring = {int(m) for m in ring}
        self.target = None if target is None else int(target)
        self.trolls = {int(m) for m in trolls}
        if days is not None:
            self.days = days
        self.list_days = np.zeros(self.n, dtype=int)
        self.picked = np.zeros(self.n, dtype=bool)
        self.first_seen: dict[int, dict[int, tuple[int, int]]] = defaultdict(dict)
        self.times_shown = np.zeros(self.n)  # times each member appeared on a list
        self.times_picked = np.zeros(self.n)  # times each member was picked

    def opens(self, today: int, opened: np.ndarray) -> np.ndarray:
        here = self.matchable & (self.join_day <= today) & ~self.paused
        return opened | (self.silent & here)

    def choose(self, i: int, todays: np.ndarray, today: int) -> list[int]:
        self.list_days[i] += 1
        self.times_shown[todays] += 1
        if self.silent[i]:
            for position, j in enumerate(todays):
                self.first_seen[i].setdefault(int(j), (today, position))
            return []
        if i in self.ring:
            chosen = [self.target] if np.any(todays == self.target) else []
            self.times_picked[chosen] += 1
            return chosen
        chosen = super().choose(i, todays, today)
        self.picked[i] |= bool(chosen)
        self.times_picked[chosen] += 1
        return chosen

    def accepts(self, chosen: int, chooser: int) -> bool:
        if chooser in self.ring and chosen == self.target:
            return True
        return super().accepts(chosen, chooser)

    def rate(self, rater: int, rated: int) -> Rating:
        r = super().rate(rater, rated)
        if rater in self.trolls or (rater in self.ring and rated == self.target):
            return Rating(rater, rated, 1, 1, 1, False)
        return r


def silent_shares(cfg: Config, seed: int, n_fake: int = 50, days: int = 28) -> tuple[float, float]:
    """(fake women, honest women): shares the silent rule stops counting after `days` days of
    the real flow, with the fakes opening the app every day and never picking (FR-LM-66)."""
    c = with_market(cfg, 300 + n_fake, 100)
    rate = c.cycle_runs.new_women_per_cycle
    pop = market(c, seed, rate)
    join = join_days(pop, c, rate)
    sc = build_scores(pop, c, seed)
    women = launch_women(pop, c, join)
    fakes = np.random.default_rng([seed, 77]).choice(women, n_fake, replace=False)
    run = AttackedRun(pop, c, seed, sc.mutual, join, silent=fakes, days=days)
    run.run("silent fakes")
    silent = (run.list_days >= c.gate.silent_after_days) & ~run.picked
    return float(silent[fakes].mean()), float(silent[np.setdiff1d(women, fakes)].mean())


# ---------------------------------------------------------------- 5. rating attacks


@dataclass(frozen=True)
class RatingTrial:
    wrongly_flagged: int  # honest members flagged
    flagged_bad: int  # bad actors flagged
    rings: dict[int, int]  # ring size -> targets paused
    targets: int  # targets tried per ring size


def rating_attacks(
    pop: Population,
    cfg: Config,
    seed: int,
    scores: np.ndarray,
    join: np.ndarray,
    trolls: int = 0,
    ring_sizes: Sequence[int] = (),
    targets: int = 8,
) -> RatingTrial:
    women = launch_women(pop, cfg, join)
    rng = np.random.default_rng([seed, 5])
    harsh = rng.choice(women, trolls, replace=False) if trolls else ()
    report = AttackedRun(pop, cfg, seed, scores, join, trolls=harsh).run("harsh raters")
    arr = MemberArrays.from_members(pop.members, cfg.values)
    eligible = eligibility_matrix(arr, cfg)
    paused = {}
    for size in ring_sizes:
        hits = 0
        for t in rng.choice(men_of(pop), targets, replace=False):
            near = women[eligible[women, t]]
            ring = rng.choice(near, min(size, near.size), replace=False)
            run = AttackedRun(pop, cfg, seed, scores, join, ring=ring, target=int(t))
            run.run("ring")
            hits += bool(run.paused[t])
        paused[size] = hits
    return RatingTrial(report.flagged - report.flagged_bad, report.flagged_bad, paused, targets)


# ---------------------------------------------------------------- 6. spies


def list_leak(
    pop: Population,
    cfg: Config,
    seed: int,
    scores: np.ndarray,
    join: np.ndarray,
    n_spies: int = 20,
    days: int = 14,
) -> float:
    """Spearman correlation between the ranking spies read from their lists (when, then where,
    each man first appeared, pooled over spies) and how much honest women like each man's
    looks (hidden truth)."""
    women = launch_women(pop, cfg, join)
    spies = np.random.default_rng([seed, 9]).choice(women, n_spies, replace=False)
    run = AttackedRun(pop, cfg, seed, scores, join, silent=spies, days=days)
    run.run("spies")
    honest = np.setdiff1d(women, spies)
    consensus = run.behavior.looks[honest].mean(axis=0)
    ranks: dict[int, list[float]] = defaultdict(list)
    for s in spies:
        seen = run.first_seen[int(s)]
        if len(seen) < 5:
            continue
        js = sorted(seen)
        order = np.array([seen[j][0] * 1000 + seen[j][1] for j in js], dtype=float)
        for j, r in zip(js, (rankdata(-order) - 0.5) / len(js), strict=True):
            ranks[j].append(float(r))
    js = sorted(ranks)
    guess = [np.mean(ranks[j]) for j in js]
    return float(spearmanr(guess, consensus[js])[0])


def no_cap(cfg: Config) -> Config:
    return dataclasses.replace(cfg, lists=dataclasses.replace(cfg.lists, exposure_multiple=1e6))


# ---------------------------------------------------------------- the report


def _mean(xs) -> str:
    return f"{np.nanmean(xs):.3f}"


def minilm_prompts(cfg: Config, seeds: Sequence[int], models_dir: str) -> None:
    """Attack 1 against the production text model (MiniLM, D20): the model files must already
    be in `models_dir` (`python -m lastfirstdate.encoders download`)."""
    models = dataclasses.replace(cfg.models, cache_dir=models_dir)
    personality = dataclasses.replace(cfg.personality, text_encoder="sentence")
    cfg = dataclasses.replace(cfg, models=models, personality=personality)
    rate = cfg.cycle_runs.new_women_per_cycle
    print("1. Stuffed prompts read by MiniLM (10 men): mean percentile in personality rankings")
    rows: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        pop = market(cfg, seed, rate)
        join = join_days(pop, cfg, rate)
        attackers = np.random.default_rng([seed, 1]).choice(men_of(pop), 10, replace=False)
        texts = {"honest": "", **stuffed_texts(pop, launch_women(pop, cfg, join))}
        for on in (False, True):
            for name, text in texts.items():
                p = with_prompts(pop, attackers, text) if text else pop
                rows[f"{name}, centring {'on' if on else 'off'}"].append(
                    personality_rank(p, centered(cfg, on), attackers)
                )
    for name, xs in rows.items():
        print(f"  {name:34s} {_mean(xs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic attackers against the engine.")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument(
        "--minilm", metavar="MODELS_DIR", help="only attack 1, read by MiniLM from MODELS_DIR"
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    cfg = load(PHASE1)
    if args.minilm:
        minilm_prompts(cfg, seeds, args.minilm)
        return
    rate = cfg.cycle_runs.new_women_per_cycle
    print(f"Abuse testing: Phase 1 zones (300 women, 100 men), seeds {', '.join(map(str, seeds))}")

    print("\n1. Stuffed prompts (10 men): mean percentile in women's personality rankings")
    rows: dict[str, list[float]] = defaultdict(list)
    exposure: dict[str, list[Exposure]] = defaultdict(list)
    for seed in seeds:
        pop = market(cfg, seed, rate)
        join = join_days(pop, cfg, rate)
        attackers = np.random.default_rng([seed, 1]).choice(men_of(pop), 10, replace=False)
        texts = stuffed_texts(pop, launch_women(pop, cfg, join))
        exposure["no attack"].append(stuffed_exposure(pop, cfg, seed, join, attackers))
        for on in (False, True):
            c = centered(cfg, on)
            rows[f"honest, centring {'on' if on else 'off'}"].append(
                personality_rank(pop, c, attackers)
            )
            for name, text in texts.items():
                stuffed = with_prompts(pop, attackers, text)
                rows[f"{name}, centring {'on' if on else 'off'}"].append(
                    personality_rank(stuffed, c, attackers)
                )
                if name == "everyone's words":
                    exposure[f"stuffed, centring {'on' if on else 'off'}"].append(
                        stuffed_exposure(stuffed, c, seed, join, attackers)
                    )
    for name, xs in rows.items():
        print(f"  {name:34s} {_mean(xs)}")
    print("  What it buys (everyone's words), attackers vs other men:")
    for name, xs in exposure.items():
        cells = []
        for field, label in (("top10", "in top 10 of"), ("shown", "shown"), ("picked", "picked")):
            a, o = zip(*(getattr(x, field) for x in xs), strict=True)
            fmt = "{:.0%}" if field == "top10" else "{:.1f}"
            cells.append(f"{label} {fmt.format(np.mean(a))} vs {fmt.format(np.mean(o))}")
        print(f"    {name:24s} " + "; ".join(cells))
    print("    (top 10: share of eligible women; shown and picked: times per cycle)")

    print("\n2-3. The starting taste: target's mean percentile in honest women's looks rankings")
    attacks = {
        "no attack": TasteAttack(0, 1),
        "1 account floods 20 exercises, no defence": TasteAttack(1, 20, one_exercise=False),
        "1 account floods 20 exercises, one exercise": TasteAttack(1, 20),
        "15 verified fakes (5%)": TasteAttack(15, 1),
        "30 verified fakes (10%)": TasteAttack(30, 1),
        "30 unverified fakes, verified-only fit": TasteAttack(30, 1, verified=False),
        "30 verified fakes, robust fit": TasteAttack(30, 1, robust=True),
        "no attack, robust fit": TasteAttack(0, 1, robust=True),
    }
    for name, attack in attacks.items():
        res = []
        for seed in seeds:
            pop = market(cfg, seed, rate)
            res.append(taste_attack(pop, cfg, seed, popular_target(pop, cfg, seed), attack))
        rank, acc = zip(*res, strict=True)
        print(f"  {name:44s} target {_mean(rank)}   honest looks AUC {_mean(acc)}")

    print("\n4. Fake women on the gate (300 women, 450 men waiting)")
    for n_fake in (0, 50):
        trials = [gate_with_fakes(cfg, seed, 300, n_fake) for seed in seeds]
        print(
            f"  {n_fake:3d} fakes: counted {np.mean([t.counted for t in trials]):.0f}, "
            f"sold {np.mean([t.admitted for t in trials]):.1f} places, forecast "
            f"{np.mean([t.forecast for t in trials]):.1%}, actual "
            f"{np.nanmean([t.actual for t in trials]):.1%}"
        )
    shares = [silent_shares(cfg, seed) for seed in seeds]
    fake, honest = zip(*shares, strict=True)
    print(
        f"  after 4 weeks of lists the silent rule stops counting {np.mean(fake):.0%} of 50 "
        f"fakes and {np.mean(honest):.1%} of honest women"
    )

    print("\n5. Rating attacks (three cycles)")
    for seed in seeds:
        pop = market(cfg, seed, rate)
        join = join_days(pop, cfg, rate)
        sc = build_scores(pop, cfg, seed)
        base = rating_attacks(pop, cfg, seed, sc.mutual, join)
        trial = rating_attacks(pop, cfg, seed, sc.mutual, join, 15, (3, 6))
        rings = ", ".join(f"ring of {k}: {v} of {trial.targets}" for k, v in trial.rings.items())
        print(
            f"  seed {seed}: honest men wrongly flagged {base.wrongly_flagged} without attack, "
            f"{trial.wrongly_flagged} with 15 harsh raters; targets paused: {rings}"
        )

    print("\n6. Spies (20 accounts, 2 weeks): ranking read from lists vs consensus looks")
    leak = defaultdict(list)
    for seed in seeds:
        pop = market(cfg, seed, rate)
        join = join_days(pop, cfg, rate)
        for name, c in (("exposure cap on", cfg), ("exposure cap off", no_cap(cfg))):
            sc = build_scores(pop, c, seed)
            leak[name].append(list_leak(pop, c, seed, sc.mutual, join))
    for name, xs in leak.items():
        print(f"  {name:20s} Spearman {_mean(xs)}")


if __name__ == "__main__":
    main()
