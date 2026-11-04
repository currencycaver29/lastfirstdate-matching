"""Synthetic member population (FR-LM-03).

Generates fake New York members so every stage can be built and tested without
real data (rule R3). The same seed always gives the same population (rule R7).

The generator also draws hidden "ground truth" for each fake member: their true
visual taste, how much they truly care about looks versus personality, and how
picky they are. The simulator uses it to decide swipes, picks and how dates go.
The matching engine must never read it: it exists only so tests can check
whether the engine learned what is really there.

Run `python -m lastfirstdate.synthetic` to print a summary of a population.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from lastfirstdate.config import Config, ValuesQuestion, load
from lastfirstdate.models import N_SLOTS, Member

# Fake photo embedding shape. Lengths are relative to a look-type center of length 1:
# photos of one type sit at cosine about 0.5 to each other and near 0 to other types,
# roughly like a real image model.
PHOTO_NOISE = 0.9  # length of each photo's random part
APPEAL_SD = 0.35  # spread along the direction most people like (real tastes overlap)
AGE_SHIFT = 0.3  # how far 12 years of age moves a photo
TASTE_NOISE = 0.7  # how far a member's taste sits from the look type they prefer

# Interest tags grouped by lifestyle. Members draw most tags from one group.
INTEREST_GROUPS: dict[str, tuple[str, ...]] = {
    "outdoors": (
        "hiking",
        "trail running",
        "climbing",
        "cycling",
        "camping",
        "kayaking",
        "skiing",
        "dogs",
    ),
    "arts": ("museums", "theater", "jazz", "galleries", "photography", "poetry", "film", "opera"),
    "food": (
        "cooking",
        "wine",
        "coffee",
        "brunch",
        "baking",
        "street food",
        "cocktails",
        "farmers markets",
    ),
    "nightlife": (
        "live music",
        "dancing",
        "techno",
        "karaoke",
        "comedy shows",
        "rooftop bars",
        "festivals",
        "djing",
    ),
    "fitness": (
        "running",
        "yoga",
        "pilates",
        "boxing",
        "crossfit",
        "tennis",
        "swimming",
        "basketball",
    ),
    "quiet": (
        "reading",
        "board games",
        "chess",
        "podcasts",
        "writing",
        "crosswords",
        "gardening",
        "meditation",
    ),
    "tech": (
        "startups",
        "coding",
        "gaming",
        "sci-fi",
        "investing",
        "ai",
        "crypto",
        "product design",
    ),
    "travel": (
        "travel",
        "languages",
        "road trips",
        "backpacking",
        "history",
        "architecture",
        "beaches",
        "volunteering",
    ),
}

BIO_TEMPLATES = (
    "I'm into {a}, {b} and {c}.",
    "Weekends are for {a} and {b}. Ask me about {c}.",
    "Big fan of {a}. Lately it's been {b} and {c}.",
    "{a}, {b}, {c}. In that order.",
    "Happiest when there's {a} involved, or {b}. Also {c}.",
)


def slot_weights() -> np.ndarray:
    """Relative chance of each weekly slot being free: evenings and weekends more often."""
    weights = np.empty(N_SLOTS)
    for day in range(7):
        weekend = day >= 4  # fri, sat, sun
        lunch, evening, late = (1.0, 1.3, 0.8) if weekend else (0.5, 1.0, 0.4)
        weights[day * 3 : day * 3 + 3] = (lunch, evening, late)
    return weights / weights.sum()


@dataclass(frozen=True)
class GroundTruth:
    """Hidden traits of fake members. Simulation only: the engine never reads these."""

    shared_taste: np.ndarray  # (d,) looks most people like; real tastes overlap like this
    taste: np.ndarray  # (n, d) each member's true visual taste, unit vectors
    looks_weight: np.ndarray  # (n,) true weight on looks versus personality, 0 to 1
    pickiness: np.ndarray  # (n,) shifts how often they like or accept, 0 = average
    look_type: np.ndarray  # (n,) which cluster each member's photos fall in
    lifestyle: np.ndarray  # (n,) index into INTEREST_GROUPS for their main interests


@dataclass(frozen=True)
class Population:
    members: list[Member]
    photo_embeddings: np.ndarray  # (n, d) unit vectors, a stand-in for a real image model
    truth: GroundTruth
    seed: int

    def __len__(self) -> int:
        return len(self.members)


def _seeking(rng: np.random.Generator, gender: str, orientation: dict[str, float]) -> frozenset:
    other = "woman" if gender == "man" else "man"
    kind = rng.choice(list(orientation), p=list(orientation.values()))
    if kind == "straight":
        return frozenset({other})
    if kind == "gay":
        return frozenset({gender})
    return frozenset({"man", "woman"})


def _values(
    rng: np.random.Generator,
    questions: tuple[ValuesQuestion, ...],
    answer_probs: list[np.ndarray],
    dealbreaker_rate: float,
) -> tuple[dict[str, str], dict[str, frozenset[str]], frozenset[str]]:
    answers: dict[str, str] = {}
    accepts: dict[str, frozenset[str]] = {}
    dealbreakers: set[str] = set()
    for q, probs in zip(questions, answer_probs, strict=True):
        own = str(rng.choice(q.options, p=probs))
        answers[q.key] = own
        if q.mode == "closeness":
            # one answer only; compatibility comes from closeness, dealbreakers from the
            # question's shared incompatible pairs
            accepts[q.key] = frozenset(q.options)
            continue
        if rng.random() < dealbreaker_rate:
            # A dealbreaker still accepts the options next to their own (options are
            # listed in order in the config) half the time, like real people do.
            dealbreakers.add(q.key)
            pos = q.options.index(own)
            neighbors = [q.options[p] for p in (pos - 1, pos + 1) if 0 <= p < len(q.options)]
            ok = {own} | {o for o in neighbors if rng.random() < 0.5}
            if "open" in q.options and own != "dont_want":
                ok.add("open")
        else:
            ok = {own} | {o for o in q.options if o != own and rng.random() < 0.6}
            if "open" in q.options and rng.random() < 0.8:
                ok.add("open")
        accepts[q.key] = frozenset(ok)
    return answers, accepts, frozenset(dealbreakers)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norm == 0, 1.0, norm)


def generate(
    cfg: Config | None = None, seed: int | None = None, n: int | None = None
) -> Population:
    """Generate a fake member population.

    Defaults come from config/lastfirstdate.toml; `seed` and `n` override them.
    """
    cfg = cfg or load()
    pop = cfg.population
    seed = cfg.simulation.seed if seed is None else seed
    n = pop.n_members if n is None else n
    rng = np.random.default_rng(seed)
    d, k_types, n_groups = pop.embedding_dim, pop.n_look_types, pop.n_audit_groups

    # Population-level draws first, so the per-member stream stays aligned.
    areas = pop.areas
    area_p = np.array([a.share for a in areas])
    area_p = area_p / area_p.sum()
    answer_probs = [rng.dirichlet(np.full(len(q.options), 2.0)) for q in cfg.values]
    type_centers = _unit(rng.normal(size=(k_types, d)))
    age_direction = _unit(rng.normal(size=d))
    shared_taste = _unit(rng.normal(size=d))
    # Each fake demographic group leans toward two look types, so photo taste can
    # silently become a group filter, which the fairness audit must be able to catch.
    group_type_p = np.ones((n_groups, k_types))
    for g in range(n_groups):
        group_type_p[g, (2 * g) % k_types] += 4.0
        group_type_p[g, (2 * g + 1) % k_types] += 4.0
    group_type_p /= group_type_p.sum(axis=1, keepdims=True)
    slot_p = slot_weights()
    group_names = list(INTEREST_GROUPS)
    all_tags = [t for tags in INTEREST_GROUPS.values() for t in tags]

    if pop.exact_counts:
        n_men = round(n * pop.share_men)
        exact_genders = rng.permutation(["man"] * n_men + ["woman"] * (n - n_men))
    members: list[Member] = []
    embeddings = np.empty((n, d))
    taste = np.empty((n, d))
    looks_weight = np.empty(n)
    pickiness = np.empty(n)
    look_type = np.empty(n, dtype=int)
    lifestyle = np.empty(n, dtype=int)

    for i in range(n):
        if pop.exact_counts:
            gender = str(exact_genders[i])
        else:
            gender = "man" if rng.random() < pop.share_men else "woman"
        seeking = _seeking(rng, gender, pop.orientation)

        span = pop.age_max - pop.age_min
        age = pop.age_min + round(rng.beta(2.0, 3.0) * span)
        younger = int(rng.integers(2, 9)) + (2 if gender == "man" else 0)
        older = int(rng.integers(2, 9)) + (2 if gender == "woman" else 0)
        age_min_pref = max(pop.age_min, age - younger)
        age_max_pref = min(pop.age_max + 10, age + older)

        area = rng.choice(len(areas), p=area_p)
        lat = areas[area].lat + rng.normal(0.0, pop.area_jitter_deg)
        lon = areas[area].lon + rng.normal(0.0, pop.area_jitter_deg)
        max_km = float(rng.uniform(pop.max_km_min, pop.max_km_max))

        verified = bool(rng.random() < pop.verified_share)
        if rng.random() < pop.active_share:
            last_active = int(min(13, rng.exponential(3.0)))
        else:
            last_active = int(rng.integers(15, 91))
        paying = bool(rng.random() < pop.paying_share[gender])

        n_slots = int(rng.integers(pop.slots_min, pop.slots_max + 1))
        free_slots = frozenset(
            int(s) for s in rng.choice(N_SLOTS, n_slots, replace=False, p=slot_p)
        )

        life = int(rng.integers(len(group_names)))
        n_tags = int(rng.integers(pop.interests_min, pop.interests_max + 1))
        main_pool = list(INTEREST_GROUPS[group_names[life]])
        tags: list[str] = []
        while len(tags) < n_tags:
            pool = main_pool if rng.random() < 0.7 else all_tags
            tag = str(rng.choice(pool))
            if tag not in tags:
                tags.append(tag)
        if pop.prompt_style == "prompts":
            # the lastfirstdate app's free-text prompts, filled from five interests
            t1, t2, t3, t4, t5 = (tags + tags)[:5]
            bio = (
                f"A perfect Sunday looks like… {t1} and {t2}. "
                f"I'm looking for… someone into {t3}, {t4} or {t5}."
            )
        else:
            a, b, c = tags[:3]
            bio = str(rng.choice(BIO_TEMPLATES)).format(a=a, b=b, c=c)
            bio = bio[0].upper() + bio[1:]

        values, accepts, dealbreakers = _values(rng, cfg.values, answer_probs, pop.dealbreaker_rate)
        big5 = tuple(float(x) for x in rng.beta(2.5, 2.5, size=5))

        group = int(rng.integers(n_groups))
        ltype = int(rng.choice(k_types, p=group_type_p[group]))
        emb = type_centers[ltype] + rng.normal(0.0, PHOTO_NOISE / np.sqrt(d), size=d)
        emb = emb + shared_taste * rng.normal(0.0, APPEAL_SD)
        emb = emb + age_direction * (age - 33) / 12 * AGE_SHIFT
        embeddings[i] = _unit(emb)

        preferred = int(rng.integers(k_types))
        personal = _unit(
            type_centers[preferred] + rng.normal(0.0, TASTE_NOISE / np.sqrt(d), size=d)
        )
        rho = rng.uniform(0.2, 0.6)
        taste[i] = _unit(rho * shared_taste + (1 - rho) * personal)
        looks_weight[i] = rng.beta(5.0, 5.0)
        pickiness[i] = rng.normal(0.0, 0.5)
        look_type[i] = ltype
        lifestyle[i] = life

        members.append(
            Member(
                id=i,
                gender=gender,
                seeking=seeking,
                age=age,
                age_min_pref=age_min_pref,
                age_max_pref=age_max_pref,
                lat=float(lat),
                lon=float(lon),
                max_km=max_km,
                verified=verified,
                last_active_days=last_active,
                paying=paying,
                free_slots=free_slots,
                interests=tuple(tags),
                bio=bio,
                values=values,
                accepts=accepts,
                dealbreakers=dealbreakers,
                big5=big5,  # type: ignore[arg-type]
                audit_group=f"G{group + 1}",
            )
        )

    truth = GroundTruth(
        shared_taste=shared_taste,
        taste=taste,
        looks_weight=looks_weight,
        pickiness=pickiness,
        look_type=look_type,
        lifestyle=lifestyle,
    )
    return Population(members=members, photo_embeddings=embeddings, truth=truth, seed=seed)


def summary(pop: Population, cfg: Config | None = None) -> str:
    """A short plain-text description of a population."""
    cfg = cfg or load()
    active_days = cfg.filter.active_within_days
    m = pop.members
    n = len(m)
    men = sum(x.gender == "man" for x in m)
    paying_men = sum(x.paying and x.gender == "man" for x in m)
    paying_women = sum(x.paying and x.gender == "woman" for x in m)
    straight = sum(len(x.seeking) == 1 and x.gender not in x.seeking for x in m)
    bi = sum(len(x.seeking) == 2 for x in m)
    verified = sum(x.verified for x in m)
    active = sum(x.last_active_days <= active_days for x in m)
    ages = np.array([x.age for x in m])
    slots = np.array([len(x.free_slots) for x in m])
    lines = [
        f"Synthetic population, seed {pop.seed}: {n} members",
        f"  men {men}, women {n - men}",
        f"  paying: {paying_men} men, {paying_women} women",
        f"  straight {straight}, gay or lesbian {n - straight - bi}, bi {bi}",
        f"  verified {verified}, active in last {active_days} days {active}",
        f"  age median {int(np.median(ages))} (range {ages.min()}-{ages.max()})",
        f"  free weekly slots median {int(np.median(slots))}",
        f"  photo embedding size {pop.photo_embeddings.shape[1]}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a summary of a synthetic population.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    print(summary(generate(seed=args.seed, n=args.n)))


if __name__ == "__main__":
    main()
