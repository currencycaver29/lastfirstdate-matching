import pytest

from lastfirstdate.config import load
from lastfirstdate.mutual_report import build_scores
from lastfirstdate.synthetic import generate


@pytest.fixture(scope="session")
def cfg():
    return load()


@pytest.fixture(scope="session")
def small_pop(cfg):
    return generate(cfg, seed=11, n=400)


@pytest.fixture(scope="session")
def city600(cfg):
    """A 600-member city with every member calibrated and all Stage 1 to 5 scores."""
    pop = generate(cfg, n=600)
    return pop, build_scores(pop, cfg, cfg.simulation.seed)


@pytest.fixture(scope="session")
def default_city(cfg):
    """The default 1,000-member city (seed 7) with all Stage 1 to 5 scores. About a minute."""
    pop = generate(cfg)
    return pop, build_scores(pop, cfg, cfg.simulation.seed)
