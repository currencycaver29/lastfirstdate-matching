import pytest

from lastfirstdate.config import load
from lastfirstdate.synthetic import generate


@pytest.fixture(scope="session")
def cfg():
    return load()


@pytest.fixture(scope="session")
def small_pop(cfg):
    return generate(cfg, seed=11, n=400)
