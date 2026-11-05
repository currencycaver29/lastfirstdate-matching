"""FR-LM-59 to FR-LM-62: the real photo and text models, tested offline with stand-ins (R4)."""

import dataclasses
import hashlib
import sqlite3
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from lastfirstdate import app, encoders, personality
from lastfirstdate.app_fill import fill_database
from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import ConfigError, from_dict, load
from lastfirstdate.encoders import ModelError, PhotoModel, SentenceEncoder, photo_model_id
from lastfirstdate.synthetic import generate

TODAY, EPOCH = date(2026, 11, 4), date(2026, 11, 2)


# ---------------------------------------------------------------- stand-ins for the models


class FakePhotoSession:
    """Returns a fixed pooled vector and remembers the pixels it was given."""

    def __init__(self, pooled):
        self.pooled, self.seen = np.asarray(pooled, dtype=np.float32), []

    def run(self, names, feeds):
        assert names == ["pooler_output"]
        self.seen.append(feeds["pixel_values"])
        return [self.pooled[None]]


class FakeTokenizer:
    """Splits on spaces; pads to the longest text in the batch, as `tokenizers` does."""

    def __init__(self):
        self.max_length, self.padding = None, False

    def enable_truncation(self, max_length):
        self.max_length = max_length

    def enable_padding(self):
        self.padding = True

    def encode_batch(self, texts):
        ids = [[1 + (sum(map(ord, w)) % 97) for w in t.split()][: self.max_length] for t in texts]
        width = max(map(len, ids))
        return [
            SimpleNamespace(
                ids=i + [0] * (width - len(i)),
                attention_mask=[1] * len(i) + [0] * (width - len(i)),
                type_ids=[0] * width,
            )
            for i in ids
        ]


class FakeTextSession:
    """A vector per token id; padding positions return huge values, which must be ignored."""

    def __init__(self, dim=8):
        self.table = np.random.default_rng(0).normal(size=(100, dim)).astype(np.float32)
        self.calls = 0

    def run(self, names, feeds):
        assert names == ["last_hidden_state"]
        self.calls += 1
        out = self.table[feeds["input_ids"]]
        out[feeds["attention_mask"] == 0] = 1e6
        return [out]


def sentence() -> SentenceEncoder:
    return SentenceEncoder(FakeTextSession(), FakeTokenizer(), max_tokens=16)


# ---------------------------------------------------------------- FR-LM-59 pinned files


def test_a_file_with_the_wrong_fingerprint_is_refused(tmp_path):
    """FR-LM-59: a download whose sha256 differs from the settings is refused and not kept."""
    m = dataclasses.replace(load("config/phase1.toml").models, cache_dir=str(tmp_path))

    def fetch(url, dest):
        dest.write_bytes(b"not the model")

    with pytest.raises(ModelError, match="doesn't match"):
        encoders.download(m, fetch=fetch)
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


def test_files_already_in_place_are_not_fetched_again(tmp_path):
    """FR-LM-59: matching files are kept; only missing ones are fetched, each to its pinned
    path (repository, revision, file)."""
    body = b"pretend model"
    h = hashlib.sha256(body).hexdigest()
    m = dataclasses.replace(
        load("config/phase1.toml").models,
        cache_dir=str(tmp_path),
        photo_sha256=h,
        text_sha256=h,
        tokenizer_sha256=h,
    )
    urls = []

    def fetch(url, dest):
        urls.append(url)
        dest.write_bytes(body)

    paths = encoders.download(m, fetch=fetch)
    assert len(urls) == 3 and all(m.photo_revision in u or m.text_revision in u for u in urls)
    assert paths["photo"].read_bytes() == body
    assert m.photo_revision in str(paths["photo"]) and paths["photo"].name == "vision_model.onnx"
    encoders.download(m, fetch=fetch)
    assert len(urls) == 3


def test_model_settings_must_be_pinned():
    """FR-LM-59: revisions must be full commit hashes and fingerprints full sha256 values."""
    import tomllib

    with open("config/phase1.toml", "rb") as fh:
        raw = tomllib.load(fh)
    for key, bad in (("photo_revision", "main"), ("text_sha256", "abc")):
        with pytest.raises(ConfigError):
            from_dict({**raw, "models": {**raw["models"], key: bad}})


# ---------------------------------------------------------------- FR-LM-60 photos


def test_photos_follow_the_siglip_recipe(tmp_path):
    """FR-LM-60: the photo is resized to the model's size, white becomes +1 and black -1, the
    output has length 1, and the same photo (as an array, a file or bytes) gives the same
    vector."""
    session = FakePhotoSession([3.0, 4.0])
    model = PhotoModel(session, size=8)
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[:, 15:] = 255  # left half black, right half white
    v = model.embed(img)
    np.testing.assert_allclose(v, [0.6, 0.8], rtol=1e-6)
    px = session.seen[0]
    assert px.shape == (1, 3, 8, 8) and px.dtype == np.float32
    assert px[0, :, :, 0].max() == pytest.approx(-1.0) and px[0, :, :, -1].min() == pytest.approx(
        1.0
    )
    path = tmp_path / "photo.png"
    Image.fromarray(img).save(path)
    np.testing.assert_array_equal(model.embed(path), v)
    np.testing.assert_array_equal(model.embed(path.read_bytes()), v)
    np.testing.assert_array_equal(session.seen[1], session.seen[0])


# ---------------------------------------------------------------- FR-LM-61 prompts


def test_padding_never_changes_a_texts_vector():
    """FR-LM-61: a text's vector is the same alone or batched with a longer text (the padding
    is left out of the average); rows have length 1; an empty text gives zeros."""
    short, long = "rock climbing", "rock climbing on weekends and long trail runs in the park"
    alone = sentence().embed_texts([short])[0]
    both = sentence().embed_texts([short, long, ""])
    np.testing.assert_allclose(both[0], alone, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(both[:2], axis=1), 1.0, rtol=1e-6)
    assert not both[2].any()


def test_texts_already_read_are_remembered():
    """FR-LM-61: the nightly run doesn't re-read prompts that haven't changed."""
    enc = sentence()
    enc.embed_texts(["a b", "c d"])
    enc.embed_texts(["c d", "a b"])
    assert enc.session.calls == 1


def test_the_personality_score_uses_the_sentence_model(monkeypatch):
    """FR-LM-61: with text_encoder = "sentence", the personality score reads prompts through
    the model; with "words" it is exactly as before."""
    cfg = load("config/phase1.toml")
    pop = generate(cfg, seed=7)
    arr = MemberArrays.from_members(pop.members, cfg.values)
    before = personality.personality_matrix(pop.members, arr, cfg)
    words = dataclasses.replace(cfg.personality, text_encoder="words")
    np.testing.assert_array_equal(
        personality.personality_matrix(
            pop.members, arr, dataclasses.replace(cfg, personality=words)
        ),
        before,
    )
    fake = sentence()
    monkeypatch.setattr(encoders, "sentence_encoder", lambda m: fake)
    sent = dataclasses.replace(cfg, personality=dataclasses.replace(words, text_encoder="sentence"))
    p = personality.personality_matrix(pop.members, arr, sent)
    assert fake.session.calls > 0 and p.shape == before.shape
    assert not np.allclose(p, before) and np.allclose(p, p.T)
    assert p.min() >= 0 and p.max() <= 1


@pytest.fixture(scope="module")
def zone():
    cfg = load("config/phase1.toml")
    pop = generate(cfg, seed=7)
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    fill_database(con, pop, cfg, TODAY, 7, subscribers=60)
    return SimpleNamespace(cfg=cfg, pop=pop, con=con)


def test_the_engine_on_the_database_accepts_the_sentence_model(zone, monkeypatch):
    """FR-LM-61: the app's nightly refresh runs with text_encoder = "sentence"."""
    fake = sentence()
    monkeypatch.setattr(encoders, "sentence_encoder", lambda m: fake)
    cfg = dataclasses.replace(
        zone.cfg, personality=dataclasses.replace(zone.cfg.personality, text_encoder="sentence")
    )
    scores = app.MarketCache(cfg, app.Calendar(EPOCH)).refresh(
        zone.con, TODAY, zone.pop.photo_embeddings
    )
    assert fake.session.calls > 0 and scores.personality.shape[0] == len(zone.pop)


# ---------------------------------------------------------------- FR-LM-62 one photo model


def test_embeddings_from_another_model_stop_the_refresh(zone):
    """FR-LM-62: an embedding made by a different model than the settings name stops the
    nightly refresh with a clear error; the right model passes."""
    con = sqlite3.connect(":memory:")
    zone.con.backup(con)
    want = photo_model_id(zone.cfg.models)
    con.execute(
        "INSERT INTO photo_embeddings VALUES (1000, ?, X'00', '2026-11-04T10:00:00Z')", (want,)
    )
    app.check_photo_models(con, zone.cfg)
    con.execute(
        "INSERT INTO photo_embeddings VALUES (1001, 'openai/clip-vit-base-patch32', X'00', "
        "'2026-11-04T10:00:00Z')"
    )
    with pytest.raises(app.HandoffError, match="openai/clip"):
        app.MarketCache(zone.cfg, app.Calendar(EPOCH)).refresh(
            con, TODAY, zone.pop.photo_embeddings
        )
