"""The production photo and text models (FR-LM-59 to FR-LM-62, decision D20).

- Photos: Google SigLIP base. `PhotoModel.embed` turns one photo into 768 numbers of length 1;
  the app stores them, encrypted, in `photo_embeddings` with `photo_model_id`.
- Prompts: MiniLM (all-MiniLM-L6-v2). `SentenceEncoder` is a `TextEncoder`: with
  `personality.text_encoder = "sentence"` it replaces the offline word counts, so "rock
  climbing" and "bouldering" count as close.

Both run with ONNX Runtime on a normal CPU. Every file is pinned by revision and sha256 in
`[models]`; `python -m lastfirstdate.encoders download` fetches them once and refuses anything
that doesn't match. The model libraries are optional (`pip install -e ".[models]"`) and are
imported only when a model is loaded, so the tests run offline with stand-ins (R4).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from lastfirstdate.config import ModelsConfig
from lastfirstdate.models import Member

HUB = "https://huggingface.co"
SIGLIP_MEAN, SIGLIP_STD = 0.5, 0.5  # SigLIP scales each colour from 0..1 to -1..1


class ModelError(RuntimeError):
    """A model file is missing, doesn't match its fingerprint, or can't be loaded."""


# ---------------------------------------------------------------- pinned files (FR-LM-59)


@dataclass(frozen=True)
class ModelFile:
    repo: str
    revision: str
    path: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{HUB}/{self.repo}/resolve/{self.revision}/{self.path}"


def model_files(m: ModelsConfig) -> dict[str, ModelFile]:
    return {
        "photo": ModelFile(m.photo_repo, m.photo_revision, m.photo_file, m.photo_sha256),
        "text": ModelFile(m.text_model, m.text_revision, m.text_file, m.text_sha256),
        "tokenizer": ModelFile(m.text_model, m.text_revision, "tokenizer.json", m.tokenizer_sha256),
    }


def local_path(m: ModelsConfig, f: ModelFile) -> Path:
    root = Path(m.cache_dir).expanduser()
    return root / f.repo.replace("/", "--") / f.revision / f.path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)


def download(m: ModelsConfig, fetch: Callable[[str, Path], None] = _fetch) -> dict[str, Path]:
    """Fetch every pinned file into `m.cache_dir`, checking its sha256. A file already in place
    with the right fingerprint is kept; a download that doesn't match is refused and removed."""
    out = {}
    for name, f in model_files(m).items():
        dest = local_path(m, f)
        if not (dest.exists() and sha256_of(dest) == f.sha256):
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
                part = Path(tmp) / dest.name
                fetch(f.url, part)
                got = sha256_of(part)
                if got != f.sha256:
                    raise ModelError(f"{f.repo}/{f.path}: sha256 {got} doesn't match {f.sha256}")
                part.replace(dest)
        out[name] = dest
    return out


def _require(m: ModelsConfig, name: str) -> Path:
    path = local_path(m, model_files(m)[name])
    if not path.exists():
        raise ModelError(f"{path} is missing: run python -m lastfirstdate.encoders download")
    return path


def _session(path: Path):
    try:
        import onnxruntime
    except ImportError as exc:
        raise ModelError('install the models: pip install -e ".[models]"') from exc
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- photos (FR-LM-60, FR-LM-62)


def photo_model_id(m: ModelsConfig) -> str:
    """What the app stores in `photo_embeddings.model`: the model and its pinned revision."""
    return f"{m.photo_model}@{m.photo_revision[:12]}"


class PhotoModel:
    """SigLIP's vision model: one photo in, its pooled embedding (length 1) out."""

    def __init__(self, session, size: int):
        self.session, self.size = session, size

    @classmethod
    def from_settings(cls, m: ModelsConfig) -> PhotoModel:
        return cls(_session(_require(m, "photo")), m.photo_size)

    def pixels(self, image) -> np.ndarray:
        """(1, 3, size, size) float32: upright, RGB, resized (bicubic), each colour scaled to
        -1..1. `image` is a path, raw bytes, a PIL image, or an (h, w, 3) uint8 array.

        Phones save many portrait photos sideways with an EXIF orientation tag; the photo is
        turned upright from the tag first (FR-LM-72), as any photo viewer would show it.
        """
        from io import BytesIO

        from PIL import Image, ImageOps

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        image = ImageOps.exif_transpose(image)
        rgb = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BICUBIC)
        x = np.asarray(rgb, dtype=np.float32) / 255.0
        x = (x - SIGLIP_MEAN) / SIGLIP_STD
        return x.transpose(2, 0, 1)[None]

    def embed(self, image) -> np.ndarray:
        pooled = self.session.run(["pooler_output"], {"pixel_values": self.pixels(image)})[0][0]
        return (pooled / np.linalg.norm(pooled)).astype(np.float32)


# ---------------------------------------------------------------- prompts (FR-LM-61)


class SentenceEncoder:
    """MiniLM sentence embeddings as a `TextEncoder`: the average of the model's token vectors
    over the real (unpadded) tokens, scaled to length 1, in 64-bit floats like the other
    encoders (so scores stay within 0..1).

    Keeps the vectors of the texts it was last asked about (FR-LM-78): unchanged prompts aren't
    re-read night to night, and edited or forgotten prompts are dropped at the next call, so
    memory doesn't grow and old text isn't kept."""

    def __init__(self, session, tokenizer, max_tokens: int, batch: int = 32):
        self.session, self.tokenizer, self.batch = session, tokenizer, batch
        tokenizer.enable_truncation(max_length=max_tokens)
        tokenizer.enable_padding()
        self.seen: dict[str, np.ndarray] = {}
        self.dim = 0  # vector length, once known

    @classmethod
    def from_settings(cls, m: ModelsConfig) -> SentenceEncoder:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ModelError('install the models: pip install -e ".[models]"') from exc
        tokenizer = Tokenizer.from_file(str(_require(m, "tokenizer")))
        return cls(_session(_require(m, "text")), tokenizer, m.text_max_tokens)

    def _run(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        feeds = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        tokens = self.session.run(["last_hidden_state"], feeds)[0].astype(np.float64)
        mask = feeds["attention_mask"][..., None].astype(np.float64)
        mean = (tokens * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        return mean / np.linalg.norm(mean, axis=1, keepdims=True)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        wanted = {t for t in texts if t.strip()}
        kept = {t: v for t, v in self.seen.items() if t in wanted}
        new = sorted(wanted - kept.keys())
        for k in range(0, len(new), self.batch):
            chunk = new[k : k + self.batch]
            kept.update(zip(chunk, self._run(chunk), strict=True))
        self.seen = kept
        if kept:
            self.dim = next(iter(kept.values())).shape[0]
        return np.array([kept[t] if t.strip() else np.zeros(self.dim) for t in texts]).reshape(
            len(texts), self.dim
        )

    def encode(self, members: Sequence[Member]) -> np.ndarray:
        return self.embed_texts([m.bio for m in members])


@lru_cache(maxsize=2)
def sentence_encoder(m: ModelsConfig) -> SentenceEncoder:
    """One encoder per settings, kept between nightly runs so unchanged prompts aren't re-read."""
    return SentenceEncoder.from_settings(m)


def main(argv: list[str] | None = None) -> None:
    from lastfirstdate.config import load

    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] != "download":
        raise SystemExit("usage: python -m lastfirstdate.encoders download [settings.toml]")
    m = load(args[1] if len(args) > 1 else "config/phase1.toml").models
    for name, path in download(m).items():
        print(f"{name:9s} {path} (sha256 checked)")


if __name__ == "__main__":
    main()
