"""Tests for ingestion/image_transcriber.py — Milestone 2 covers the local-path (corpus)
branch only. The invariant under test everywhere: exactly one ImageTranscription per
article.images entry, in order, never shorter.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest
from PIL import Image

import ingestion.image_transcriber as it
from models import Article, ImageRef


def _png(seed: int = 1, size=(220, 180)) -> bytes:
    b = io.BytesIO()
    Image.effect_noise(size, 100 + seed).convert("RGB").save(b, "PNG")
    return b.getvalue()


TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"  # signature + a real 1x1 png
    + bytes.fromhex(
        "0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c6300010000050001a5f645400000000049454e44ae426082"
    )
)


class FakeVision:
    """Answers the type-detection prompt with a fixed type and the transcription prompt
    with a long body. Counts calls so cache hits are observable."""

    model_name = "mock-vision"
    supports_vision = True

    def __init__(self, image_type="chart"):
        self.image_type = image_type
        self.transcribe_calls = 0
        self.detect_calls = 0

    def complete_with_image(self, image_bytes, media_type, text_prompt, max_tokens=400):
        from llm_provider import TextResponse

        if "one word" in text_prompt:
            self.detect_calls += 1
            return TextResponse(self.image_type, self.model_name, 10, 1)
        self.transcribe_calls += 1
        text = (
            "Line chart. X axis years 2010 to 2024. Y axis trillions of USD. Money supply "
            "grows steadily then surges after 2020 before a decline through 2023."
        )
        return TextResponse(text, self.model_name, 1500, 200)


@pytest.fixture
def vision(monkeypatch):
    v = FakeVision()
    monkeypatch.setattr(it, "get_vision_provider", lambda: v)
    return v


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "IMAGE_CACHE_DB", str(tmp_path / "cache.db"))
    monkeypatch.setattr(it, "IMAGE_CACHE_DIR", str(tmp_path / "img_cache"))


def _article(images: list[ImageRef], url="https://example.com/a") -> Article:
    return Article(
        url=url,
        title="Article A",
        author="W",
        published_at=datetime(2021, 6, 27, tzinfo=UTC),
        fetched_at=datetime.now(UTC),
        tags=[],
        body_text="body",
        tables_md=[],
        images=images,
        word_count=1,
        content_hash="h" * 64,
        is_stub=False,
        source="corpus",
        source_path="/c/a.md",
    )


def _ref(path, *, position=0, local=True, paywall=False) -> ImageRef:
    return ImageRef(
        src=str(path) if not local else f"images/x/{position:02d}.png",
        local_path=str(path) if local else None,
        alt="a chart",
        caption="M2 money supply",
        position=position,
        is_paywall=paywall,
    )


def _write(tmp_path, name, data) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


# ── the invariant ──────────────────────────────────────────────────────────


async def test_one_entry_per_image_in_order(tmp_path, vision):
    refs = [
        _ref(_write(tmp_path, "0.png", _png(1)), position=0),
        _ref(_write(tmp_path, "1.png", _png(2)), position=1),
        _ref("/gone/2.png", position=2),  # missing local file
    ]
    result = await it.transcribe_images(_article(refs))
    assert len(result) == 3
    assert [r.image_ref.position for r in result] == [0, 1, 2]
    assert result[2].skipped and result[2].skip_reason == "unavailable"
    assert not result[0].skipped and result[0].transcription


async def test_local_read_makes_no_http_and_calls_vision(tmp_path, vision, monkeypatch):
    # there is no httpx import in the module; assert the vision provider is what ran
    ref = _ref(_write(tmp_path, "c.png", _png()))
    await it.transcribe_images(_article([ref]))
    assert vision.detect_calls == 1 and vision.transcribe_calls == 1


async def test_missing_local_file_skips_that_one_only(tmp_path, vision):
    good = _ref(_write(tmp_path, "g.png", _png()), position=0)
    gone = _ref("/no/such.png", position=1)
    result = await it.transcribe_images(_article([good, gone]))
    assert not result[0].skipped
    assert result[1].skipped and result[1].skip_reason == "unavailable"


async def test_paywall_web_image_without_context_is_skipped(vision):
    ref = _ref("https://site/chart.png", local=False, paywall=True)
    result = await it.transcribe_images(_article([ref]), browser_context=None)
    assert result[0].skipped and result[0].skip_reason == "no_browser_context"
    assert vision.transcribe_calls == 0


# ── classification gate ────────────────────────────────────────────────────


async def test_photo_is_skipped(tmp_path, monkeypatch):
    v = FakeVision(image_type="photo")
    monkeypatch.setattr(it, "get_vision_provider", lambda: v)
    ref = _ref(_write(tmp_path, "p.png", _png()))
    result = await it.transcribe_images(_article([ref]))
    assert result[0].skipped and result[0].skip_reason == "photo"
    assert v.transcribe_calls == 0


async def test_unexpected_type_becomes_unknown_and_is_skipped(tmp_path, monkeypatch):
    v = FakeVision(image_type="banana")
    monkeypatch.setattr(it, "get_vision_provider", lambda: v)
    ref = _ref(_write(tmp_path, "u.png", _png()))
    result = await it.transcribe_images(_article([ref]))
    assert result[0].skipped and result[0].skip_reason == "unknown"


async def test_tiny_image_is_skipped_too_small(tmp_path, vision):
    ref = _ref(_write(tmp_path, "t.png", TINY_PNG))
    result = await it.transcribe_images(_article([ref]))
    assert result[0].skipped and result[0].skip_reason == "too_small"
    assert vision.detect_calls == 0


async def test_oversized_image_is_skipped(tmp_path, vision, monkeypatch):
    monkeypatch.setattr(it, "MAX_IMAGE_MB", 0.01)  # 10 KB cap
    ref = _ref(_write(tmp_path, "big.png", _png()))  # ~75 KB
    result = await it.transcribe_images(_article([ref]))
    assert result[0].skipped and result[0].skip_reason == "too_large"


async def test_unopenable_bytes_are_invalid_image(tmp_path, vision):
    ref = _ref(_write(tmp_path, "junk.png", b"not an image at all " * 400))
    result = await it.transcribe_images(_article([ref]))
    assert result[0].skipped and result[0].skip_reason == "invalid_image"


# ── cap ────────────────────────────────────────────────────────────────────


async def test_over_cap_images_still_get_an_entry(tmp_path, vision, monkeypatch):
    monkeypatch.setattr(it, "MAX_IMAGES_PER_ARTICLE", 2)
    refs = [_ref(_write(tmp_path, f"{i}.png", _png(i)), position=i) for i in range(5)]
    result = await it.transcribe_images(_article(refs))
    assert len(result) == 5
    assert [r.skipped for r in result] == [False, False, True, True, True]
    assert all(r.skip_reason == "over_cap" for r in result if r.skipped)
    assert vision.transcribe_calls == 2


# ── cache ──────────────────────────────────────────────────────────────────


async def test_second_run_of_same_file_hits_the_cache(tmp_path, vision):
    path = _write(tmp_path, "c.png", _png())
    art = _article([_ref(path)])
    await it.transcribe_images(art)
    await it.transcribe_images(art)
    assert vision.transcribe_calls == 1  # not re-paid


async def test_touching_the_bytes_forces_a_new_call(tmp_path, vision):
    path = _write(tmp_path, "c.png", _png(1))
    art = _article([_ref(path)])
    await it.transcribe_images(art)
    _write(tmp_path, "c.png", _png(2))  # same name, different bytes -> new source_key
    await it.transcribe_images(art)
    assert vision.transcribe_calls == 2


async def test_cache_under_a_different_vision_model_is_not_reused(tmp_path, monkeypatch):
    path = _write(tmp_path, "c.png", _png())
    art = _article([_ref(path)])

    v1 = FakeVision()
    v1.model_name = "vision-a"
    monkeypatch.setattr(it, "get_vision_provider", lambda: v1)
    await it.transcribe_images(art)

    v2 = FakeVision()
    v2.model_name = "vision-b"
    monkeypatch.setattr(it, "get_vision_provider", lambda: v2)
    await it.transcribe_images(art)
    assert v2.transcribe_calls == 1  # re-transcribed for the new model


# ── count_uncached ─────────────────────────────────────────────────────────


async def test_count_uncached_is_pure(tmp_path, vision, monkeypatch):
    path = _write(tmp_path, "c.png", _png())
    art = _article([_ref(path), _ref("/gone.png", position=1)])

    monkeypatch.setattr(it, "VISION_MODEL", "mock-vision")
    assert it.count_uncached(art.images) == 2  # nothing cached yet

    await it.transcribe_images(art)
    assert it.count_uncached(art.images) == 1  # the resolvable one is now cached


async def test_count_uncached_counts_a_web_ref_by_url(tmp_path):
    web = ImageRef(
        src="https://site/x.png", local_path=None, alt="", caption="", position=0, is_paywall=True
    )
    assert it.count_uncached([web]) == 1  # web ref keyed by URL, nothing cached


async def test_small_pixel_dimensions_are_too_small(tmp_path, vision):
    b = io.BytesIO()
    Image.effect_noise((60, 60), 200).convert("RGB").save(b, "PNG")
    assert len(b.getvalue()) > 5000  # passes the byte check
    ref = _ref(_write(tmp_path, "small.png", b.getvalue()))
    result = await it.transcribe_images(_article([ref]))
    assert result[0].skipped and result[0].skip_reason == "too_small"


async def test_transcription_prompt_without_alt_or_caption(tmp_path, vision):
    ref = ImageRef(
        src="images/x/0.png",
        local_path=_write(tmp_path, "c.png", _png()),
        alt="",
        caption="",
        position=0,
        is_paywall=False,
    )
    result = await it.transcribe_images(_article([ref]))
    assert not result[0].skipped  # no alt/caption lines, still transcribes fine
