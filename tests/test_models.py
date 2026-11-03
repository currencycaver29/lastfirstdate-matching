"""FR-LM-02: the member schema holds real profile fields and no global attractiveness score."""

import dataclasses

import pytest

from lastfirstdate.models import N_SLOTS, Member, slot_id, slot_name

FORBIDDEN = ("attract", "desirab", "elo", "hot", "rating", "rank", "score", "beauty")


def test_member_has_no_global_attractiveness_field():
    """FR-LM-02 / rule R2: nothing on a member can hold an overall looks rank."""
    names = [f.name for f in dataclasses.fields(Member)]
    for name in names:
        assert not any(bad in name.lower() for bad in FORBIDDEN), name


def test_member_has_no_photo_or_embedding_field():
    """FR-LM-02 / rule R3: photo embeddings are stored apart from the profile."""
    names = [f.name for f in dataclasses.fields(Member)]
    assert not any("photo" in n or "embed" in n for n in names)


def test_slots_round_trip():
    """FR-LM-02: 21 weekly slots, 7 days by lunch, evening and late."""
    assert N_SLOTS == 21
    assert slot_id("fri", "evening") == 13
    assert slot_name(13) == "fri evening"
    for s in range(N_SLOTS):
        day, part = slot_name(s).split()
        assert slot_id(day, part) == s
    with pytest.raises(ValueError):
        slot_name(21)
