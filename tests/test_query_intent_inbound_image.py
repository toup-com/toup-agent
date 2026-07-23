"""An inbound image must expose the image tools regardless of the caption text.

Regression for the founder's "Make a six pack" bug (2026-07-23): a photo sent
with a short caption that names no image-noun classified as ``question`` and the
`edit_image` tool was never offered to the model. The model could *see* the
image but had no tool to act on it, so it falsely told the user it "can't
edit/render the image in this chat".

`with_inbound_image()` is the fix: whenever the turn carries an image, the media
toolset (`edit_image`/`generate_image`/`analyze_image`/…) is merged onto whatever
the text classified into — additively, so existing tools survive.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.query_intent import (  # noqa: E402
    classify_query_intent,
    with_inbound_image,
    TOOLS_MEDIA,
    INTENT_FULL,
)

_IMAGE_TOOLS = ("edit_image", "generate_image", "analyze_image")

# Real-world captions people type on a photo they want edited. NONE of these
# name an image-noun, so text-only classification never routes them to media.
_SHORT_EDIT_CAPTIONS = [
    "Make a six pack",
    "make me a six pack",
    "make me look muscular",
    "fix this",
    "fix my face",
    "remove the background",
    "whiten my teeth",
    "make it brighter",
    "add sunglasses",
    "",              # no caption at all — just a photo
    "hey",           # greeting-only caption
    "what do you think",
]


def _exposes(intent, name: str) -> bool:
    # "full" means every tool (empty tool_names sentinel); otherwise explicit.
    return intent.category == "full" or name in intent.tool_names


def test_short_captions_do_not_expose_edit_image_on_text_alone():
    # Establishes the precondition: without the image signal these captions
    # would leave edit_image unexposed. If this ever changes, the fix below is
    # still correct but this guard documents WHY it was needed.
    missing = [
        c for c in _SHORT_EDIT_CAPTIONS
        if _exposes(classify_query_intent(c), "edit_image")
    ]
    # At least the truly bare ones must miss on text alone.
    assert not _exposes(classify_query_intent("Make a six pack"), "edit_image")
    assert not _exposes(classify_query_intent(""), "edit_image")


def test_inbound_image_exposes_image_tools_for_every_caption():
    for caption in _SHORT_EDIT_CAPTIONS:
        base = classify_query_intent(caption)
        aug = with_inbound_image(base)
        for tool in _IMAGE_TOOLS:
            assert _exposes(aug, tool), (
                f"caption {caption!r} (text-cat={base.category}) should expose "
                f"{tool} once an image is attached"
            )


def test_augment_is_additive_preserves_existing_tools():
    # A caption that DOES carry non-media intent must keep those tools too —
    # the user could attach a photo AND ask for a reminder in the same turn.
    base = classify_query_intent("remind me to call mom in 5 minutes")
    aug = with_inbound_image(base)
    assert base.tool_names, "precondition: scheduling caption has tools"
    assert base.tool_names <= aug.tool_names, "existing tools must survive the merge"
    assert TOOLS_MEDIA <= aug.tool_names, "media tools must be added"


def test_full_intent_is_a_noop():
    # "full" already exposes every tool; augmenting must not change or copy it.
    assert with_inbound_image(INTENT_FULL) is INTENT_FULL


def test_original_intent_is_not_mutated():
    # QueryIntent is frozen; the augmented copy must not leak back.
    base = classify_query_intent("Make a six pack")
    with_inbound_image(base)
    assert "edit_image" not in base.tool_names


if __name__ == "__main__":
    test_short_captions_do_not_expose_edit_image_on_text_alone()
    test_inbound_image_exposes_image_tools_for_every_caption()
    test_augment_is_additive_preserves_existing_tools()
    test_full_intent_is_a_noop()
    test_original_intent_is_not_mutated()
    print("all inbound-image intent tests passed")
