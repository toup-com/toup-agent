"""Regression: slang 'pic'/'pics' and 'photo' must expose the image tools.

Bug (found via the founder's live test): "generate pic" scored 0 for the media
intent — the keyword list had "picture"/"image" but not the extremely common
slang "pic" — so generate_image was never offered to the model, which then
wrongly told the user "I can't generate images." Word-boundary matching keeps
"pic" from firing inside epic / typical / picnic / spicy / pickup / topic.
"""
from app.agent.query_intent import classify_query_intent, filter_tools_by_intent

_IMAGE_REQUESTS = [
    "generate pic",
    "i meant generate pic",
    "make me a photo",
    "make a pic of a sunset",
    "create a picture of a cat",
    "edit my pic",
    "edit this pic please",
    "send me a pic",
    "draw a selfie of me",
    "generate an avatar",
]

# Contain the letters "pic" but are NOT image requests — must NOT route to media.
_NOT_IMAGE = [
    "epic story about a hero",
    "that is pretty typical",
    "the topic of today",
    "i love spicy food",
    "grab my pickup truck",
    "picnic tomorrow at noon",
]


def test_slang_pic_photo_routes_to_media():
    for msg in _IMAGE_REQUESTS:
        intent = classify_query_intent(msg)
        assert intent.category == "media", f"{msg!r} -> {intent.category} (expected media)"


def test_pic_substring_words_do_not_false_trigger_media():
    for msg in _NOT_IMAGE:
        intent = classify_query_intent(msg)
        assert intent.category != "media", f"{msg!r} wrongly classified as media"


def test_generate_image_tool_is_exposed_for_pic_request():
    all_tools = [{"name": "generate_image"}, {"name": "edit_image"},
                 {"name": "send_photo"}, {"name": "web_search"}, {"name": "run_terminal"}]
    for msg in ("generate pic", "edit my pic", "make me a photo"):
        intent = classify_query_intent(msg)
        names = {t["name"] for t in filter_tools_by_intent(all_tools, intent)}
        assert "generate_image" in names and "edit_image" in names, \
            f"{msg!r}: image tools not exposed; got {names}"
