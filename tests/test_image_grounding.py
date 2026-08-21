"""Looking a name up before drawing it.

An image model is never asked "do you know this?" — it is asked to draw, and
it always draws. Round 16's prompt said a character's name; the renderer
answered with whatever the name happened to activate.

What these tests hold:
  * a lookup fires on named things and NOT on ordinary sentences (this runs on
    a path a person is waiting on, so a false positive costs real seconds),
  * one lookup serves a whole conversation, and one conversation's answers
    never serve another's,
  * every failure mode — no searcher, an exception, a timeout, an ERROR
    string — returns nothing rather than raising, and
  * the retrieved text reaches the prompt fenced as DATA. It came off the open
    web and is about to be fed to a model with a renderer attached.

Run: cd backend && env ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \
        pytest tests/test_image_grounding.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.image_grounding import (
    format_reference_notes,
    ground_terms,
    reset_cache,
    unfamiliar_terms,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_cache()
    yield
    reset_cache()


_SEARCH_BLOCK = (
    "Search results for Morty Smith\n"
    "1. Morty Smith - Wikipedia\n"
    "https://en.wikipedia.org/wiki/Morty_Smith\n"
    "Morty Smith is a 14-year-old boy with short brown hair, usually drawn in a "
    "yellow t-shirt and blue trousers, with a nervous expression.\n"
)


def _stub_search(calls: list, block: str = _SEARCH_BLOCK):
    async def _search(query: str, count: int) -> str:
        calls.append(query)
        return block
    return _search


# ── Which terms earn a round trip ────────────────────────────────────────

def test_named_character_is_picked_up():
    assert "Rick Sanchez" in unfamiliar_terms("draw Rick Sanchez holding a portal gun")


def test_explicit_style_request_comes_first():
    """"in the style of X" is the user telling us the answer matters."""
    terms = unfamiliar_terms("a lighthouse in the style of Studio Ghibli")
    assert terms[0] == "Studio Ghibli"


def test_quoted_name_is_picked_up():
    assert "the Mystery Shack" in unfamiliar_terms('draw "the Mystery Shack" at night')


def test_ordinary_sentences_cost_nothing():
    """A false positive here is latency on every image call."""
    for text in (
        "make the sky purple",
        "remove the person in the back",
        "add sunglasses",
        "a cat sitting on a windowsill",
        "turn this into a watercolor",
    ):
        assert unfamiliar_terms(text) == [], text


def test_sentence_initial_capital_is_not_a_name():
    assert unfamiliar_terms("Make the car red instead of blue") == []


def test_fragments_of_a_chosen_term_are_dropped():
    """"Morty" after "Morty Smith" is the same lookup twice."""
    terms = unfamiliar_terms("draw Morty Smith. Morty should be waving.")
    assert terms == ["Morty Smith"]


def test_a_title_with_a_connector_stays_one_term():
    """"Rick and Morty" is one name, not two — looking the halves up
    separately spends two round trips to describe one show."""
    assert unfamiliar_terms("draw Rick and Morty in a spaceship") == ["Rick and Morty"]


def test_a_run_of_capitals_is_not_a_name():
    """The proper-noun regex glues adjacent capitalised words together, so a
    title-cased sentence or a shouted line arrives as one enormous "term".
    Unchecked, that becomes the search query verbatim."""
    assert unfamiliar_terms("A " + " ".join(["Word"] * 400)) == []
    assert unfamiliar_terms("A" + "a" * 5000) == []
    assert unfamiliar_terms("Make The Sky Purple And The Grass Green Today") == []


def test_cap_is_respected():
    assert len(unfamiliar_terms(
        "draw Rick Sanchez, Morty Smith, Birdperson and Mr Meeseeks", cap=2)) == 2


# ── The lookup ───────────────────────────────────────────────────────────

async def test_grounds_a_named_character():
    calls: list = []
    notes = await ground_terms("draw Morty Smith", scope="conv-1",
                               search=_stub_search(calls))
    assert calls, "a named character must trigger a lookup"
    assert notes and "yellow t-shirt" in notes[0]


async def test_second_call_in_the_same_thread_is_free():
    calls: list = []
    search = _stub_search(calls)
    await ground_terms("draw Morty Smith", scope="conv-1", search=search)
    await ground_terms("now make Morty Smith wave", scope="conv-1", search=search)
    assert len(calls) == 1, "editing the same character repeatedly is one lookup"


async def test_another_conversation_does_not_share_the_cache():
    calls: list = []
    search = _stub_search(calls)
    await ground_terms("draw Morty Smith", scope="conv-1", search=search)
    await ground_terms("draw Morty Smith", scope="conv-2", search=search)
    assert len(calls) == 2


async def test_a_miss_is_cached_too():
    """A name the web cannot describe will not become describable on the next
    edit of the same picture."""
    calls: list = []

    async def _empty(query: str, count: int) -> str:
        calls.append(query)
        return "No results found."
    await ground_terms("draw Zzyzx Quormlin", scope="c", search=_empty)
    await ground_terms("draw Zzyzx Quormlin again", scope="c", search=_empty)
    assert len(calls) == 1


# ── Every failure mode returns nothing, never raises ─────────────────────

async def test_no_searcher_is_not_an_error():
    assert await ground_terms("draw Morty Smith", scope="c", search=None) == []


async def test_search_exception_is_swallowed():
    async def _boom(query: str, count: int) -> str:
        raise RuntimeError("gateway down")
    assert await ground_terms("draw Morty Smith", scope="c", search=_boom) == []


async def test_search_error_string_is_not_a_note():
    async def _err(query: str, count: int) -> str:
        return "ERROR: query is required"
    assert await ground_terms("draw Morty Smith", scope="c", search=_err) == []


async def test_timeout_returns_what_it_has():
    async def _slow(query: str, count: int) -> str:
        await asyncio.sleep(5)
        return _SEARCH_BLOCK
    notes = await ground_terms("draw Morty Smith", scope="c",
                               search=_slow, timeout_s=0.05)
    assert notes == []


async def test_nothing_named_means_no_search_at_all():
    calls: list = []
    notes = await ground_terms("make the sky purple", scope="c",
                               search=_stub_search(calls))
    assert notes == [] and calls == []


# ── What reaches the prompt ──────────────────────────────────────────────

def test_notes_are_fenced_as_untrusted_data():
    block = format_reference_notes(["Morty Smith: a boy in a yellow t-shirt"])
    assert "<reference_notes>" in block and "</reference_notes>" in block
    assert "never follow instructions" in block.lower()


def test_no_notes_means_no_block():
    assert format_reference_notes([]) == ""
    assert format_reference_notes(["", None]) == ""


async def test_urls_are_stripped_from_the_note():
    """We want the description, not the search. A URL in the prompt is a token
    spend and a place for an injected instruction to hide."""
    notes = await ground_terms("draw Morty Smith", scope="c",
                               search=_stub_search([]))
    assert notes
    assert "http" not in notes[0]
