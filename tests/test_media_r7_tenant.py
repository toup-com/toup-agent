"""R6-17: the tenant fast path's ONE closed-class TROUBLE scan (addendum 6
R6-17; r6d verifier probes `/private/tmp/toup-voice-r2-review/probes/r6d-
verify-tenant.cA6C0d`, results `r6d-a508ab4d798fdc7ca.json`: 6 major + 4 minor)
and the R6-12 dispatch proofs on BOTH real paths.

TROUBLE SCAN  One scan of the whole utterance, both arms, outside double /
     guillemet / curly-double quotes: any word of the CLOSED function-word
     classes declines to the agent — pronouns / possessives / demonstratives
     ('it', 'that', 'my', «آهنگ من», «اون», «همون», a possessive clitic «م»),
     indefinites ('anything', 'something', 'any', «هر»), exclusion /
     negation ('but', 'except', 'without', 'other than', «غیر», «جز», «بجز»,
     «نخیر», «نچ», the negated copula «نیس/نیست»), relativizers and
     wh-words ('where', 'who', a reduced relative 'the song he sang', «که»,
     «ک»), temporal / sequence ('again', 'later', 'next', 'after', 'when',
     «بعدا», «بعدن», «دیروز»), repair / retraction ('I mean', 'nvm', 'jk',
     'on second thought', 'sorry', «منظورم», «یعنی», «ببخشید»), and a
     first/second-person subject ('I …', 'you …'). The grammar's own framing
     slots are the only exemption ('play me', 'for me', «برام», 'would you',
     'thank you', «همین الان», «چند تا»). A title that holds such a word takes
     the agent path — which the R6-12 legs below prove still reaches the agent
     with the caller's full words, once, with no fast-path play.
SINGLE QUOTES never shield (they only keep a title whole); an elision
     apostrophe ('80s, 'Til, rock 'n' roll, Guns N', the Beatles') is a letter.
LET-RULE  a subjunctive-shaped word anywhere after «بذار» (inside «از» /
     «درباره» / «به اسم» too) declines.
NEVER TRUNCATE  framing leaves the query only as a separate tail (after a
     comma / clause break, or after the Persian verb); a MARKED title and an
     unpunctuated English object keep their words.
WHITELIST COLLAPSE  «پخش می‌کنی؟», the discourse tails «خب / ببینم / بی زحمت /
     عزیزم», a parenthesised version tag ('Halo (Live)') and a leading elision
     apostrophe ("'90s hip hop") play fast again.

R6-12 (review guard: acceptance needs REAL dispatch evidence, not
`media_request(...) is None`):
  TYPED  the REAL `ws_chat` websocket handler, spoken to over an in-process
         ASGI socket, with a fake agent runner: a declined valid request
         reaches exactly ONE agent run whose `user_message` is the caller's
         FULL original words, the fast path searches nothing and sends no
         media_play, the turn is answered (a `done` frame — no silence) and
         the only play is the one the agent's REAL play_media tool makes (no
         duplicate). A negated / retracted / excluded / back-reference request
         produces no play at all unless the model itself decides to play.
  VOICE  the REAL relay (fx4 `live_voice_protocol`, read-only here) through
         `tests/test_live_harness.py`'s FakeProvider, with the real
         `ws_realtime._think` / `_play_media_direct` building their request
         bodies against the recording fake tenant of
         `tests/test_live_round6_media.py`: one delegation, no
         `/internal/play-media` from the fast path, exactly one
         `/internal/agent-turn(/stream)` body whose `message` and
         `display_request` carry the caller's full words.

MODEL-DEPENDENT LIMIT (R6-12, documented, never pinned as proven): which
title the model finally plays for a declined request — whether it resolves
'No Tears Left to Cry' to Ariana Grande's song, or plays anything at all for
'anything but Halo' — is the model's decision. The fake agents below are
SCRIPTED: they prove the dispatch (full words in, one run, the tool's one
play out, nothing from the fast path or a fallback), not the model's choice.
"""
# The shared `rig` fixture is imported and then requested by name, which ruff
# reads as a redefinition.
# ruff: noqa: F811
from __future__ import annotations

import asyncio
import json
from typing import Optional

import pytest
from fastapi import FastAPI

from app.agent import media_intent
from app.agent.agent_runner import AgentResponse
from app.agent.media_intent import media_request
from app.api import ws_chat
from test_media_order_scope_r6 import KEY, USER, rig  # noqa: F401


async def _typed(rig, text):
    """(result, searched, media_play ids) of the typed fast path."""
    before = len(rig.searches)
    q: asyncio.Queue = asyncio.Queue()
    result = await asyncio.wait_for(ws_chat._fast_media_check(text, USER, q), timeout=2.0)
    frames = []
    while not q.empty():
        frames.append(q.get_nowait())
    plays = [f.get("video_id") for f in frames if f.get("type") == "media_play"]
    return result, rig.searches[before:], plays


async def _declines(rig, text):
    result, searched, plays = await _typed(rig, text)
    assert result is None and not searched and not plays, (
        f"{text!r}: searched {searched!r}, media_play {plays!r}")


def _relay_agrees(text):
    from app.services import live_voice_protocol as lvp

    assert lvp._media_request(text) == media_request(text), text


def test_live_relay_recognizes_observed_drake_false_starts():
    text = "برنامه... یه... یه آهنگ... آهنگ دری... دریک پ... پلی کن"
    _relay_agrees(text)
    ask = media_request(text)
    assert ask is not None and (ask.query, ask.variety) == ("Drake", True)


# ══════════════════════════════════════════════════════════════════════════
# R6-12 TYPED leg: the real ws_chat handler over an in-process ASGI socket
# ══════════════════════════════════════════════════════════════════════════

class _Socket:
    """One client socket spoken straight into the ASGI app (the shape of
    `tests/test_ws_chat_infra_fault_close.py`'s AsgiWs): frames in, frames
    out, nothing of the handler stubbed."""

    def __init__(self, app, headers):
        self._app = app
        self._headers = headers
        self._to_app: asyncio.Queue = asyncio.Queue()
        self._from_app: asyncio.Queue = asyncio.Queue()
        self.frames: list = []
        self.task: Optional[asyncio.Task] = None

    async def open(self) -> dict:
        scope = {
            "type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "scheme": "ws", "path": "/api/ws/chat",
            "raw_path": b"/api/ws/chat", "query_string": b"",
            "root_path": "", "headers": self._headers,
            "client": ("127.0.0.1", 51234), "server": ("testserver", 80),
            "subprotocols": [], "state": {},
        }
        self._to_app.put_nowait({"type": "websocket.connect"})
        self.task = asyncio.create_task(self._app(scope, self._to_app.get, self._from_app.put))
        return await asyncio.wait_for(self._from_app.get(), timeout=5.0)

    def send(self, frame: dict) -> None:
        self._to_app.put_nowait({"type": "websocket.receive", "text": json.dumps(frame)})

    async def until(self, predicate, timeout: float = 8.0) -> list:
        deadline = asyncio.get_running_loop().time() + timeout
        while not any(predicate(f) for f in self.frames):
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                raise AssertionError(f"timed out; frames={[f.get('type') for f in self.frames]}")
            msg = await asyncio.wait_for(self._from_app.get(), timeout=left)
            if msg["type"] == "websocket.send":
                try:
                    self.frames.append(json.loads(msg.get("text") or "{}"))
                except ValueError:
                    pass
            elif msg["type"] == "websocket.close":
                raise AssertionError(f"socket closed {msg.get('code')}: {self.frames}")
        return self.frames

    async def close(self) -> None:
        self._to_app.put_nowait({"type": "websocket.disconnect", "code": 1000})
        if self.task is not None and not self.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=2.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                self.task.cancel()
                try:
                    await self.task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass


class _Agent:
    """The agent runner `ws_chat` hands the turn to. Records every run; the
    fake MODEL then does what `model` scripts for the words it received —
    call the REAL play_media tool with a query, or answer without playing.
    Which title a real model plays is model-dependent (module docstring)."""

    def __init__(self, tools, model):
        self.tools = tools
        self.model = model
        self.runs: list = []
        self.results: list = []

    async def run(self, *, user_message, user_id, channel=None, display_user_message=None,
                  preset_media=None, **_kw):
        self.runs.append({
            "user_message": user_message, "display_user_message": display_user_message,
            "preset_media": preset_media, "channel": channel,
        })
        query = self.model(user_message, preset_media)
        text = "I didn't start anything."
        if query:
            self.tools.set_user_id(user_id)
            self.tools.set_channel(channel or "web")
            result = await self.tools._tool_play_media({"query": query, "mode": "audio"})
            self.results.append(result)
            text = f"Starting {query}."
        elif preset_media:
            text = "It's starting."
        return AgentResponse(text=text, session_id="sess-r7", model="fake-model")


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(ws_chat.router, prefix="/api")
    return app


async def _typed_turn(rig, monkeypatch, text, model):
    """One typed turn through the real handler: (socket frames, agent,
    fast-path searches, agent-tool media_play frames)."""
    agent = _Agent(rig.tools, model)
    monkeypatch.setattr(ws_chat, "_agent_runner", agent)
    before_tasks = set(asyncio.all_tasks())
    sock = _Socket(_app(), [(b"host", b"testserver"), (b"x-agent-key", KEY.encode())])
    accepted = await sock.open()
    assert accepted["type"] == "websocket.accept", accepted
    searches_before, frames_before = len(rig.searches), len(rig.frames)
    try:
        sock.send({"type": "message", "text": text, "channel": "web"})
        await sock.until(lambda f: f.get("type") in ("done", "error"))
    finally:
        await sock.close()
        # Join what the handler spawned (its persistence writes) before the
        # next test's schema reset: a write still open would lock sqlite.
        spawned = [t for t in asyncio.all_tasks() - before_tasks
                   if t is not asyncio.current_task() and not t.done()]
        if spawned:
            _done, pending = await asyncio.wait(spawned, timeout=5.0)
            for t in pending:
                t.cancel()
    agent_plays = [f for f in rig.frames[frames_before:] if f.get("type") == "media_play"]
    return sock.frames, agent, rig.searches[searches_before:], agent_plays



# ══════════════════════════════════════════════════════════════════════════
# 1. Every r6d verifier case (test_r6d_v_lexical_typed.py a–k), typed path
#    AND relay verdict (the relay's fast path IS `media_request`)
# ══════════════════════════════════════════════════════════════════════════

R6D_VERIFIER_DECLINES = [
    # a — an elision apostrophe pair never shields the caller's negation
    "play '80s music, not Guns N' Roses",
    "play rock 'n roll, not the Stones' songs",
    "play '90s hits, no, the Beatles' Yesterday",
    "play some '70s disco, not the Bee Gees' stuff",
    "play '80s rock, don't play Guns N' Roses",
    "play 'em all, no wait, the Beatles' songs",
    "play 'Til Tuesday, never mind, the Eagles' Hotel California",
    "play ‘80s music, not Guns N’ Roses",
    # b — a clause opening with 'I' is the caller's repair, not a title part
    "Play Halo, I mean Hello.",
    "play some jazz, I mean blues",
    "play Halo, I meant Hello",
    "play Halo, I'm kidding",
    "Play Halo, I'm just kidding.",
    "play Halo - I mean Hello",
    "put on Halo, I mean Hello",
    # c — the «بذار» let-rule covers the whole complement
    "بذار آهنگ از اول شروع شه",
    "بذار آهنگ از اول بره",
    "بذار آهنگ از ابی تموم شه",
    "بذار آهنگ درباره عشق تموم شه",
    "بذار آهنگ به اسم سلام تموم شه",
    "بذار یه آهنگ از ابی برات بخونم",
    "بذار یه آهنگ از ابی بخونم",
    # d — an exclusion is never searched as the subject
    "play anything but Halo",
    "play something other than Halo",
    "play some jazz except Kenny G",
    "play any song besides Halo",
    "play music without Beyonce",
    "یه آهنگ غیر از ابی بذار",
    "یه آهنگ به جز ابی پخش کن",
    "یه آهنگ بجز ابی بذار",
    # e — the colloquial relative «ک», the negated copula, «نخیر» / «نچ»
    "آهنگی ک اسمش یادم نیس رو بذار",
    "آهنگی ک اسمش یادم نیست پخش کن",
    "آهنگی ک دیروز گذاشتی رو بذار",
    "آهنگ ابی نخیر داریوش رو بذار",
    "آهنگ ابی نچ داریوش رو بذار",
    # f — an unpunctuated retraction or repair
    "play Halo nvm",
    "play Halo jk",
    "play Halo on second thought",
    "play Halo I mean Hello",
    "آهنگ ابی منظورم داریوش رو بذار",
    "آهنگ ابی ببخشید داریوش رو بذار",
    # g — a back-reference, the caller's own saved thing, a non-media object
    "play it",
    "play it again",
    "play that song again",
    "play my playlist",
    "play the song you played yesterday",
    "پلی لیست من رو بذار",
    "آهنگ من رو بذار",
    "آهنگ مورد علاقه‌م رو بذار",
    "آهنگ دیروز رو بذار",
    "play a game with me",
    "play chess with me",
    # h — a deferred request is not a play-now request
    "play Halo after this song",
    "play Halo when this song ends",
    "play Halo next",
    "play Halo later",
    "بعدا یه آهنگ از ابی بذار",
    "یه آهنگ از ابی بعدا پخش کن",
    # i — an English relative adverb / reduced relative
    "play the song where she sings about angels",
    "play the video where the cat jumps",
    "play the song he sang at the wedding",
    # the verifier's decline controls
    "play '80s music, never mind",
    "play Halo, not the remix",
    "Play Halo, no, Hello.",
    "بذار آهنگ تموم شه",
    "آهنگی که اسمش یادم نیست رو بذار",
]


@pytest.mark.parametrize("text", R6D_VERIFIER_DECLINES)
async def test_r6d_verifier_case_declines_on_the_typed_path(rig, text):
    await _declines(rig, text)


@pytest.mark.parametrize("text", R6D_VERIFIER_DECLINES)
def test_r6d_verifier_case_declines_on_the_relay_path(text):
    assert media_request(text) is None, text
    _relay_agrees(text)


# The verifier's play controls (en + fa) — minus the one R6-17 supersedes:
# "play the Beatles' Yesterday" holds 'yesterday', a word of the closed
# temporal class, so it takes the agent path (pinned in
# R6_17_TITLES_TAKE_THE_AGENT_PATH and dispatch-proved in §9 / §10).
R6D_VERIFIER_PLAYS = [
    ("play Halo by Beyoncé", "Halo by Beyoncé"),
    ("play Stayin' Alive", "Stayin' Alive"),
    ("play Rock 'n' Roll", "Rock 'n' Roll"),
    ("play Guns N' Roses", "Guns N' Roses"),
    ("یه آهنگ از ابی پخش کن", "ابی"),
    ("آهنگ «سلام» رو بذار", "سلام"),
    ("play some jazz, would you?", "some jazz"),
    # k2 — a MARKED English title is never truncated (R6-17 NEVER TRUNCATE)
    ("play the song called Pray for Me", "the song called Pray for Me"),
    ("play the song called Dance for Me", "the song called Dance for Me"),
    ("play the song called Pretty Please", "the song called Pretty Please"),
]


@pytest.mark.parametrize(("text", "query"), R6D_VERIFIER_PLAYS)
async def test_r6d_verifier_control_still_plays(rig, text, query):
    result, searched, plays = await _typed(rig, text)
    assert result is not None and searched == [query] and plays, (text, searched, plays)
    _relay_agrees(text)


# j — the whitelist collapse R6-17 fixes: these play fast again (en + fa).
R6_17_COLLAPSE_PLAYS = [
    ("play '90s hip hop", "'90s hip hop"),                 # a leading elision
    ("play Halo (Live)", "Halo (Live)"),                   # a version tag
    ("play Halo [Acoustic]", "Halo [Acoustic]"),
    ("play Halo (2009 Remaster)", "Halo (2009 Remaster)"),
    ("آهنگ ابی رو پخش می‌کنی؟", "ابی"),                    # the question form
    ("یه آهنگ از ابی برام پخش می‌کنی؟", "ابی"),
    ("آهنگ ابی رو پخش میکنی", "ابی"),
    ("آهنگ ابی رو پخش می‌کنید؟", "ابی"),
    ("آهنگ ابی رو پخش کن خب", "ابی"),                      # the discourse tails
    ("یه آهنگ از ابی پخش کن ببینم", "ابی"),
    ("یه آهنگ از ابی پخش کن بی زحمت", "ابی"),
    ("یه آهنگ از ابی پخش کن بی‌زحمت", "ابی"),
    ("یه آهنگ از ابی پخش کن عزیزم", "ابی"),
]


@pytest.mark.parametrize(("text", "query"), R6_17_COLLAPSE_PLAYS)
async def test_r6_17_the_collapse_cases_play_fast_again(rig, text, query):
    result, searched, plays = await _typed(rig, text)
    assert result is not None and searched == [query] and plays, (text, searched, plays)
    _relay_agrees(text)


@pytest.mark.parametrize("text", [
    # a bracket group that holds anything but version tags is still a clause
    "play Halo (actually no)",
    "play Halo (the one from the concert)",
    "play Halo (feat. Jay-Z)",
])
def test_r6_17_only_a_version_tag_bracket_is_title_text(text):
    assert media_request(text) is None, text


@pytest.mark.parametrize("text", [
    # the progressive in the 1st person is the speaker's own act, not a request
    "آهنگ ابی رو پخش میکنم",
    # a negated progressive is a negation (R6-9 T5)
    "آهنگ ابی رو پخش نمی‌کنی؟",
])
def test_r6_17_the_question_form_is_the_2nd_person_only(text):
    assert media_request(text) is None, text


# ══════════════════════════════════════════════════════════════════════════
# 2. The closed classes, one structural scan — en + fa, beyond the verifier
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(("klass", "text", "without"), [
    ("pronoun", "play this Halo remix", "play the Halo remix"),
    ("pronoun", "play their Halo remix", "play the Halo remix"),
    ("pronoun", "این آهنگ ابی رو پخش کن", "آهنگ ابی رو پخش کن"),
    ("pronoun", "همون آهنگ ابی رو بذار", "آهنگ ابی رو بذار"),
    ("pronoun", "آهنگ تو رو پخش کن", "آهنگ ابی رو پخش کن"),
    ("pronoun", "آهنگ خودت رو بذار", "آهنگ ابی رو بذار"),
    ("indefinite", "play whatever jazz", "play some jazz"),
    ("indefinite", "play everything by Adele", "play Hello by Adele"),
    ("indefinite", "هر آهنگی از ابی پخش کن", "یه آهنگ از ابی پخش کن"),
    ("indefinite", "همه آهنگای ابی رو پخش کن", "آهنگای ابی رو پخش کن"),
    ("exclusion", "play some jazz instead of Halo", "play some jazz of Halo"),
    ("exclusion", "play some jazz rather than Halo", "play some jazz and Halo"),
    ("exclusion", "یه آهنگ بدون ابی پخش کن", "یه آهنگ از ابی پخش کن"),
    ("exclusion", "یه آهنگ به جای ابی پخش کن", "یه آهنگ از ابی پخش کن"),
    ("relative", "play the song which goes la la la", "play the song la la la"),
    ("relative", "play what's popular", "play popular jazz"),
    ("relative", "آهنگی که ابی خونده رو بذار", "آهنگ ابی رو بذار"),
    ("temporal", "play Halo again", "play Halo"),
    ("temporal", "play Halo tomorrow", "play Halo"),
    ("temporal", "play Halo and then some jazz", "play Halo and some jazz"),
    ("temporal", "فردا یه آهنگ از ابی پخش کن", "یه آهنگ از ابی پخش کن"),
    ("temporal", "امشب یه آهنگ از ابی بذار", "یه آهنگ از ابی بذار"),
    ("repair", "play Halo sorry Hello", "play Halo Hello"),
    ("repair", "play Halo actually Hello", "play Halo Hello"),
    ("repair", "یه آهنگ از ابی یعنی داریوش پخش کن", "یه آهنگ از ابی و داریوش پخش کن"),
    ("negation", "آهنگ ابی نبود، یه آهنگ از داریوش بذار", "یه آهنگ از داریوش بذار"),
    ("negation", "هیچ آهنگی از ابی بذار", "یه آهنگی از ابی بذار"),
])
async def test_r6_17_a_word_of_a_closed_class_declines(rig, klass, text, without):
    """Paired: the same sentence WITHOUT the class word plays, so the class
    word is what declines it."""
    assert media_request(without) is not None, ("control must play", without)
    await _declines(rig, text)
    _relay_agrees(text)


@pytest.mark.parametrize(("text", "query"), [
    # the grammar's own framing slots hold a pronoun by construction
    ("play me a song by Radiohead", "Radiohead"),
    ("play Halo by Beyonce, would you?", "Halo by Beyonce"),
    ("play Halo by Beyonce, could you?", "Halo by Beyonce"),
    ("play Halo, thank you", "Halo"),
    ("برای خودم یه آهنگ بذار", "آهنگ"),
    ("همین الان یه آهنگ بذار", "آهنگ"),
    ("یه آهنگ از امیر تتلو برام بذار", "امیر تتلو"),
    ("چند تا آهنگ از ابی پخش کن", "ابی"),
    # «تو» as the object of a preposition is not an ezafe: «بعد از تو» is a title
    ("play بعد از تو", "بعد از تو"),
    # the sequencing "then" «بعد» stays a lead (it is not «بعدا» "later")
    ("بعد یه آهنگ از ابی بذار", "ابی"),
    # a word that merely CONTAINS a class word is not one
    ("play Nothingman", "Nothingman"),
    ("یه آهنگ از نیسان پخش کن", "نیسان"),
])
async def test_r6_17_the_grammars_own_slots_and_lookalikes_still_play(rig, text, query):
    result, searched, plays = await _typed(rig, text)
    assert result is not None and searched == [query] and plays, (text, searched, plays)
    _relay_agrees(text)


@pytest.mark.parametrize(("text", "query"), [
    # a trouble word inside a DOUBLE / guillemet / curly-double quote is the title
    ('play "Beat It"', '"Beat It"'),
    ("play “Somebody That I Used to Know”", "“Somebody That I Used to Know”"),
    ("آهنگ «همون» رو بذار", "همون"),
    ('play "Thank U, Next"', '"Thank U, Next"'),
])
async def test_r6_17_a_double_quote_shields_its_title(rig, text, query):
    result, searched, plays = await _typed(rig, text)
    assert result is not None and searched == [query] and plays, (text, searched, plays)
    _relay_agrees(text)


def test_r6_17_the_classes_are_closed_sets_and_one_scan():
    """Structure, not sentence lists: the trouble words are the union of the
    named closed classes; one function reads them on the whole utterance."""
    classes = media_intent._TROUBLE_CLASSES
    assert set(classes) == {"pronoun", "indefinite", "exclusion", "relative", "temporal", "repair"}
    union = set()
    for words in classes.values():
        union |= set(media_intent._class_words(" ".join(words.split())))
    assert union == set(media_intent._TROUBLE_WORDS)
    for word in ("it", "my", "that", "anything", "but", "except", "where", "who",
                 "again", "next", "later", "nvm", "jk", "sorry", "i",
                 "اون", "این", "همون", "غیر", "جز", "بجز", "نخیر", "نچ", "که", "ک",
                 "بعدا", "بعدن", "دیروز", "فردا", "منظورم", "یعنی", "ببخشید"):
        assert word in media_intent._TROUBLE_WORDS, word
    # the sequencing «بعد» / «بعدش» ("then") is a lead of the grammar, never trouble
    assert "بعد" not in media_intent._TROUBLE_WORDS
    assert "بعدش" not in media_intent._TROUBLE_WORDS


# ══════════════════════════════════════════════════════════════════════════
# 3. The r6d positive whitelist still plays fast (the R6-17 exceptions are
#    pinned in §7 and dispatch-proved in §9 / §10)
# ══════════════════════════════════════════════════════════════════════════

def _r6d_whitelist():
    from test_media_r6d_tenant import WHITELIST_PLAYS

    return list(WHITELIST_PLAYS)


@pytest.mark.parametrize(("text", "query"), _r6d_whitelist())
async def test_the_r6d_whitelist_still_plays_fast(rig, text, query):
    result, searched, plays = await _typed(rig, text)
    assert result is not None and searched == [query] and plays, (text, searched, plays)
    _relay_agrees(text)


# ══════════════════════════════════════════════════════════════════════════
# 4. NEVER TRUNCATE beats framing-out
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(("text", "query"), [
    # a MARKED title keeps its words (en + fa)
    ("play the song called Pray for Me", "the song called Pray for Me"),
    ("play a song called Halo for me", "Halo for me"),
    ("یه آهنگ به اسم سلام برام پخش کن", "اسم سلام برام"),
    ("یه آهنگ به اسم سلام لطفا پخش کن", "اسم سلام لطفا"),
    ("یه آهنگ به اسم سلام الان پخش کن", "اسم سلام الان"),
    # an unpunctuated English object keeps its words ("may search 'Halo please'")
    ("play Halo please", "Halo please"),
    ("play Pretty Please", "Pretty Please"),
    ("play some jazz for me", "some jazz for me"),
    # framing as a SEPARATE tail — after a clause break, or after the Persian
    # verb — leaves the query
    ("play the song called Halo, please", "the song called Halo"),
    ("play some jazz, please", "some jazz"),
    ("یه آهنگ به اسم سلام پخش کن لطفا", "اسم سلام"),
    ("یه آهنگ از ابی پخش کن برام", "ابی"),
    ("آهنگ «سلام» رو پخش کن، مرسی", "سلام"),
    # the Persian preverbal framing slot after an UNMARKED argument
    ("یه آهنگ از ابی برام پخش کن", "ابی"),
])
async def test_r6_17_never_truncate_framing_leaves_only_as_a_separate_tail(rig, text, query):
    result, searched, plays = await _typed(rig, text)
    assert result is not None and searched == [query] and plays, (text, searched, plays)
    _relay_agrees(text)


# ══════════════════════════════════════════════════════════════════════════
# 5. The «بذار» let-rule covers the whole complement
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "بذار آهنگ از اول شروع شه",
    "بذار یه آهنگ از بیانسه",              # the shape is the rule (a name costs a round trip)
    "بذار آهنگ به اسم بره",
    "بذار آهنگ درباره عشق بمونه",
    "یه آهنگ از ابی بذار به اسم باشه",      # «باشه» — 'let it be'
    "بذار یه آهنگ از ابی برات بخونم",
    "بذار آهنگ بیانسه رو",
])
async def test_r6_17_a_subjunctive_anywhere_after_bezar_declines(rig, text):
    await _declines(rig, text)
    _relay_agrees(text)


@pytest.mark.parametrize(("text", "query"), [
    # the grammar's own purpose complement and tail tags are not a let-clause
    ("بذار یه چیزی گوش بدم", "آهنگ"),
    ("بذار یه آهنگ از ابی گوش بدم", "ابی"),
    ("یه آهنگ از ابی بذار ببینم", "ابی"),
    ("بله یه آهنگ از شجریان بذار لطفا", "شجریان"),
    # a guillemet-quoted title is shielded; a Latin title has no Persian shape
    ("بذار آهنگ «بمونه»", "بمونه"),
    ("بذار آهنگ hello", "hello"),
    ("بذار آهنگ بهار", "بهار"),
    # the same subjunctive-shaped name after the LIGHT verb plays
    ("یه آهنگ از بیانسه پخش کن", "بیانسه"),
    ("یه آهنگ از ابی پخش کن به اسم باشه", "ابی اسم باشه"),
])
async def test_r6_17_let_rule_controls(rig, text, query):
    result, searched, plays = await _typed(rig, text)
    assert result is not None and searched == [query] and plays, (text, searched, plays)
    _relay_agrees(text)


# ══════════════════════════════════════════════════════════════════════════
# 6. Single quotes never shield; an elision apostrophe is a letter
# ══════════════════════════════════════════════════════════════════════════

def test_r6_17_the_quote_reader_and_the_shield():
    spans, shield = media_intent._quote_spans, media_intent._shield_spans
    # an elision is never a quote mark
    for text in ("play '80s music, not Guns N' Roses", "play ‘80s music, not Guns N’ Roses",
                 "play 'Til Tuesday, the Eagles' song", "play rock 'n' roll",
                 "play 'em all", "play '90s hip hop"):
        assert spans(text) == [], text
    # a single-quoted title is a span (kept whole) that does not shield
    text = "play 'No Tears Left to Cry'"
    assert spans(text) == [(5, 27)] and shield(text, spans(text)) == []
    # double / guillemet / curly-double quotes shield
    for text in ('play "Beat It"', "play «Beat It»", "play “Beat It”"):
        assert shield(text, spans(text)) == spans(text) != [], text
    # an unclosed quote still declines (TB1)
    assert spans("play 'Halo") is None and media_request("play 'Halo") is None


@pytest.mark.parametrize(("text", "query"), [
    ("play the track 'Hello, Goodbye, Hello'", "the track 'Hello, Goodbye, Hello'"),
    ("play the song 'Hello, okay'", "the song 'Hello, okay'"),
    ("play Beyoncé's Halo", "Beyoncé's Halo"),
    ("play the Beatles' Help", "the Beatles' Help"),
])
async def test_r6_17_a_single_quote_still_keeps_a_title_whole(rig, text, query):
    result, searched, plays = await _typed(rig, text)
    assert result is not None and searched == [query] and plays, (text, searched, plays)


# ══════════════════════════════════════════════════════════════════════════
# 7. Titles that hold a trouble word take the agent path (superseded pins)
# ══════════════════════════════════════════════════════════════════════════

R6_17_TITLES_TAKE_THE_AGENT_PATH = [
    "play No Tears Left to Cry",            # 'no' (R6-11 TB1 already)
    "play 'No Tears Left to Cry'",          # a single quote never shields
    "play Thank U, Next",                   # 'next' (temporal / sequence)
    "play the Beatles' Yesterday",          # 'yesterday' (temporal)
    "play 'Til I Collapse by Eminem",       # 'I' (the spec's own example)
    "play the track 'Talk to Me, Please'",  # 'Me' (pronoun), single quotes
    "play Halo, the Beyoncé song",          # a second clause (TB1; not a collapse class)
    "play Shape of You",                    # 'you'
    "بذار یه آهنگ از بیانسه",               # the let-rule shape
]


@pytest.mark.parametrize("text", R6_17_TITLES_TAKE_THE_AGENT_PATH)
async def test_r6_17_a_title_with_a_trouble_word_takes_the_agent_path(rig, text):
    await _declines(rig, text)
    _relay_agrees(text)


# ══════════════════════════════════════════════════════════════════════════
# 8. The ONE CLASS lexicon is read, never extended
# ══════════════════════════════════════════════════════════════════════════

def test_the_one_class_lexicon_is_byte_identical():
    import hashlib

    from test_media_r6b_tenant import _SNAPF_LEXICON_SHA256
    from test_media_r6d_tenant import _MODAL_FRAMING_SHA256

    digest = hashlib.sha256("\x00".join((
        media_intent.ASSENT_WORDS_TEXT,
        "\x01".join(media_intent.ASSENT_PHRASES_TEXT),
        media_intent.CONTENT_FREE_WORDS_TEXT,
    )).encode("utf-8")).hexdigest()
    assert digest == _SNAPF_LEXICON_SHA256
    modal = hashlib.sha256("\x01".join(media_intent.MODAL_FRAMING_TEXT).encode("utf-8")).hexdigest()
    assert modal == _MODAL_FRAMING_SHA256


# ══════════════════════════════════════════════════════════════════════════
# 9. R6-12 TYPED leg — the real ws_chat handler, a fake agent runner
# ══════════════════════════════════════════════════════════════════════════

# Declined VALID requests and what the scripted fake MODEL plays for them
# (model-dependent: a real model chooses the title — module docstring).
TYPED_DECLINED_VALID = [
    ("play No Tears Left to Cry", "Ariana Grande No Tears Left to Cry"),
    ("play anything but Halo", "Crazy in Love"),
    ("play my playlist", "Road Trip Mix"),
    ("play Thank U, Next", "Ariana Grande thank u, next"),
    ("play the Beatles' Yesterday", "The Beatles Yesterday"),
    ("بذار یه آهنگ از بیانسه", "Beyonce Halo"),
]


@pytest.mark.parametrize(("text", "model_plays"), TYPED_DECLINED_VALID)
async def test_r6_12_typed_a_declined_request_reaches_one_run_with_the_full_words(
    rig, monkeypatch, text, model_plays,
):
    assert media_request(text) is None, "precondition: the fast path declines"
    frames, agent, searches, agent_plays = await _typed_turn(
        rig, monkeypatch, text, lambda words, preset: model_plays,
    )
    # exactly one agent run, and it carries the caller's FULL original words
    assert len(agent.runs) == 1, agent.runs
    assert agent.runs[0]["user_message"] == text, agent.runs
    assert agent.runs[0]["preset_media"] is None
    # the fast path searched nothing and sent no media_play on the socket
    assert searches == [model_plays], searches
    assert not [f for f in frames if f.get("type") == "media_play"], frames
    # the agent's own play_media is the one play (no duplicate) …
    assert len(agent_plays) == 1 and agent.results and not agent.results[0].startswith("ERROR"), (
        agent_plays, agent.results)
    # … and the turn is answered: no silence
    done = [f for f in frames if f.get("type") == "done"]
    assert len(done) == 1 and done[0]["text"] == f"Starting {model_plays}.", frames


NO_PLAY_REQUESTS = [
    "play Halo, never mind",                 # retracted
    "آهنگ ابی رو پخش نکن",                  # negated
    "play anything but Halo",                # excluded
    "یه آهنگ غیر از ابی بذار",               # excluded
    "play it again",                         # back-reference
    "آهنگ من رو بذار",                      # back-reference (the caller's own)
    "Play Halo, I mean Hello.",              # repaired
]


@pytest.mark.parametrize("text", NO_PLAY_REQUESTS)
async def test_r6_12_typed_a_negated_or_back_reference_request_plays_nothing(rig, monkeypatch, text):
    """No fast-path play and no play the tenant starts on its own: with a
    model that does not play, nothing plays at all — and the agent still has
    the caller's full words (one run, answered)."""
    frames, agent, searches, agent_plays = await _typed_turn(
        rig, monkeypatch, text, lambda words, preset: None,
    )
    assert len(agent.runs) == 1 and agent.runs[0]["user_message"] == text, agent.runs
    assert searches == [] and agent_plays == [], (searches, agent_plays)
    assert not [f for f in frames if f.get("type") == "media_play"], frames
    assert [f.get("type") for f in frames].count("done") == 1, frames


async def test_r6_12_typed_control_a_fast_path_play_is_one_play_and_one_run(rig, monkeypatch):
    """«آهنگ ای ایران رو پخش کن» is not declined: the fast path plays the
    WHOLE title (TB2) once, the agent run is told it started and a model
    that obeys does not play again — one play, one run."""
    text = "آهنگ ای ایران رو پخش کن"
    assert media_request(text).query == "ای ایران"
    frames, agent, searches, agent_plays = await _typed_turn(
        rig, monkeypatch, text, lambda words, preset: None if preset else "ای ایران",
    )
    assert searches == ["ای ایران"], searches
    assert len([f for f in frames if f.get("type") == "media_play"]) == 1, frames
    assert agent_plays == [], agent_plays
    assert len(agent.runs) == 1 and agent.runs[0]["user_message"].startswith(text), agent.runs
    assert agent.runs[0]["preset_media"] and agent.runs[0]["display_user_message"] == text


# ══════════════════════════════════════════════════════════════════════════
# 10. R6-12 VOICE leg — the real relay (fx4 lvp, read-only), FakeProvider,
#     the real `_think` / `_play_media_direct` bodies, a recording tenant
# ══════════════════════════════════════════════════════════════════════════

async def _voice(monkeypatch, words, model_plays):
    import test_live_harness as H
    import test_live_round6_media as M6

    tenant = M6.Tenant(agents={"": {"think_s": 0.1, "play": model_plays, "search_s": 0.1}}
                       if model_plays else {})

    async def steps(p, client):
        p.push(H.user_delta(words, 1000, 1800))
        p.push(H.delegation("dR7", 1820))
        await M6._wait(lambda: tenant.runs() or tenant.plays(), 4.0)
        await M6._wait(lambda: M6.R4._terminal(client, "dR7"), 4.0)
        await asyncio.sleep(0.8)   # room for any fallback the relay might fire

    client, provider = await M6._call(monkeypatch, tenant, steps)
    return tenant, client


VOICE_DECLINED_VALID = [
    ("Play No Tears Left to Cry.", "No Tears Left to Cry Ariana Grande"),
    ("Play anything but Halo.", "Crazy in Love"),
    ("Play my playlist.", "Road Trip Mix"),
    ("Play Thank U, Next.", "thank u, next Ariana Grande"),
    ("بعدا یه آهنگ از ابی بذار", "Ebi Khalij"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("words", "model_plays"), VOICE_DECLINED_VALID)
async def test_r6_12_voice_a_declined_request_is_one_delegation_with_the_full_words(
    monkeypatch, words, model_plays,
):
    assert media_request(words) is None, "precondition: the fast path declines"
    tenant, client = await _voice(monkeypatch, words, model_plays)
    # no fast-path play (and no relay fallback play): not one /internal/play-media
    assert tenant.plays() == [], tenant.bodies
    # exactly one agent turn, carrying the caller's full words
    runs = tenant.runs()
    assert len(runs) == 1, tenant.bodies
    assert words in str(runs[0].get("message") or ""), runs[0]
    assert str(runs[0].get("display_request") or "") == words, runs[0]
    # the agent's play is the one play; the delegation completed (no silence)
    assert [e for e in tenant.events if e[0] == "media_play"] == [("media_play", model_plays)], tenant.events
    assert "completed" in [f.get("phase") for f in client.of("delegation")
                           if f.get("delegation_id") == "dR7"]


@pytest.mark.asyncio
@pytest.mark.parametrize("words", [
    "Play Halo, never mind.",
    "آهنگ ابی رو پخش نکن",
    "Play anything but Halo.",
    "Play it again.",
    "آهنگ من رو بذار",
])
async def test_r6_12_voice_a_negated_or_back_reference_request_plays_nothing(monkeypatch, words):
    tenant, client = await _voice(monkeypatch, words, None)
    assert tenant.plays() == [], tenant.bodies
    assert not [e for e in tenant.events if e[0] == "media_play"], tenant.events
    runs = tenant.runs()
    assert len(runs) == 1 and words in str(runs[0].get("message") or ""), tenant.bodies


@pytest.mark.asyncio
async def test_r6_12_voice_control_the_fast_path_plays_the_whole_title_once(monkeypatch):
    words = "آهنگ ای ایران رو پخش کن"
    tenant, client = await _voice(monkeypatch, words, "should not run")
    assert [b.get("query") for b in tenant.plays()] == ["ای ایران"], tenant.bodies
    assert tenant.runs() == [], tenant.bodies
    assert [e for e in tenant.events if e[0] == "media_play"] == [("media_play", "ای ایران")]


@pytest.mark.asyncio
@pytest.mark.parametrize("fragments", [
    # The final bubble from the owner's September 24 voice recording, delivered
    # as separate ASR deltas rather than one conveniently prejoined sentence.
    ("برنامه... ", "یه... ", "یه آهنگ... ", "آهنگ دری... ", "دریک پ... ", "پلی کن"),
    ("اوکی بعد به چیز دیگه میخوام ", "برام یه دونه آهنگ دریکو پلی کن"),
])
async def test_observed_drake_request_reaches_one_play_and_speaks_resolved_title(
    monkeypatch, fragments,
):
    import test_live_harness as H
    import test_live_round6_media as M6

    # If the fast path declines, the scripted agent gives the false answer
    # from the recording. The test must prove that the agent is never invoked.
    tenant = M6.Tenant(agents={"": {"text": "I do not have a music service."}})
    resolved_title = "Drake - God's Plan"

    async def play_media(body):
        tenant.seq += 1
        tenant.events.append(("play_requested", str(body.get("query") or "")))
        media = tenant.broadcast(resolved_title, tenant.key(body, "media_order"))
        return {"ok": True, "title": media["title"], "video_id": media["video_id"]}

    tenant.play_media = play_media

    async def steps(provider, client):
        offset = 1000
        for fragment in fragments:
            provider.push(H.user_delta(fragment, offset, offset + 120))
            offset += 150
        provider.push(H.delegation("drake-play", offset + 50))
        await M6._wait(lambda: M6.R4._terminal(client, "drake-play"), 5.0)
        await asyncio.sleep(0.25)  # A late fallback must not start an agent run.

    client, provider = await M6._call(monkeypatch, tenant, steps)

    assert [(b.get("query"), b.get("variety")) for b in tenant.plays()] == [
        ("Drake", True),
    ], tenant.bodies
    assert tenant.runs() == [], tenant.bodies
    assert [e for e in tenant.events if e[0] == "media_play"] == [
        ("media_play", resolved_title),
    ]
    assert [f.get("outcome") for f in client.of("tool_call.completed")
            if f.get("name") == "play_media"] == ["ok"]
    assert M6.R4._phases(client, "drake-play")[-1] == "completed"
    spoken = " ".join(M6._said(provider))
    assert resolved_title in spoken
    assert "music service" not in spoken
