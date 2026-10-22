"""PR-14: a task's card is titled with a short summary (the 2026-10-02 call, D2).

Findings (FINDINGS-REPORT §2 P7, §5 PR-14; cluster report p7p8p10-cards §4 D2):
GPT-Live's `session.delegation.created` carries no text (the relay reads only
its id, target and offset), so every title surface was built from the caller's
raw words — job 1's card read "Everything is good. Can you search Who is most
experienced officer…", and "retry" read "Task", then "That last one".  Decision
D2: a job's title is a short task summary (at most six words) produced at
delegation time, never the raw utterance; the utterance may stay as secondary
text.

The relay now asks ONE small model for that title (`summarize_task_text`, the
seam the harness stubs) — concurrently with the dispatch and never awaited by
it — and delivers it:

  * to a client that lists `task_summary`: `summary` on its lifecycle
    `delegation` frames once known, a dedicated `task_summary` frame when the
    title lands between two observations, and `ready.capabilities.task_summary`;
  * to every client's saved row, as `voice.task_summary` (the day chat is read
    later, by whatever build the user has then — AF6);
  * never in `delegation.title`, which keeps PR-10's stripped request for every
    client, so build 138 / TF132 / web are byte for byte unchanged.

Driven through the real relay (`test_live_harness`) with provider-shaped
events; the pure seams (`task_summary_instructions`, `task_summary_request`,
`clean_task_summary`) are called directly.
"""

import asyncio
import logging
import time

import pytest

import test_live_harness as H
from app.config import settings
from app.services import live_voice_protocol as live
from app.services.live_voice_protocol import (
    DelegatedTask,
    clean_task_summary,
    display_request_text,
    request_title,
    task_summary_instructions,
    task_summary_request,
)


REQUEST = (
    "Everything is good. Can you search who is the most experienced professor "
    "in LLM at U of T"
)
#: PR-10's stripped request: the deterministic title, and the fallback.
TITLE = request_title(display_request_text(REQUEST))
SUMMARY = "Find U of T LLM professors"
#: Voice round 2026-10-07 (D-R4 (5)): the deterministic floor the card wears
#: from its FIRST frame until the model's title replaces it.
FALLBACK = live.fallback_task_title(DelegatedTask(delegation_id="x", offset_ms=0, transcript=REQUEST))
ANSWER = "Professor Jane Doe leads the LLM group at U of T."
NEGOTIATED = H.config()["features"] + ["task_summary"]
OLDER = H.config()["features"]
LOGGER = "app.services.live_voice_protocol"


class StampedClient(H.FakeClient):
    """The phone, with WHEN each frame arrived."""

    def __init__(self, script=None):
        super().__init__(script)
        self.at: list[float] = []

    async def send_json(self, frame):
        self.at.append(time.monotonic())
        await super().send_json(frame)


def _frames_for(client, did="d1"):
    return [f for f in client.of("delegation") if f.get("delegation_id") == did]


def _terminal(client, did="d1"):
    return any(
        f.get("phase") in {"completed", "failed", "cancelled", "superseded", "expired"}
        for f in _frames_for(client, did)
    )


def _rows_for(recorded, did="d1"):
    return [
        s for s in recorded["saves"]
        if str(s.get("assistant_ref") or "").endswith(f":{did}")
        and str(s.get("assistant_ref") or "").startswith("live-delegation:")
    ]


async def _wait(pred, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def _summarizer(answer=SUMMARY, *, delay=0.0, calls=None, raises=None):
    """A stub for the summarizer's one model call, recording what it was asked."""

    async def summarize(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        if delay:
            await asyncio.sleep(delay)
        if raises is not None:
            raise raises
        return answer

    return summarize


async def _call(
    monkeypatch, *, features, summarize, db, think_delay=0.0, budget=1.5,
    enabled=True, request=REQUEST, until=None, think=None,
):
    """One request, delegated as d1; the phone waits for `until(client,
    recorded)` (default: d1's terminal frame), then hangs up.  `events` is
    when the agent call and the summarizer call were each ENTERED."""

    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=3.0,
        voice_live_task_summary_timeout_s=budget,
        voice_live_task_summary_enabled=enabled,
    )
    events: list[tuple] = []

    async def default_think(user_id, task, session_id, relay=None, out=None, **kw):
        events.append(("think", time.monotonic()))
        if think_delay:
            await asyncio.sleep(think_delay)
        return ANSWER, "m"

    async def recording_summarize(**kwargs):
        events.append(("summarize", time.monotonic()))
        return await summarize(**kwargs)

    recorded = H.patch_relay(monkeypatch, think=think or default_think, summarize=recording_summarize)

    def on_send(p, e):
        if e["type"] == "session.start":
            p.push(H.user_delta(request, 0, 2400))
            p.push(H.delegation("d1", 2450))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def settle():
        await _wait(lambda: until(client, recorded) if until else _terminal(client))
        # Let a row revision or a late frame land before the hang-up.
        await asyncio.sleep(0.3)

    client = StampedClient([H.config(features=list(features)), settle, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=15, db_session_id=db)
    return client, recorded, events


# ══════════════════════════════════════════════════════════════════════
# the wire: negotiated, additive, `title` untouched
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.parametrize("negotiated", [True, False], ids=["task_summary", "older_client"])
async def test_the_summary_reaches_only_a_client_that_negotiated_it(monkeypatch, negotiated):
    """The title lands while the task runs: a negotiated client gets the
    `task_summary` frame, then `summary` on every later lifecycle frame; an
    older client gets none of it.  `title` is PR-10's stripped request for
    both, on every frame."""

    calls: list[dict] = []
    client, recorded, _events = await _call(
        monkeypatch, features=NEGOTIATED if negotiated else OLDER,
        summarize=_summarizer(calls=calls), think_delay=0.4,
        db=f"db-summary-wire-{negotiated}",
    )

    assert len(calls) == 1, calls
    frames = _frames_for(client)
    assert frames and _terminal(client), client.phases()
    assert all(f["title"] == TITLE for f in frames), [f["title"] for f in frames]
    ready = client.of("ready")[0]["capabilities"]
    if negotiated:
        assert client.of("task_summary") == [
            {"type": "task_summary", "delegation_id": "d1", "summary": SUMMARY,
             "summary_source": "model"},
        ]
        assert ready.get("task_summary") is True
        terminal = [f for f in frames if f.get("phase") == "completed" and "task_revision" in f]
        assert terminal and terminal[-1]["summary"] == SUMMARY, terminal
        assert terminal[-1]["summary_source"] == "model"
        # 2026-10-07 (D-R4 (5)): the created frame went out before the model's
        # title existed — it wears the deterministic fallback, never the raw
        # sentence, and says so.
        first = frames[0]
        assert first["phase"] == "created", first
        assert FALLBACK and FALLBACK != TITLE
        assert first["summary"] == FALLBACK and first["summary_source"] == "fallback", first
    else:
        assert client.of("task_summary") == []
        assert all("summary" not in f for f in client.frames), [
            f for f in client.frames if "summary" in f
        ]
        assert "task_summary" not in ready
    # Persistence is not negotiated: both calls name the saved row.
    rows = _rows_for(recorded)
    assert rows and rows[-1]["assistant_voice"]["task_summary"] == SUMMARY, rows
    assert rows[-1]["assistant_voice"]["task_title"] == TITLE


def _comparable(client):
    """Every frame the feature could touch, minus the provider clock and the
    per-run DB session id."""

    return [
        {k: v for k, v in f.items() if k not in {"updated_ms", "session_id"}}
        for f in client.frames if f.get("type") in {"delegation", "ready", "task_summary"}
    ]


@pytest.mark.asyncio
async def test_an_older_client_gets_byte_identical_frames_whatever_the_summarizer_says(monkeypatch):
    """Build 138 / TF132 / web: the summarizer answering, failing or switched
    off changes none of their frames — `ready` and every `delegation` frame
    are the same dicts (the provider clock aside)."""

    answered, _r1, _e1 = await _call(
        monkeypatch, features=OLDER, summarize=_summarizer(), think_delay=0.3,
        db="db-summary-older-answered",
    )
    silent, _r2, _e2 = await _call(
        monkeypatch, features=OLDER, summarize=_summarizer(answer=None), think_delay=0.3,
        db="db-summary-older-silent",
    )
    off, _r3, _e3 = await _call(
        monkeypatch, features=OLDER, summarize=_summarizer(), think_delay=0.3,
        enabled=False, db="db-summary-older-off",
    )
    assert _comparable(answered) == _comparable(silent) == _comparable(off)
    assert not answered.of("task_summary")


@pytest.mark.asyncio
async def test_the_kill_switch_makes_no_call_and_advertises_nothing(monkeypatch):
    calls: list[dict] = []
    client, recorded, _events = await _call(
        monkeypatch, features=NEGOTIATED, summarize=_summarizer(calls=calls),
        enabled=False, db="db-summary-off",
    )
    assert calls == []
    assert "task_summary" not in client.of("ready")[0]["capabilities"]
    assert client.of("task_summary") == []
    assert all("summary" not in f for f in _frames_for(client))
    assert all("task_summary" not in r["assistant_voice"] for r in _rows_for(recorded))


# ══════════════════════════════════════════════════════════════════════
# the budget: concurrent, bounded, never on the dispatch path
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_a_timeout_yields_no_summary_and_never_delays_dispatch(monkeypatch, caplog):
    """The summarizer is SLOWER than its budget.  The agent call is entered
    before the summarizer is even asked, the task ends without waiting for
    it, and the late title is dropped — no frame, no `summary` key, nothing on
    the row — and only logged, with how late it was (the measurement the
    budget is set from)."""

    caplog.set_level(logging.INFO, logger=LOGGER)
    budget = 0.4
    client, recorded, events = await _call(
        monkeypatch, features=NEGOTIATED,
        summarize=_summarizer(delay=1.2), budget=budget,
        db="db-summary-timeout",
    )
    order = [name for name, _at in events]
    assert order[:2] == ["think", "summarize"], order
    think_at = dict(events)["think"]
    summarize_at = dict(events)["summarize"]
    # The task ended well before the summarizer's budget ran out: nothing on
    # its path waited for a title.
    done_at = next(
        at for at, f in zip(client.at, client.frames)
        if f.get("type") == "delegation" and f.get("phase") == "completed"
    )
    assert done_at - think_at < budget, (done_at - think_at)
    assert done_at < summarize_at + budget

    # 2026-10-07 (D-R4 (5)): no model title, so the card keeps the fallback it
    # wore from its first frame — on every lifecycle frame and on the row.
    assert client.of("task_summary") == []
    lifecycle = [f for f in _frames_for(client) if "task_revision" in f]
    assert lifecycle and all(
        f.get("summary") == FALLBACK and f.get("summary_source") == "fallback" for f in lifecycle
    ), lifecycle
    rows = _rows_for(recorded)
    assert rows and all(r["assistant_voice"].get("task_summary") == FALLBACK for r in rows), rows
    assert all(r["assistant_voice"]["task_title"] == TITLE for r in rows)

    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert any(
        m.startswith("[LIVE] task summary id=d1 outcome=timeout") and "budget_ms=400" in m
        for m in messages
    ), messages
    assert any(m.startswith("[LIVE] task summary late id=d1 outcome=answered") for m in messages), messages
    # Counts and latency only — never the words.
    assert not any(SUMMARY in m or "professor" in m.lower() for m in messages if "task summary" in m)


@pytest.mark.asyncio
async def test_a_late_summary_arrives_on_its_own_frame_after_the_task_ended(monkeypatch):
    """The agent answers first; the title lands after the terminal frame.  It
    comes on the dedicated `task_summary` frame — no new `delegation`
    observation, no new revision, no state — and the saved row is revised in
    place with it (same ref, next revision)."""

    client, recorded, _events = await _call(
        monkeypatch, features=NEGOTIATED, summarize=_summarizer(delay=0.35),
        db="db-summary-late",
        until=lambda client, recorded: bool(client.of("task_summary")) and any(
            r["assistant_voice"].get("task_summary") for r in _rows_for(recorded)
        ),
    )

    kinds = [
        ("summary" if f.get("type") == "task_summary" else f.get("phase"))
        for f in client.frames
        if f.get("type") == "task_summary"
        or (f.get("type") == "delegation" and f.get("delegation_id") == "d1")
    ]
    assert "completed" in kinds and kinds.index("summary") > kinds.index("completed"), kinds
    summary_at = next(i for i, f in enumerate(client.frames) if f.get("type") == "task_summary")
    terminal = [f for f in _frames_for(client) if f.get("phase") == "completed" and "task_revision" in f]
    assert terminal and terminal[0].get("summary") == FALLBACK, terminal
    assert terminal[0].get("summary_source") == "fallback"
    later = [
        f for f in client.frames[summary_at + 1:]
        if f.get("type") == "delegation" and f.get("delegation_id") == "d1" and "task_revision" in f
    ]
    assert later == [], later
    assert client.of("task_summary") == [
        {"type": "task_summary", "delegation_id": "d1", "summary": SUMMARY,
         "summary_source": "model"},
    ]

    rows = _rows_for(recorded)
    assert rows, recorded["saves"]
    assert rows[0]["assistant_voice"].get("task_summary") == FALLBACK, rows[0]
    revised = [r for r in rows if r["assistant_voice"].get("task_summary") == SUMMARY]
    assert revised, rows
    first = revised[0]
    assert first["assistant_ref"] == rows[0]["assistant_ref"]
    assert first["assistant_revision"] > (rows[0].get("assistant_revision") or 1)
    assert "assistant_occurred_at" not in first, "the title moved the row"
    # Every revision after it keeps it (the tenant REPLACES `voice`).
    assert all(r["assistant_voice"].get("task_summary") == SUMMARY for r in rows[rows.index(first):])


@pytest.mark.asyncio
async def test_a_title_never_writes_a_row_nothing_else_names(monkeypatch):
    """A task that failed with nothing to show keeps only a LATENT record row
    (C10: written only if a relation later names it).  A title that lands
    after that failure is sent to the card and kept for the latent row — it
    never writes the row itself, so no "That didn't go through" record
    appears in the day chat because of a title."""

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        raise RuntimeError("agent container gone")

    client, recorded, _events = await _call(
        monkeypatch, features=NEGOTIATED, summarize=_summarizer(delay=0.3), think=think,
        db="db-summary-latent",
        until=lambda client, _recorded: _terminal(client) and bool(client.of("task_summary")),
    )
    frames = _frames_for(client)
    assert frames and frames[-1]["phase"] == "failed", client.phases()
    assert client.of("task_summary") == [
        {"type": "task_summary", "delegation_id": "d1", "summary": SUMMARY,
         "summary_source": "model"},
    ]
    assert _rows_for(recorded) == [], _rows_for(recorded)


@pytest.mark.asyncio
async def test_delivering_a_title_moves_no_media_ledger_transition(monkeypatch):
    """The dedicated frame is modelled on `send_relation_update`, NOT on
    `send_delegation`, whose first act is a media-ledger transition: a title
    is learned ABOUT a task, it is no transition of it.  The ledger sees the
    same transitions, in the same order, whether a title arrives (mid-run and
    after the end) or not."""

    real = live._LiveSession.media_transition

    async def run(summarize, db, until=None, think_delay=0.0):
        seen: list[tuple[str, str]] = []

        def spy(self, task, event, **info):
            seen.append((task.delegation_id, event))
            return real(self, task, event, **info)

        monkeypatch.setattr(live._LiveSession, "media_transition", spy)
        client, _recorded, _events = await _call(
            monkeypatch, features=NEGOTIATED, summarize=summarize, db=db,
            until=until, think_delay=think_delay,
        )
        return client, seen

    plain, base = await run(_summarizer(answer=None), "db-summary-ledger-none")
    late, after = await run(
        _summarizer(delay=0.35), "db-summary-ledger-late",
        until=lambda client, _recorded: bool(client.of("task_summary")),
    )
    mid, during = await run(_summarizer(), "db-summary-ledger-mid", think_delay=0.4)
    assert not plain.of("task_summary") and late.of("task_summary") and mid.of("task_summary")
    assert base and after == base and during == base, {"none": base, "late": after, "mid": during}


@pytest.mark.asyncio
@pytest.mark.parametrize("tag, answer, raises, outcome", [
    ("seven_words", "Find the most experienced LLM professor at the University of Toronto", None, "invalid"),
    ("two_lines", "Find U of T\nLLM professors", None, "invalid"),
    ("wrong_language", "استادهای LLM دانشگاه تورنتو", None, "invalid"),
    ("none", None, None, "no_answer"),
    ("empty", "", None, "no_answer"),
    ("raises", None, RuntimeError("provider down"), "error"),
], ids=["seven_words", "two_lines", "wrong_language", "none", "empty", "raises"])
async def test_an_answer_that_is_not_a_title_keeps_the_fallback(
    monkeypatch, caplog, tag, answer, raises, outcome,
):
    """Re-baselined 2026-10-07 (D-R4 (5)): no model title keeps the
    deterministic fallback the card wore from its first frame (it used to
    keep the raw request).  An INVALID answer is asked for once more, inside
    the budget, with its reason named; a timeout, an error or no answer is
    not retried."""

    caplog.set_level(logging.INFO, logger=LOGGER)
    calls: list[dict] = []
    client, recorded, _events = await _call(
        monkeypatch, features=NEGOTIATED,
        summarize=_summarizer(answer, raises=raises, calls=calls),
        db=f"db-summary-bad-{tag}",
    )
    assert client.of("task_summary") == []
    lifecycle = [f for f in _frames_for(client) if "task_revision" in f]
    assert all(f["title"] == TITLE for f in _frames_for(client))
    assert lifecycle and all(
        f.get("summary") == FALLBACK and f.get("summary_source") == "fallback" for f in lifecycle
    ), lifecycle
    assert all(r["assistant_voice"].get("task_summary") == FALLBACK for r in _rows_for(recorded))
    assert len(calls) == (2 if outcome == "invalid" else 1), len(calls)
    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert any(m.startswith(f"[LIVE] task summary id=d1 outcome={outcome} ") for m in messages), messages
    if outcome == "invalid":
        assert any(
            m.startswith("[LIVE] task summary id=d1 outcome=invalid reason=") and "attempt=2" in m
            for m in messages
        ), messages


# ══════════════════════════════════════════════════════════════════════
# the summarizer's instruction and input
# ══════════════════════════════════════════════════════════════════════


def test_the_instruction_pins_six_words_and_the_callers_language():
    for lang in ("en", "fa"):
        text = task_summary_instructions(lang)
        assert f"At most {live.TASK_SUMMARY_MAX_WORDS} words" in text, text
        assert live.TASK_SUMMARY_MAX_WORDS == 6
        assert "Do not answer the request" in text
        assert "one line" in text
    assert "(Persian/Farsi)" in task_summary_instructions("fa")
    assert "Persian" not in task_summary_instructions("en")
    assert "the language of their own words" in task_summary_instructions("en")


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text, lang", [
    (REQUEST, "en"),
    ("می‌خوای استادهای هوش مصنوعی دانشگاه تورنتو رو پیدا کنی؟", "fa"),
], ids=["english", "persian"])
async def test_the_relay_asks_for_six_words_in_the_language_the_caller_spoke(
    monkeypatch, request_text, lang,
):
    calls: list[dict] = []
    answer = SUMMARY if lang == "en" else "استادهای هوش مصنوعی تورنتو"
    client, _recorded, _events = await _call(
        monkeypatch, features=NEGOTIATED, summarize=_summarizer(answer, calls=calls),
        request=request_text, db=f"db-summary-lang-{lang}",
        until=lambda client, _recorded: _terminal(client) and bool(client.of("task_summary")),
    )
    (call,) = calls
    assert call["instructions"] == task_summary_instructions(lang)
    assert call["model"] == settings.voice_live_task_summary_model == "gpt-4o-mini"
    assert call["timeout_s"] == 1.5
    assert call["user_id"] == "user-1"
    # A plain request: exactly the caller's words, as the agent gets them.
    assert call["request"] == request_text.strip()
    assert client.of("task_summary")[0]["summary"] == answer


def test_a_retry_is_summarized_from_the_request_it_resolves_to():
    """PR-10's resolved request (a correction applied, the failed task's whole
    chain) is what the summarizer names — the caller's "retry" alone names
    nothing — and none of what is not this task reaches it."""

    officer = "find the most experienced officer in the llm program at u of t"
    correction = "not officer, professor"
    resolved = "find the most experienced professor in the llm program at u of t"
    task = DelegatedTask(delegation_id="d3", offset_ms=0, transcript="retry", intent="retry")
    task.relation_kind, task.related_task_id = "retries", "d2"
    task.related_request = officer + "\n" + correction
    task.resolved_request = resolved
    task.title_request = resolved
    text = task_summary_request(task)
    assert "resolved request: «" + resolved + "»" in text, text
    assert officer in text and correction in text
    assert text.endswith("Current caller request: retry"), text[-120:]
    for absent in (
        "Follow-up rules", "Reply in the caller's language", "Accepted backend result",
        # Voice round 2026-10-06 (D-6a): nor the new-question rule.
        "The current request is a NEW question",
        # Voice round 2026-10-07 (T10): nor the scope sentences.
        "Answer ONLY the current request",
    ):
        assert absent not in text, absent
    # The scope sentences ride only the CONTEXT block, so prove their absence
    # on a task that carries one (running and previous alike).
    for kind in ("running", "previous"):
        ctx = DelegatedTask(delegation_id=f"d-ctx-{kind}", offset_ms=0,
                            transcript="what about the academic one", intent="new")
        ctx.context_request = "search the IELTS general reading test"
        ctx.context_kind = kind
        ctx_text = task_summary_request(ctx)
        assert "search the IELTS general reading test" in ctx_text
        for absent in ("Answer ONLY the current request",
                       "That request is being answered separately",
                       "That request was already answered"):
            assert absent not in ctx_text, (kind, absent)
    # …and the language follows the words the card names, not the bare "retry".
    persian = DelegatedTask(delegation_id="d4", offset_ms=0, transcript="retry", intent="retry")
    persian.title_request = "استاد هوش مصنوعی دانشگاه تورنتو رو پیدا کن"
    assert live.task_summary_language(persian) == "fa"


def test_clean_task_summary_keeps_titles_and_refuses_everything_else():
    keep = {
        '"Find U of T LLM professors."': "Find U of T LLM professors",
        "Title: Find a dentist downtown": "Find a dentist downtown",
        "  Toronto   weather  ": "Toronto weather",
        "Who leads U of T AI?": "Who leads U of T AI?",
        "“Book a table for two”": "Book a table for two",
    }
    for raw, want in keep.items():
        assert clean_task_summary(raw, "en") == want, raw
    assert clean_task_summary("«هوای تورنتو»", "fa") == "هوای تورنتو"
    assert clean_task_summary("استادهای LLM دانشگاه تورنتو", "fa") == "استادهای LLM دانشگاه تورنتو"
    refuse = [
        "one two three four five six seven",          # past six words: refused, not cut
        "x" * 61,                                      # past the character bound
        "Find U of T\nLLM professors",                 # two lines
        "هوای تورنتو",                                 # wrong script for an English caller
        "", "   ", None, 42, ["Find"],
        "Live-session context from earlier accepted delegations follows",
        "Find\x00professors",
    ]
    for raw in refuse:
        assert clean_task_summary(raw, "en") == "", raw
    assert clean_task_summary("Toronto weather", "fa") == "", "an English title for a Persian caller"
    # D-R5 (2026-10-06 call): the card read «ها» — the summarizer echoing a
    # one-word filler is not a title, and neither is a one-word request said
    # back.  A one-word title of a LONGER request still is one.
    assert clean_task_summary("ها", "fa") == ""
    assert clean_task_summary("«ها»", "fa", "ها") == ""
    assert clean_task_summary("okay", "en") == ""
    assert clean_task_summary("فورمولوان", "fa", "فورمولوان") == ""
    assert clean_task_summary("Weather", "en", "weather") == ""
    assert clean_task_summary("Weather", "en", "what's the weather in toronto") == "Weather"
    assert clean_task_summary("فرمول‌یک", "fa", "فورمولوان") == "فرمول‌یک"


# ══════════════════════════════════════════════════════════════════════
# a retry wears the title of the task it re-runs
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_a_retry_wears_the_title_of_the_task_it_reruns(monkeypatch):
    """d1's title lands, then d1 fails; "retry" links d2 to it (PR-10).  d2's
    card wears d1's title from its FIRST frame — no second summarizer call —
    and its saved row carries it."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=3.0, voice_live_task_summary_timeout_s=1.5,
    )
    calls: list[dict] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        if kw.get("delegation_id") == "d1":
            await asyncio.sleep(0.3)
            raise RuntimeError("agent container gone")
        return ANSWER, "m"

    recorded = H.patch_relay(monkeypatch, think=think, summarize=_summarizer(calls=calls))

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(REQUEST, 0, 2400))
            p.push(H.delegation("d1", 2450))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        # Past the failure line's speak budget, so its claim is withdrawn.
        await asyncio.sleep(0.7)
        box["p"].push(H.user_delta("retry", 15000, 15400))
        box["p"].push(H.delegation("d2", 15450))
        await _wait(lambda: _terminal(client, "d2"))
        await asyncio.sleep(0.3)

    features = NEGOTIATED + ["retry_relation"]
    client = StampedClient([H.config(features=features), script, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=20, db_session_id="db-summary-retry")

    assert len(calls) == 1, [c["request"][-40:] for c in calls]
    d2 = [f for f in _frames_for(client, "d2") if "task_revision" in f]
    assert d2 and d2[0]["phase"] == "created", d2
    assert all(f.get("summary") == SUMMARY for f in d2), d2
    assert all(f.get("relation") == {"kind": "retries", "task_id": "d1"} for f in d2), d2
    # The retry's own title is the failed request (PR-10), not "retry".
    assert all(f["title"] == TITLE for f in d2), [f["title"] for f in d2]
    rows = _rows_for(recorded, "d2")
    assert rows and rows[-1]["assistant_voice"]["task_summary"] == SUMMARY, rows


# ══════════════════════════════════════════════════════════════════════
# the seam to the tenant
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_the_saved_title_passes_the_tenant_allowlist(monkeypatch):
    """The relay's actual row, handed to the tenant's `_clean_voice`: the title
    survives exactly as sent (the tenant half — the write route and every read
    surface — is pinned in tests/test_voice_record_metadata.py, agent lane)."""

    from app.api.sessions import _VOICE_KEYS, _VOICE_TEXT_KEYS, _clean_voice

    assert "task_summary" in _VOICE_KEYS and "task_summary" in _VOICE_TEXT_KEYS
    _client, recorded, _events = await _call(
        monkeypatch, features=NEGOTIATED, summarize=_summarizer(), think_delay=0.3,
        db="db-summary-allowlist",
    )
    rows = _rows_for(recorded)
    assert rows
    for row in rows:
        voice = row["assistant_voice"]
        assert voice["task_summary"] == SUMMARY, voice
        assert _clean_voice(dict(voice))["task_summary"] == SUMMARY



# ══════════════════════════════════════════════════════════════════════
# Voice round 2026-10-07 (D-R4 (5)): a title from the first frame
# ══════════════════════════════════════════════════════════════════════
# Card 2 «اوکی. بعد به نظرت همین توی آیسآکادمیک چجوریه» wore the raw sentence
# for its whole life: the summarizer answered (15 tokens) and the answer was
# refused, with no reason logged and no second chance.

CARD_2 = "اوکی. بعد به نظرت همین توی آیسآکادمیک چجوریه"


@pytest.mark.parametrize(("raw", "lang", "asked", "reason"), [
    ("Find U of T LLM professors", "en", "", "ok"),
    (None, "en", "", "empty"),
    ("   ", "en", "", "empty"),
    ("Find U of T\nLLM professors", "en", "", "lines"),
    ("Find\x00professors", "en", "", "control"),
    ("Live-session context from earlier accepted delegations follows", "en", "", "scaffold"),
    ("one two three four five six seven", "en", "", "words"),
    ("x" * 61, "en", "", "chars"),
    ("IELTS Academic Reading", "fa", CARD_2, "lang"),
    ("ها", "fa", "ها", "one_word"),
])
def test_a_refused_title_names_its_reason(raw, lang, asked, reason):
    title, why = live.task_summary_verdict(raw, lang, asked)
    assert why == reason, (raw, why)
    assert bool(title) is (reason == "ok")
    assert live.clean_task_summary(raw, lang, asked) == title
    if reason != "ok":
        assert reason in live.TASK_SUMMARY_REASON_TEXT


@pytest.mark.parametrize(("request_text", "want"), [
    (CARD_2, "آیسآکادمیک"),                            # one word of ≥ 4 letters
    ("کیه", ""),                                         # nothing that names
    ("Toronto weather", ""),                             # IS the display text
    ("نه یه آهنگی برام پخش کن", ""),                     # a bare song noun names no task
    ("حالا یه دنه آهنگ دریک واسم پخش کن", "آهنگ دریک"),
    ("find me a dentist near union station", "dentist near union station"),
    # Phase D (search-play-title-7): a negator or "near me" is the question,
    # never dropped — but a LEADING «نه، …» / "no, …" is a correction.
    ("look up flights to paris that don't stop in london", "flights paris don't stop london"),
    ("find coffee shops near me that are not chains", "coffee shops near me not chains"),
    ("no, search flights to rome", "flights rome"),
])
def test_the_fallback_title_is_never_the_raw_sentence(request_text, want):
    task = DelegatedTask(delegation_id="x", offset_ms=0, transcript=request_text)
    title = live.fallback_task_title(task)
    assert title == want, title
    if title:
        assert title != display_request_text(request_text)
        assert len(title.split()) <= live.TASK_SUMMARY_MAX_WORDS


@pytest.mark.parametrize(("request_text", "want"), [
    ("search which phones have no headphone jack", "phones no headphone jack"),
    ("search for restaurants that aren't expensive", "restaurants aren't expensive"),
    ("سرچ کن کدوم داروخانه نزدیک من بازه", "داروخانه نزدیک من بازه"),
    ("سرچ کن ببین کدوم گوشی‌ها جک هدفون ندارن", "گوشی‌ها جک هدفون ندارن"),
    ("find me a dentist near union station", "dentist near union station"),
    ("نه، منظورم قیمت دلار بود سرچ کن", "منظورم قیمت دلار"),
    (CARD_2, ""),   # one word: the tenant searches its own cleaning (search-play-title-1)
])
def test_the_search_query_keeps_what_inverts_or_anchors_the_question(request_text, want):
    """Phase D (search-play-title-7): the relay's `search_query` WINS over the
    display text on the tenant (T12), and the cleaner dropped every negator
    and "me" — 'phones have no headphone jack' searched 'phones headphone
    jack', the opposite question, injected into round 0 as "already this
    turn's search"."""

    task = DelegatedTask(delegation_id="x", offset_ms=0, transcript=request_text)
    assert live.task_search_query(task) == want


@pytest.mark.asyncio
async def test_an_invalid_title_is_retried_once_with_its_reason(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    calls: list[dict] = []
    answers = iter([
        "Find the most experienced LLM professor at the University of Toronto",
        SUMMARY,
    ])

    async def summarize(**kwargs):
        calls.append(kwargs)
        return next(answers)

    client, recorded, _events = await _call(
        monkeypatch, features=NEGOTIATED, summarize=summarize, think_delay=0.4,
        db="db-summary-retry-once",
    )
    assert len(calls) == 2, calls
    assert "could not be used: it was longer than 6 words" in calls[1]["instructions"]
    assert calls[0]["instructions"] == task_summary_instructions("en")
    assert client.of("task_summary") == [
        {"type": "task_summary", "delegation_id": "d1", "summary": SUMMARY,
         "summary_source": "model"},
    ]
    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert any(
        m.startswith("[LIVE] task summary id=d1 outcome=invalid reason=words attempt=1")
        for m in messages
    ), messages
    assert any(
        m.startswith("[LIVE] task summary id=d1 outcome=ok ") and "attempt=2" in m
        for m in messages
    ), messages


@pytest.mark.asyncio
async def test_the_first_frame_wears_the_fallback_for_card_2(monkeypatch):
    client, _recorded, _events = await _call(
        monkeypatch, features=NEGOTIATED, summarize=_summarizer(answer=None),
        request=CARD_2, db="db-summary-card2",
    )
    created = [f for f in _frames_for(client) if f.get("phase") == "created"]
    assert created and created[0]["summary"] == "آیسآکادمیک", created
    assert created[0]["summary_source"] == "fallback"
    assert created[0]["title"] != created[0]["summary"]


@pytest.mark.asyncio
async def test_the_body_task_title_is_the_fallback_not_the_raw_request(monkeypatch):
    seen: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        seen.update(kw)
        return ANSWER, "m"

    await _call(
        monkeypatch, features=NEGOTIATED, summarize=_summarizer(answer=None), think=think,
        db="db-summary-body-title",
    )
    assert seen.get("task_title") == FALLBACK, seen.get("task_title")
    assert seen.get("task_title") != TITLE


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 404], ids=["accepted", "older_image"])
async def test_the_model_title_is_forwarded_to_the_tenant(monkeypatch, caplog, status):
    """The tenant's job was named at dispatch from the fallback; the model's
    title follows it on the addendum route with an EMPTY text (D-R3 (3)),
    once, idempotent on `title:<task>`; an image without the route answers
    404, logged `unsupported`."""

    from app.api import ws_realtime as rt

    caplog.set_level(logging.INFO, logger=LOGGER)
    posts: list[tuple[str, dict]] = []

    async def vps(_user_id):
        return ("http://tenant.invalid", "key")

    async def vps_api(url, key, method, path, params=None, json_body=None, timeout=15.0,
                      outcome=None):
        if path.endswith("/voice-delegations/status"):
            return {"tasks": {}}
        posts.append((path, dict(json_body or {})))
        if not path.endswith("/voice-delegations/addendum"):
            return None
        if status == 404:
            if outcome is not None:
                outcome.update(status=404, failure="permanent")
            return None
        return {"accepted": True, "state": "queued"}

    real_patch = H.patch_relay

    def patch_with_tenant(monkeypatch_, **kwargs):
        recorded = real_patch(monkeypatch_, vps=vps, **kwargs)
        monkeypatch_.setattr(rt, "_vps_api", vps_api)
        return recorded

    monkeypatch.setattr(H, "patch_relay", patch_with_tenant)
    await _call(
        monkeypatch, features=NEGOTIATED, summarize=_summarizer(delay=0.1), think_delay=0.6,
        db=f"db-summary-forward-{status}",
    )
    titles = [(p, b) for p, b in posts if p.endswith("/voice-delegations/addendum")]
    assert len(titles) == 1, posts
    path, body = titles[0]
    assert body == {
        "session_id": f"db-summary-forward-{status}", "task_id": "d1",
        "addendum_id": "title:d1", "text": "", "lang": "en", "task_title": SUMMARY,
    }, body
    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    want = "outcome=accepted" if status == 200 else "outcome=unsupported"
    assert any(
        m.startswith("[LIVE] task title forwarded id=d1 ") and want in m for m in messages
    ), messages


# ── Phase C seam (ruling 5): ONE addendum poster, ONE unsupported flag ────

@pytest.mark.asyncio
async def test_a_title_and_a_tail_share_one_addendum_poster_and_one_unsupported_flag(monkeypatch):
    """The late model title rides the same poster a caller's tail does
    (`post_tenant_addendum`), so an image without the route answers 404
    ONCE for the session — not once per kind of post."""

    from app.api import ws_realtime as rt
    from app.services import live_voice_protocol as live

    session = live._LiveSession.__new__(live._LiveSession)
    session.user_id = "user-1"
    session.db_session_id = "db-one-poster"
    posts: list[dict] = []

    async def vps(_user_id):
        return ("http://tenant.invalid", "key")

    async def vps_api(url, key, method, path, params=None, json_body=None, timeout=15.0,
                      outcome=None):
        posts.append(dict(json_body or {}))
        if outcome is not None:
            outcome.update(status=404, failure="permanent")
        return None

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)
    task = live.DelegatedTask(
        delegation_id="d1", offset_ms=0, transcript="museum hours", summary="Museum hours",
    )
    assert await session.forward_task_title(task) == "unsupported"
    assert session.tenant_addenda is False
    assert [(p["addendum_id"], p["text"], p["task_title"]) for p in posts] == [
        ("title:d1", "", "Museum hours"),
    ], posts
    assert await session.post_tenant_addendum(task, "live-utt:p:2", "کیه") == "unsupported"
    assert len(posts) == 1, "a tail was posted to a tenant that already answered 404"


# ── Phase C (ruling 8): a retry inherits the title's SOURCE too ──────────

@pytest.mark.asyncio
@pytest.mark.parametrize("first_title", ["model", "late_model", "fallback"])
async def test_a_retry_of_a_fallback_titled_task_still_asks_the_model(monkeypatch, first_title):
    """d1 fails.  When d1 wore a MODEL title, the retry wears it (one
    summarizer call, `inherited`).  When d1 only ever wore the deterministic
    FALLBACK (its summarizer call failed), the retry wears that floor on its
    first frame AND asks the model once more — the record carries the
    source, so a floor is never mistaken for a final title."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=3.0, voice_live_task_summary_timeout_s=1.5,
    )
    calls: list[dict] = []
    box: dict = {}

    async def summarize(**kwargs):
        calls.append(kwargs)
        if first_title == "fallback" and len(calls) == 1:
            raise RuntimeError("summarizer down")
        if first_title == "late_model" and len(calls) == 1:
            # Lands after d1 failed: the record was written with the fallback.
            await asyncio.sleep(0.9)
        return SUMMARY

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        if kw.get("delegation_id") == "d1":
            await asyncio.sleep(0.3)
            raise RuntimeError("agent container gone")
        await asyncio.sleep(0.4)
        return ANSWER, "m"

    H.patch_relay(monkeypatch, think=think, summarize=summarize)

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(REQUEST, 0, 2400))
            p.push(H.delegation("d1", 2450))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.7)
        box["p"].push(H.user_delta("retry", 15000, 15400))
        box["p"].push(H.delegation("d2", 15450))
        await _wait(lambda: _terminal(client, "d2"))
        await asyncio.sleep(0.3)

    features = NEGOTIATED + ["retry_relation"]
    client = StampedClient([H.config(features=features), script, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=20, db_session_id=f"db-retry-src-{first_title}")

    d2 = [f for f in _frames_for(client, "d2") if "summary" in f]
    assert d2, _frames_for(client, "d2")
    if first_title in ("model", "late_model"):
        assert len(calls) == 1, len(calls)
        assert {(f["summary"], f.get("summary_source")) for f in d2} == {(SUMMARY, "inherited")}, d2
    else:
        assert len(calls) == 2, len(calls)
        assert d2[0].get("summary_source") == "fallback", d2[0]
        assert d2[-1]["summary"] == SUMMARY and d2[-1].get("summary_source") == "model", d2[-1]
