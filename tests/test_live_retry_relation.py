"""PR-10: retry, relation, order and self-contained task text (the 2026-10-02 call).

Findings (FINDINGS-REPORT §2 P10, P7, P2.5; §5 PR-10), reproduced offline:

  P10  job 2 — the caller's correction "Not officer, Professor" of job 1 —
       failed; the caller asked "What happened", then said "retry".  The relay
       had no notion of a retry: "retry" was dispatched as a NEW request, its
       card titled "Task" / "That last one", its agent turn handed the bare
       word plus job 1's UNCORRECTED officer request, and the "What happened"
       turn in between broke the only link back to the failed task
       (`latest_task_before`).  A status answer would also have replayed job
       1's older answer over the newer failure.
  D2   a correction reached the agent as the earlier request plus the
       caller's bare words, never stated in full.
  P7   job 1's card, row and tenant job were titled "Everything is good. Can
       yo…": the caller's wellbeing answer, said in the same breath.
  P10  order: the card has no fact to sit after the model's own "Okay, I'll
       check that out." — the created frame and that epoch opened in the same
       millisecond.

Driven through the real relay (`test_live_harness`) with provider-shaped
events; the pure seams are called directly.  The `retries` relation kind and
the card anchor are negotiated (`retry_relation`); every older client gets the
frames a new request gets.
"""

import asyncio

import pytest

import test_live_harness as H
from app.services import live_voice_protocol as live
from app.services.live_voice_protocol import (
    delegated_agent_input,
    request_title,
    sanitize_user_visible_answer,
)


OFFICER = "find the most experienced officer in the llm program at u of t"
CORRECTION = "not officer, professor"
RESOLVED = "find the most experienced professor in the llm program at u of t"
RETRY_LABEL = "Request the caller asked to run again"
NEGOTIATED = H.config()["features"] + ["retry_relation"]


# ── helpers ───────────────────────────────────────────────────────────


def _frames_for(client, did):
    return [f for f in client.of("delegation") if f.get("delegation_id") == did]


def _lifecycle(frames):
    return [f for f in frames if "task_revision" in f]


def _terminal(client, did):
    return any(
        f.get("phase") in {"completed", "failed", "cancelled", "superseded", "expired"}
        for f in _frames_for(client, did)
    )


async def _wait(pred, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def _first_turn(box, text, end_ms, did="d1"):
    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(text, 0, end_ms))
            p.push(H.delegation(did, end_ms + 50))
        elif e["type"] == "session.close":
            p.push(H.closed())
    return on_send


def _rows_for(recorded, did):
    return [
        s for s in recorded["saves"]
        if str(s.get("assistant_ref") or "").endswith(f":{did}")
    ]


async def _run_incident(monkeypatch, *, features, db, middle=(("what happened", "model"),),
                        last="retry"):
    """The call's shape: d1 (the officer request) answers; d2 (the correction
    "not officer, professor") FAILS; the caller says each `middle` turn —
    answered by the model itself ("model", as on the call) or delegated to the
    relay ("delegated"); then `last` is delegated as d3."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list[dict] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id")
        calls.append({"did": did, "task": task, "display": kw.get("display_request")})
        if did == "d2":
            raise RuntimeError("agent container gone")
        return f"answer for {did}", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    async def script():
        p = box["p"]
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.2)
        p.push(H.user_delta(CORRECTION, 5000, 5800))
        p.push(H.delegation("d2", 5850))
        await _wait(lambda: _terminal(client, "d2"))
        # Past the failure line's speak budget, so its claim is withdrawn.
        await asyncio.sleep(0.7)
        at = 15000
        for index, (words, mode) in enumerate(middle):
            p.push(H.user_delta(words, at, at + 600))
            if mode == "delegated":
                did = f"dm{index}"
                p.push(H.delegation(did, at + 650))
                await _wait(lambda: any(
                    e.get("delegation_id") == did
                    for e in provider.of("session.commentary.append")
                ) or bool(_frames_for(client, did)))
                await asyncio.sleep(0.7)
            else:
                await asyncio.sleep(0.3)
                p.push(H.out_text("Sorry, that one did not go through.", at + 800, at + 1400))
                p.push(H.out_audio("X"))
                await asyncio.sleep(0.4)
            # Further apart than the request-span gap: separate turns.
            at += 10000
        p.push(H.user_delta(last, at, at + 400))
        p.push(H.delegation("d3", at + 450))
        await _wait(lambda: _terminal(client, "d3"))

    client = H.FakeClient([H.config(features=features), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_first_turn(box, OFFICER, 900), auto_ack=True)
    await H.run_relay(client, provider, timeout=20, db_session_id=db)
    return client, provider, {c["did"]: c for c in calls}, recorded


# ══════════════════════════════════════════════════════════════════════
# P10 / D6 — "retry" re-runs the task that did not go through
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.parametrize("negotiated", [True, False], ids=["retry_relation", "older_client"])
async def test_a_retry_after_a_failure_links_the_failed_task(monkeypatch, negotiated):
    features = NEGOTIATED if negotiated else H.config()["features"]
    client, _provider, calls, recorded = await _run_incident(
        monkeypatch, features=features, db=f"db-retry-link-{negotiated}",
    )

    assert set(calls) == {"d1", "d2", "d3"}, calls
    d3 = _frames_for(client, "d3")
    lifecycle = _lifecycle(d3)
    assert lifecycle, d3
    if negotiated:
        assert all(
            f.get("relation") == {"kind": "retries", "task_id": "d2"} for f in lifecycle
        ), lifecycle
        assert client.of("ready")[0]["capabilities"].get("retry_relation") is True
    else:
        # An older client gets the frames of a new request: no relation, no key.
        assert all("relation" not in f for f in d3), d3
        assert "retry_relation" not in client.of("ready")[0]["capabilities"]
    # The card names what runs again — the request the failed correction
    # resolved to — never "retry" (nor the bare "not officer, professor").
    assert all(f["title"] == request_title(RESOLVED) for f in d3), d3
    assert calls["d3"]["display"] == RESOLVED
    # The agent is handed the failed task's whole chain and what it resolves
    # to; the caller's own words stay last.
    message = calls["d3"]["task"]
    assert RETRY_LABEL in message, message
    assert OFFICER + " — changed since, oldest first: " + CORRECTION in message, message
    assert "resolved request: «" + RESOLVED + "»" in message, message
    assert message.rstrip().endswith(
        "Current caller request: retry\n\nReply in the caller's language."
    ), message[-200:]
    rows = _rows_for(recorded, "d3")
    assert rows and rows[-1]["assistant_voice"]["related_task_id"] == "d2"
    assert rows[-1]["assistant_voice"]["task_title"] == request_title(RESOLVED)


@pytest.mark.asyncio
async def test_say_that_again_is_not_a_retry(monkeypatch):
    """Bare "again" / «دوباره» belong to other grammars; "say that again" asks
    to HEAR something again.  Only a whole-utterance retry phrase links."""

    for text in (
        "retry", "Retry.", "try again", "Try again, please.", "can you try that again?",
        "one more time", "do it again", "run it again", "okay, try again",
        "دوباره", "دوباره؟", "آره دوباره امتحان کن", "یه بار دیگه",
    ):
        assert live.is_retry_request(text), text
    for text in (
        "say that again", "can you say that again?", "again", "again?",
        "find the dean again", "speak Persian again", "try again tomorrow",
        "retry the search for dentists", "دوباره بگو", "دوباره فارسی حرف بزن",
        "what happened", "",
    ):
        assert not live.is_retry_request(text), text

    client, _provider, calls, _recorded = await _run_incident(
        monkeypatch, features=NEGOTIATED, db="db-retry-say-again", last="say that again",
    )
    d3 = _frames_for(client, "d3")
    assert d3 and all(
        (f.get("relation") or {}).get("kind") != "retries" for f in d3
    ), d3
    assert all(f["title"] == "say that again" for f in d3), d3
    assert RETRY_LABEL not in calls["d3"]["task"]


@pytest.mark.asyncio
@pytest.mark.parametrize("middle, linked", [
    ((("what happened", "model"),), True),
    ((("what happened", "delegated"),), True),
    ((("what happened", "model"), ("okay", "model")), True),
    ((("what is the weather in paris", "model"),), False),
], ids=["check_in_answered_by_model", "check_in_answered_by_relay", "check_in_and_okay",
        "a_request_in_between"])
async def test_a_status_question_does_not_break_the_retry_link(monkeypatch, middle, linked):
    """`latest_task_before` returns nothing once ANY caller turn sits in
    between, so "What happened" severed the link.  A check-in, "okay" or a
    relay-answered status question keep it; a request of the caller's own
    breaks it (then "retry" is dispatched exactly as before)."""

    client, _provider, calls, _recorded = await _run_incident(
        monkeypatch, features=NEGOTIATED, middle=middle,
        db=f"db-retry-status-{len(middle)}-{middle[0][1]}-{linked}",
    )
    d3 = _lifecycle(_frames_for(client, "d3"))
    assert d3
    if linked:
        assert all(f.get("relation") == {"kind": "retries", "task_id": "d2"} for f in d3), d3
        assert RETRY_LABEL in calls["d3"]["task"]
    else:
        assert all("relation" not in f for f in d3), d3
        assert all(f["title"] == "retry" for f in d3), d3
        assert RETRY_LABEL not in calls["d3"]["task"]


@pytest.mark.asyncio
async def test_a_retry_never_reruns_a_relay_owned_media_command(monkeypatch):
    """A spoken stop/pause is the relay's own command (R6-4 ledger, tenant
    order).  "Try again" after one failed is NOT linked: re-running it as a
    binding agent request would execute a halt outside that ledger.  It is
    dispatched exactly as before (no relation, its own words)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list[str] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(task)
        return "Done.", "m"

    async def control(_user_id, action):
        return {"ok": False, "action": action, "reason": "error"}

    H.patch_relay(monkeypatch, think=think, control=control)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.7)
        box["p"].push(H.user_delta("try again", 5000, 5400))
        box["p"].push(H.delegation("d2", 5450))
        await _wait(lambda: _terminal(client, "d2"))

    features = NEGOTIATED + ["media_transport"]
    client = H.FakeClient([H.config(features=features), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_first_turn(box, "stop the music", 600), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-retry-media-guard")

    d1 = _frames_for(client, "d1")
    assert d1 and d1[-1].get("phase") == "failed", d1
    d2 = _frames_for(client, "d2")
    assert d2 and all("relation" not in f for f in d2), d2
    assert all(f["title"] == "try again" for f in d2), d2
    assert calls and RETRY_LABEL not in calls[-1]


@pytest.mark.asyncio
async def test_a_retried_consent_carries_the_question_it_answered(monkeypatch):
    """A failed CONSENT ("yes" to the assistant's own proposal) re-runs with
    the question it answered: its request chain alone is the bare «yes», so
    without the R6-5 answer context the agent would be told to run «yes» "in
    full".  The card keeps the proposal's title."""

    question = "Should I check what the weather is like tomorrow?"
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: dict = {}
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id")
        calls[did] = {"task": task, "display": kw.get("display_request")}
        if did == "d1":
            raise RuntimeError("agent container gone")
        return "Sunny, 21 degrees.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait(lambda: "p" in box)
        p = box["p"]
        p.push(H.user_delta("I have a dentist appointment tomorrow", 1000, 1600))
        p.push(H.out_text(question, 1700, 2400))
        p.push(H.out_audio("Q"))
        await asyncio.sleep(0.6)
        p.push(H.user_delta("yes", 3000, 3300))
        p.push(H.delegation("d1", 3350))
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.7)
        p.push(H.user_delta("try again", 13000, 13400))
        p.push(H.delegation("d2", 13450))
        await _wait(lambda: _terminal(client, "d2"))

    client = H.FakeClient([H.config(features=NEGOTIATED), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-retry-consent")

    assert set(calls) == {"d1", "d2"}, calls
    assert "agree to what it proposed" in calls["d1"]["task"], calls["d1"]["task"]
    d2 = _lifecycle(_frames_for(client, "d2"))
    assert d2 and all(
        f.get("relation") == {"kind": "retries", "task_id": "d1"} for f in d2
    ), d2
    assert all(f["title"] == "check what the weather is like tomorrow" for f in d2), d2
    message = calls["d2"]["task"]
    assert "agree to what it proposed" in message and question in message, message
    assert message.rstrip().endswith(
        "Current caller request: try again\n\nReply in the caller's language."
    ), message[-200:]


# ══════════════════════════════════════════════════════════════════════
# F6 — a status answer reports the NEWEST finished work
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_a_status_question_after_a_failure_reports_the_failure(monkeypatch):
    """F6: the NEWEST finished work decides a status answer.  Job 1's older
    officer answer must not be replayed as "what you found" after the newer
    correction failed — the caller hears that it did not go through."""

    counters = H.capture_counters(monkeypatch)
    _client, provider, _calls, _recorded = await _run_incident(
        monkeypatch, features=NEGOTIATED, middle=(("what happened", "delegated"),),
        db="db-retry-status-recency",
    )
    answered = [fields for name, fields in counters if name == "live_status_answered"]
    assert answered == [{"source": "failed"}], answered
    replays = [
        e["content"] for e in provider.of("session.instructions.append")
        if "answer for d1" in e["content"] and "asked what you found" in e["content"]
    ]
    assert replays == [], "an older answer was replayed over a newer failure"
    failure_line = live.relay_line("delegation_failed", "en")
    assert any(
        e.get("delegation_id") == "dm0" and e["content"] == failure_line
        for e in provider.of("session.commentary.append")
    )


# ══════════════════════════════════════════════════════════════════════
# D2 — a correction is handed over self-contained
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_correction_hands_the_agent_a_self_contained_request(monkeypatch):
    """Job 2's agent turn carried the officer request (labelled) and the bare
    words "not officer, professor", never the request they resolve to.  It
    now carries one self-contained line — earlier request, correction,
    resolved request — before the caller's own words, which stay last."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list[str] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(task)
        return "Professor Ada Lovelace.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.2)
        box["p"].push(H.user_delta(CORRECTION, 5000, 5800))
        box["p"].push(H.delegation("d2", 5850))
        await _wait(lambda: _terminal(client, "d2"))

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_first_turn(box, OFFICER, 900), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-retry-self-contained")

    assert len(calls) == 2
    corrected = calls[1]
    assert (
        "earlier request: «" + OFFICER + "»; correction: «" + CORRECTION
        + "»; resolved request: «" + RESOLVED + "»"
    ) in corrected, corrected
    # Unchanged around it: the labelled request it corrects, the card title,
    # and the caller's own words last.
    assert "Request the caller is correcting (keep its constraints" in corrected
    assert corrected.rstrip().endswith(
        "Current caller request: " + CORRECTION + "\n\nReply in the caller's language."
    )
    d2 = _lifecycle(_frames_for(client, "d2"))
    assert d2 and all(f.get("relation") == {"kind": "replaces", "task_id": "d1"} for f in d2)
    assert d2[0]["title"] == CORRECTION


def test_the_self_contained_line_is_honest_when_nothing_resolves():
    """Shapes that do not NAME the wrong words resolve to nothing.  Then no
    restatement line is written at all: an "earlier, corrected by: …"
    concatenation is not a resolution, and the labelled related request plus
    the caller's own words (still last) already carry both parts."""

    for correction in ("no, only the downtown campus", "actually make it tuesday", "not that one"):
        assert live.resolve_correction(OFFICER, correction) == "", correction
    for correction in (
        "Not officer Professor", "not officer, professor", "no, not officer — professor",
        "professor, not officer", "I meant professor, not officer",
        "professor instead of officer", "instead of officer, professor",
    ):
        assert live.resolve_correction(OFFICER, correction).lower() == RESOLVED, correction
    assert live.resolve_correction(
        "استادهای ال ال ام دانشگاه تورنتو رو پیدا کن", "نه ال ال ام، رباتیک",
    ) == "استادهای رباتیک دانشگاه تورنتو رو پیدا کن"
    # «نه،» is the interjection "No," — never a negation of what follows it.
    assert live.resolve_correction("استادهای دانشگاه تورنتو", "نه، فقط پردیس مرکزی") == ""

    message = delegated_agent_input(
        "no, only the downtown campus", [],
        related_request="find iranian professors at the university of toronto",
        relation_kind="replaces",
    )
    assert "Self-contained restatement" not in message, message
    assert "resolved request" not in message and "corrected by" not in message, message
    assert (
        "Request the caller is correcting (keep its constraints unless the current "
        "request changes them): find iranian professors at the university of toronto"
    ) in message, message
    assert message.rstrip().endswith("Current caller request: no, only the downtown campus")
    # The same correction WITH a resolution keeps its line.
    resolved = delegated_agent_input(
        CORRECTION, [], related_request=OFFICER, relation_kind="replaces",
        resolved_request=RESOLVED,
    )
    assert "resolved request: «" + RESOLVED + "»" in resolved, resolved


def test_every_new_scaffold_line_is_blocked_from_a_user_visible_answer():
    """The retry label and the restatement are relay scaffolding: an answer
    that echoes them — whole or truncated — is dropped like the others."""

    for kind, resolved in (("replaces", RESOLVED), ("retries", ""), ("retries", RESOLVED)):
        message = delegated_agent_input(
            "retry" if kind == "retries" else CORRECTION,
            [(OFFICER, "Officer X.")],
            related_request=OFFICER + ("\n" + CORRECTION if kind == "retries" else ""),
            relation_kind=kind,
            resolved_request=resolved,
        )
        scaffold = message.split("Current caller request:")[0]
        lines = [line for line in scaffold.splitlines() if line.strip()]
        assert any("Self-contained restatement" in line for line in lines), lines
        for line in lines:
            if line.startswith(("Prior caller request", "Accepted backend")):
                continue
            assert sanitize_user_visible_answer(line) == "", line
            assert sanitize_user_visible_answer(line[: max(24, len(line) * 2 // 3)]) == "", line
    ok = "Professor Ada Lovelace runs the LLM program at the University of Toronto."
    assert sanitize_user_visible_answer(ok) == ok


# ══════════════════════════════════════════════════════════════════════
# P7 (light) — a display title starts at the request
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_leading_clause_is_stripped_from_the_title(monkeypatch):
    """The card, the saved row and the tenant job were all titled "Everything
    is good. Can yo…".  Deterministic, no model call, titles only: the agent
    is still handed every word the caller said."""

    spoken = (
        "Everything is good. Can you search who is the most experienced officer "
        "in the LLM program at U of T"
    )
    title = "Can you search who is the most experienced officer in the LLM program at U of T"
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list[dict] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"task": task, "display": kw.get("display_request")})
        return "Professor Ada Lovelace.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_first_turn(box, spoken, 2600), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-retry-title-strip")

    frames = _frames_for(client, "d1")
    assert frames and all(f["title"] == request_title(title) for f in frames), frames
    assert calls and calls[0]["display"] == title
    assert spoken in calls[0]["task"], "the request's words are never dropped"
    rows = _rows_for(recorded, "d1")
    assert rows and rows[-1]["assistant_voice"]["task_title"] == request_title(title)

    for text, expected in (
        ("Hi, can you hear me? Find a dentist near union station",
         "Find a dentist near union station"),
        ("Yeah, everything is fine, thanks. Search for flights to Paris.",
         "Search for flights to Paris."),
        ("سلام. ساعت توکیو چنده؟", "ساعت توکیو چنده؟"),
        ("همه چی خوبه. استادهای رباتیک تورنتو رو پیدا کن", "استادهای رباتیک تورنتو رو پیدا کن"),
        # Unchanged: a correction's opener, a bare "okay" (may be consent), a
        # filler before a comma (tests/test_live_three_videos.py pins it), a
        # wholly phatic text, and a request with no leading clause.
        ("No, only the downtown campus", "No, only the downtown campus"),
        ("Okay. Find a dentist", "Okay. Find a dentist"),
        ("خب، پس استاد یوآفتی نیست؟", "خب، پس استاد یوآفتی نیست؟"),
        ("Thanks. Bye.", "Thanks. Bye."),
        ("Dr. Smith's office hours", "Dr. Smith's office hours"),
        ("find a plumber", "find a plumber"),
    ):
        assert live.display_request_text(text) == expected, text
    # The wellbeing answers are TITLE-only: the per-turn verdict is unchanged.
    assert not live.text_asks_nothing("Everything is good.")


# ══════════════════════════════════════════════════════════════════════
# P10 — an ordering fact for the card
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.parametrize("negotiated", [True, False], ids=["retry_relation", "older_client"])
async def test_the_card_anchor_names_the_reply_that_opened_before_dispatch(
    monkeypatch, negotiated,
):
    """The model's "Okay, I'll check that out." opened in the same millisecond
    as the dispatch, and the card had no fact to sit after it.  A negotiated
    client is told which reply (by its `response_id`) had opened, parented to
    the task's turn, by dispatch — and a later reply on that turn (the result
    being spoken) never moves it.  Older clients get no new key."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    box: dict = {}
    request = "find a plumber near union station"

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(0.3)
        return "Union Plumbing on Front Street.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(request, 0, 900))
            # The model's own ack, then its delegation.
            p.push(H.out_text("Okay, let me look up plumbers near Union Station.", 1000, 1700))
            p.push(H.out_audio("ACK"))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait(lambda: bool(client.of("audio_delta")))
        box["p"].push(H.delegation("d1", 950))
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.3)

    features = NEGOTIATED if negotiated else H.config()["features"]
    client = H.FakeClient([H.config(features=features), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=f"db-retry-anchor-{negotiated}")

    frames = _frames_for(client, "d1")
    assert frames
    ack_id = client.of("audio_delta")[0]["response_id"]
    if not negotiated:
        assert all("anchor_assistant_turn_id" not in f for f in frames), frames
        return
    lifecycle = _lifecycle(frames)
    assert lifecycle and all(
        f.get("anchor_assistant_turn_id") == ack_id for f in lifecycle
    ), (ack_id, lifecycle)
    # The key names the ack epoch by its `assistant_turn_id` — the value its
    # `response_text` frames carry, equal to its `response_id`.
    ack_texts = [f for f in client.of("response_text") if "Okay, let me look up" in f["text"]]
    assert ack_texts and {f.get("assistant_turn_id") for f in ack_texts} == {ack_id}, ack_texts
    assert all("anchor_output_id" not in f for f in frames), "the pre-contract name is gone"
    # The result was spoken on the same turn, in a LATER epoch: not the anchor.
    later = {f["response_id"] for f in client.of("audio_delta")} - {ack_id}
    assert later, "the spoken result opened its own epoch"


# ══════════════════════════════════════════════════════════════════════
# Contract — the card's Retry control: `client_event` `retry_task`
# ══════════════════════════════════════════════════════════════════════


def _retry_task(task_id):
    return {"type": "client_event", "name": "retry_task", "detail": task_id}


def _relay_retry_ids(client):
    seen = []
    for frame in client.of("delegation"):
        did = str(frame.get("delegation_id") or "")
        if did.startswith(live.RELAY_RETRY_PREFIX) and did not in seen:
            seen.append(did)
    return seen


async def _run_failed_then_event(monkeypatch, *, features, db, task_id, expect_retry=True):
    """The incident's first two jobs — d1 (the officer request) answers, d2
    (the correction) FAILS — then the phone sends ONE `retry_task` event for
    `task_id` (the card's Retry control) and nothing is said."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list[dict] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id")
        calls.append({"did": did, "task": task, "display": kw.get("display_request")})
        if did == "d2":
            raise RuntimeError("agent container gone")
        return f"answer for {did}", "m"

    recorded = H.patch_relay(monkeypatch, think=think)
    counters = H.capture_counters(monkeypatch)

    async def jobs():
        p = box["p"]
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.2)
        p.push(H.user_delta(CORRECTION, 5000, 5800))
        p.push(H.delegation("d2", 5850))
        await _wait(lambda: _terminal(client, "d2"))
        # Past the failure line's speak budget, as in `_run_incident`.
        await asyncio.sleep(0.7)

    async def retried():
        if not expect_retry:
            await asyncio.sleep(0.5)
            return
        await _wait(lambda: any(_terminal(client, rid) for rid in _relay_retry_ids(client)), 3.0)
        # D-P1: d1's unvoiced answer now gets a second delivery on an open call,
        # so the retry's answer waits one more speak budget for its turn.
        await _wait(lambda: any(
            "answer for " + rid in str(e.get("content") or "")
            for rid in _relay_retry_ids(client)
            for e in box["p"].of("session.instructions.append")
        ), 4.0)

    client = H.FakeClient([
        H.config(features=features), jobs, _retry_task(task_id), retried, 0.3,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=_first_turn(box, OFFICER, 900), auto_ack=True)
    await H.run_relay(client, provider, timeout=20, db_session_id=db)
    return client, provider, calls, recorded, counters


@pytest.mark.asyncio
async def test_the_cards_retry_control_links_the_failed_task_like_a_spoken_retry(monkeypatch):
    """`client_event` `retry_task` naming the failed task re-runs it exactly
    as a spoken whole-utterance "retry" does: the same relation, card title,
    display request and agent message (the failed task's whole chain and what
    it resolved to), and the same saved row — anchored on the failed task's
    own request turn, since the caller said nothing.  GPT-Live never issued
    the relay's id, so no provider event names it: its answer goes in as an
    instruction to say."""

    spoken_client, _p, spoken_calls, spoken_rows = await _run_incident(
        monkeypatch, features=NEGOTIATED, db="db-retry-task-spoken", middle=(),
    )
    spoken = _lifecycle(_frames_for(spoken_client, "d3"))
    assert spoken, spoken_client.frames

    client, provider, calls, recorded, counters = await _run_failed_then_event(
        monkeypatch, features=NEGOTIATED, db="db-retry-task-event", task_id="d2",
    )
    (rid,) = _relay_retry_ids(client)
    by_id = {c["did"]: c for c in calls}
    assert set(by_id) == {"d1", "d2", rid}, calls
    frames = _frames_for(client, rid)
    lifecycle = _lifecycle(frames)
    assert lifecycle and [f["phase"] for f in frames][-1] == "completed", frames
    # Like the spoken retry's card…
    for key in ("relation", "title"):
        assert {repr(f.get(key)) for f in lifecycle} == {repr(spoken[0].get(key))}, key
    assert lifecycle[0]["relation"] == {"kind": "retries", "task_id": "d2"}
    assert lifecycle[0]["title"] == request_title(RESOLVED)
    # …anchored on the request it re-runs (the caller said nothing new).
    d2_turn = _frames_for(client, "d2")[0]["turn_id"]
    assert {f["turn_id"] for f in frames} == {d2_turn}
    assert all("request_turn_ids" not in f for f in frames), frames
    # …the same agent turn, word for word.
    assert by_id[rid]["display"] == spoken_calls["d3"]["display"] == RESOLVED
    assert by_id[rid]["task"] == spoken_calls["d3"]["task"], (
        by_id[rid]["task"], spoken_calls["d3"]["task"],
    )
    assert "resolved request: «" + RESOLVED + "»" in by_id[rid]["task"]
    # …and the same saved row.
    rows = _rows_for(recorded, rid)
    assert rows and rows[-1]["assistant_voice"]["related_task_id"] == "d2"
    assert rows[-1]["assistant_voice"]["task_title"] == (
        _rows_for(spoken_rows, "d3")[-1]["assistant_voice"]["task_title"]
    )
    assert ("live_retry_task", {"outcome": "linked"}) in counters
    # GPT-Live never sees the relay's id; the answer is an instruction to say.
    assert all(e.get("delegation_id") != rid for e in provider.sent), [
        e for e in provider.sent if e.get("delegation_id") == rid
    ]
    said = [
        e for e in provider.of("session.instructions.append")
        if f"answer for {rid}" in str(e.get("content") or "")
    ]
    assert said and said[0]["content"].startswith("Update on the caller's request."), said


@pytest.mark.asyncio
@pytest.mark.parametrize("features, task_id, reason", [
    (NEGOTIATED, "no-such-task", "unknown_task"),
    (NEGOTIATED, "", "unknown_task"),
    (NEGOTIATED, "d1", "not_failed"),
    (H.config()["features"], "d2", "not_negotiated"),
], ids=["unknown_id", "empty_id", "answered_task", "older_client"])
async def test_a_retry_task_event_that_names_no_failed_task_is_ignored(
    monkeypatch, features, task_id, reason,
):
    """An unknown id, a task that answered, or a client that did not
    negotiate `retry_relation`: nothing is created, nothing runs, nothing is
    said — the event is only counted."""

    client, provider, calls, _recorded, counters = await _run_failed_then_event(
        monkeypatch, features=features, task_id=task_id, expect_retry=False,
        db=f"db-retry-task-ignored-{reason}-{bool(task_id)}",
    )
    assert _relay_retry_ids(client) == [], client.of("delegation")
    assert [c["did"] for c in calls] == ["d1", "d2"], calls
    assert ("live_retry_task", {"outcome": "ignored", "reason": reason}) in counters, [
        c for c in counters if c[0] == "live_retry_task"
    ]
    assert not [
        e for e in provider.sent
        if str(e.get("delegation_id") or "").startswith(live.RELAY_RETRY_PREFIX)
    ]
