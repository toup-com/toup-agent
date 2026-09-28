"""A voice card must not claim an outcome the user never received (R48, A5-05).

Until now every close of a voice turn's job went through
``close_job_completed``, including the two cancellations the Live relay
introduced — a barge-in and a supersession — where the turn was stopped
*before* it said anything. The card therefore read "Done in 3 steps" over work
that produced nothing, and tapping it showed real steps and real sources, which
makes the claim more convincing rather than less.

Three outcomes now, and the third is the one that was missing:

* ``completed`` — the turn produced its answer (unchanged);
* ``interrupted_after_answer`` — the caller hung up after hearing the reply.
  Still a completed card; "Didn't finish" over work the agent already spoke
  aloud is the lie ``job_status.turn_interrupted`` exists to stop telling;
* ``cancelled`` — stopped before anything was delivered.

The only thing this process can see is the THREAD: on voice the answer row is
written by the platform relay under an id this container never learns, so the
proof of delivery is "an assistant message landed in this conversation after
the turn opened" — the same proof ``reconcile_delivered_turn_jobs`` trusts, and
it has to be checked here because that watchdog only ever looks at ``running``
rows and can never upgrade a terminal one.

Also pinned here: the title never carries prompt scaffolding, and never a
one-word caption fragment (A5-01 / A5-11).

Needs RUN_MODE=agent: build_jobs / job_events / messages / conversations are
AGENT_ONLY. See COVERAGE_DEBT.txt.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

import pytest


# ── Harness ─────────────────────────────────────────────────────────────

async def _make_user() -> str:
    from app.db import async_session_maker, User
    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"{user_id[:8]}@test.local",
                    hashed_password="x" * 60, name="R48"))
        await db.commit()
    return user_id


async def _make_conversation(user_id: str) -> str:
    from app.db import async_session_maker
    from app.db.models import Conversation
    cid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Conversation(id=cid, user_id=user_id, channel="voice"))
        await db.commit()
    return cid


async def _add_assistant_row(
    conversation_id: str, *, when: datetime, voice: dict | None = None,
) -> None:
    """One assistant row. ``voice`` is the relay's provenance stamp — the
    thing that says WHICH writer produced it (`sessions._clean_voice` →
    `metadata_json.voice`)."""
    import json

    from app.db import async_session_maker
    from app.db.models import Message
    async with async_session_maker() as db:
        db.add(Message(
            id=str(uuid.uuid4()), conversation_id=conversation_id,
            role="assistant", content="Imagen 4 Ultra leads right now.",
            created_at=when,
            metadata_json=json.dumps({"voice": voice}) if voice else None,
        ))
        await db.commit()


async def _sweep_with_row(voice_job_env, *, voice: dict | None,
                          delegation_id: str | None = None) -> str:
    """Open a voice card, land ONE assistant row stamped ``voice`` while it is
    open, then hang up. Returns the BuildJob status."""
    vj = voice_job_env["vj"]
    user_id = await _make_user()
    conv = await _make_conversation(user_id)
    job = vj.VoiceTurnJob(user_id=user_id, conversation_id=conv,
                          request_text="research how the new pricing works",
                          delegation_id=delegation_id)
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()
    await _add_assistant_row(conv, when=datetime.utcnow() + timedelta(seconds=1),
                             voice=voice)
    vj.sweep_current_voice_job()
    await voice_job_env["drain"]()
    return (await _job_rows(user_id))[0].status


async def _job_rows(user_id: str) -> list:
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import BuildJob
    async with async_session_maker() as db:
        return list((await db.execute(
            select(BuildJob).where(BuildJob.user_id == user_id)
        )).scalars().all())


@pytest.fixture
def voice_job_env(monkeypatch):
    """Keep the card's background writes so the assertions can see them, and
    take the delivery-proof grace to zero.

    The writes are deliberately OFF the turn's critical path; a test that
    awaited them inline could not tell the difference, which is the property
    the design is built on.
    """
    import app.agent.voice_jobs as vj
    import app.api.ws_chat as ws
    from app.services import agent_notify_client as anc

    bg: list = []

    def _keep(coro, **kw):
        t = asyncio.get_event_loop().create_task(coro)
        bg.append(t)
        return t

    frames: list = []

    async def _capture(user_id, event, **kw):
        frames.append(dict(event))
        return 0

    monkeypatch.setattr(vj, "_spawn_bg", _keep)
    # The proof is a bounded POLL now, so "take the grace to zero" means a
    # ceiling of zero: one read, no sleep. A row already in the thread is still
    # found — that is the point of asking immediately first.
    monkeypatch.setattr(vj, "_CANCEL_PROOF_POLL_S", 0.0)
    monkeypatch.setattr(vj, "_CANCEL_PROOF_MAX_WAIT_S", 0.0)
    monkeypatch.setattr(ws, "broadcast_to_user", _capture)
    monkeypatch.setattr(anc, "OPPORTUNISTIC_FLUSH", False)

    async def _drain():
        for _ in range(4):
            if not bg:
                break
            pending, bg[:] = list(bg), []
            await asyncio.gather(*pending, return_exceptions=True)

    return {"bg": bg, "frames": frames, "drain": _drain, "vj": vj}


# ── The regression ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_turn_stopped_before_any_answer_closes_cancelled(voice_job_env):
    """The A5-05 headline. Nothing was delivered, so nothing may claim it was."""
    vj = voice_job_env["vj"]
    user_id = await _make_user()
    conv = await _make_conversation(user_id)

    job = vj.VoiceTurnJob(
        user_id=user_id, conversation_id=conv,
        request_text="research how the new pricing works",
    )
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()

    vj.sweep_current_voice_job()
    await voice_job_env["drain"]()

    rows = await _job_rows(user_id)
    assert len(rows) == 1
    assert rows[0].status == "cancelled"
    assert rows[0].completed_at is not None


@pytest.mark.asyncio
async def test_the_cancelled_card_keeps_the_steps_it_actually_reached(voice_job_env):
    """The work genuinely happened and stays worth opening — what must not
    happen is `finish_all_steps` painting the unreached steps done."""
    import json

    vj = voice_job_env["vj"]
    user_id = await _make_user()
    conv = await _make_conversation(user_id)
    job = vj.VoiceTurnJob(user_id=user_id, conversation_id=conv,
                          request_text="compare the two pricing pages")
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()
    vj.sweep_current_voice_job()
    await voice_job_env["drain"]()

    steps = json.loads((await _job_rows(user_id))[0].steps_json)
    labels = [s["label"] for s in steps]
    assert labels == ["Search the web", "Put the answer together"]
    # The answer step was never reached. A card that greens it is claiming the
    # answer was put together, which is the whole defect.
    assert steps[-1]["status"] != "done", steps


@pytest.mark.asyncio
async def test_a_hung_up_turn_whose_answer_landed_still_closes_completed(voice_job_env):
    """Today's intended behaviour, guarded. The caller hanging up AFTER the
    reply is how a voice call normally ends; the relay has already written the
    assistant row, and that row is the only proof this process can see."""
    vj = voice_job_env["vj"]
    user_id = await _make_user()
    conv = await _make_conversation(user_id)
    job = vj.VoiceTurnJob(user_id=user_id, conversation_id=conv,
                          request_text="what did the pricing page say")
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()

    # The relay's write, landing between the answer and the hang-up.
    await _add_assistant_row(conv, when=datetime.utcnow() + timedelta(seconds=1))

    vj.sweep_current_voice_job()
    await voice_job_env["drain"]()

    rows = await _job_rows(user_id)
    assert rows[0].status == "completed", (
        "an answer that was delivered and then interrupted is not a "
        "cancellation — that mapping is what job_status.turn_interrupted "
        "was written for"
    )


@pytest.mark.asyncio
async def test_an_assistant_row_from_BEFORE_the_turn_is_not_proof(voice_job_env):
    """The proof is scoped in TIME as well as by conversation. Every voice
    session has earlier assistant rows in the same day thread; taking any of
    them as proof would make the cancelled outcome unreachable."""
    vj = voice_job_env["vj"]
    user_id = await _make_user()
    conv = await _make_conversation(user_id)
    await _add_assistant_row(conv, when=datetime.utcnow() - timedelta(minutes=5))

    job = vj.VoiceTurnJob(user_id=user_id, conversation_id=conv,
                          request_text="now check the changelog for me")
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()
    vj.sweep_current_voice_job()
    await voice_job_env["drain"]()

    assert (await _job_rows(user_id))[0].status == "cancelled"


@pytest.mark.asyncio
async def test_a_normal_seal_is_untouched(voice_job_env):
    """The happy path is the one this round must not move: a turn that
    produced its answer seals `completed`, with its preview."""
    vj = voice_job_env["vj"]
    user_id = await _make_user()
    conv = await _make_conversation(user_id)
    job = vj.VoiceTurnJob(user_id=user_id, conversation_id=conv,
                          request_text="find the strongest image model")
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()
    job.seal(final_text="Imagen 4 Ultra.", total_tokens=42, model="gpt-5.5")
    await voice_job_env["drain"]()

    rows = await _job_rows(user_id)
    assert rows[0].status == "completed"
    assert rows[0].total_tokens == 42


@pytest.mark.asyncio
async def test_the_cancelled_card_is_announced_as_cancelled(voice_job_env):
    """A DB write does not close a Live Activity — only a terminal
    notification does — and the in-app frame must not say `completed` for a
    row that says `cancelled`."""
    vj = voice_job_env["vj"]
    user_id = await _make_user()
    conv = await _make_conversation(user_id)
    job = vj.VoiceTurnJob(user_id=user_id, conversation_id=conv,
                          request_text="look up the release notes")
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()
    vj.sweep_current_voice_job()
    await voice_job_env["drain"]()

    terminal = [f for f in voice_job_env["frames"]
                if f.get("type") == "job_update" and f.get("status") != "running"]
    assert terminal, voice_job_env["frames"]
    assert terminal[-1]["status"] == "cancelled"
    assert terminal[-1]["job_id"] == job._job_id  # noqa: SLF001


# ── WHOSE answer (addendum C1) ──────────────────────────────────────────
# "An assistant row landed" is not "this turn's answer landed", and the Live
# prompt this same round added GUARANTEES the difference: it tells the model
# to "say one short line and delegate", so a spoken filler row lands in the
# same conversation a second after the card opens. The cancelled outcome was
# therefore unreachable in exactly the scenario it was written for.

@pytest.mark.asyncio
async def test_a_spoken_filler_row_is_not_proof_that_the_work_was_delivered(
        voice_job_env):
    """The blocker. `live_spoken` is the relay's word for "the provider said
    this out loud" — it is never the delegated answer."""
    status = await _sweep_with_row(
        voice_job_env, voice={"source": "live_spoken", "epoch": 3})
    assert status == "cancelled", (
        "a filler line spoken while the research was still running closed the "
        "card as if the research had been delivered"
    )


@pytest.mark.asyncio
async def test_a_delegated_row_stamped_cancelled_is_not_proof(voice_job_env):
    """The relay persists a cancelled delegation's row too — that is D4/§4,
    the work is never thrown away. A row that says it was cancelled cannot be
    the proof that it was delivered."""
    status = await _sweep_with_row(
        voice_job_env, voice={"source": "delegated", "cancelled": True})
    assert status == "cancelled"


@pytest.mark.asyncio
async def test_a_superseded_delegations_row_is_not_proof(voice_job_env):
    status = await _sweep_with_row(
        voice_job_env, voice={"source": "delegated", "superseded": True})
    assert status == "cancelled"


@pytest.mark.asyncio
async def test_another_delegations_answer_is_not_this_cards_proof(voice_job_env):
    """Two concurrent delegations are an ordinary Live state (D9 allows two
    and queues the third). Without the id, the fast one's answer closes the
    slow one's card as completed."""
    status = await _sweep_with_row(
        voice_job_env,
        voice={"source": "delegated", "delegation_id": "dlg-OTHER"},
        delegation_id="dlg-MINE",
    )
    assert status == "cancelled"


@pytest.mark.asyncio
async def test_this_delegations_own_answer_IS_proof(voice_job_env):
    """The positive case — and the one that must not regress into "nothing is
    ever proof", which would make every hung-up voice turn a cancellation."""
    status = await _sweep_with_row(
        voice_job_env,
        voice={"source": "delegated", "delegation_id": "dlg-MINE"},
        delegation_id="dlg-MINE",
    )
    assert status == "completed"


@pytest.mark.asyncio
async def test_a_delegated_row_is_proof_for_a_card_that_has_no_delegation_id(
        voice_job_env):
    """The relay ships before the agent image rolls, so for a window the id is
    absent on the card while the provenance is already on the row. Absent must
    mean "any delegated answer", not "none"."""
    status = await _sweep_with_row(
        voice_job_env, voice={"source": "delegated", "delegation_id": "dlg-7"})
    assert status == "completed"


@pytest.mark.asyncio
async def test_a_relay_authored_record_row_is_not_proof(voice_job_env):
    """L1V2-R1, the seam's other side.

    A delegated turn whose `_think` raised finalizes nothing, so the sweep
    seals CANCELLED and `_close` then polls ~6 s for a late answer row — a
    window the relay's own POST lands inside. That row exists to carry the
    tool trail and the files of a turn that failed, and its whole text is
    relay copy ("That didn't go through"), so the relay stamps it
    `delegated_record`. Accepted as proof, the card announces success for work
    that crashed. The producer is pinned in the platform lane
    (tests/test_live_delegation.py); this is the consumer, and only a test on
    each side keeps the two agreeing.
    """
    record = {"source": "delegated_record", "delegation_id": "dlg-MINE",
              "superseded": False, "cancelled": False}
    status = await _sweep_with_row(
        voice_job_env, voice=dict(record), delegation_id="dlg-MINE")
    assert status == "cancelled", (
        "the relay's record of a turn that never answered is not the answer"
    )

    # The same row, same flags, same id — only the provenance differs. Without
    # this half the rule could be satisfied by rejecting everything.
    status = await _sweep_with_row(
        voice_job_env, voice={**record, "source": "delegated"},
        delegation_id="dlg-MINE")
    assert status == "completed"


@pytest.mark.asyncio
async def test_an_id_longer_than_the_wires_bound_still_matches_its_own_answer(
        voice_job_env):
    """The id is PROVIDER-supplied and unbounded, and the two sides of this
    comparison are bounded differently on the way in: the card's copy came
    through the relay body (`[:64]`) and `ChatRequest(max_length=64)`, while
    the row's copy is stamped as the relay read it. Compared raw, a longer id
    makes the proof permanently False — so every DELIVERED delegated answer
    would close `cancelled`: the A5-05 lie inverted, and it looks correct.
    """
    long_id = "dlg-" + "a" * 70          # 74 chars, > the 64 the wire allows
    assert len(long_id) > 64
    status = await _sweep_with_row(
        voice_job_env,
        voice={"source": "delegated", "delegation_id": long_id},
        delegation_id=long_id,
    )
    assert status == "completed"
    # …and the narrowing still holds at that length — the bound must not have
    # replaced one lie with another by making every long id equal.
    status = await _sweep_with_row(
        voice_job_env,
        voice={"source": "delegated", "delegation_id": "dlg-OTHER-" + "b" * 70},
        delegation_id=long_id,
    )
    assert status == "cancelled"


@pytest.mark.asyncio
async def test_a_row_with_no_voice_provenance_is_still_proof(voice_job_env):
    """The Realtime path, unchanged: there the only assistant writer is the
    turn itself, and its rows carry no `voice` key at all."""
    status = await _sweep_with_row(voice_job_env, voice=None)
    assert status == "completed"


@pytest.mark.asyncio
async def test_an_answer_that_lands_while_the_card_is_closing_is_still_proof(
        voice_job_env, monkeypatch):
    """The proof is a bounded POLL, not one guessed interval.

    The row that counts is the DELEGATED one, and the relay submits it after
    the delegation completes, through a persistence queue and an HTTP POST to
    the tenant — an interval nothing in either process measures. A single
    sleep therefore had to be either too short (a delivered answer announced
    as cancelled: silent, and safe-looking) or a flat cost on every real
    cancellation. Re-reading until a ceiling makes the covered window a number
    instead of a guess, and this row lands well after the first read.
    """
    vj = voice_job_env["vj"]
    monkeypatch.setattr(vj, "_CANCEL_PROOF_POLL_S", 0.02)
    monkeypatch.setattr(vj, "_CANCEL_PROOF_MAX_WAIT_S", 3.0)

    user_id = await _make_user()
    conv = await _make_conversation(user_id)
    job = vj.VoiceTurnJob(user_id=user_id, conversation_id=conv,
                          request_text="research how the new pricing works",
                          delegation_id="dlg-LATE")
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()

    async def _late_write():
        await asyncio.sleep(0.15)
        await _add_assistant_row(
            conv, when=datetime.utcnow() + timedelta(seconds=1),
            voice={"source": "delegated", "delegation_id": "dlg-LATE"},
        )

    vj.sweep_current_voice_job()
    writer = asyncio.get_event_loop().create_task(_late_write())
    await voice_job_env["drain"]()
    await writer
    await voice_job_env["drain"]()

    assert (await _job_rows(user_id))[0].status == "completed"


# ── A cancelled card is cancelled everywhere (addendum C4) ──────────────

@pytest.mark.asyncio
async def test_no_step_of_a_cancelled_card_is_left_running(voice_job_env):
    """A terminal row whose steps still say `running` renders a live spinner
    forever — the Round-27 "archived frozen as running" class, on the durable
    surface. Closing the window must not GREEN it either."""
    import json

    vj = voice_job_env["vj"]
    user_id = await _make_user()
    conv = await _make_conversation(user_id)
    job = vj.VoiceTurnJob(user_id=user_id, conversation_id=conv,
                          request_text="compare the two pricing pages")
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()
    vj.sweep_current_voice_job()
    await voice_job_env["drain"]()

    steps = json.loads((await _job_rows(user_id))[0].steps_json)
    assert [s["status"] for s in steps] == ["skipped", "skipped"], steps
    # The step that was running really ran, and for that long.
    assert steps[0].get("completed_at"), steps[0]
    assert "duration_ms" in steps[0], steps[0]
    # The one that never started has no window at all — a duration nobody
    # measured is unknown, not zero and not the enclosing window.
    assert "duration_ms" not in steps[1], steps[1]


@pytest.mark.asyncio
async def test_a_cancelled_card_never_announces_a_FAILURE(voice_job_env, monkeypatch):
    """`mission_failed` is a terminal notify kind that maps to the widget's
    `failed` phase — a red mark and "Didn't finish" over work the user
    STOPPED. No neutral terminal kind exists (KNOWN_NOTIFY_KINDS is a closed
    enum and the widget's phase vocabulary has no `cancelled`), so the push is
    suppressed and the `job_update` frame is the announcement."""
    import app.agent.subagent_orchestrator as so

    vj = voice_job_env["vj"]
    pushes: list = []

    async def _spy(**kw):
        pushes.append(kw)

    # Patched on the module the call site imports FROM, so a push re-added
    # inside the function (a call-time `from … import`) still lands on the spy.
    monkeypatch.setattr(so, "_notify_job_event", _spy)

    user_id = await _make_user()
    conv = await _make_conversation(user_id)
    job = vj.VoiceTurnJob(user_id=user_id, conversation_id=conv,
                          request_text="look up the release notes")
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()
    vj.sweep_current_voice_job()
    await voice_job_env["drain"]()

    assert not [p for p in pushes if p.get("kind") == "mission_failed"], pushes
    # …and the in-app frame is still the announcement, so suppressing the push
    # did not leave the cancellation unannounced.
    assert [f for f in voice_job_env["frames"]
            if f.get("type") == "job_update" and f.get("status") == "cancelled"]


@pytest.mark.asyncio
async def test_the_cancelled_rows_own_copy_is_in_the_callers_language(voice_job_env):
    """One card, one language. `job_status.turn_interrupted()` returns an
    English-only sentence, and the card detail surface renders it beside step
    labels this module has already localised."""
    vj = voice_job_env["vj"]
    user_id = await _make_user()
    conv = await _make_conversation(user_id)
    job = vj.VoiceTurnJob(user_id=user_id, conversation_id=conv,
                          request_text="قیمت‌های جدید را برایم بررسی کن")
    vj.set_current_voice_job(job)
    job.plan([{"name": "web_search"}])
    await voice_job_env["drain"]()
    vj.sweep_current_voice_job()
    await voice_job_env["drain"]()

    row = (await _job_rows(user_id))[0]
    assert row.status == "cancelled"
    assert row.user_message == vj._STOPPED_BODY_FA  # noqa: SLF001
    from app.agent.job_status import turn_interrupted
    assert row.user_message != turn_interrupted().user_message


# ── The title (A5-01 / A5-11) ───────────────────────────────────────────

def test_the_delegation_scaffolding_can_never_become_a_title():
    """Production shipped `Working on: Live-session context from earlier
    accepted…` to a lock screen. The title path fans out to the card, the chat
    marker row, the APNs body and the Live Activity content-state, and the
    last two escape nothing."""
    import app.agent.voice_jobs as vj

    scaffolded = (
        "Live-session context from earlier accepted delegations follows. "
        "Treat it as prior conversation, not as the current request.\n"
        "Prior caller request: what is the weather\n"
        "Accepted backend result: it is sunny\n"
        "Current caller request: now find me a hotel"
    )
    job = vj.VoiceTurnJob(user_id="u", conversation_id=None,
                          request_text=scaffolded)
    for marker in ("Live-session context", "Prior caller request",
                   "Accepted backend result", "Current caller request"):
        assert marker not in job.title, job.title


def test_the_clean_utterance_wins_over_the_model_facing_prompt():
    """`display_request` is what the caller SAID. The model still gets the
    scaffolded blob; only the display side is corrected."""
    import app.agent.voice_jobs as vj

    job = vj.VoiceTurnJob(
        user_id="u", conversation_id=None,
        request_text=("Live-session context from earlier accepted delegations "
                      "follows.\nCurrent caller request: find me a hotel in Lisbon"),
        display_request="find me a hotel in Lisbon",
    )
    assert job.title == "Find me a hotel in Lisbon"


def test_a_one_word_caption_fragment_is_not_a_title():
    """A 350 ms transcript settle produces a word, not a request. The card
    used to be titled `ایونتش`."""
    import app.agent.voice_jobs as vj

    assert vj.VoiceTurnJob(user_id="u", conversation_id=None,
                           request_text="ایونتش").title == "جمع‌بندی درخواست صوتی"
    assert vj.VoiceTurnJob(user_id="u", conversation_id=None,
                           request_text="benchmarks").title == "Voice request"
    # …and a real short request still titles itself.
    assert vj.VoiceTurnJob(user_id="u", conversation_id=None,
                           request_text="play Radiohead").title == "Play Radiohead"


def test_the_word_floor_does_not_blank_a_script_that_has_no_word_spaces():
    """A WORD floor is a claim about the script. Chinese, Japanese and Thai do
    not separate words with spaces, so a whole sentence is one `split()` token
    — and every request in those languages was titled "Voice request",
    unconditionally, for every speaker of them."""
    import app.agent.voice_jobs as vj

    for said in ("播放周杰伦的歌", "ニュースを教えて", "ข่าววันนี้เป็นอย่างไร"):
        title = vj.VoiceTurnJob(user_id="u", conversation_id=None,
                                request_text=said).title
        assert title not in ("Voice request", "جمع‌بندی درخواست صوتی"), said
        assert said[:4] in title, (said, title)
    # The floor still bites on a fragment in those scripts: two ideographs is
    # a word, not a request.
    assert vj.VoiceTurnJob(user_id="u", conversation_id=None,
                           request_text="天气").title == "Voice request"


def test_clean_request_text_keeps_an_utterance_that_merely_mentions_a_marker():
    """The markers are matched as the prompt's OPENING, not as a substring
    anywhere: a user is allowed to say the words."""
    import app.agent.voice_jobs as vj

    said = "tell me what a prior caller request even means"
    assert vj.clean_request_text(said) == said
