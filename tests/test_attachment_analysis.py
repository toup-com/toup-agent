"""Natural-chat full-file work: original coverage, central model, and resume."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
import attachments as fx  # noqa: E402

from app.agent import attachment_analysis as AA
from app.agent.agent_runner import AgentRunner
from app.agent.attachment_provenance import (
    history_analysis_ref, history_attachment_refs, recent_user_file_refs,
)
from app.agent.tool_executor import ToolExecutor
from app.api import chat_attachments as CA
from app.services.file_storage import get_storage_backend

USER = "file-user-0042"
AID = "a" * 32


@pytest.fixture(autouse=True)
def storage(tmp_path):
    fx.local_storage(tmp_path)
    AA._active.clear()
    AA._start_locks.clear()
    yield
    from app.services import file_storage
    file_storage._backend = None


async def _stored(data: bytes, mime: str = "application/pdf", aid: str = AID) -> dict:
    key = f"{USER}/{aid}.original"
    await get_storage_backend().put(key, data)
    rec = {
        "attachment_id": aid, "name": "body.pdf" if mime == "application/pdf" else "long.txt",
        "mime": mime, "attachment": {"storage_path": key, "attachment_id": aid},
        "ingest": {"status": "truncated"},
    }
    await CA._write_json(CA._record_key(USER, aid), rec)
    return rec


class FakeModel:
    def __init__(self):
        self.pages: list[int] = []
        self.chunks: list[int] = []
        self.peak = 0
        self.active = 0

    async def page(self, task, number, count, native_text, image):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.001)
            self.pages.append(number)
            return f"Page {number}: image examined; task={task[:30]}"
        finally:
            self.active -= 1

    async def text_chunk(self, task, number, count, text):
        self.chunks.append(number)
        return f"Chunk {number}: {text[-40:]}"

    async def section(self, task, first, last, pages, unit_kind):
        return f"Section {first}-{last}"

    async def volume(self, task, first, last, sections, unit_kind):
        return f"Volume {first}-{last}"

    async def overview(self, task, count, volumes, failed, unit_kind, selected_range=None):
        return f"Answer for {count} {unit_kind}s; failed={failed}"

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_83_page_scan_is_all_rendered_with_bounded_concurrency(monkeypatch):
    rec = await _stored(b"%PDF-fake")
    monkeypatch.setattr(AA, "inspect_pdf", lambda data: 83)
    monkeypatch.setattr(AA, "_native_texts", lambda data, count: [""] * count)
    rendered = []
    monkeypatch.setattr(AA, "render_page", lambda data, index: rendered.append(index + 1) or b"webp")
    model = FakeModel()
    monkeypatch.setattr(AA, "_make_model", lambda: model)

    task = "هر اسلاید را با جزئیات خلاصه کن"
    state = await AA.start_analysis(USER, AID, rec, task)
    await AA._active[(USER, AID, state["analysis_id"])]
    final = AA.load_state(USER, AID, state["analysis_id"])
    assert final["status"] == "completed"
    assert sorted(rendered) == list(range(1, 84))
    assert sorted(model.pages) == list(range(1, 84))
    assert 1 < model.peak <= AA.PAGE_MODEL_CONCURRENCY
    parts = AA._delivery_parts(final)
    assert len(parts) == 10  # overview + nine 10-page batches
    assert "Page 83" in parts[-1]
    assert all(len(part) < 15_000 for part in parts[1:])


@pytest.mark.asyncio
async def test_restart_resumes_missing_pages_without_model_redo(monkeypatch):
    await _stored(b"%PDF-fake")
    monkeypatch.setattr(AA, "inspect_pdf", lambda data: 23)
    monkeypatch.setattr(AA, "_native_texts", lambda data, count: [""] * count)
    rendered = []
    monkeypatch.setattr(AA, "render_page", lambda data, index: rendered.append(index + 1) or b"webp")
    model = FakeModel()
    monkeypatch.setattr(AA, "_make_model", lambda: model)
    task = "Summarize every page"
    analysis_id = AA.analysis_id_for(task)
    state = AA.new_state(USER, AID, analysis_id, task, "body.pdf", "page", 23)
    state["status"] = "running"
    state["pages"] = [
        {"page_number": n, "status": "completed", "summary": f"Page {n}"}
        for n in range(1, 11)
    ]
    await AA.save_state(state)
    await AA.reconcile_local_analyses()
    await AA._active[(USER, AID, analysis_id)]
    final = AA.load_state(USER, AID, analysis_id)
    assert final["status"] == "completed"
    assert sorted(rendered) == list(range(11, 24))
    assert sorted(model.pages) == list(range(11, 24))


@pytest.mark.asyncio
async def test_late_page_followup_rerenders_only_page_83(monkeypatch):
    rec = await _stored(b"%PDF-fake")
    monkeypatch.setattr(AA, "inspect_pdf", lambda data: 83)
    monkeypatch.setattr(AA, "_native_texts", lambda data, count: [""] * count)
    rendered = []
    monkeypatch.setattr(AA, "render_page", lambda data, index: rendered.append(index + 1) or b"webp")
    model = FakeModel()
    monkeypatch.setattr(AA, "_make_model", lambda: model)
    state = await AA.start_analysis(
        USER, AID, rec, "What is the tiny label on slide 83?", page_start=83, page_end=83,
    )
    await AA._active[(USER, AID, state["analysis_id"])]
    final = AA.load_state(USER, AID, state["analysis_id"])
    assert final["status"] == "completed"
    assert final["overview"].startswith("Page 83:")
    assert rendered == model.pages == [83]
    assert AA.public_state(final)["selected_range"] == [83, 83]


@pytest.mark.asyncio
async def test_mixed_text_pdf_still_renders_late_visual_page(monkeypatch):
    from reportlab.pdfgen import canvas
    from app.agent.attachment_ingest import ingest_one
    from PIL import Image

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    for page in (1, 2):
        pdf.drawString(40, 700, ("Text layer headers and footers " * 8)[:240])
        pdf.showPage()
    pdf.setFillColorRGB(1, 0, 0)
    pdf.rect(100, 300, 300, 300, fill=1, stroke=0)
    pdf.showPage()
    pdf.save()
    data = buf.getvalue()
    preview = ingest_one(data, "mixed.pdf", "application/pdf")
    assert len(preview.text or "") > 200
    assert not preview.page_images  # normal chat's heuristic misses the red slide

    rec = await _stored(data)
    model = FakeModel()
    visual_pages = []

    async def observe(task, number, count, native_text, image):
        with Image.open(io.BytesIO(image)) as picture:
            rgb = picture.convert("RGB")
            # The third page has a large pure-red diagram, absent from text.
            pixels = rgb.resize((80, 100)).getdata()
            if sum(1 for r, g, b in pixels if r > 180 and g < 90 and b < 90) > 100:
                visual_pages.append(number)
        return await FakeModel.page(model, task, number, count, native_text, image)

    model.page = observe
    monkeypatch.setattr(AA, "_make_model", lambda: model)
    state = await AA.start_analysis(USER, AID, rec, "Describe every page")
    await AA._active[(USER, AID, state["analysis_id"])]
    assert AA.load_state(USER, AID, state["analysis_id"])["status"] == "completed"
    assert sorted(model.pages) == [1, 2, 3]
    assert visual_pages == [3]


@pytest.mark.asyncio
async def test_long_plain_text_uses_original_beyond_preview_and_is_user_scoped(monkeypatch, tmp_path):
    text = "A" * 220_000 + " LATE FACT: mitochondria generate ATP"
    rec = await _stored(text.encode(), "text/plain")
    model = FakeModel()
    monkeypatch.setattr(AA, "_make_model", lambda: model)
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.set_user_id("different-user")
    denied = await executor.execute("analyze_attachment", {
        "attachment_id": AID, "task": "Summarize this file",
    })
    assert denied.startswith("ERROR:")
    assert not AA._active

    executor.set_user_id(USER)
    result = json.loads(await executor.execute("analyze_attachment", {
        "attachment_id": AID, "task": "Summarize this file",
    }))
    assert result["status"] == "queued"
    await AA._active[(USER, AID, result["analysis_id"])]
    final = AA.load_state(USER, AID, result["analysis_id"])
    assert final["status"] == "completed"
    assert final["page_count"] > 17  # past the 200k turn preview
    assert model.chunks == list(range(1, final["page_count"] + 1))
    assert "mitochondria" in final["pages"][-1]["summary"]


@pytest.mark.asyncio
async def test_persian_user_request_survives_model_task_paraphrase(monkeypatch, tmp_path):
    await _stored(b"%PDF-fake")
    monkeypatch.setattr(AA, "inspect_pdf", lambda data: 4)
    monkeypatch.setattr(AA, "_native_texts", lambda data, count: [""] * count)
    monkeypatch.setattr(AA, "render_page", lambda data, index: b"webp")
    monkeypatch.setattr(AA, "_make_model", lambda: FakeModel())
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.set_user_id(USER)
    executor.set_original_user_text("هر اسلاید را جداگانه و با جزئیات خلاصه کن")
    task = "Give a detailed summary of the deck"  # model paraphrase omits 'each slide'
    result = json.loads(await executor.execute("analyze_attachment", {
        "attachment_id": AID, "task": task,
    }))
    await AA._active[(USER, AID, result["analysis_id"])]
    state = AA.load_state(USER, AID, result["analysis_id"])
    assert state["include_unit_details"] is True
    assert "Page 4" in AA._delivery_parts(state)[-1]


@pytest.mark.asyncio
async def test_central_gpt_model_override_never_comes_from_user(monkeypatch):
    from app.config import settings
    from app.services import bundle_client

    seen = []

    class FakeClient:
        def with_options(self, **kwargs):
            return self

        async def close(self):
            pass

    client = FakeClient()

    async def create(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Page evidence"))])

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    monkeypatch.setattr(bundle_client, "make_openai_client", lambda: client)
    monkeypatch.setattr(settings, "attachment_analysis_model", "owner-gpt-vision")
    model = AA.AnalysisModel()
    assert await model._call("System", "data", 100) == "Page evidence"
    assert seen[0]["model"] == "owner-gpt-vision"
    assert not any("provider" in key or "api_key" in key for key in seen[0])


def test_chat_manifest_and_followup_history_keep_scoped_original_id():
    rec = {
        "name": "body.pdf", "kind": "document", "mime": "application/pdf",
        "attachment_id": AID, "attachment": {"attachment_id": AID},
        "page_count": 83, "truncated": True, "text": "Only first pages",
    }
    blocks = AgentRunner._build_attachment_blocks(
        AgentRunner.__new__(AgentRunner), "Summarize every slide", records=[rec],
    )
    prompt = "\n".join(b["text"] for b in blocks if b["type"] == "text")
    assert AID in prompt
    assert "analyze_attachment" in prompt
    assert "all diagrams" in prompt
    assert AID in history_attachment_refs([{
        "attachment_id": AID, "filename": "body.pdf",
    }])
    assert AA.analysis_id_for("task") in history_analysis_ref(json.dumps({
        "attachment_id": AID, "attachment_analysis_id": AA.analysis_id_for("task"),
    }))


def test_request_gate_only_auto_starts_an_explicit_whole_file_summary():
    assert AA.explicit_full_document_summary("Summarize every slide in this PDF")
    assert AA.explicit_full_document_summary("هر اسلاید این فایل را خلاصه کن")
    assert AA.explicit_full_document_summary("Summarize the attached PDF")
    assert not AA.explicit_full_document_summary("I attached the PDF for later")
    assert not AA.explicit_full_document_summary("Summarize page 83 of the PDF")
    assert not AA.explicit_full_document_summary("What does the chart on page 83 say?")
    assert not AA.explicit_full_document_summary("Summarize all points on page 83 of this PDF")
    assert not AA.explicit_full_document_summary("Summarize all slides 1-10 of this PDF")
    assert not AA.explicit_full_document_summary("همه صفحات ۱ تا ۱۰ این فایل را خلاصه کن")
    assert AA.explicit_current_attachment_summary("Summarize this")
    assert AA.explicit_current_attachment_summary("Please recap it for me!")
    assert AA.explicit_current_attachment_summary("این رو خلاصه کن")
    assert not AA.explicit_current_attachment_summary("Summarize this page")
    assert not AA.explicit_current_attachment_summary("I attached this for later")
    assert AA.references_prior_attachment("Summarize the earlier PDF")
    assert AA.references_prior_attachment("Summarize every slide of the previous PDF")
    assert AA.references_prior_attachment("Summarize the PDF I sent earlier")
    assert AA.references_prior_attachment("Summarize the PDF from yesterday")
    assert AA.references_prior_attachment("Summarize the PDF I uploaded before")
    assert AA.references_prior_attachment("Summarize the other PDF")
    assert AA.references_prior_attachment("Summarize the PDF I sent last week")
    assert AA.references_prior_attachment("Compare this PDF with the other PDF")
    assert AA.references_prior_attachment("فایل قبلی را خلاصه کن")
    assert AA.references_prior_attachment("پی دی اف دیروز را خلاصه کن")
    assert AA.references_prior_attachment("فایل دیگری را خلاصه کن")
    assert not AA.references_prior_attachment("Summarize the attached PDF")
    assert not AA.references_prior_attachment("Summarize the last page of this PDF")
    assert not AA.references_prior_attachment("What is the last version of Python?")
    assert not AA.references_prior_attachment("Summarize this PDF before my meeting")


@pytest.mark.asyncio
async def test_final_83_page_batches_persist_once_in_anchor_day_and_broadcast_background(monkeypatch):
    from sqlalchemy import select
    from app.db.database import async_session_maker
    from app.db.models import Conversation, DayChat, Message, User
    from app.api import ws_chat

    broadcasts = []

    async def capture(uid, event, exclude=None):
        broadcasts.append((uid, event))
        return 1

    monkeypatch.setattr(ws_chat, "broadcast_to_user", capture)
    session_id = "c" * 36
    anchor_id = "m" * 36
    old_day_id = "d" * 36
    actual_day_id = "e" * 36
    async with async_session_maker() as db:
        db.add(User(id=USER, email="file-user@example.test", hashed_password="x"))
        db.add(DayChat(id=old_day_id, user_id=USER, local_date=date.today() - timedelta(days=1)))
        db.add(DayChat(id=actual_day_id, user_id=USER, local_date=date.today()))
        db.add(Conversation(id=session_id, user_id=USER, day_chat_id=old_day_id, channel="app"))
        db.add(Message(id=anchor_id, conversation_id=session_id, day_chat_id=actual_day_id,
                       role="assistant", channel="app", content="I am processing the file."))
        await db.commit()

    task = "هر اسلاید را با جزئیات خلاصه کن"
    state = AA.new_state(USER, AID, AA.analysis_id_for(task), task, "body.pdf", "page", 83,
                         session_id, "app", anchor_id, True)
    state["status"] = "completed"
    state["stage"] = "complete"
    state["overview"] = "Whole document overview"
    state["pages"] = [
        {"page_number": n, "status": "completed", "summary": f"Page {n} summary"}
        for n in range(1, 84)
    ]
    await AA.save_state(state)
    await AA.deliver_analysis(state)
    await AA.deliver_analysis(state)

    async with async_session_maker() as db:
        rows = (await db.execute(select(Message).where(
            Message.conversation_id == session_id,
            Message.source == "attachment_analysis",
        ).order_by(Message.created_at))).scalars().all()
    assert len(rows) == 10
    assert len({m.id for m in rows}) == 10
    assert all(m.day_chat_id == actual_day_id for m in rows)
    assert any("Page 83" in m.content for m in rows)
    assert len(broadcasts) == 10
    assert all(event["source"] == "attachment_analysis" and event["background"] is True
               for uid, event in broadcasts)
    assert all(event["id"] in {m.id for m in rows} for uid, event in broadcasts)


@pytest.mark.asyncio
async def test_agent_db_claim_uses_server_clock_and_fences_stale_worker(monkeypatch):
    from app.db.database import async_session_maker
    from app.db.models import AttachmentAnalysisJob, User

    class SkewedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(minutes=14)

        @classmethod
        def utcnow(cls):
            return datetime.utcnow() + timedelta(minutes=14)

    monkeypatch.setattr(AA, "_agent_mode", lambda: True)
    monkeypatch.setattr(AA, "datetime", SkewedDateTime)
    async with async_session_maker() as db:
        db.add(User(id=USER, email="lease-user@example.test", hashed_password="x"))
        await db.commit()

    analysis_id = AA.analysis_id_for("Summarize every page")
    state = AA.new_state(USER, AID, analysis_id, "Summarize every page", "body.pdf", "page", 83)
    await AA.save_state(state)
    first = await AA._claim_analysis(USER, AID, analysis_id)
    assert first and first["_claim_token"]
    assert await AA._claim_analysis(USER, AID, analysis_id) is None

    async with async_session_maker() as db:
        row = await db.get(AttachmentAnalysisJob, AA._job_id(USER, AID, analysis_id))
        now = await AA._server_now(db)
        assert 118 <= (row.claim_expires_at - now).total_seconds() <= 120
        row.claim_expires_at = now - timedelta(seconds=1)
        await db.commit()

    second = await AA._claim_analysis(USER, AID, analysis_id)
    assert second and second["_claim_token"] != first["_claim_token"]
    first["pages"].append({"page_number": 1, "status": "completed", "summary": "stale"})
    with pytest.raises(AA.LeaseLost):
        await AA._checkpoint(first)
    second["pages"].append({"page_number": 1, "status": "completed", "summary": "winner"})
    await AA._checkpoint(second)
    committed = await AA.load_state_async(USER, AID, analysis_id)
    assert committed["pages"][0]["summary"] == "winner"


@pytest.mark.asyncio
async def test_agent_db_reconciler_resumes_abandoned_page_47_without_user_poll(monkeypatch):
    from app.db.database import async_session_maker
    from app.db.models import AttachmentAnalysisJob, User

    await _stored(b"%PDF-fake")
    monkeypatch.setattr(AA, "_agent_mode", lambda: True)
    monkeypatch.setattr(AA, "inspect_pdf", lambda data: 83)
    monkeypatch.setattr(AA, "_native_texts", lambda data, count: [""] * count)
    rendered = []
    monkeypatch.setattr(AA, "render_page", lambda data, index: rendered.append(index + 1) or b"webp")
    model = FakeModel()
    monkeypatch.setattr(AA, "_make_model", lambda: model)
    async with async_session_maker() as db:
        db.add(User(id=USER, email="resume-user@example.test", hashed_password="x"))
        await db.commit()

    task = "Summarize every page"
    analysis_id = AA.analysis_id_for(task)
    state = AA.new_state(USER, AID, analysis_id, task, "body.pdf", "page", 83)
    state["status"] = "running"
    state["pages"] = [
        {"page_number": n, "status": "completed", "summary": f"Page {n}"}
        for n in range(1, 48)
    ]
    await AA.save_state(state)
    async with async_session_maker() as db:
        row = await db.get(AttachmentAnalysisJob, AA._job_id(USER, AID, analysis_id))
        row.claim_token = "crashed-container"
        row.claim_expires_at = await AA._server_now(db) - timedelta(seconds=1)
        await db.commit()

    await AA.reconcile_local_analyses()
    await AA._active[(USER, AID, analysis_id)]
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["status"] == "completed"
    assert sorted(rendered) == list(range(48, 84))
    assert sorted(model.pages) == list(range(48, 84))
    assert final["delivered"] is True  # no chat target on this synthetic job


@pytest.mark.asyncio
async def test_agent_db_start_reloads_winning_insert_and_preserves_both_sessions(monkeypatch):
    from app.db.database import async_session_maker
    from app.db.models import User

    rec = await _stored(b"%PDF-fake")
    monkeypatch.setattr(AA, "_agent_mode", lambda: True)
    monkeypatch.setattr(AA, "inspect_pdf", lambda data: 2)
    monkeypatch.setattr(AA, "ensure_running", lambda *args: None)
    async with async_session_maker() as db:
        db.add(User(id=USER, email="race-user@example.test", hashed_password="x"))
        await db.commit()

    original_save = AA.save_state
    inserted_winner = False

    async def simulate_racing_insert(state):
        nonlocal inserted_winner
        if not inserted_winner:
            inserted_winner = True
            winner = dict(state)
            winner["delivery_targets"] = [{
                "session_id": "first-session", "channel": "app",
                "anchor_message_id": "first-anchor", "delivered": False,
            }]
            await original_save(winner)
            # This caller's deterministic INSERT lost in another process.
            # Its local state still lists only second-session.
            return
        await original_save(state)

    monkeypatch.setattr(AA, "save_state", simulate_racing_insert)
    state = await AA.start_analysis(
        USER, AID, rec, "Summarize every page", session_id="second-session",
        channel="app", anchor_message_id="second-anchor",
    )
    targets = (await AA.load_state_async(USER, AID, state["analysis_id"]))["delivery_targets"]
    assert {target["session_id"] for target in targets} == {"first-session", "second-session"}


@pytest.mark.asyncio
async def test_agent_db_terminal_checkpoint_keeps_target_added_during_delivery(monkeypatch):
    from app.db.database import async_session_maker
    from app.db.models import User

    monkeypatch.setattr(AA, "_agent_mode", lambda: True)
    async with async_session_maker() as db:
        db.add(User(id=USER, email="delivery-race@example.test", hashed_password="x"))
        await db.commit()
    analysis_id = AA.analysis_id_for("Summarize every page")
    state = AA.new_state(USER, AID, analysis_id, "Summarize every page", "body.pdf", "page", 2,
                         "first-session", "app", "first-anchor")
    state["status"] = "completed"
    state["overview"] = "Overview"
    await AA.save_state(state)

    stale_delivery = await AA.load_state_async(USER, AID, analysis_id)
    newer = await AA.load_state_async(USER, AID, analysis_id)
    newer["delivery_targets"].append({
        "session_id": "second-session", "channel": "app",
        "anchor_message_id": "second-anchor", "delivered": False,
    })
    await AA.save_state(newer)
    stale_delivery["delivery_targets"][0]["delivered"] = True
    stale_delivery["delivered"] = True
    await AA.save_state(stale_delivery)
    final = await AA.load_state_async(USER, AID, analysis_id)
    targets = {target["session_id"]: target for target in final["delivery_targets"]}
    assert targets["first-session"]["delivered"] is True
    assert targets["second-session"]["delivered"] is False
    assert final["delivered"] is False


@pytest.mark.asyncio
async def test_agent_db_missing_destination_is_terminal_but_new_chat_can_join(monkeypatch):
    from app.db.database import async_session_maker
    from app.db.models import User

    monkeypatch.setattr(AA, "_agent_mode", lambda: True)
    async with async_session_maker() as db:
        db.add(User(id=USER, email="missing-session@example.test", hashed_password="x"))
        await db.commit()
    task = "Summarize every page"
    analysis_id = AA.analysis_id_for(task)
    state = AA.new_state(USER, AID, analysis_id, task, "body.pdf", "page", 1,
                         "deleted-session", "app", None)
    state["status"] = "completed"
    state["overview"] = "summary"
    await AA.save_state(state)
    await AA.deliver_analysis(state)
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["delivered"] is True
    assert final["delivery_targets"][0]["delivery_failed"] == "destination_unavailable"

    # The completed analysis remains reusable in a new, owned session.
    newer = await AA.load_state_async(USER, AID, analysis_id)
    newer["delivery_targets"].append({
        "session_id": "new-session", "channel": "app",
        "anchor_message_id": None, "delivered": False,
    })
    newer["delivered"] = False
    await AA.save_state(newer)
    assert (await AA.load_state_async(USER, AID, analysis_id))["delivered"] is False


@pytest.mark.asyncio
async def test_stale_worker_cannot_deliver_a_nonterminal_canonical_checkpoint(monkeypatch):
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message, User

    monkeypatch.setattr(AA, "_agent_mode", lambda: True)
    session_id = "stale-worker-session"
    async with async_session_maker() as db:
        db.add(User(id=USER, email="stale-worker@example.test", hashed_password="x"))
        db.add(Conversation(id=session_id, user_id=USER, channel="app"))
        await db.commit()
    task = "Summarize every page"
    analysis_id = AA.analysis_id_for(task)
    canonical = AA.new_state(USER, AID, analysis_id, task, "body.pdf", "page", 83,
                             session_id, "app", None)
    await AA.save_state(canonical)
    stale = dict(canonical)
    stale["status"] = "completed"
    stale["overview"] = "stale answer"
    await AA.deliver_analysis(stale)
    async with async_session_maker() as db:
        rows = (await db.execute(select(Message).where(
            Message.source == "attachment_analysis",
        ))).scalars().all()
    assert rows == []
    assert (await AA.load_state_async(USER, AID, analysis_id))["status"] == "queued"


@pytest.mark.asyncio
async def test_recent_file_locator_is_user_scoped_across_days_and_gated_for_plain_chat(monkeypatch):
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message, User

    other_user = "file-user-other"
    async with async_session_maker() as db:
        db.add_all([
            User(id=USER, email="recent-owner@example.test", hashed_password="x"),
            User(id=other_user, email="recent-other@example.test", hashed_password="x"),
            Conversation(id="old-owner-session", user_id=USER, channel="app"),
            Conversation(id="old-other-session", user_id=other_user, channel="app"),
            Message(id="old-owner-message", conversation_id="old-owner-session", role="user",
                    content="Please read this PDF", created_at=datetime.utcnow() - timedelta(days=1),
                    attachments=[{"attachment_id": AID, "filename": "body.pdf"}]),
            Message(id="old-other-message", conversation_id="old-other-session", role="user",
                    content="Private PDF", created_at=datetime.utcnow() - timedelta(days=1),
                    attachments=[{"attachment_id": "b" * 32, "filename": "secret.pdf"}]),
        ])
        await db.commit()
    async with async_session_maker() as db:
        refs = await recent_user_file_refs(db, USER)
    assert AID in refs
    assert "b" * 32 not in refs
    assert AA.likely_file_followup("What was on page 83 of that PDF?")
    assert AA.likely_file_followup("صفحه ۸۳ آن فایل چه بود؟")
    assert AA.likely_file_followup("Hi, summarize that PDF")
    assert not AA.likely_file_followup("Hi")
    assert not AA.likely_file_followup("How is the weather today?")


@pytest.mark.parametrize(
    "precreate_session,channel,original,ack,unit_details",
    [
        (True, "app", "هر اسلاید این فایل را جداگانه خلاصه کن", "دارم تمام فایل", True),
        (False, "app", "هر اسلاید این فایل را جداگانه خلاصه کن", "دارم تمام فایل", True),
        (True, "mobile", "هر اسلاید این فایل را جداگانه خلاصه کن", "دارم تمام فایل", True),
        (True, "mobile", "Summarize this", "I’m analyzing the full file", False),
    ],
    ids=["existing-session", "first-turn", "owner-mobile-canary", "deictic-current-file"],
)
@pytest.mark.asyncio
async def test_ordinary_chat_explicit_summary_starts_original_job_without_preview_answer(
    monkeypatch, tmp_path, precreate_session, channel, original, ack, unit_details,
):
    import uuid
    import app.agent.agent_runner as ar
    from app.config import settings
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message, User

    rec = await _stored(b"%PDF-fake")
    monkeypatch.setattr(AA, "inspect_pdf", lambda data: 3)
    monkeypatch.setattr(AA, "_native_texts", lambda data, count: [""] * count)
    rendered = []
    monkeypatch.setattr(AA, "render_page", lambda data, index: rendered.append(index + 1) or b"webp")
    monkeypatch.setattr(AA, "_make_model", lambda: FakeModel())

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    class NoPreviewLLM:
        async def create_message_stream(self, **kwargs):
            raise AssertionError("ordinary model must not answer from first-page preview")
            yield  # pragma: no cover

    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    # Exercise the PDF kickoff with the owner wire pilot enabled. A current
    # attachment/full-file request must still take the durable job path before
    # any ordinary preview-based model answer.
    monkeypatch.setattr(ar, "intent_wire_prune_enabled", lambda uid: channel == "mobile" and uid == USER)
    monkeypatch.setattr(settings, "agent_model", "gpt-4o")
    session_id = str(uuid.uuid4()) if precreate_session else None
    async with async_session_maker() as db:
        if (not precreate_session or channel == "mobile") and db.bind.dialect.name == "sqlite":
            # The shared in-memory SQLite fixture lets the mobile/day-chat
            # path's unrelated async DB sessions share one connection and
            # roll back its Conversation. PostgreSQL exercises persistence.
            pytest.skip("first-turn/mobile persistence requires independent DB connections")
        db.add(User(id=USER, email="persian-chat@example.test", hashed_password="x"))
        if session_id:
            db.add(Conversation(id=session_id, user_id=USER, channel="app"))
        await db.commit()

    runner = AgentRunner(NoPreviewLLM(), ToolExecutor(workspace=str(tmp_path)))
    chunks = []

    async def capture(chunk):
        chunks.append(chunk)

    response = await runner.run(
        user_message="[CONTEXT: app reply quote]\n" + original,
        display_user_message=original,
        display_request=original,
        user_id=USER, session_id=session_id, channel=channel,
        attachment_records=[{
            "attachment_id": AID, "name": "body.pdf", "kind": "document",
            "mime": "application/pdf", "status": "truncated", "page_count": 3,
            "truncated": True, "text": "Only page one preview",
        }],
        inbound_attachments=[rec["attachment"]],
        client_msg_id="client-persian-1", on_text_chunk=capture,
        disable_post_processing=True,
    )
    assert ack in response.text
    assert chunks == [response.text]
    analysis_id = AA.analysis_id_for(original)
    await AA._active[(USER, AID, analysis_id)]
    state = AA.load_state(USER, AID, analysis_id)
    assert sorted(rendered) == [1, 2, 3]
    assert state["include_unit_details"] is unit_details
    assert state["delivery_targets"][0]["session_id"] == response.session_id
    async with async_session_maker() as db:
        conv = await db.get(Conversation, response.session_id)
        assert conv is not None and conv.user_id == USER
        rows = (await db.execute(
            select(Message).where(Message.conversation_id == response.session_id)
        )).scalars().all()
    assert any(m.id == response.asst_message_id and ack in m.content for m in rows)
    assert any(m.source == "attachment_analysis" and "Page 3" in m.content for m in rows)
    assert any(m.role == "user" and m.client_msg_id == "client-persian-1" for m in rows)


@pytest.mark.parametrize(
    "task_text", ["Summarize the PDF from yesterday", "Compare this PDF with the other PDF"],
)
@pytest.mark.asyncio
async def test_new_pdf_b_does_not_steal_explicit_earlier_pdf_a_request(monkeypatch, tmp_path, task_text):
    import uuid
    import app.agent.agent_runner as ar
    from app.config import settings
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message, User
    import app.agent.attachment_provenance as provenance

    bid = "b" * 32
    b_record = await _stored(b"%PDF-new", aid=bid)
    old_session = str(uuid.uuid4())
    current_session = str(uuid.uuid4())
    async with async_session_maker() as db:
        if db.bind.dialect.name == "sqlite":
            pytest.skip("mobile/day-chat persistence requires independent DB connections")
        db.add_all([
            User(id=USER, email="earlier-pdf@example.test", hashed_password="x"),
            Conversation(id=old_session, user_id=USER, channel="mobile"),
            Conversation(id=current_session, user_id=USER, channel="mobile"),
            Message(
                id=str(uuid.uuid4()), conversation_id=old_session, role="user",
                content="Read this earlier PDF", created_at=datetime.utcnow() - timedelta(days=1),
                attachments=[{"attachment_id": AID, "filename": "earlier.pdf"}],
            ),
        ])
        await db.commit()

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    model_calls = []

    class NoPreviewLLM:
        async def create_message_stream(self, **kwargs):
            model_calls.append(kwargs)
            raise AssertionError("earlier-PDF request must clarify before preview model")
            yield  # pragma: no cover

    starts = []

    async def forbidden_start(*args, **kwargs):
        starts.append((args, kwargs))
        raise AssertionError("new PDF B must not be analyzed for an earlier-PDF request")

    async def forbidden_locator(*args, **kwargs):
        raise AssertionError("current B must not be offered as an earlier file")

    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    monkeypatch.setattr(ar, "intent_wire_prune_enabled", lambda uid: uid == USER)
    monkeypatch.setattr(AA, "start_analysis", forbidden_start)
    monkeypatch.setattr(provenance, "recent_user_file_refs", forbidden_locator)
    monkeypatch.setattr(settings, "agent_model", "gpt-4o")

    chunks = []

    async def capture(chunk):
        chunks.append(chunk)

    response = await AgentRunner(
        NoPreviewLLM(), ToolExecutor(workspace=str(tmp_path)),
    ).run(
        user_message=task_text,
        user_id=USER, session_id=current_session, channel="mobile",
        attachment_records=[{
            "attachment_id": bid, "name": "new.pdf", "kind": "document",
            "mime": "application/pdf", "status": "truncated", "page_count": 3,
            "truncated": True, "text": "Preview of new PDF B",
        }],
        inbound_attachments=[b_record["attachment"]],
        on_text_chunk=capture,
        disable_post_processing=True,
    )
    assert starts == []
    assert model_calls == []
    assert "Which file should I use for your request?" in response.text
    assert "analyzing" not in response.text.lower()
    assert chunks == [response.text]


@pytest.mark.asyncio
async def test_unrelated_ordinary_chat_never_scans_recent_file_rows(monkeypatch, tmp_path):
    import uuid
    import app.agent.agent_runner as ar
    import app.agent.attachment_provenance as provenance
    from app.config import settings
    from app.db.database import async_session_maker
    from app.db.models import User
    from app.services.openai_agent_service import StreamEvent

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    async def forbidden_locator(db, user_id):
        raise AssertionError("unrelated turn queried seven days of file history")

    class OneSentenceLLM:
        async def create_message_stream(self, **kwargs):
            yield StreamEvent(type="text", text="Hello.")
            yield StreamEvent(type="message_end", stop_reason="end_turn",
                              usage={"input_tokens": 5, "output_tokens": 2})

    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(provenance, "recent_user_file_refs", forbidden_locator)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    monkeypatch.setattr(settings, "agent_model", "gpt-4o")
    async with async_session_maker() as db:
        db.add(User(id=USER, email="plain-chat@example.test", hashed_password="x"))
        await db.commit()
    runner = AgentRunner(OneSentenceLLM(), ToolExecutor(workspace=str(tmp_path)))
    response = await runner.run(
        user_message="Hello, how are you?", user_id=USER,
        session_id=str(uuid.uuid4()), channel="app", disable_post_processing=True,
    )
    assert response.text == "Hello."


@pytest.mark.asyncio
async def test_production_agent_reconcile_loop_runs_and_cancels(monkeypatch):
    import agent_main

    called = asyncio.Event()

    async def one_pass():
        called.set()

    monkeypatch.setattr(AA, "reconcile_local_analyses", one_pass)
    task = asyncio.create_task(agent_main.attachment_analysis_reconcile_loop())
    await asyncio.wait_for(called.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    source = Path(agent_main.__file__).read_text()
    assert "attachment_reconciler_task = asyncio.create_task(" in source
    assert "attachment_analysis_reconcile_loop(), name=\"attachment-analysis-reconcile\"" in source


SAMPLE = Path(os.environ.get("TOUP_LONG_PDF_SAMPLE") or "/nonexistent/optional-long-pdf.pdf")


@pytest.mark.skipif(not SAMPLE.exists(), reason="owner's local 83-page scan not present")
def test_real_owner_sample_all_pages_are_renderable_without_model_calls():
    data = SAMPLE.read_bytes()
    assert AA.inspect_pdf(data) == 83
    for index in range(83):
        assert AA.render_page(data, index).startswith(b"RIFF")
