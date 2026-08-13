"""The first view of a generated document must not pay for the conversion.

Measured 2026-08-13 against a real user file on their own agent:

    cached          0.2 s
    warm convert    3.3 s
    cold convert   10.3 s

Every generated DOCX/PPTX is cold exactly once — on the view immediately
after generation, which is the view the user always takes. So the single
time the conversion cost is paid is the single time it is visible, and
for a while it was not merely slow but fatal: the platform proxy allowed
15 s, and a request that overran rendered as pdf.js's "Missing PDF <url>",
which names the file and so reads as a corrupt document.

Warming at `_persist` overlaps the conversion with the rest of the agent's
turn, so the pane opens from cache.

Three properties, each of which has a way of going wrong quietly:

  * it warms the types that HAVE a conversion, and nothing else;
  * it never fails the tool call that produced the document;
  * the task is strongly referenced, because a bare `create_task` may be
    garbage-collected mid-flight and drop the warm it just scheduled.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from app.agent import doc_generators as dg


class _FakeBackend:
    def __init__(self):
        self.written = []

    async def put(self, key, data):
        self.written.append(key)


@pytest.fixture
def warmed(monkeypatch):
    """Capture the storage keys handed to the converter."""
    monkeypatch.setattr(dg, "get_storage_backend", lambda: _FakeBackend())
    seen: list[str] = []
    mod = types.ModuleType("app.services.doc_preview")

    async def render_to_pdf(key):
        seen.append(key)

    mod.render_to_pdf = render_to_pdf  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.services.doc_preview", mod)
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("mime,should_warm", [
    (dg.MIME_DOCX, True),
    (dg.MIME_PPTX, True),
    (dg.MIME_PDF, False),   # already a PDF — nothing to convert
    (dg.MIME_XLSX, False),  # rendered as HTML tables, never through LibreOffice
    (dg.MIME_MD, False),
])
async def test_warms_only_what_libreoffice_converts(warmed, mime, should_warm):
    await dg._persist(b"x", "f.bin", mime, "u1")
    await asyncio.sleep(0.05)
    assert bool(warmed) is should_warm, (
        f"{mime}: expected warm={should_warm}, got {warmed}. Warming a type "
        f"with no conversion burns a LibreOffice process per file; skipping "
        f"one that has a conversion puts the 10.3 s back on the user."
    )


@pytest.mark.asyncio
async def test_a_failing_warm_never_fails_the_document(monkeypatch):
    """The tool call that made the file must survive a broken converter."""
    monkeypatch.setattr(dg, "get_storage_backend", lambda: _FakeBackend())
    mod = types.ModuleType("app.services.doc_preview")

    async def render_to_pdf(key):
        raise RuntimeError("libreoffice is on fire")

    mod.render_to_pdf = render_to_pdf  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.services.doc_preview", mod)

    att = await dg._persist(b"x", "report.docx", dg.MIME_DOCX, "u1")
    await asyncio.sleep(0.05)
    assert att.filename == "report.docx"
    assert att.storage_path.endswith("report.docx")


@pytest.mark.asyncio
async def test_the_warm_task_is_strongly_referenced(warmed):
    """A bare create_task can be collected mid-flight — asyncio keeps only
    a weak reference. The set is what makes the warm survive to run."""
    before = len(dg._PREWARM_TASKS)
    await dg._persist(b"x", "a.docx", dg.MIME_DOCX, "u1")
    assert len(dg._PREWARM_TASKS) > before, (
        "_persist scheduled no tracked task — a warm that is only weakly "
        "referenced can vanish before it converts anything"
    )
    await asyncio.sleep(0.05)
    assert warmed, "the tracked task never ran"
    # ...and it must clean up after itself, or the set grows unbounded.
    assert len(dg._PREWARM_TASKS) == before


def test_no_running_loop_is_a_no_op():
    """`_persist` is awaited in production, but the helper must not explode
    if it is ever reached from a synchronous context."""
    dg._prewarm_preview("some/key.docx", dg.MIME_DOCX)  # must not raise
