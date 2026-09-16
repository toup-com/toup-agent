"""The reported case, at the deterministic layer: three documents, one
instruction, and what the model is actually handed.

No live model. A FakeLLM records the exact `messages` array that would go on
the wire, and the assertions are about the CONTENT the model receives — which
is the only thing the client can control. Whether a given model then writes a
good résumé is not a thing a test can pin; whether it was given all three
sources, in order, with a handle for each and the instruction as the anchor,
is.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
import attachments as fx  # noqa: E402

from app.agent.agent_runner import AgentRunner  # noqa: E402
from app.agent.attachment_ingest import ingest_one, record_from_ingested  # noqa: E402

pypdf_only = pytest.mark.skipif(not fx.have("pypdf"), reason="pypdf not installed here")


class FakeLLM:
    """Records the messages it was asked to complete. It is never called with a
    network; the point is the array."""

    def __init__(self):
        self.seen = None

    def complete(self, messages, **_kw):
        self.seen = messages
        return "ok"


def build_turn():
    # Real-length CVs: a document whose whole text layer is one word is
    # treated as SCANNED (SCAN_MIN_CHARS), which is correct behaviour and not
    # the case under test here.
    docs = [
        ("cv-2019.pdf", fx.long_text_pdf("MARINEBIOLOGIST ", 2000)),
        ("cv-2022.pdf", fx.long_text_pdf("QUANTSURVEYOR ", 2000)),
        ("cv-2024.pdf", fx.long_text_pdf("STAFFENGINEER ", 2000)),
    ]
    records = []
    for i, (name, data) in enumerate(docs):
        ing = ingest_one(data, name, "application/pdf")
        r = record_from_ingested(ing)
        r["attachment_id"] = f"att{i}"
        records.append(r)
    blocks = AgentRunner._build_attachment_blocks(
        AgentRunner.__new__(AgentRunner),
        "combine these into one resume",
        None,
        records=records,
    )
    return blocks


@pypdf_only
def test_all_three_documents_reach_the_model_with_their_own_tokens():
    llm = FakeLLM()
    llm.complete([{"role": "user", "content": build_turn()}])
    blocks = llm.seen[0]["content"]
    joined = "\n".join(b.get("text", "") for b in blocks)
    for token in ("MARINEBIOLOGIST", "QUANTSURVEYOR", "STAFFENGINEER"):
        assert token in joined, f"{token} never reached the model"


@pypdf_only
def test_each_source_has_a_handle_the_model_can_refer_to():
    blocks = build_turn()
    joined = "\n".join(b.get("text", "") for b in blocks)
    assert "[1] cv-2019.pdf" in joined
    assert "[2] cv-2022.pdf" in joined
    assert "[3] cv-2024.pdf" in joined


@pypdf_only
def test_the_sources_appear_in_the_order_they_were_attached():
    blocks = build_turn()
    joined = "\n".join(b.get("text", "") for b in blocks)
    i1 = joined.index("MARINEBIOLOGIST")
    i2 = joined.index("QUANTSURVEYOR")
    i3 = joined.index("STAFFENGINEER")
    assert i1 < i2 < i3


@pypdf_only
def test_the_instruction_is_the_last_block_not_the_first():
    """It used to be prepended to the document blob, i.e. buried under
    thousands of tokens of extracted text."""
    blocks = build_turn()
    assert blocks[-1]["text"] == "combine these into one resume"
    assert blocks[0]["text"].startswith("The user attached 3 files:")
    # …and exactly ONCE. A copy at the front as well is not "the anchor is
    # last": it is the same instruction buried under the sources plus a repeat.
    assert sum(1 for b in blocks
               if b.get("text") == "combine these into one resume") == 1


@pypdf_only
def test_the_manifest_names_every_source_up_front():
    blocks = build_turn()
    head = blocks[0]["text"]
    for name in ("cv-2019.pdf", "cv-2022.pdf", "cv-2024.pdf"):
        assert name in head
