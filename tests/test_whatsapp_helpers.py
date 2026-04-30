"""Unit tests for app.agent.channels.whatsapp_helpers.

These are pure-Python tests. They intentionally do not import any DB
or runtime fixtures so they pass without the project's conftest
infrastructure standing up.
"""

from __future__ import annotations

import pytest

from app.agent.channels.whatsapp_helpers import (
    RETRY_STATUS_CODES,
    WHATSAPP_TEXT_LIMIT,
    chunk_for_whatsapp,
    markdown_to_whatsapp,
    redact_phone,
    retry_status_codes,
)


# ── markdown_to_whatsapp ──────────────────────────────────────────


class TestMarkdownToWhatsapp:
    def test_empty_passthrough(self):
        assert markdown_to_whatsapp("") == ""

    def test_plain_text_unchanged(self):
        assert markdown_to_whatsapp("hello world") == "hello world"

    def test_double_star_to_single(self):
        assert markdown_to_whatsapp("hello **world**") == "hello *world*"

    def test_double_underscore_to_single_star(self):
        # GFM bold __X__ → WhatsApp bold *X*
        assert markdown_to_whatsapp("__bold__ text") == "*bold* text"

    def test_double_tilde_to_single(self):
        assert markdown_to_whatsapp("~~strike~~") == "~strike~"

    def test_single_underscore_italic_unchanged(self):
        # _italic_ is valid in both dialects; do not rewrite.
        assert markdown_to_whatsapp("_italic_") == "_italic_"

    def test_single_star_italic_unchanged(self):
        assert markdown_to_whatsapp("*italic*") == "*italic*"

    def test_fenced_code_preserved_verbatim(self):
        src = "before\n```python\nx = **2**\n```\nafter"
        out = markdown_to_whatsapp(src)
        # The ** inside the code fence must NOT be rewritten.
        assert "x = **2**" in out
        assert out.startswith("before\n")
        assert out.endswith("after")

    def test_inline_code_preserved_verbatim(self):
        src = "use `**raw**` for emphasis"
        out = markdown_to_whatsapp(src)
        assert "`**raw**`" in out

    def test_mixed_bold_outside_and_inside_code(self):
        src = "**bold** then `**code**` then **more**"
        out = markdown_to_whatsapp(src)
        assert out == "*bold* then `**code**` then *more*"

    def test_multiple_fences_round_trip_in_order(self):
        src = "```\nA\n``` and ```\nB\n```"
        out = markdown_to_whatsapp(src)
        # Both fences restored in original order.
        assert out.index("```\nA\n```") < out.index("```\nB\n```")

    def test_strike_with_other_formatting(self):
        src = "**bold** ~~gone~~ _kept_"
        assert markdown_to_whatsapp(src) == "*bold* ~gone~ _kept_"


# ── chunk_for_whatsapp ────────────────────────────────────────────


class TestChunkForWhatsapp:
    def test_empty_returns_empty_list(self):
        assert chunk_for_whatsapp("") == []
        assert chunk_for_whatsapp("   \n\n  ") == []

    def test_short_text_one_chunk(self):
        assert chunk_for_whatsapp("hello world") == ["hello world"]

    def test_under_limit_unchanged(self):
        text = "x" * (WHATSAPP_TEXT_LIMIT - 1)
        assert chunk_for_whatsapp(text) == [text]

    def test_at_limit_one_chunk(self):
        text = "x" * WHATSAPP_TEXT_LIMIT
        assert chunk_for_whatsapp(text) == [text]

    def test_paragraph_break_preferred(self):
        a = "para a " * 100  # ~700 chars
        b = "para b " * 100
        text = a.strip() + "\n\n" + b.strip()
        chunks = chunk_for_whatsapp(text, limit=900)
        # Split on the paragraph boundary, not mid-paragraph.
        assert len(chunks) == 2
        assert chunks[0].endswith("para a")
        assert chunks[1].startswith("para b")

    def test_newline_when_no_paragraph(self):
        text = "\n".join(["line " + str(i) for i in range(60)])  # ~480 chars
        chunks = chunk_for_whatsapp(text, limit=200)
        assert len(chunks) >= 2
        # No chunk exceeds the limit.
        assert all(len(c) <= 200 for c in chunks)
        # No chunk starts with whitespace.
        assert all(not c.startswith((" ", "\n", "\t")) for c in chunks)

    def test_sentence_boundary_when_no_newline(self):
        text = (
            "First sentence here. Second sentence here. "
            "Third sentence here. Fourth sentence here. "
            "Fifth sentence here."
        )
        chunks = chunk_for_whatsapp(text, limit=50)
        assert len(chunks) >= 2
        for c in chunks[:-1]:
            # Each non-final chunk should end on a sentence terminator.
            assert c.rstrip().endswith((".", "!", "?")), c

    def test_word_boundary_when_no_sentence(self):
        text = "wordone wordtwo wordthree wordfour wordfive wordsix wordseven"
        chunks = chunk_for_whatsapp(text, limit=20)
        assert len(chunks) >= 2
        # Never split a word mid-character at a word-boundary fallback.
        for c in chunks:
            for word in c.split():
                assert word in text

    def test_hard_cut_for_solid_blob(self):
        text = "X" * 50  # one solid blob, no breaks at all
        chunks = chunk_for_whatsapp(text, limit=10)
        assert len(chunks) == 5
        assert all(len(c) == 10 for c in chunks)
        assert "".join(chunks) == text

    def test_negative_limit_raises(self):
        with pytest.raises(ValueError):
            chunk_for_whatsapp("hello", limit=0)
        with pytest.raises(ValueError):
            chunk_for_whatsapp("hello", limit=-1)


# ── redact_phone ──────────────────────────────────────────────────


class TestRedactPhone:
    def test_empty(self):
        assert redact_phone("") == ""

    def test_short(self):
        # ≤4 chars: no usable margin, mask completely.
        assert redact_phone("12") == "***"
        assert redact_phone("+1") == "***"
        assert redact_phone("+123") == "***"

    def test_e164(self):
        assert redact_phone("+14155552671") == "+1***71"

    def test_long_number(self):
        assert redact_phone("123456789012345") == "12***45"


# ── retry_status_codes ────────────────────────────────────────────


class TestRetryStatusCodes:
    def test_includes_429_and_5xx(self):
        codes = set(retry_status_codes())
        assert 429 in codes
        for c in (500, 502, 503, 504):
            assert c in codes

    def test_excludes_other_4xx(self):
        codes = set(retry_status_codes())
        for c in (400, 401, 403, 404, 422):
            assert c not in codes

    def test_constant_matches_function(self):
        assert set(retry_status_codes()) == RETRY_STATUS_CODES
