"""Tests for `send_infra_alert` (onboarding incident 2026-09-12, L3-6..L3-9).

`alerting.py` had no test file. These execute the REAL `send_infra_alert` with
the HTTP client mocked, pinning the two behaviours the incident needed:

  * L3-8 — the rate-limit window is keyed on (category, subject), so one
    permanently-stuck account cannot blanket-suppress every OTHER self-heal
    event fleet-wide for the whole window (and `subject=None` preserves the old
    category-wide behaviour for callers that have no per-subject identity).
  * L3-7 — the window is consumed only after a CONFIRMED 2xx from Telegram, so
    a 429 (the canonical result of two replicas posting 0.3 s apart) does not
    silently suppress the next window against a message nobody received.
"""
from __future__ import annotations

import asyncio

import pytest

from app.services import alerting


class _Resp:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _telegram_config(monkeypatch):
    """Make the token/chat_id truthy so `send_infra_alert` does not early-return
    on 'no config', and start each test with a clean rate-limit table."""
    monkeypatch.setattr(alerting.settings, "infra_alert_telegram_token", "T", raising=False)
    monkeypatch.setattr(alerting.settings, "infra_alert_telegram_chat_id", "C", raising=False)
    alerting.reset_for_tests()
    yield
    alerting.reset_for_tests()


def _install_fake_client(monkeypatch, statuses):
    """Patch `alerting.httpx.AsyncClient`. `statuses` is consumed across ALL
    sends in the test (each send opens its own client); returns a `posts` list
    that records the JSON body of every POST that was actually issued."""
    posts: list = []
    seq = list(statuses)

    class _Client:
        def __init__(self, *a, **k):
            ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kw):
            posts.append(kw.get("json", {}))
            status = seq[min(len(posts) - 1, len(seq) - 1)]
            return _Resp(status)

    monkeypatch.setattr(alerting.httpx, "AsyncClient", _Client)
    return posts


def test_rate_limit_is_keyed_on_category_and_subject(monkeypatch):
    """Two DIFFERENT subjects in the same category must both send inside one
    window; the SAME subject again is suppressed. One stuck account cannot
    blanket-suppress alerts about a different one."""
    posts = _install_fake_client(monkeypatch, [200])

    ok_a = asyncio.run(alerting.send_infra_alert(
        "selfheal", "warning", "a stuck", subject="aaaa1111"))
    ok_b = asyncio.run(alerting.send_infra_alert(
        "selfheal", "warning", "b stuck", subject="bbbb2222"))

    assert ok_a is True
    assert ok_b is True, "a stuck subject blanket-suppressed a DIFFERENT subject"
    assert len(posts) == 2

    # The SAME (category, subject) within the window is suppressed.
    ok_a2 = asyncio.run(alerting.send_infra_alert(
        "selfheal", "warning", "a stuck again", subject="aaaa1111"))
    assert ok_a2 is False
    assert len(posts) == 2, "the same (category, subject) must be suppressed in-window"


def test_no_subject_is_category_wide_for_legacy_callers(monkeypatch):
    """Callers with no per-subject identity (`subject=None`) keep the old
    category-wide window, so existing call sites are unaffected."""
    posts = _install_fake_client(monkeypatch, [200])
    a = asyncio.run(alerting.send_infra_alert("cat", "warning", "m1"))
    b = asyncio.run(alerting.send_infra_alert("cat", "warning", "m2"))
    assert a is True
    assert b is False
    assert len(posts) == 1


def test_window_not_consumed_on_send_failure(monkeypatch):
    """A 429 must NOT consume the window: the next send goes through instead of
    being suppressed against a message Telegram rejected."""
    # First POST 429 (two replicas raced), then 200, then would be 200.
    posts = _install_fake_client(monkeypatch, [429, 200, 200])

    first = asyncio.run(alerting.send_infra_alert(
        "selfheal", "warning", "boom", subject="aaaa1111"))
    assert first is False, "a 429 is not a delivered alert"

    # Window was NOT consumed → an immediate retry (same subject, same window)
    # SENDS rather than being suppressed.
    second = asyncio.run(alerting.send_infra_alert(
        "selfheal", "warning", "boom", subject="aaaa1111"))
    assert second is True
    assert len(posts) == 2, "the failed send silently suppressed the next window"

    # NOW the window is consumed (a 2xx landed) → a third within the window is
    # suppressed.
    third = asyncio.run(alerting.send_infra_alert(
        "selfheal", "warning", "boom", subject="aaaa1111"))
    assert third is False
    assert len(posts) == 2


def test_window_not_consumed_when_post_raises(monkeypatch):
    """A transport exception on the POST must also leave the window open."""
    calls = {"n": 0}

    class _Client:
        def __init__(self, *a, **k):
            ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("network down")
            return _Resp(200)

    monkeypatch.setattr(alerting.httpx, "AsyncClient", _Client)

    first = asyncio.run(alerting.send_infra_alert(
        "selfheal", "critical", "x", subject="s1"))
    assert first is False
    second = asyncio.run(alerting.send_infra_alert(
        "selfheal", "critical", "x", subject="s1"))
    assert second is True
    assert calls["n"] == 2, "the exception path consumed the window"


def test_confirmed_send_consumes_window_and_reports_suppressed_count(monkeypatch):
    """A confirmed 2xx consumes the window; suppressed repeats are counted and
    folded into the next successful send's body."""
    posts = _install_fake_client(monkeypatch, [200, 200])

    assert asyncio.run(alerting.send_infra_alert(
        "cat", "warning", "m", subject="s")) is True
    # Two suppressed within the window.
    assert asyncio.run(alerting.send_infra_alert(
        "cat", "warning", "m", subject="s")) is False
    assert asyncio.run(alerting.send_infra_alert(
        "cat", "warning", "m", subject="s")) is False

    # Force the window open by rewinding the last-sent clock, then send again.
    alerting._last_sent[("cat", "s")][0] = 0.0
    assert asyncio.run(alerting.send_infra_alert(
        "cat", "warning", "m", subject="s")) is True

    assert len(posts) == 2
    assert "suppressed" in posts[1]["text"], (
        "the count of suppressed repeats must be carried into the next send"
    )
    # The subject appears in the alert body so a stuck account is named.
    assert "s" in posts[1]["text"]


# ── The per-category subject cap (L10 §5c caution (i)) ────────────────
#
# Keying the window on (category, subject) is the right fix for L3-8, and it
# multiplies volume by the number of stuck subjects. On 13 Sep that number was
# nine — four ambiguous-ownership accounts plus five permanently-MCP-401
# tenants — and with W1's lease making only ONE replica send, nine pages per
# window is still a pager nobody reads. The cap collects; it never drops.

def test_category_cap_collects_further_subjects_instead_of_paging(monkeypatch):
    monkeypatch.setattr(
        alerting.settings, "infra_alert_category_subject_cap", 3, raising=False,
    )
    posts = _install_fake_client(monkeypatch, [200])

    sent = [
        asyncio.get_event_loop().run_until_complete(
            alerting.send_infra_alert("pool-selfheal-stuck", "warning",
                                      f"user {i} is 401ing", subject=f"u{i}")
        )
        for i in range(6)
    ]

    assert sent == [True, True, True, False, False, False]
    assert len(posts) == 3, "past the cap nothing is POSTed"
    # …and the capped subjects are remembered, not dropped.
    assert alerting._cat_window["pool-selfheal-stuck"]["capped"] == {"u3", "u4", "u5"}


def test_capped_subjects_are_named_in_a_digest_on_the_next_window(monkeypatch):
    monkeypatch.setattr(
        alerting.settings, "infra_alert_category_subject_cap", 2, raising=False,
    )
    posts = _install_fake_client(monkeypatch, [200])
    loop = asyncio.get_event_loop()

    for i in range(5):
        loop.run_until_complete(
            alerting.send_infra_alert("cat", "warning", f"m{i}", subject=f"u{i}")
        )
    assert len(posts) == 2

    # Roll every window: the per-subject ones AND the category one.
    for entry in alerting._last_sent.values():
        entry[0] = 0.0
    alerting._cat_window["cat"]["start"] = 0.0

    loop.run_until_complete(
        alerting.send_infra_alert("cat", "warning", "m0 again", subject="u0")
    )
    assert len(posts) == 3
    text = posts[2]["text"]
    assert "per-category cap" in text
    for name in ("u2", "u3", "u4"):
        assert name in text, f"{name} was capped and must be named: {text!r}"


def test_digest_is_cleared_once_it_has_been_delivered(monkeypatch):
    monkeypatch.setattr(
        alerting.settings, "infra_alert_category_subject_cap", 1, raising=False,
    )
    posts = _install_fake_client(monkeypatch, [200])
    loop = asyncio.get_event_loop()

    loop.run_until_complete(
        alerting.send_infra_alert("cat", "warning", "a", subject="u0"))
    loop.run_until_complete(
        alerting.send_infra_alert("cat", "warning", "b", subject="u1"))
    for entry in alerting._last_sent.values():
        entry[0] = 0.0
    alerting._cat_window["cat"]["start"] = 0.0
    loop.run_until_complete(
        alerting.send_infra_alert("cat", "warning", "a2", subject="u0"))
    assert "u1" in posts[1]["text"]

    for entry in alerting._last_sent.values():
        entry[0] = 0.0
    alerting._cat_window["cat"]["start"] = 0.0
    loop.run_until_complete(
        alerting.send_infra_alert("cat", "warning", "a3", subject="u0"))
    assert "per-category cap" not in posts[2]["text"], (
        "a digest already delivered must not repeat forever"
    )


def test_category_wide_alerts_are_never_capped(monkeypatch):
    """`subject=None` means there is only ever ONE of this alert — capping it
    would silence the sweep-quorum / fleet-level pages the cap exists to keep
    readable."""
    monkeypatch.setattr(
        alerting.settings, "infra_alert_category_subject_cap", 1, raising=False,
    )
    posts = _install_fake_client(monkeypatch, [200])
    loop = asyncio.get_event_loop()

    loop.run_until_complete(
        alerting.send_infra_alert("q", "critical", "a", subject="u0"))
    loop.run_until_complete(
        alerting.send_infra_alert("q", "critical", "fleet-wide"))
    assert len(posts) == 2
    assert "fleet-wide" in posts[1]["text"]


def test_a_failed_send_does_not_burn_a_cap_slot(monkeypatch):
    """The cap counts CONFIRMED pages. A 429 that consumed a slot would let a
    rejected message silence a real one."""
    monkeypatch.setattr(
        alerting.settings, "infra_alert_category_subject_cap", 1, raising=False,
    )
    posts = _install_fake_client(monkeypatch, [429, 200])
    loop = asyncio.get_event_loop()

    assert loop.run_until_complete(
        alerting.send_infra_alert("c", "warning", "a", subject="u0")) is False
    assert loop.run_until_complete(
        alerting.send_infra_alert("c", "warning", "b", subject="u1")) is True
    assert len(posts) == 2


def test_cap_of_zero_disables_it(monkeypatch):
    monkeypatch.setattr(
        alerting.settings, "infra_alert_category_subject_cap", 0, raising=False,
    )
    posts = _install_fake_client(monkeypatch, [200])
    loop = asyncio.get_event_loop()
    for i in range(8):
        loop.run_until_complete(
            alerting.send_infra_alert("c", "warning", f"m{i}", subject=f"u{i}"))
    assert len(posts) == 8
