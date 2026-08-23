"""Round 25, item 3: the look is honest about not having run, and it keeps
the picture it was shown.

Two defects, both found by reading the downgrade paths in `verify._smoke` and
both reproduced with a real browser before being fixed:

* The blocked-request path — the third of three ways the browser pass gets
  downgraded — cleared both screenshots and left `downgrade_reason` unset, so
  the card said the bare "the code checks out", which is what a build with no
  JavaScript at all also says. It also kept `skill.py`'s substitution from
  firing (that guard needs a reason to substitute), so the look row blamed the
  renderer for a run in which the renderer was fine.
* The judged screenshot — the one the vision reviewer is actually shown —
  existed only in memory for the length of one publish, so no verdict could be
  checked afterwards.

Everything here except the first test runs without a browser and without a
model, on purpose: a check that cannot run on the machine reading it is not a
check.
"""

import asyncio
import hashlib
import os
import tempfile

import pytest

from app.agent.skills.builtins.app_html import store, verify


#: ONE loop for the whole file, deliberately. `verify` keeps a shared browser
#: (round 24) and both it and the playwright driver are bound to the loop that
#: started them, so a second call on a fresh loop awaits a handle that will
#: never answer — which surfaces as the 45 s `SMOKE_TIMEOUT_S` and a report
#: reading "page never settled", i.e. as a plausible-looking wrong answer
#: rather than as an error. Observed here before it was written down.
_LOOP = asyncio.new_event_loop()


def _run(coro):
    return _LOOP.run_until_complete(coro)


#: A document that is fine in every way the gate measures.
_CLEAN = (
    "<!doctype html><html><head><title>t</title></head><body>"
    '<h1 style="color:#000;background:#fff;font-size:24px">Hello</h1>'
    "<script>document.title = 'ran';</script>"
    "</body></html>"
)

#: The same document, plus one subresource that cannot be fetched. Any of
#: them will do: the app runs under `default-src 'self'` on an opaque origin,
#: so an unreachable host and a CSP-refused one both arrive as `requestfailed`.
_BLOCKED = _CLEAN.replace(
    "</body>", '<img src="https://cdn.example.invalid/sprite.png"></body>'
)


def _can_really_render() -> bool:
    """A browser binary is NOT enough — the driver has to be importable too.

    This guard originally asked only `find_browser_bin() is None`, and that is
    exactly wrong for CI: the self-hosted runner is the fleet's own box and
    HAS Brave on PATH, while neither of the workflow's two inline pip lists
    installs playwright. So the skip did not fire, `verify_app` downgraded for
    want of a driver, and the control assertion below failed the sweep — a
    host fact reported as a defect. Ask for both halves.
    """
    if verify.find_browser_bin() is None:
        return False
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        try:
            import patchright.async_api  # noqa: F401
        except ImportError:
            return False
    return True


@pytest.mark.skipif(
    not _can_really_render(),
    reason="no browser binary AND playwright driver here — the downgrade "
           "cannot be driven, and a host fact is not a defect",
)
def test_a_request_that_never_arrived_is_named_on_the_card():
    """The third downgrade path says WHY, like the other two.

    Driven through the real browser rather than a stub, because the fault was
    in which of `_smoke`'s exits ran: a stub at that layer would have been
    written against the same wrong belief.
    """
    control = _run(verify.verify_app(_CLEAN, deep=True))
    assert "runtime" in control.ran, "the control did not open — nothing here is a test"
    assert control.downgrade_reason is None
    assert control.screenshot

    report = _run(verify.verify_app(_BLOCKED, deep=True))
    assert "runtime" in report.skipped and "runtime" not in report.ran
    # The picture goes, as it did before — a page whose subresource never
    # arrived must not be handed to the reviewer as though it were the app.
    assert report.screenshot is None and report.cover is None
    # …and the reason arrives, which is the fix.
    assert report.downgrade_reason
    assert "cdn.example.invalid" in report.downgrade_reason
    assert report.downgrade_reason in report.summary()
    assert report.summary() != "the code checks out"


def test_a_reason_is_short_enough_to_sit_inside_the_card_sentence():
    """It is rendered inside another sentence, so it is cut at one length."""
    assert verify._clip("short") == "short"
    long_url = "https://cdnjs.cloudflare.com/ajax/libs/" + "x" * 200 + ".js"
    clipped = verify._clip(long_url)
    assert len(clipped) == verify._REASON_MAX and clipped.endswith("…")
    # The same clip the exception path uses — one length, not two.
    assert verify._downgrade_reason_of(RuntimeError(long_url)) == clipped


def test_summary_distinguishes_a_dead_browser_from_a_page_with_no_script():
    """The whole point of the field, asserted at the line that reads it."""
    bare = verify.Report(ran=["syntax"])
    named = verify.Report(ran=["syntax"], downgrade_reason="couldn't reach cdnjs")
    assert bare.summary() == "the code checks out"
    assert named.summary() != bare.summary()
    assert "couldn't reach cdnjs" in named.summary()


@pytest.fixture()
def apps_root(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("TOUP_HTML_APPS_DIR", tmp)
        yield tmp


def test_the_judged_screenshot_is_kept_under_the_hash_of_what_was_judged(apps_root):
    png = b"\x89PNG\r\n\x1a\n" + b"pixels"
    digest = store.judged_digest(_CLEAN)
    # The name is derivable from the app file alone, which is the property an
    # audit needs: `shasum -a 256 ~/apps/<slug>.html | cut -c1-16`.
    assert digest == hashlib.sha256(_CLEAN.encode("utf-8")).hexdigest()[:16]

    path = store.write_judged(digest, png)
    assert path and os.path.basename(path) == f"{digest}.png"
    assert store.read_judged(digest) == png
    # 0644, not the 0600 `mkstemp` hands out — the sandboxed shell runs as an
    # unprivileged uid and an unreadable artifact is not an artifact.
    assert oct(os.stat(path).st_mode & 0o777) == "0o644"

    # An edit moves the evidence rather than overwriting it: the verdict
    # belongs to the bytes it was given.
    other = store.judged_digest(_CLEAN.replace("Hello", "Hello there"))
    assert other != digest
    store.write_judged(other, png)
    assert store.read_judged(digest) == png


def test_retention_never_fails_a_build(apps_root, monkeypatch):
    digest = store.judged_digest(_CLEAN)
    assert store.write_judged(digest, b"") is None
    assert store.write_judged(digest, b"x" * (store.MAX_JUDGED_BYTES + 1)) is None
    with pytest.raises(store.AppStoreError):
        store.judged_path("../../etc/passwd")
    # No app root on disk: nothing to be beside, and nothing conjured either.
    missing = os.path.join(apps_root, "not-there", "apps")
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", missing)
    assert store.write_judged(digest, b"\x89PNG") is None
    assert not os.path.exists(missing)


def test_the_kept_screenshots_are_capped(apps_root, monkeypatch):
    monkeypatch.setattr(store, "MAX_JUDGED_FILES", 3)
    for i in range(8):
        assert store.write_judged(store.judged_digest(f"app-{i}"), b"\x89PNG%d" % i)
    kept = os.listdir(os.path.join(apps_root, store.JUDGED_DIR))
    assert len(kept) == 3, kept


def test_nothing_is_kept_when_nothing_was_judged(apps_root, monkeypatch):
    """A downgraded run has no picture, so it must leave no evidence — and an
    operator who turns retention off gets none either."""
    store.ensure_root()
    assert verify._retain_judged(_CLEAN, None) is None
    monkeypatch.setenv("TOUP_APP_KEEP_JUDGED", "0")
    assert verify.keep_judged_enabled() is False
    assert verify._retain_judged(_CLEAN, b"\x89PNG") is None
    assert not os.path.isdir(os.path.join(apps_root, store.JUDGED_DIR))
    monkeypatch.setenv("TOUP_APP_KEEP_JUDGED", "1")
    assert verify._retain_judged(_CLEAN, b"\x89PNG")
