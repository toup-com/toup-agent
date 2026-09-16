"""R46 F9a — deleting a Toup account logs the WhatsApp device out.

The founder saw a previous test account's agent still sitting in their own
WhatsApp → Linked Devices long after the account was gone. `grep -in whatsapp`
over `user_deletion.py` found only the name of an unrelated table: nothing in
the product ever called the logout. The capability existed and had exactly one
caller — the user's own "Disconnect" button in Settings.

Two properties, and they pull in opposite directions:
  * it must actually happen, BEFORE the container is destroyed (after that
    there is nobody to ask);
  * it must never block the deletion. A user who asked to be deleted is
    deleted whether or not their phone can be reached.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test \\
      pytest tests/test_user_deletion_whatsapp_logout.py -q -p no:cacheprovider
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve().parents[1]
_SRC = (_HERE / "app" / "services" / "user_deletion.py").read_text()

_UID = "11111111-2222-3333-4444-555555555555"


class _Row:
    def __init__(self, url, key, status):
        self.agent_url = url
        self.agent_api_key = key
        self.whatsapp_session_status = status


class _Db:
    def __init__(self, row):
        self._row = row

    async def execute(self, _stmt):
        row = self._row

        class _R:
            def first(_s):
                return row
        return _R()


class _Resp:
    def __init__(self, code):
        self.status_code = code


def _patch_httpx(monkeypatch, *, code=200, raises=None, calls=None):
    import httpx

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None):
            if calls is not None:
                calls.append((url, sorted((headers or {}).keys())))
            if raises is not None:
                raise raises
            return _Resp(code)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


@pytest.mark.asyncio
async def test_a_linked_account_is_logged_out_exactly_once(monkeypatch):
    from app.services import user_deletion as mod

    calls: list = []
    _patch_httpx(monkeypatch, code=200, calls=calls)

    ok = await mod._whatsapp_logout_best_effort(
        _Db(_Row("https://agent-abc.agents.toup.ai", "k", "linked")), _UID,
    )

    assert ok is True
    assert len(calls) == 1
    url, header_names = calls[0]
    assert url.endswith("/api/whatsapp/qr/logout")
    assert header_names == ["X-Agent-Key"]


@pytest.mark.asyncio
async def test_a_pairing_in_progress_is_also_torn_down(monkeypatch):
    """`linking` means a socket is open and an auth dir exists — leaving it is
    the same carry-over hazard as `linked`."""
    from app.services import user_deletion as mod

    calls: list = []
    _patch_httpx(monkeypatch, code=200, calls=calls)
    await mod._whatsapp_logout_best_effort(
        _Db(_Row("https://a", "k", "linking")), _UID,
    )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_even_an_explicit_never_linked_status_is_asked(monkeypatch):
    """No status gate at all (R46 fix lane A, decision D21).

    `whatsapp_session_status` is a best-effort column: it is written by a
    detached task off the qr-status poll, `_wa_poll_may_persist` can refuse the
    write outright, and the whole persist sits inside a swallowing `except`. A
    stale `not_linked` on a device that IS linked is therefore reachable, and
    it used to skip the only logout a NAMED-container tenant ever gets. The
    agent answers in microseconds when it has no adapter, so asking always is
    the cheap side of the trade.
    """
    from app.services import user_deletion as mod

    calls: list = []
    _patch_httpx(monkeypatch, code=503, calls=calls)

    for status in ("not_linked", "logged_out"):
        await mod._whatsapp_logout_best_effort(
            _Db(_Row("https://a", "k", status)), _UID,
        )
    assert len(calls) == 2, "a recorded status skipped the logout again"


def test_the_logout_decision_does_not_read_the_best_effort_status_column():
    """The behavioural case above passes for a gate on any OTHER column too;
    this one pins the specific coupling that was removed."""
    body = _SRC.split("async def _whatsapp_logout_best_effort", 1)[1].split(
        "\nasync def ", 1
    )[0]
    code = "\n".join(
        ln for ln in body.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "whatsapp_session_status" not in code, (
        "the deletion-time logout is gated on a column written only by "
        "best-effort paths again"
    )


@pytest.mark.asyncio
async def test_an_unknown_status_still_asks(monkeypatch):
    """`whatsapp_session_status` is written only by best-effort paths — the
    qr-status poll's detached task, which `_wa_poll_may_persist` can refuse
    outright — so the users whose link never persisted read as NULL. They are
    exactly the founder's case, and for a NAMED-container tenant there is no
    pool-release backstop. The agent answers 503 in microseconds when it has
    no adapter, so asking on NULL costs one round-trip.
    """
    from app.services import user_deletion as mod

    calls: list = []
    _patch_httpx(monkeypatch, code=503, calls=calls)

    assert await mod._whatsapp_logout_best_effort(
        _Db(_Row("https://a", "k", None)), _UID,
    ) is False
    assert len(calls) == 1, "a NULL status skipped the logout again"


@pytest.mark.asyncio
async def test_no_agent_row_makes_no_call(monkeypatch):
    from app.services import user_deletion as mod

    calls: list = []
    _patch_httpx(monkeypatch, code=200, calls=calls)
    assert await mod._whatsapp_logout_best_effort(_Db(None), _UID) is False
    assert calls == []


@pytest.mark.asyncio
async def test_an_unreachable_agent_does_not_abort_the_deletion(monkeypatch):
    from app.services import user_deletion as mod

    _patch_httpx(monkeypatch, raises=RuntimeError("connect timeout"))
    assert await mod._whatsapp_logout_best_effort(
        _Db(_Row("https://a", "k", "linked")), _UID,
    ) is False


@pytest.mark.asyncio
async def test_a_500_from_the_agent_does_not_abort_the_deletion(monkeypatch):
    from app.services import user_deletion as mod

    _patch_httpx(monkeypatch, code=500)
    assert await mod._whatsapp_logout_best_effort(
        _Db(_Row("https://a", "k", "linked")), _UID,
    ) is False


def test_the_logout_runs_before_the_container_is_destroyed():
    """ORDER is the whole fix. After `destroy_container` there is no agent to
    ask, and the linked device stays in the user's WhatsApp forever — which is
    exactly what the founder observed."""
    tree = ast.parse(_SRC)
    body = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "delete_user_completely":
            body = ast.unparse(node)
            break
    assert body, "delete_user_completely not found"

    logout = body.find("_whatsapp_logout_best_effort(")
    destroy = body.find("destroy_container(")
    assert logout != -1, "the deletion never logs WhatsApp out"
    assert destroy != -1
    assert logout < destroy, (
        "the WhatsApp logout runs after the container teardown — there is "
        "nothing left to answer it"
    )


def test_the_logout_is_not_wrapped_in_a_hard_fail_step():
    """A deletion must not be abortable by a phone that is off.

    Structural, not a text window: the previous version sliced 800 characters
    and split on the marker comment "BEST-EFFORT", so editing that comment away
    made the assertion vacuous. What matters is that no `try` ENCLOSING the
    call raises `DeletionAbortedError` — which is exactly what the hard-fail
    steps on either side of it do.
    """
    tree = ast.parse(_SRC)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "delete_user_completely":
            target = node
            break
    assert target is not None, "delete_user_completely not found"

    def _calls_logout(node) -> bool:
        return any(
            isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None))
            == "_whatsapp_logout_best_effort"
            for n in ast.walk(node)
        )

    assert _calls_logout(target), "the deletion never logs WhatsApp out"

    for node in ast.walk(target):
        if isinstance(node, ast.Try) and _calls_logout(node):
            for handler in node.handlers:
                raised = [
                    getattr(r.exc.func, "id", getattr(r.exc.func, "attr", None))
                    for r in ast.walk(handler)
                    if isinstance(r, ast.Raise) and isinstance(r.exc, ast.Call)
                ]
                assert "DeletionAbortedError" not in raised, (
                    "the WhatsApp logout sits inside a hard-fail step — an "
                    "unreachable phone would abort the deletion"
                )


def test_the_logout_never_logs_the_number_or_the_url():
    body = _SRC.split("async def _whatsapp_logout_best_effort", 1)[1].split("\nasync def ", 1)[0]
    for line in body.splitlines():
        if "logger." in line:
            assert "agent_url" not in line and "self_e164" not in line
            assert "user_id[:8]" in line or "%s" not in line
