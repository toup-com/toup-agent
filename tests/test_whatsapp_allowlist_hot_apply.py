"""R46 F1 — seeding the pairing number must not restart the WhatsApp adapter.

Incident 2026-09-15, container toup-agent-pool-82. The user tapped "Connect"
and waited ~14 s on "Waking Aria and creating your code…". The trail:

    14:48:19.685 [admin/bind] Binding container to user_id=… (fields: 10)
    14:48:20.7   sidecar SIGTERM
    14:48:20.990 POST /api/whatsapp/qr/pair-code → 503
    14:48:23.367 POST /api/whatsapp/qr/pair-code → 503
    14:48:26.413 POST /api/whatsapp/qr/pair-code → 200

The bind at 14:48:19.685 was the platform seeding the user's own number into
`whatsapp_baileys_allowlist` before minting the code. That reached the agent
as a config push, and `_restart_whatsapp_locked`'s fingerprint contained
`frozenset(_allowlist)` — so a list the sidecar never sees SIGTERMed a sidecar
that was mid-pairing.

The fix is structural: the allowlist leaves the fingerprint and is applied in
place. These cases pin BOTH halves — the no-restart behaviour AND the reason
it is safe (nothing reads `allowed_numbers` at socket-construction time).

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test \\
      pytest tests/test_whatsapp_allowlist_hot_apply.py -q -p no:cacheprovider
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

_HERE = pathlib.Path(__file__).resolve().parents[1]
_AGENT_MAIN_SRC = (_HERE / "agent_main.py").read_text()
_BAILEYS_SRC = (_HERE / "app" / "agent" / "channels" / "whatsapp_baileys.py").read_text()


# ── fakes ─────────────────────────────────────────────────────────────


class FakeAdapter:
    """A registered, healthy WhatsApp adapter that records what was done to it."""

    def __init__(self, allowed=None, healthy=True):
        from app.agent.channels.base import ChannelType

        self.channel_type = ChannelType.WHATSAPP
        self.allowed_numbers = set(allowed or [])
        self.applied: list = []
        self.stopped = False
        self._healthy = healthy

    def apply_allowlist(self, numbers):
        new = set(numbers or [])
        self.applied.append(sorted(new))
        if new == self.allowed_numbers:
            return False
        self.allowed_numbers = new
        return True

    def health(self):
        return {
            "started": self._healthy,
            "is_current": True,
            "sidecar_alive": self._healthy,
            "adopted_sidecar": False,
        }

    async def stop(self):
        self.stopped = True


@pytest.fixture
def wired(monkeypatch):
    """agent_main with a registered healthy adapter and a matching fingerprint."""
    import agent_main as am
    from app.agent.channels.registry import ChannelRegistry
    from app.agent.channels.base import ChannelType
    from app.config import settings

    monkeypatch.setattr(settings, "whatsapp_mode", "qr_link", raising=False)
    monkeypatch.setattr(settings, "whatsapp_phone_number_id", "", raising=False)
    monkeypatch.setattr(settings, "whatsapp_access_token", "", raising=False)
    monkeypatch.setattr(settings, "whatsapp_baileys_allowlist", "", raising=False)
    monkeypatch.setattr(am, "_agent_runner", object(), raising=False)

    adapter = FakeAdapter(allowed=set())
    ChannelRegistry.unregister(ChannelType.WHATSAPP)
    ChannelRegistry.register(adapter)
    # The fingerprint the CURRENT implementation must build for this config.
    # Hard-coded on purpose: putting the allowlist back would change the shape
    # and every case below would fall through to a restart.
    monkeypatch.setattr(
        am, "_wa_config_fingerprint", ("qr_link", "", False), raising=False,
    )
    try:
        yield am, adapter, settings
    finally:
        ChannelRegistry.unregister(ChannelType.WHATSAPP)


# ── the fingerprint ───────────────────────────────────────────────────


def _fingerprint_source() -> str:
    """The `fp = (...)` tuple inside `_restart_whatsapp_locked`."""
    tree = ast.parse(_AGENT_MAIN_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_restart_whatsapp_locked":
            for stmt in ast.walk(node):
                if (
                    isinstance(stmt, ast.Assign)
                    and any(getattr(t, "id", None) == "fp" for t in stmt.targets)
                ):
                    return ast.unparse(stmt.value)
    raise AssertionError("fp assignment not found in _restart_whatsapp_locked")


def test_the_config_fingerprint_contains_no_allowlist():
    """The one line that caused the incident. A fingerprint over the allowlist
    means every allowlist write is a channel restart."""
    src = _fingerprint_source()
    assert "allowlist" not in src.lower(), (
        f"the allowlist is back in the config fingerprint: {src}"
    )
    assert "frozenset" not in src, f"fingerprint still folds a set: {src}"


def test_the_fingerprint_still_covers_mode_and_credentials():
    """Removing the allowlist must not remove the things that ARE socket
    parameters — a mode or credential change still has to restart."""
    src = _fingerprint_source()
    assert "_wa_mode" in src
    assert "whatsapp_phone_number_id" in src
    assert "whatsapp_access_token" in src


# ── behaviour ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_allowlist_only_change_is_hot_applied_and_never_restarts(wired):
    am, adapter, settings = wired
    settings.whatsapp_baileys_allowlist = "+14155552671"

    await am._restart_whatsapp_locked()

    assert adapter.applied == [["+14155552671"]], (
        f"apply_allowlist not called with the new list: {adapter.applied}"
    )
    assert adapter.stopped is False, "the adapter was stopped for an allowlist change"
    assert adapter.allowed_numbers == {"+14155552671"}


@pytest.mark.asyncio
async def test_an_unchanged_allowlist_is_still_a_no_op(wired):
    """The hot-apply must not turn every routine config push into a log line
    and a set rebuild."""
    am, adapter, settings = wired
    adapter.allowed_numbers = {"+14155552671"}
    settings.whatsapp_baileys_allowlist = "+14155552671"

    await am._restart_whatsapp_locked()

    assert adapter.applied == [["+14155552671"]]
    assert adapter.stopped is False


@pytest.mark.asyncio
async def test_a_mode_change_still_restarts(wired, monkeypatch):
    """The fingerprint must still DO something: flipping to cloud_api has to
    tear the Baileys adapter down."""
    am, adapter, settings = wired
    monkeypatch.setattr(settings, "whatsapp_mode", "cloud_api", raising=False)
    monkeypatch.setattr(settings, "whatsapp_phone_number_id", "", raising=False)
    monkeypatch.setattr(settings, "whatsapp_access_token", "", raising=False)

    await am._restart_whatsapp_locked()

    assert adapter.stopped is True, "a mode change did not stop the old adapter"


@pytest.mark.asyncio
async def test_an_unhealthy_adapter_still_restarts_on_an_allowlist_change(wired):
    """The hot-apply sits INSIDE the healthy branch. A dead sidecar must still
    take the restart path — a config push is its only automatic recovery."""
    am, adapter, settings = wired
    adapter._healthy = False
    settings.whatsapp_baileys_allowlist = "+14155552671"

    await am._restart_whatsapp_locked()

    assert adapter.stopped is True


@pytest.mark.asyncio
async def test_an_adapter_without_apply_allowlist_does_not_break_the_push(wired):
    """The Cloud API adapter has no `apply_allowlist`. A push must degrade to
    today's behaviour, not raise inside a fire-and-forget restart."""
    from app.agent.channels.registry import ChannelRegistry
    from app.agent.channels.base import ChannelType

    am, adapter, settings = wired

    class NoHotApply(FakeAdapter):
        apply_allowlist = None  # not callable → the push must skip it

    adapter = NoHotApply(allowed=set())
    ChannelRegistry.unregister(ChannelType.WHATSAPP)
    ChannelRegistry.register(adapter)
    settings.whatsapp_baileys_allowlist = "+14155552671"

    await am._restart_whatsapp_locked()  # must not raise

    assert adapter.stopped is False


# ── the invariant that makes the hot-apply safe ───────────────────────


def test_allowed_numbers_is_only_read_by_the_dispatch_gate_and_health():
    """Why a hot-apply is correct at all: `allowed_numbers` is never read at
    socket-construction time. If a future consumer reads it while building the
    sidecar connection, the hot-apply silently stops being applied and this
    case is the only thing that would say so."""
    # AST, not text: a docstring that MENTIONS the attribute is not a reader,
    # and a dict key that happens to share the name is not one either.
    tree = ast.parse(_BAILEYS_SRC)
    owners: set[str] = set()

    class _Walk(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def _fn(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _fn
        visit_AsyncFunctionDef = _fn

        def visit_Attribute(self, node):
            if node.attr == "allowed_numbers":
                owners.add(self.stack[-1] if self.stack else "<module>")
            self.generic_visit(node)

    _Walk().visit(tree)

    # `start` reads it only to log the size; the other four are the writers,
    # the dispatch gate and the health surface. None of them builds a socket
    # from it, which is the whole licence for a hot-apply.
    expected = {"__init__", "apply_allowlist", "_handle_inbound_message", "health", "start"}
    assert owners <= expected, (
        "a new reader of self.allowed_numbers appeared in "
        f"{sorted(owners - expected)} — if it runs at socket construction, "
        "apply_allowlist no longer applies and this must go back to a restart"
    )


def test_apply_allowlist_normalises_the_same_way_the_constructor_does():
    """A hot-applied list that skipped `_normalize_e164` would silently stop
    matching inbound senders — the allowlist would be 'applied' and block
    everything."""
    from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel

    ch = BaileysWhatsAppChannel(allowed_numbers=["+1 (415) 555-2671"])
    built = set(ch.allowed_numbers)
    assert built, "constructor normalisation produced nothing"

    ch2 = BaileysWhatsAppChannel(allowed_numbers=[])
    assert ch2.apply_allowlist(["+1 (415) 555-2671"]) is True
    assert ch2.allowed_numbers == built


def test_apply_allowlist_stamps_the_health_field():
    from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel

    ch = BaileysWhatsAppChannel(allowed_numbers=[])
    before = ch.health()["allowlist_applied_at"]
    assert isinstance(before, float)
    ch.apply_allowlist(["+14155552671"])
    assert ch.health()["allowlist_applied_at"] >= before


def test_the_restart_path_logs_the_hot_apply():
    """The whole point is that a restart did NOT happen — the only way an
    operator can tell that apart from 'nothing was pushed' is the line."""
    assert re.search(r"allowlist hot-applied size=", _AGENT_MAIN_SRC), (
        "no hot-apply log line in agent_main"
    )
