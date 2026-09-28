"""Toup for Mac — the relay bridge, and the tool family that rides it.

What this file pins
-------------------
`desktop/docs/RELAY_PROTOCOL.md` §5 (frames, limits, at-most-once) and
`agent-tool-relay.md` §5.8 / §2.6 precedent A — availability is an
EXECUTION-time question answered with an actionable `ERROR:` string, never a
tools-array question.

The assertion that carries the most weight here is the dullest-looking one:
**an offline Mac produces an error string that tells the model not to invent
a result.** Asked to read a file from a machine it cannot see, a model with
no instruction to the contrary will describe a plausible file. The audit asks
for the error to be the signal (§5.8); this family also has to ask the model
not to answer around it, and a description is the text the model reads at the
moment it decides.

Lane: RUN_MODE=platform. `desktop_pending_actions` is PLATFORM_ONLY; nothing
here touches an AGENT_ONLY table.

Run:
    cd backend && DATABASE_URL="sqlite+aiosqlite:///:memory:" RUN_MODE=platform \
        python -m pytest tests/test_desktop_bridge_and_skill.py -q
"""

from __future__ import annotations

import ast
import asyncio
import json
import pathlib
import uuid
from typing import Any, Dict, List

import pytest

from app.agent import desktop_bridge
from app.agent.skills.base import SkillContext
from app.agent.skills.builtins.desktop.skill import (
    ALL_TOOLS,
    CONSENT_TOOLS,
    READ_TOOLS,
    DesktopSkill,
)
from app.agent.tool_display import display_of
from app.agent.tool_executor import TOOL_OUTPUT_LIMITS
from app.config import settings
from app.db.database import async_session_maker
from app.db.models import AgentConfig, DesktopDevice, DesktopPendingAction

USER = "00000000-0000-4000-8000-0000000000aa"


@pytest.fixture(autouse=True)
def _enabled_account_cohort(monkeypatch):
    # These tests exercise the enabled protocol; rollout-off has its own tests.
    monkeypatch.setattr(settings, "desktop_connections_rollout_pct", 100)
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)


@pytest.fixture(autouse=True)
def _enabled_agent(monkeypatch):
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)

# ── A device socket that records what the server sent ────────────────
class FakeSocket:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []
        self.closed_with: Any = None
        self.fail_send = False

    async def send_text(self, text: str) -> None:
        if self.fail_send:
            raise RuntimeError("socket is gone")
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed_with = (code, reason)


@pytest.fixture(autouse=True)
def _clean_registry():
    desktop_bridge.reset_for_tests()
    yield
    desktop_bridge.reset_for_tests()


async def _connect(user_id: str = USER, device_id: str = "dev-1"):
    sock = FakeSocket()
    ep = await desktop_bridge.register(
        user_id, sock, device_id=device_id, token_jti="jti-1",
    )
    return sock, ep


def _reply(user_id: str, task_id: str, **over) -> None:
    frame = {"type": "result", "id": task_id, "ok": True, "status": "ok",
             "took_ms": 12, "summary": "Read 2 files.", "data": {"n": 2}}
    frame.update(over)
    desktop_bridge.deliver_inbound(user_id, frame)


# ══════════════════════════════════════════════════════════════════════
# 1. §5.3/§5.4 — the round trip
# ══════════════════════════════════════════════════════════════════════
async def test_a_dispatch_round_trip_resolves_the_awaited_future():
    sock, _ = await _connect()

    async def answer():
        for _ in range(50):
            if sock.sent:
                _reply(USER, sock.sent[0]["id"])
                return
            await asyncio.sleep(0.01)
        raise AssertionError("no task frame was ever sent")

    task = asyncio.create_task(answer())
    res = await desktop_bridge.dispatch(
        USER, "desktop__fs_read", {"path": "Projects/toup"}, timeout_s=5.0,
    )
    await task

    assert res["status"] == "ok"
    assert res["data"] == {"n": 2}
    assert res["summary"] == "Read 2 files."

    frame = sock.sent[0]
    assert frame["type"] == "task"
    assert frame["action"] == "desktop__fs_read"
    assert frame["params"] == {"path": "Projects/toup"}
    assert frame["timeout_ms"] == 5000
    # §5.3: "There is no account field, deliberately." The device stamps the
    # request with the account it is paired to; if the account travelled in
    # the frame, the device's "must equal the paired account" check would be
    # comparing a value to itself.
    assert not ({"user_id", "account_id", "sub", "account"} & set(frame))


async def test_a_replayed_task_id_is_honoured_so_rm_cannot_run_twice():
    """§5.3: `id` is the seam's idempotency key, and an approved pending
    action reuses the id it was staged with."""
    sock, _ = await _connect()
    pinned = "task-from-the-staged-row"

    async def answer():
        for _ in range(50):
            if sock.sent:
                _reply(USER, sock.sent[0]["id"])
                return
            await asyncio.sleep(0.01)

    t = asyncio.create_task(answer())
    await desktop_bridge.dispatch(
        USER, "desktop__exec_run", {"command": "ls"}, timeout_s=5.0,
        task_id=pinned,
    )
    await t
    assert sock.sent[0]["id"] == pinned


async def test_a_result_for_a_task_from_a_previous_socket_still_resolves():
    """§5.4: results survive a reconnect. The device keeps a bounded outbox
    and flushes it after `hello` on the next connection — and the agent is
    awaiting a future keyed by that id, so a dropped result is a HUNG TURN,
    not a lost log line. That is why the pending map is per USER and a
    disconnect deliberately leaves futures alone."""
    sock, ep = await _connect()

    async def reconnect_then_answer():
        for _ in range(50):
            if sock.sent:
                break
            await asyncio.sleep(0.01)
        task_id = sock.sent[0]["id"]
        await desktop_bridge.unregister(USER, ep)       # the Mac sleeps
        await _connect(device_id="dev-1")               # …and comes back
        _reply(USER, task_id, summary="Flushed from the outbox.")

    t = asyncio.create_task(reconnect_then_answer())
    res = await desktop_bridge.dispatch(
        USER, "desktop__fs_list", {"path": "x"}, timeout_s=5.0,
    )
    await t
    assert res["summary"] == "Flushed from the outbox."


async def test_an_id_resolves_at_most_once():
    sock, _ = await _connect()

    async def answer_twice():
        for _ in range(50):
            if sock.sent:
                break
            await asyncio.sleep(0.01)
        tid = sock.sent[0]["id"]
        _reply(USER, tid)
        _reply(USER, tid, summary="a duplicate from the outbox")

    t = asyncio.create_task(answer_twice())
    res = await desktop_bridge.dispatch(
        USER, "desktop__fs_list", {"path": "x"}, timeout_s=5.0,
    )
    await t
    assert res["summary"] == "Read 2 files."
    assert desktop_bridge._active_eps(USER)[-1].counters["orphan_results"] == 1


async def test_a_denied_result_is_not_an_error():
    """§5.4: `denied` means the user (or standing policy) refused — it is not
    a fault, and the agent should stop asking."""
    sock, _ = await _connect()

    async def answer():
        for _ in range(50):
            if sock.sent:
                _reply(USER, sock.sent[0]["id"], ok=False, status="denied",
                       summary="Outside the folders you granted.", data={})
                return
            await asyncio.sleep(0.01)

    t = asyncio.create_task(answer())
    with pytest.raises(desktop_bridge.DesktopDenied) as exc:
        await desktop_bridge.dispatch(
            USER, "desktop__fs_read", {"path": "/etc/passwd"}, timeout_s=5.0,
        )
    await t
    assert "granted" in exc.value.summary


async def test_a_result_with_no_legible_status_is_never_read_as_ok():
    """§5.4: `ok` is DERIVED from `status`, never set apart from it. Guessing
    from `ok: true` is exactly that inversion, and it would hand the model a
    payload the device never claimed was good."""
    sock, _ = await _connect()

    async def answer():
        for _ in range(50):
            if sock.sent:
                desktop_bridge.deliver_inbound(USER, {
                    "type": "result", "id": sock.sent[0]["id"], "ok": True,
                    "data": {"contents": "anything at all"},
                })
                return
            await asyncio.sleep(0.01)

    t = asyncio.create_task(answer())
    with pytest.raises(desktop_bridge.DesktopError) as exc:
        await desktop_bridge.dispatch(
            USER, "desktop__fs_read", {"path": "x"}, timeout_s=5.0,
        )
    await t
    assert exc.value.code == "bad_result"


# ══════════════════════════════════════════════════════════════════════
# 2. §5.7 — the limits, refused on this side too
# ══════════════════════════════════════════════════════════════════════
async def test_no_mac_connected_raises_unavailable():
    with pytest.raises(desktop_bridge.DesktopUnavailable):
        await desktop_bridge.dispatch(USER, "desktop__fs_list", {}, timeout_s=1)


async def test_more_than_four_tasks_in_flight_is_busy():
    sock, _ = await _connect()
    running = [
        asyncio.create_task(desktop_bridge.dispatch(
            USER, "desktop__fs_list", {"i": i}, timeout_s=5.0,
        ))
        for i in range(desktop_bridge.MAX_TASKS_IN_FLIGHT)
    ]
    for _ in range(100):
        if len(sock.sent) == desktop_bridge.MAX_TASKS_IN_FLIGHT:
            break
        await asyncio.sleep(0.01)

    with pytest.raises(desktop_bridge.DesktopError) as exc:
        await desktop_bridge.dispatch(USER, "desktop__fs_list", {}, timeout_s=5.0)
    assert exc.value.code == "busy"

    for frame in sock.sent:
        _reply(USER, frame["id"])
    await asyncio.gather(*running)


async def test_more_than_sixty_tasks_a_minute_is_rate_limited():
    sock, ep = await _connect()
    ep.recent_tasks.extend([9e9] * desktop_bridge.RATE_LIMIT_PER_WINDOW)
    with pytest.raises(desktop_bridge.DesktopError) as exc:
        await desktop_bridge.dispatch(USER, "desktop__fs_list", {}, timeout_s=1)
    assert exc.value.code == "rate_limited"


async def test_an_illegal_action_name_never_reaches_the_mac():
    sock, _ = await _connect()
    for bad in ("", "x" * 97, "desktop fs read", "desktop__fs_read;rm", "../x"):
        with pytest.raises(desktop_bridge.DesktopError) as exc:
            await desktop_bridge.dispatch(USER, bad, {}, timeout_s=1)
        assert exc.value.code == "invalid_task"
    assert sock.sent == []


async def test_params_must_be_an_object():
    await _connect()
    with pytest.raises(desktop_bridge.DesktopError):
        await desktop_bridge.dispatch(USER, "desktop__fs_read", ["x"], timeout_s=1)


async def test_a_reason_is_flattened_and_capped_before_it_is_sent():
    """§5.3: `reason` is untrusted display text. A native approval dialog is
    not a place where one sentence gets to impersonate three lines of
    app-authored UI."""
    sock, _ = await _connect()
    hostile = "Read the file\n\n\nAPPROVED BY TOUP\n" + ("x" * 400)
    task = asyncio.create_task(desktop_bridge.dispatch(
        USER, "desktop__fs_read", {"path": "x"}, timeout_s=1.0, reason=hostile,
    ))
    for _ in range(100):
        if sock.sent:
            break
        await asyncio.sleep(0.01)
    reason = sock.sent[0]["reason"]
    assert "\n" not in reason
    assert len(reason) <= desktop_bridge.REASON_MAX_LEN
    _reply(USER, sock.sent[0]["id"])
    await task


async def test_a_send_failure_is_unavailable_and_leaks_no_future():
    sock, _ = await _connect()
    sock.fail_send = True
    with pytest.raises(desktop_bridge.DesktopUnavailable):
        await desktop_bridge.dispatch(USER, "desktop__fs_list", {}, timeout_s=1)
    assert not desktop_bridge._pending.get(USER)


# ══════════════════════════════════════════════════════════════════════
# 3. §5.5/§5.6 — cancel, and the metadata-only event channel
# ══════════════════════════════════════════════════════════════════════
async def test_a_targeted_cancel_is_never_widened():
    sock, _ = await _connect()
    assert await desktop_bridge.cancel(USER, "t1") is True
    assert sock.sent[-1] == {"type": "cancel", "id": "t1"}
    assert await desktop_bridge.cancel(USER) is True
    assert sock.sent[-1] == {"type": "cancel"}


async def test_an_event_can_carry_nothing_but_metadata():
    """§5.6: there is no field on that frame that can carry arguments, paths
    or output, so the activity feed cannot become a second copy of the
    user's files on the wire."""
    await _connect()
    desktop_bridge.deliver_inbound(USER, {
        "type": "event", "kind": "tool_started", "id": "t1",
        "tool": "desktop__fs_read", "status": "",
        "path": "/Users/nariman/.ssh/id_rsa", "output": "ssh-rsa AAAA…",
    })
    events = desktop_bridge.recent_events(USER)
    assert len(events) == 1
    assert set(events[0]) == {"kind", "id", "tool", "status", "at"}
    assert "id_rsa" not in json.dumps(events[0])


async def test_an_unknown_event_kind_is_counted_not_stored():
    await _connect()
    desktop_bridge.deliver_inbound(
        USER, {"type": "event", "kind": "exfiltrate", "id": "t1"},
    )
    assert desktop_bridge.recent_events(USER) == []
    assert desktop_bridge._active_eps(USER)[-1].counters["unknown_events"] == 1


async def test_close_for_device_with_no_selector_closes_every_socket():
    a, _ = await _connect(device_id="dev-a")
    b, _ = await _connect(device_id="dev-b")
    closed = await desktop_bridge.close_for_device(USER)
    assert closed == 2
    assert a.closed_with[0] == 4003 and b.closed_with[0] == 4003
    assert desktop_bridge.is_connected(USER) is False


async def test_close_for_device_leaves_the_other_mac_alone():
    a, _ = await _connect(device_id="dev-a")
    b, _ = await _connect(device_id="dev-b")
    assert await desktop_bridge.close_for_device(USER, device_id="dev-a") == 1
    assert a.closed_with[0] == 4003
    assert b.closed_with is None
    assert desktop_bridge.connected_device_ids(USER) == ["dev-b"]


async def test_the_newest_socket_wins_a_dispatch():
    old, _ = await _connect(device_id="dev-old")
    new, _ = await _connect(device_id="dev-new")
    task = asyncio.create_task(desktop_bridge.dispatch(
        USER, "desktop__fs_list", {}, timeout_s=1.0,
    ))
    for _ in range(100):
        if new.sent:
            break
        await asyncio.sleep(0.01)
    assert new.sent and old.sent == []
    _reply(USER, new.sent[0]["id"])
    await task


# ══════════════════════════════════════════════════════════════════════
# 4. The tool family — shape, and the instructions in the descriptions
# ══════════════════════════════════════════════════════════════════════
def _tools() -> List[Dict[str, Any]]:
    return DesktopSkill().get_tools()


def test_the_family_is_the_thirteen_names_the_protocol_names():
    expected = {
        "desktop__fs_list", "desktop__fs_read", "desktop__fs_search",
        "desktop__fs_write", "desktop__fs_mkdir", "desktop__fs_move",
        "desktop__fs_trash", "desktop__exec_run", "desktop__screen_capture",
        "desktop__ui_snapshot", "desktop__ui_click", "desktop__ui_type",
        "desktop__ui_key",
    }
    names = [t["name"] for t in _tools()]
    assert len(names) == len(set(names)) == 13
    assert set(names) == expected == ALL_TOOLS
    assert READ_TOOLS.isdisjoint(CONSENT_TOOLS)
    assert READ_TOOLS | CONSENT_TOOLS == expected


def test_every_name_is_namespaced_so_the_proxy_cap_cannot_eat_it():
    """§2.2: the 128-tool cap's mechanism is contested in this repo —
    tail-trim vs un-namespaced-drop — and the two give OPPOSITE answers for
    a bare `desktop_*`. Namespaced is safe under both."""
    assert all(t["name"].startswith("desktop__") for t in _tools())


def test_every_description_says_whose_machine_and_what_a_refusal_means():
    for tool in _tools():
        d = tool["description"]
        assert "OWN Mac" in d, tool["name"]
        assert "error" in d.lower(), tool["name"]
        # The clause that stops the model answering from imagination.
        assert ("never invent" in d.lower() or "never act on it" in d.lower()), (
            tool["name"]
        )
        if tool["name"] in CONSENT_TOOLS:
            assert "FINAL for this turn" in d, tool["name"]
            assert "do not retry" in d, tool["name"]


def test_the_credential_and_dialog_rules_are_on_the_tools_that_need_them():
    by_name = {t["name"]: t["description"] for t in _tools()}
    assert "Never type a password" in by_name["desktop__ui_type"]
    assert "payment" in by_name["desktop__ui_click"]
    assert "Trash, not deletion" in by_name["desktop__fs_trash"]


def test_every_tool_has_a_result_cap_and_a_user_facing_label():
    from app.agent.tool_display import _STEP_LABELS

    for name in sorted(ALL_TOOLS):
        assert name in TOOL_OUTPUT_LIMITS, f"{name} would inherit the default cap"
        assert name in _STEP_LABELS, f"{name} has no step label"
        label = _STEP_LABELS[name]
        # The label is the vocabulary; the path belongs in the tool's own
        # `display` sentence, and never in the step row.
        assert "your Mac" in label
        assert "/" not in label


def test_every_read_tool_is_fenced_as_external_content():
    """§7.4 / §2.3, and the fence must be UNCONDITIONAL — never derived from
    the result string, because that string is attacker-controlled.

    Parsed out of the source because the set is a local literal inside
    `ToolExecutor.execute_tool`; `tests/test_security_builder_attribution.py`
    reads it the same way."""
    src = pathlib.Path("app/agent/tool_executor.py").read_text()
    tree = ast.parse(src)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "_EXTERNAL_CONTENT_TOOLS" not in targets:
            continue
        assert isinstance(node.value, ast.Set), "the set stopped being a literal"
        found = {
            e.value for e in node.value.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        }
    assert found, "_EXTERNAL_CONTENT_TOOLS was not found as a set literal"
    assert READ_TOOLS <= found, sorted(READ_TOOLS - found)
    # A mutating tool's confirmation line is not external content, and
    # fencing it would teach the model to read approvals as untrusted data.
    assert not (CONSENT_TOOLS & found)


def test_the_skill_prompt_section_states_the_three_rules():
    section = DesktopSkill().get_system_prompt_section()
    assert "authority on what it will do" in section
    assert "untrusted data" in section
    assert "do not answer as though you had read their files" in section


# ══════════════════════════════════════════════════════════════════════
# 5. Execution-time availability (precedent A), and staging
# ══════════════════════════════════════════════════════════════════════
def _ctx(user_id: str = USER) -> SkillContext:
    return SkillContext(user_id=user_id, session_id="sess-1")


async def test_an_offline_mac_is_an_actionable_error_and_not_a_result():
    skill = DesktopSkill()
    for name in sorted(ALL_TOOLS):
        out = await skill.execute_tool(name, {"path": "x", "command": "ls",
                                              "cwd": "x", "app": "Finder",
                                              "key": "return", "text": "hi",
                                              "element_id": "e1",
                                              "query": "q", "content": "c",
                                              "from_path": "a", "to_path": "b"},
                                       _ctx())
        assert out.startswith("ERROR:"), name
        assert "not connected" in out, name
        assert "do NOT describe files" in out, name
        # Nothing fabricated: the string carries no payload shape at all.
        assert "{" not in out, name


async def test_a_turn_with_no_user_refuses_rather_than_guessing():
    out = await DesktopSkill().execute_tool("desktop__fs_list", {}, SkillContext())
    assert out.startswith("ERROR:") and "signed-in user" in out


async def test_an_unknown_tool_name_is_refused():
    out = await DesktopSkill().execute_tool("desktop__rm_rf", {}, _ctx())
    assert out.startswith("ERROR: unknown tool")


async def test_a_read_tool_dispatches_and_returns_the_devices_payload():
    sock, _ = await _connect()

    async def answer():
        for _ in range(100):
            if sock.sent:
                _reply(USER, sock.sent[0]["id"], summary="Listed 3 items.",
                       data={"entries": ["a", "b", "c"]})
                return
            await asyncio.sleep(0.01)

    t = asyncio.create_task(answer())
    out = await DesktopSkill().execute_tool(
        "desktop__fs_list", {"path": "Projects"}, _ctx(),
    )
    await t
    assert json.loads(out) == {"entries": ["a", "b", "c"]}
    assert display_of(out) == "Listed 3 items."


async def test_a_refusal_tells_the_model_to_stop_rather_than_reroute():
    sock, _ = await _connect()

    async def answer():
        for _ in range(100):
            if sock.sent:
                _reply(USER, sock.sent[0]["id"], ok=False, status="denied",
                       summary="That folder was not granted.", data={})
                return
            await asyncio.sleep(0.01)

    t = asyncio.create_task(answer())
    out = await DesktopSkill().execute_tool(
        "desktop__fs_read", {"path": "/etc/passwd"}, _ctx(),
    )
    await t
    assert out.startswith("REFUSED:")
    assert "do not" in out and "another tool" in out


async def test_an_unavailable_capability_is_explained_not_retried():
    sock, _ = await _connect()

    async def answer():
        for _ in range(100):
            if sock.sent:
                _reply(USER, sock.sent[0]["id"], ok=False, status="unavailable",
                       summary="Screen Recording is not granted.", data={})
                return
            await asyncio.sleep(0.01)

    t = asyncio.create_task(answer())
    out = await DesktopSkill().execute_tool("desktop__screen_capture", {}, _ctx())
    await t
    assert "Screen Recording" in out
    assert "ever script System Settings" in out


async def test_an_oversize_result_is_reported_as_having_none_of_it():
    """§5.4: an oversize result is REPLACED, never truncated — "a shortened
    payload that still says ok: true is a lie the model cannot detect, and
    it would act on half a file as though it were the file"."""
    sock, _ = await _connect()

    async def answer():
        for _ in range(100):
            if sock.sent:
                _reply(USER, sock.sent[0]["id"], ok=False, status="error",
                       summary="", data={"code": "result_too_large"})
                return
            await asyncio.sleep(0.01)

    t = asyncio.create_task(answer())
    out = await DesktopSkill().execute_tool(
        "desktop__fs_read", {"path": "big.bin"}, _ctx(),
    )
    await t
    assert "NOT" in out and "you have none of it" in out


# ── Staging: a mutating call stops, and the row is the record ────────
@pytest.fixture
async def tenant(test_user_id):
    """A user with an AgentConfig, so the skill's staging hop authenticates
    the way it does in production (X-Agent-Key → tenant)."""
    key = f"agent-key-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=test_user_id, agent_api_key=key))
        db.add(DesktopDevice(
            id="dev-1", user_id=test_user_id, device_name="A Mac",
            flavor="direct", token_jti="jti-1",
        ))
        await db.commit()
    prev = settings.agent_api_key
    settings.agent_api_key = key
    yield {"user_id": test_user_id, "key": key}
    settings.agent_api_key = prev


async def test_a_mutating_tool_stages_a_card_and_runs_nothing(tenant):
    sock, _ = await _connect(user_id=tenant["user_id"])
    out = await DesktopSkill().execute_tool(
        "desktop__exec_run",
        {"command": "npm", "args": ["test"], "cwd": "Projects/toup",
         "reason": "Run the test suite\nand report"},
        _ctx(tenant["user_id"]),
    )

    assert out.startswith("NOT RUN YET")
    assert "STOP" in out and "Do not call this tool again" in out
    # Nothing reached the Mac. The approval is a precondition, and the
    # dispatch happens from the approve route.
    assert sock.sent == []

    async with async_session_maker() as db:
        rows = (await db.execute(
            DesktopPendingAction.__table__.select()
        )).mappings().all()
    assert len(rows) == 1
    row = rows[0]
    assert row["tool_name"] == "desktop__exec_run"
    assert row["status"] == "pending"
    assert row["device_id"] == "dev-1"
    assert row["task_id"]
    # `reason` is flattened, and it is NOT part of the payload the device
    # will be handed.
    assert row["reason"] == "Run the test suite and report"
    assert json.loads(row["payload_json"]) == {
        "command": "npm", "args": ["test"], "cwd": "Projects/toup",
    }


async def test_every_consent_tool_stages_and_no_read_tool_does(tenant):
    sock, _ = await _connect(user_id=tenant["user_id"])
    skill = DesktopSkill()
    args = {"path": "x", "command": "ls", "cwd": "x", "app": "Finder",
            "key": "return", "text": "hi", "element_id": "e1",
            "content": "c", "from_path": "a", "to_path": "b"}
    for name in sorted(CONSENT_TOOLS):
        out = await skill.execute_tool(name, dict(args), _ctx(tenant["user_id"]))
        assert out.startswith("NOT RUN YET"), name
    assert sock.sent == [], "a consent tool reached the Mac without approval"

    async with async_session_maker() as db:
        rows = (await db.execute(
            DesktopPendingAction.__table__.select()
        )).mappings().all()
    assert {r["tool_name"] for r in rows} == CONSENT_TOOLS
