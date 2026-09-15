"""The platform must never persist a WhatsApp DOWNGRADE from a poll (round 44).

After a successful pairing, clients kept rendering "Not connected / Connect".
Half of that was the agent's cache (see `tests/test_whatsapp_baileys.py`,
`TestSidecarReconciler`); the other half is here. Both
`/agent-setup/whatsapp/health` and `/agent-setup/whatsapp/qr-status` copied
the agent's `session_status` straight into `AgentConfig` — including when it
read LOWER than what was already stored. That happens routinely and for two
independent reasons:

  * the agent's value is a CACHE fed by a replay-less SSE stream, so a
    `connection_open` emitted while no consumer was attached leaves it on
    `linking` indefinitely; and
  * sidecar.mjs writes `not_linked` on any non-logout Baileys close, i.e. for
    the seconds a reconnect takes.

Either one, persisted over a stored `linked`, un-links the user in the UI for
a session that is working. So a POLL may RAISE the stored status and may
record an explicit `logged_out` — the phone unlinked the device, which is a
fact the agent is entitled to report — but it may never write `linking` or
`not_linked` over `linked`. Un-linking stays with the routes that mean it:
`/whatsapp/qr-reset`, `/whatsapp/logout`, and a user-initiated re-pair
(`qr-start` / `pair-code`), all of which assign directly.

Lane: platform. Nothing here touches a table — `_get_or_create_config` and
the agent call are both replaced, so the cases are about the decision, not
about SQLAlchemy.

    cd backend && RUN_MODE=platform PYTHONPATH=. pytest tests/test_whatsapp_no_downgrade.py
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest


class FakeConfig:
    """Just the four columns these two routes read and write."""

    def __init__(
        self,
        session_status: Optional[str] = None,
        self_e164: Optional[str] = None,
        allowlist: Optional[str] = None,
    ):
        self.whatsapp_session_status = session_status
        self.whatsapp_self_e164 = self_e164
        self.whatsapp_baileys_allowlist = allowlist
        self.whatsapp_mode = "qr_link"
        self.updated_at = None


class FakeDb:
    """`execute()` answers the one SELECT `/whatsapp/health` runs."""

    def __init__(self, row: Any = None):
        self._row = row
        self.commits = 0

    async def execute(self, *_a: Any, **_kw: Any):
        row = self._row
        return MagicMock(first=lambda: row)

    async def commit(self) -> None:
        self.commits += 1


def _agent_row(url: str = "https://agent.example", key: str = "k"):
    row = MagicMock()
    row.agent_url = url
    row.agent_api_key = key
    return row


class _Resp:
    def __init__(self, body: dict, status_code: int = 200):
        self.status_code = status_code
        self._body = body
        self.content = b"x"

    def json(self) -> dict:
        return self._body


class _FakeHttpx:
    """Stands in for the `httpx` module attribute on agent_setup."""

    def __init__(self, resp: _Resp):
        self._resp = resp

    def AsyncClient(self, **_kw: Any):  # noqa: N802
        resp = self._resp

        class _C:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *_exc):
                return False

            async def get(self_inner, _url, **_kw2):
                return resp

        return _C()


@pytest.fixture
def setup(monkeypatch):
    """The module with its two outside seams replaced, plus a config box."""
    import app.api.agent_setup as mod

    state: dict[str, Any] = {"config": None, "pushes": []}

    async def _get_or_create_config(_user_id, _db):
        return state["config"]

    async def _push(coro, **kw):
        # The real one awaits a bridge call; close the coroutine so the
        # loop does not warn about it, and record that a push happened.
        try:
            coro.close()
        except Exception:
            pass
        state["pushes"].append(kw.get("what"))
        return True

    monkeypatch.setattr(mod, "_get_or_create_config", _get_or_create_config)
    monkeypatch.setattr(mod, "_bridge_push_within_budget", _push)
    monkeypatch.setattr(mod, "_env_push_worker", lambda uid: MagicMock())
    return mod, state


def _user():
    return MagicMock(id="11111111-2222-3333-4444-555555555555")


async def _call_health(mod, state, agent_whatsapp: dict, config: FakeConfig,
                       monkeypatch):
    state["config"] = config
    monkeypatch.setattr(
        mod, "httpx",
        _FakeHttpx(_Resp({"channels": {"whatsapp": agent_whatsapp}})),
    )
    return await mod.whatsapp_health(
        current_user=_user(), db=FakeDb(_agent_row()),
    )


async def _call_qr_status(mod, state, snapshot: dict, config: FakeConfig,
                          monkeypatch):
    state["config"] = config

    async def _proxy(*_a: Any, **_kw: Any):
        return snapshot

    monkeypatch.setattr(mod, "_agent_qr_proxy", _proxy)
    return await mod.whatsapp_qr_status(current_user=_user(), db=FakeDb())


# ── the decision, in isolation ───────────────────────────────────


class TestThePollRule:
    def test_a_poll_may_raise_but_never_lower_a_link(self, setup):
        mod, _ = setup
        may = mod._wa_poll_may_persist
        # Upgrades and first writes are free.
        assert may(None, "linking") is True
        assert may("linking", "linked") is True
        assert may("not_linked", "linked") is True
        assert may("logged_out", "linked") is True
        # An explicit logout is a fact, not a cache artefact.
        assert may("linked", "logged_out") is True
        # These two are the defect, and an unproven downgrade stays refused.
        assert may("linked", "linking") is False
        assert may("linked", "not_linked") is False
        # No-ops, empties and nonsense write nothing.
        assert may("linked", "linked") is False
        assert may("linked", None) is False
        assert may("linked", "") is False
        assert may("linking", "wat") is False, (
            "an unknown status reached the column — both routes now share "
            "one validator and it has to do the known-status check"
        )

    def test_a_downgrade_needs_all_three_proofs(self, setup):
        """Refusing forever is its own defect: a user who unlinks from their
        phone while nothing polls, followed by a container restart, leaves a
        credentialless sidecar that can never clear the stored link. The
        agent can PROVE that case — no number, held long enough that no
        reconnect explains it, and heard from the sidecar rather than from
        the agent's own optimism."""
        mod, _ = setup
        may = mod._wa_poll_may_persist

        def clear(**kw):
            base = dict(self_e164=None, stable_s=90, source="reconcile")
            base.update(kw)
            return may("linked", "linking", **base)

        assert clear() is True
        # …and each leg alone withholds it.
        assert clear(self_e164="+14155552671") is False, (
            "a reconnect keeps the number — sidecar.mjs clears it only next "
            "to wipeAuthDir()"
        )
        assert clear(stable_s=59) is False
        assert clear(source="local") is False, (
            "the agent's own optimistic write is not evidence"
        )
        assert clear(source="init") is False
        assert clear(stable_s=None) is False, (
            "an agent image with no provenance keys must degrade to the "
            "strict rule, not to the defect"
        )
        assert clear(stable_s="nonsense") is False
        # A proven clear works from `not_linked` too, and never from linked.
        assert may("linked", "not_linked", self_e164=None, stable_s=90,
                   source="boot") is True
        assert may("linked", "linked", self_e164=None, stable_s=9999,
                   source="reconcile") is False


# ── /whatsapp/health ─────────────────────────────────────────────


class TestHealthProxyPersistence:
    async def test_linking_does_not_clear_a_stored_link(self, setup, monkeypatch):
        """The reported defect. The agent's cache missed a
        `connection_open`; the phone is linked and the DB knows it."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        out = await _call_health(
            mod, state,
            {"mode": "qr_link", "session_status": "linking", "self_e164": None},
            cfg, monkeypatch,
        )
        assert out["reachable"] is True
        assert cfg.whatsapp_session_status == "linked", (
            "a stale `linking` from the agent's cache un-linked the user"
        )
        assert cfg.whatsapp_self_e164 == "+14155552671"

    async def test_not_linked_does_not_clear_a_stored_link(self, setup, monkeypatch):
        """sidecar.mjs writes `not_linked` on ANY non-logout close — i.e.
        for the seconds a reconnect takes."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        await _call_health(
            mod, state,
            {"mode": "qr_link", "session_status": "not_linked", "self_e164": None},
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "linked"
        assert cfg.whatsapp_self_e164 == "+14155552671"

    async def test_linked_is_persisted_whenever_the_agent_reports_it(
        self, setup, monkeypatch
    ):
        """ANTI-VACUITY: the rule is 'never downgrade', not 'never write'."""
        mod, state = setup
        cfg = FakeConfig(session_status="linking", self_e164=None)
        await _call_health(
            mod, state,
            {"mode": "qr_link", "session_status": "linked",
             "self_e164": "+14155552671"},
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "linked"
        assert cfg.whatsapp_self_e164 == "+14155552671"

    async def test_an_explicit_logged_out_still_clears_the_link(
        self, setup, monkeypatch
    ):
        """The user unlinked the device from their phone. That is the one
        downgrade a poll is allowed to carry, and the hero has to go."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        await _call_health(
            mod, state,
            {"mode": "qr_link", "session_status": "logged_out", "self_e164": None},
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "logged_out"
        assert cfg.whatsapp_self_e164 is None

    async def test_an_unknown_status_writes_nothing(self, setup, monkeypatch):
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        await _call_health(
            mod, state,
            {"mode": "qr_link", "session_status": "wat", "self_e164": None},
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "linked"


# ── /whatsapp/qr-status ──────────────────────────────────────────


class TestQrStatusPersistence:
    async def test_linked_is_persisted_and_seeds_the_allowlist(
        self, setup, monkeypatch
    ):
        """Self-chat needs the user's own number allowlisted — an empty
        allowlist blocks everything — so the seed rides the first `linked`."""
        mod, state = setup
        cfg = FakeConfig(session_status="linking", self_e164=None, allowlist=None)
        out = await _call_qr_status(
            mod, state,
            {"session_status": "linked", "connected": True,
             "self_e164": "+14155552671"},
            cfg, monkeypatch,
        )
        assert out["session_status"] == "linked"
        assert cfg.whatsapp_session_status == "linked"
        assert cfg.whatsapp_self_e164 == "+14155552671"
        assert cfg.whatsapp_baileys_allowlist == "+14155552671"
        assert state["pushes"] == ["whatsapp-qr-allowlist"], (
            "the seeded allowlist never reached the running container"
        )

    async def test_a_stale_linking_snapshot_leaves_the_link_alone(
        self, setup, monkeypatch
    ):
        """Same cache, different route: the modal polls this one every
        ~1.5 s, so a single stale snapshot used to be enough."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671",
                         allowlist="+14155552671")
        await _call_qr_status(
            mod, state,
            {"session_status": "linking", "connected": False, "self_e164": None},
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "linked"
        assert cfg.whatsapp_self_e164 == "+14155552671"

    async def test_a_deliberate_repair_can_still_move_off_linked(
        self, setup, monkeypatch
    ):
        """ANTI-VACUITY for the guard: re-pairing writes `linking` from
        `/whatsapp/qr-start` and `/whatsapp/pair-code` DIRECTLY, so the
        stored value is already `linking` by the time polling resumes and
        the flow is not frozen on the old link."""
        mod, state = setup
        cfg = FakeConfig(session_status="linking", self_e164="+14155552671")
        await _call_qr_status(
            mod, state,
            {"session_status": "not_linked", "connected": False, "self_e164": None},
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "not_linked"


# ── the poll is a READ (round 44, client-side follow-up) ─────────
#
# The mobile Channels panel now calls this route once per visit to
# Connectors → Channels and once per return to the foreground, for EVERY
# user whose stored status is not `linked` — including users who never set
# WhatsApp up. So the whole chain has to be inert: no container work, no
# allowlist push, no AgentConfig row invented for someone who has no agent,
# and a typed answer rather than a hang.


class TestQrStatusIsAReadOnlyPoll:
    async def test_an_undeployed_agent_creates_no_config_row(self, setup, monkeypatch):
        """`_agent_qr_proxy` raises BEFORE `_get_or_create_config` is
        reached, so polling on behalf of a user who never set anything up
        writes nothing at all."""
        from fastapi import HTTPException

        mod, state = setup
        touched: list[str] = []

        async def _no_config(*_a: Any, **_kw: Any):
            touched.append("config")
            raise AssertionError("a status poll created an AgentConfig row")

        async def _proxy(*_a: Any, **_kw: Any):
            raise HTTPException(status_code=503, detail="Your agent isn't deployed yet.")

        monkeypatch.setattr(mod, "_get_or_create_config", _no_config)
        monkeypatch.setattr(mod, "_agent_qr_proxy", _proxy)

        with pytest.raises(HTTPException) as exc:
            await mod.whatsapp_qr_status(current_user=_user(), db=FakeDb())
        assert exc.value.status_code == 503
        assert touched == []
        assert state["pushes"] == []

    async def test_a_channel_that_is_not_started_is_a_503_not_a_write(
        self, setup, monkeypatch
    ):
        """The agent answers 503 when `BaileysWhatsAppChannel` is not
        active. That is the steady state for most users and must stay free."""
        from fastapi import HTTPException

        mod, state = setup
        cfg = FakeConfig(session_status=None, self_e164=None)
        state["config"] = cfg

        async def _proxy(*_a: Any, **_kw: Any):
            raise HTTPException(
                status_code=503, detail="QR-link mode isn't active on your agent yet.",
            )

        monkeypatch.setattr(mod, "_agent_qr_proxy", _proxy)
        with pytest.raises(HTTPException) as exc:
            await mod.whatsapp_qr_status(current_user=_user(), db=FakeDb())
        assert exc.value.status_code == 503
        assert cfg.whatsapp_session_status is None
        assert cfg.whatsapp_baileys_allowlist is None
        assert state["pushes"] == []

    async def test_a_never_linked_user_settles_after_one_write(
        self, setup, monkeypatch
    ):
        """A poll may record a real transition once. Repeating the same
        answer must then be free — this now runs on every foreground."""
        mod, state = setup
        cfg = FakeConfig(session_status=None, self_e164=None)
        db = FakeDb()
        state["config"] = cfg

        async def _proxy(*_a: Any, **_kw: Any):
            return {"session_status": "not_linked", "connected": False,
                    "self_e164": None, "qr_data_url": None}

        monkeypatch.setattr(mod, "_agent_qr_proxy", _proxy)
        for _ in range(4):
            await mod.whatsapp_qr_status(current_user=_user(), db=db)

        assert cfg.whatsapp_session_status == "not_linked"
        assert db.commits == 1, (
            f"{db.commits} writes for four identical polls — the poll is "
            f"rewriting a value it already stored"
        )
        assert state["pushes"] == [], "a not-linked poll pushed container env"
        assert cfg.whatsapp_baileys_allowlist is None, (
            "a poll minted an allowlist for a user with no link"
        )

    async def test_a_steady_linked_poll_pushes_nothing(self, setup, monkeypatch):
        """The allowlist seed fires only where there is something to repair
        — a link with no allowlist, which silently blocks every message. A
        healthy linked config must not push env on every poll."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671",
                         allowlist="+14155552671")
        db = FakeDb()
        state["config"] = cfg

        async def _proxy(*_a: Any, **_kw: Any):
            return {"session_status": "linked", "connected": True,
                    "self_e164": "+14155552671"}

        monkeypatch.setattr(mod, "_agent_qr_proxy", _proxy)
        for _ in range(3):
            await mod.whatsapp_qr_status(current_user=_user(), db=db)
        assert db.commits == 0
        assert state["pushes"] == []

    def test_the_poll_is_on_a_short_leash(self):
        """A read behind a screen the user is looking at cannot inherit the
        10 s default: the agent's own sidecar read is bounded at 2 s, so
        anything past a few seconds is an unreachable agent."""
        import inspect
        import app.api.agent_setup as mod

        src = inspect.getsource(mod.whatsapp_qr_status)
        assert "timeout_s=" in src, (
            "the status poll uses _agent_qr_proxy's 10 s default"
        )
        assert "/api/whatsapp/qr/status" in src


# ── the stored link has to be clearable (review F1) ──────────────
#
# "Never downgrade" alone is unclearable-forever. The scenario, end to end:
# the user removes the linked device on their phone while nothing is polling
# (sidecar.mjs sets `logged_out` in Node memory ONLY and wipes the auth dir),
# then the container restarts — this rollout restarts ~90 of them. The new
# sidecar boots credentialless at `not_linked`, `start()` auto-kicks
# `/pair/start` → `linking`, and every poll from then on is a refused
# downgrade. Both UIs would read "Linked as +1…" against a session with no
# credentials until the user found Disconnect.


def _snapshot(status, self_e164=None, stable_s=600, source="reconcile"):
    return {
        "session_status": status,
        "connected": status == "linked",
        "self_e164": self_e164,
        "session_status_stable_s": stable_s,
        "session_status_source": source,
    }


class TestAStoredLinkStaysClearable:
    async def test_wiped_creds_then_restart_clears_the_link(
        self, setup, monkeypatch
    ):
        """The F1 scenario. No number (credentials gone), and the state has
        held past the floor — nothing but a wipe explains that."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        await _call_qr_status(
            mod, state,
            _snapshot("linking", self_e164=None, stable_s=61),
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "linking", (
            "a credentialless sidecar could not clear the stored link — the "
            "user is shown 'Linked as +1…' against a dead session forever"
        )

    async def test_the_same_scenario_through_the_health_route(
        self, setup, monkeypatch
    ):
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        await _call_health(
            mod, state,
            dict(_snapshot("not_linked", self_e164=None, stable_s=300),
                 mode="qr_link"),
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "not_linked"

    async def test_a_reconnect_blip_never_clears_it(self, setup, monkeypatch):
        """The routine close path sets `linking` and LEAVES `selfE164` in
        place, which is exactly what makes it distinguishable."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        await _call_qr_status(
            mod, state,
            _snapshot("linking", self_e164="+14155552671", stable_s=3),
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "linked"
        assert cfg.whatsapp_self_e164 == "+14155552671"

    async def test_a_ten_minute_outage_reconnect_never_clears_it(
        self, setup, monkeypatch
    ):
        """Duration is NOT the discriminator on its own — a long outage is
        still a reconnect, and it still has the number."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        await _call_qr_status(
            mod, state,
            _snapshot("linking", self_e164="+14155552671", stable_s=600),
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "linked"

    async def test_a_fresh_boot_inside_the_floor_never_clears_it(
        self, setup, monkeypatch
    ):
        """The one window where a CREDENTIALLED sidecar reports a null
        number is between container boot and its first `connection.open`.
        That is seconds; the floor is a minute."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        for stable in (0, 5, 59):
            cfg.whatsapp_session_status = "linked"
            await _call_health(
                mod, state,
                dict(_snapshot("not_linked", self_e164=None, stable_s=stable,
                               source="boot"), mode="qr_link"),
                cfg, monkeypatch,
            )
            assert cfg.whatsapp_session_status == "linked", (
                f"a {stable}s-old boot state cleared the link"
            )

    async def test_an_old_agent_image_cannot_clear_it(self, setup, monkeypatch):
        """The platform half deploys on a Railway push; the agent half needs
        a fleet rollout. In between, agents send no provenance keys — and
        must land on the STRICT rule, never on the defect."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        await _call_health(
            mod, state,
            {"mode": "qr_link", "session_status": "linking", "self_e164": None},
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "linked"

    async def test_the_agents_own_optimism_is_not_proof(self, setup, monkeypatch):
        """`local` is the agent having just POSTed `/pair/start` itself. It
        describes an intention, not the sidecar."""
        mod, state = setup
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        await _call_qr_status(
            mod, state,
            _snapshot("linking", self_e164=None, stable_s=900, source="local"),
            cfg, monkeypatch,
        )
        assert cfg.whatsapp_session_status == "linked"


# ── the refusal log is not per-poll (review F3) ──────────────────


class TestRefusalLogging:
    async def test_a_refusal_is_logged_once_not_twelve_times_a_minute(
        self, setup, monkeypatch, caplog
    ):
        """`/whatsapp/health` runs every 5 s for as long as Settings is
        open. One line per refused VALUE, not one per poll."""
        import logging

        mod, state = setup
        mod._WA_REFUSAL_SEEN.clear()
        cfg = FakeConfig(session_status="linked", self_e164="+14155552671")
        with caplog.at_level(logging.INFO, logger="app.api.agent_setup"):
            for _ in range(6):
                await _call_health(
                    mod, state,
                    {"mode": "qr_link", "session_status": "linking",
                     "self_e164": None},
                    cfg, monkeypatch,
                )
        lines = [r for r in caplog.records if "keeping stored" in r.getMessage()]
        assert len(lines) == 1, (
            f"{len(lines)} refusal lines for six polls of the same value"
        )
        # A DIFFERENT refused value is new information and does log.
        with caplog.at_level(logging.INFO, logger="app.api.agent_setup"):
            await _call_health(
                mod, state,
                {"mode": "qr_link", "session_status": "not_linked",
                 "self_e164": None},
                cfg, monkeypatch,
            )
        lines = [r for r in caplog.records if "keeping stored" in r.getMessage()]
        assert len(lines) == 2
        mod._WA_REFUSAL_SEEN.clear()

    def test_the_refusal_memo_is_bounded(self, setup):
        """Per-process state on a route every user hits. It may not grow."""
        mod, _ = setup
        mod._WA_REFUSAL_SEEN.clear()
        for i in range(mod._WA_REFUSAL_SEEN_MAX + 50):
            mod._wa_log_refusal("T", f"user-{i}", "linked", "linking")
        assert len(mod._WA_REFUSAL_SEEN) <= mod._WA_REFUSAL_SEEN_MAX
        mod._WA_REFUSAL_SEEN.clear()
