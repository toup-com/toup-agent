#!/usr/bin/env python3
"""Live automations harness (Round 28-D) — the production topology on
a laptop, no kinder cage.

Where `e2e_automations.py` runs the agent side in-process against a
monolith, THIS harness boots the real two-process topology:

  * platform_main under uvicorn (RUN_MODE=platform, its own sqlite) —
    the Railway role: proxy routes, RPC, grants, vault, feature flags,
    WS chat proxy.
  * agent_main under uvicorn (RUN_MODE=agent, its OWN sqlite) — the
    tenant-container role: chat WS, the automations engine, the REAL
    routine scheduler, the REAL outbox flush loop, the REAL 60-s
    reconciler sweep.

and drives everything the way production traffic does: user JWT over
HTTP to the platform, WebSocket through the platform's ws_chat proxy,
platform→agent hooks with X-Agent-Key. The agent's sqlite file is
opened read-only by the harness for ASSERTIONS only — every ACTION
travels the wire.

Proof sections (R28-D brief):

  FLAG        the surface 404s dark; the admin API flips
              `automations.rollout_pct`; the same routes go 200.
              (The per-tenant half, AUTOMATIONS_ENABLED=true, is the
              agent process env — on a real fleet it is a bridge
              blue-green env append.)
  SOCKET      template tap → REAL LLM plan turn → connector card frame
              arrives on the live socket through the platform proxy.
  BACKGROUND  socket CLOSED (app backgrounded); the platform-shaped
              `_connector_connected` hook flips the card in place; a
              reload (thread fetch) sees the new status.
  GRANT       grant card frame live; the user's approve on the platform
              fires `_grant_decided`; the approved frame arrives on the
              open socket.
  UNDO        the real scheduler fires a poll; the staged outbox row is
              undone over HTTP inside the 6-s window while the REAL
              flush loop is running; a second write is left alone, the
              flush wins, and the late undo 409s.
  AUTOPAUSE   a revoked grant makes 3 REAL schedule fires fail; the
              60-s reconciler sweep flips the automation to error,
              posts exactly ONE notice IN the session thread, with the
              navigate chip.
  ENDPOINTS   /thread and /memory served over the platform proxy.

Requires an LLM key for the plan turn (read from E2E_LLM_ENV_FILE,
default backend/.env). Without one the harness FAILS rather than
skipping — a plan turn that cannot run is not a plan turn that passed.
Set E2E_ALLOW_NO_LLM=1 to explicitly downgrade SOCKET/GRANT to the
deterministic rails only.

Usage:  make e2e-automations-live     (from the repo root)
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent

_P_PORT = int(os.environ.get("E2E_LIVE_PLATFORM_PORT", "8975"))
_A_PORT = int(os.environ.get("E2E_LIVE_AGENT_PORT", "8976"))
_SECRET = "e2e-automations-live-secret-not-production"
_USER_EMAIL = os.environ.get("E2E_USER_EMAIL", "nariman@toup.ai")
_ADMIN_EMAIL = "e2e-admin@toup.ai"


def _fresh_fernet_key() -> str:
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


def _read_env_file(path: Path) -> dict:
    """KEY=VALUE lines; used ONLY to lift LLM keys for the plan turn."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


_FERNET = _fresh_fernet_key()
_TMP = Path(tempfile.mkdtemp(prefix="toup-e2e-live-"))
_P_DB = _TMP / "platform.db"
_A_DB = _TMP / "agent.db"

_LLM_ENV = _read_env_file(
    Path(os.environ.get("E2E_LLM_ENV_FILE", str(BACKEND / ".env")))
)
# OpenAI only: the local .env's Anthropic key is subscription-authed
# (usage-window errors masquerade as turn failures — iteration 5), and
# with no Anthropic key the router's "anthropic disabled → openai"
# path keeps the whole turn on one accountable provider.
_LLM_KEYS = {
    k: v for k, v in _LLM_ENV.items()
    if k in ("OPENAI_API_KEY",) and v
}
if _LLM_KEYS:
    # The fleet default, NOT whatever model the .env pins — a local
    # .env pinning a subscription-authed Claude model turns the plan
    # turn into "usage exhausted" noise (iteration 3).
    _LLM_KEYS["AGENT_MODEL"] = os.environ.get(
        "E2E_AGENT_MODEL", "gpt-5.6-terra")
    _LLM_KEYS["AGENT_FALLBACK_MODEL"] = _LLM_KEYS["AGENT_MODEL"]

_COMMON = {
    "ENVIRONMENT": "development",
    "SECRET_KEY": _SECRET,
    "JWT_SECRET": _SECRET,
    "PLATFORM_ENCRYPTION_KEY": _FERNET,
    "AUTOMATIONS_E2E": "1",
    "ENABLE_CONNECTOR_OAUTH": "true",
}

_P_ENV = {
    **os.environ, **_COMMON,
    "RUN_MODE": "platform",
    "DATABASE_URL": f"sqlite+aiosqlite:///{_P_DB}",
    # Deliberately NOT setting AUTOMATIONS_ROLLOUT_PCT: the env floor is
    # 0, so the platform boots DARK and only the admin API turns it on —
    # that flip is the first proof.
}
_P_ENV.pop("AUTOMATIONS_ROLLOUT_PCT", None)

_A_ENV = {
    **os.environ, **_COMMON, **_LLM_KEYS,
    "RUN_MODE": "agent",
    "DATABASE_URL": f"sqlite+aiosqlite:///{_A_DB}",
    "AUTOMATIONS_ENABLED": "true",          # the per-tenant half
    "AUTOMATIONS_DEV_FAST_LANE": "1",       # 5-s floors (dev only)
    "PLATFORM_API_URL": f"http://127.0.0.1:{_P_PORT}/api",
    "ENABLE_SCHEDULER": "true",
    # The container's writable paths — locally, /app is somebody
    # else's read-only root and agent init dies before constructing
    # the runner (chat turns then 4503 with code=agent_starting).
    "AGENT_WORKSPACE_DIR": str(_TMP / "workspace"),
    "SKILLS_DIR": str(_TMP / "skills"),
}
(_TMP / "workspace").mkdir(exist_ok=True)
(_TMP / "skills").mkdir(exist_ok=True)
_A_ENV.pop("ANTHROPIC_API_KEY", None)   # never the subscription key

PASS = 0
FAIL = 0
_T0 = time.time()


def check(name: str, ok, detail: str = "") -> None:
    global PASS, FAIL
    ok = bool(ok)   # callers pass truthy expressions, not just bools
    mark = "✅" if ok else "❌"
    print(f"  [{time.time() - _T0:6.1f}s] {mark} {name}"
          + (f" — {detail}" if detail and not ok else ""), flush=True)
    PASS += ok
    FAIL += not ok


def section(name: str) -> None:
    print(f"\n═══ {name} ═══", flush=True)


def _wait_port(port: int, timeout: float = 90.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(1.0)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.5)
    return False


def _agent_db(query: str, params: tuple = ()) -> list[tuple]:
    """Read-only assertion peek at the agent's tenant DB."""
    con = sqlite3.connect(f"file:{_A_DB}?mode=ro", uri=True, timeout=5.0)
    try:
        return con.execute(query, params).fetchall()
    finally:
        con.close()


async def _await_agent_rows(query: str, params: tuple, pred,
                            timeout: float = 30.0, every: float = 0.2):
    """Poll the agent DB until pred(rows) is truthy (or timeout).
    Returns the rows either way."""
    end = time.time() + timeout
    rows: list[tuple] = []
    while time.time() < end:
        try:
            rows = _agent_db(query, params)
        except sqlite3.OperationalError:
            rows = []
        if pred(rows):
            return rows
        await asyncio.sleep(every)
    return rows


class WSClient:
    """A real browser-shaped client: subprotocol JWT auth through the
    platform's ws_chat proxy. Collects every inbound frame."""

    def __init__(self, jwt: str, label: str = "ws"):
        self._jwt = jwt
        self.label = label
        self.frames: list[dict] = []
        self._ws = None
        self._reader: asyncio.Task | None = None

    async def connect(self) -> None:
        import websockets
        self._ws = await websockets.connect(
            f"ws://127.0.0.1:{_P_PORT}/api/ws/chat",
            subprotocols=["toup.auth.v1", f"bearer.{self._jwt}"],
            open_timeout=20, ping_interval=20,
        )
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    self.frames.append(json.loads(raw))
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass

    async def send(self, payload: dict) -> None:
        # The web/app clients ride "tz" on every message frame; the
        # server persists it to the tenant User row (the self-healing
        # path the routine runner depends on for registration).
        if payload.get("type") == "message":
            payload.setdefault("tz", "America/Toronto")
        await self._ws.send(json.dumps(payload))

    async def wait_for(self, pred, timeout: float = 90.0):
        """First frame (ever received) matching pred, else None.
        Fails FAST on a turn-level error frame — a dead turn must not
        burn the whole timeout looking like a slow one."""
        end = time.time() + timeout
        seen = 0
        while time.time() < end:
            while seen < len(self.frames):
                f = self.frames[seen]
                seen += 1
                if pred(f):
                    return f
                if f.get("type") == "error":
                    print(f"  [{self.label}] turn error frame: "
                          f"{str(f)[:200]}", flush=True)
                    return None
            await asyncio.sleep(0.2)
        return None

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self._ws:
            await self._ws.close()


def _boot(name: str, module: str, port: int, env: dict) -> subprocess.Popen:
    log_path = _TMP / f"{name}.log"
    print(f"  ({name} log: {log_path})", flush=True)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", f"{module}:app",
         "--port", str(port), "--log-level", "warning"],
        cwd=str(BACKEND), env=env,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
    )


async def main() -> int:
    user_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    agent_key = f"e2e-live-key-{uuid.uuid4().hex}"
    _A_ENV["USER_ID"] = user_id
    _A_ENV["AGENT_API_KEY"] = agent_key

    if not _LLM_KEYS and os.environ.get("E2E_ALLOW_NO_LLM") != "1":
        print("❌ No LLM key found (E2E_LLM_ENV_FILE) — the plan-turn "
              "proof cannot run. Set E2E_ALLOW_NO_LLM=1 to explicitly "
              "downgrade.")
        return 1

    section("BOOT — the production topology, two real processes")
    p_proc = _boot("platform", "platform_main", _P_PORT, _P_ENV)
    a_proc = _boot("agent", "agent_main", _A_PORT, _A_ENV)
    try:
        check("platform port up", _wait_port(_P_PORT))
        check("agent port up", _wait_port(_A_PORT))

        # Harness identity: PLATFORM-side (seeding the platform DB the
        # way signup + provisioning would).
        os.environ.update(_P_ENV)
        sys.path.insert(0, str(BACKEND))
        import httpx
        from app.db.database import async_session_maker
        from app.db.models import AgentConfig, User
        from app.services.auth_service import (
            create_access_token, get_password_hash,
        )

        async with async_session_maker() as db:
            db.add(User(id=user_id, email=_USER_EMAIL,
                        hashed_password=get_password_hash("e2e-pass-1"),
                        name="E2E Live", timezone="America/Toronto"))
            db.add(User(id=admin_id, email=_ADMIN_EMAIL,
                        hashed_password=get_password_hash("e2e-pass-2"),
                        name="E2E Admin", role="admin"))
            db.add(AgentConfig(
                user_id=user_id,
                agent_url=f"http://127.0.0.1:{_A_PORT}",
                agent_api_key=agent_key, deploy_status="active",
            ))
            await db.commit()
        user_jwt = create_access_token(user_id)
        admin_jwt = create_access_token(admin_id)
        user_h = {"Authorization": f"Bearer {user_jwt}"}
        admin_h = {"Authorization": f"Bearer {admin_jwt}"}
        rpc_h = {"X-Agent-Key": agent_key, "X-Agent-User-Id": user_id}

        base = f"http://127.0.0.1:{_P_PORT}"
        agent_base = f"http://127.0.0.1:{_A_PORT}"
        async with httpx.AsyncClient(timeout=60) as http:
            r = await http.get(f"{base}/api/health")
            check("platform health", r.status_code == 200)
            r = await http.get(f"{agent_base}/agent/health")
            check("agent health", r.status_code == 200,
                  f"{r.status_code}")


            # ════ FLAG — the dark launch ends through the admin API ═══
            section("FLAG — flip automations ON through the admin API")
            r = await http.get(f"{base}/api/system/feature-flags",
                               headers=user_h)
            flags = r.json()
            check("boot state: automations OFF (env floor 0)",
                  flags.get("automations") is False, str(flags))
            r = await http.get(f"{base}/api/automations/templates",
                               headers=user_h)
            check("dark: the platform surface does not exist (404)",
                  r.status_code == 404, f"{r.status_code}")

            r = await http.put(
                f"{base}/api/admin/feature-flags/flag/automations",
                headers=user_h, json={"rollout_pct": 100},
            )
            check("non-admin cannot flip the flag (403)",
                  r.status_code == 403, f"{r.status_code}")
            r = await http.put(
                f"{base}/api/admin/feature-flags/flag/automations",
                headers=admin_h, json={"rollout_pct": 100},
            )
            check("admin PUT flips rollout to 100",
                  r.status_code == 200
                  and r.json().get("rollout_pct") == 100,
                  f"{r.status_code} {r.text[:120]}")
            r = await http.get(f"{base}/api/system/feature-flags",
                               headers=user_h)
            check("public readout now says automations ON",
                  r.json().get("automations") is True, r.text[:120])
            r = await http.get(f"{base}/api/automations/templates",
                               headers=user_h)
            check("the same surface now serves (200, catalog present)",
                  r.status_code == 200
                  and len(r.json().get("templates", [])) >= 20,
                  f"{r.status_code} {r.text[:120]}")

            # ── seed: the stub connector's vault identity (platform DB,
            # exactly what a completed OAuth stores) ───────────────────
            from app.services import connector_vault as vault
            async with async_session_maker() as db:
                await vault.put(
                    db, user_id, "stub",
                    access_token="stub_access_token_e2e_live",
                    refresh_token="stub_refresh_token_e2e_live",
                    access_expires_at=datetime.utcnow() + timedelta(hours=2),
                    scopes=[], provider_account_id=_USER_EMAIL,
                    event_type="connected",
                )
            r = await http.get(f"{base}/api/v1/automations/registry",
                               headers=rpc_h)
            reg_ids = {c["connector_id"]
                       for c in r.json().get("connectors", [])}
            check("RPC registry serves stub (dev) + slack",
                  {"stub", "slack"} <= reg_ids, str(sorted(reg_ids)))

            # ════ SOCKET — template tap → plan turn → card frame ══════
            section("SOCKET — template tap → REAL plan turn → "
                    "connector card on the live socket")
            connector_id = "slack"
            card_id = None
            if not _LLM_KEYS:
                print("  ⚠️ E2E_ALLOW_NO_LLM=1 — SOCKET/BACKGROUND/GRANT "
                      "did NOT run. This is a DOWNGRADED result.",
                      flush=True)
                ws0 = WSClient(user_jwt, "ws0")
                await ws0.connect()
                await ws0.send({"type": "message", "text": "hi"})
                rows = await _await_agent_rows(
                    "SELECT timezone FROM users WHERE id=?", (user_id,),
                    lambda r: bool(r) and r[0][0], timeout=15,
                )
                await ws0.close()
                check("client tz rode the frame and persisted "
                      "(runner registration precondition)",
                      bool(rows) and rows[0][0] == "America/Toronto",
                      str(rows))
            else:
                ws1 = WSClient(user_jwt, "ws1")
                await ws1.connect()
                tap = (
                    "[template tap] Please set up the automation "
                    "template 'Morning work brief' for me. Slack is "
                    "not connected yet — request the Slack connection "
                    "(read_write) with automations__request_connection "
                    "now, then stop and wait for me."
                )
                await ws1.send({"type": "message", "text": tap})
                rows = await _await_agent_rows(
                    "SELECT timezone FROM users WHERE id=?", (user_id,),
                    lambda r: bool(r) and r[0][0], timeout=15,
                )
                check("client tz rode the message frame and persisted "
                      "(production self-healing path)",
                      bool(rows) and rows[0][0] == "America/Toronto",
                      str(rows))
                card = await ws1.wait_for(
                    lambda f: f.get("type") == "automation_connector_card",
                    timeout=240,
                )
                check("connector card frame arrived over the live socket",
                      card is not None,
                      "no automation_connector_card frame; last frames: "
                      + str([f.get("type") for f in ws1.frames][-12:]))
                if card:
                    check("card is offered, names the connector, carries "
                          "the connect URL",
                          card.get("status") == "offered"
                          and card.get("connector_id")
                          and (card.get("connect_url") or "")
                          .startswith("/api/oauth/connect/"),
                          str(card)[:200])
                    connector_id = card.get("connector_id") or "slack"
                    card_id = card.get("id")

                # ════ BACKGROUND — socket closed; OAuth lands; the
                # wake-up hook rewrites the card in place ══════════════
                section("BACKGROUND — card flips while the app is away")
                await ws1.close()
                # The platform's OAuth callback sequence: store tokens,
                # then wake the tenant (oauth.py:_connector_connected).
                async with async_session_maker() as db:
                    await vault.put(
                        db, user_id, connector_id,
                        access_token="e2e-live-oauth-token",
                        refresh_token="e2e-live-oauth-refresh",
                        access_expires_at=datetime.utcnow()
                        + timedelta(hours=2),
                        scopes=["channels:read", "chat:write"],
                        provider_account_id=_USER_EMAIL,
                        event_type="connected",
                    )
                r = await http.post(
                    f"{agent_base}/api/automations/_connector_connected",
                    headers=rpc_h,
                    json={"connector_id": connector_id, "ok": True},
                )
                check("wake-up hook accepted and updated the open card",
                      r.status_code == 200
                      and r.json().get("updated", 0) >= 1,
                      f"{r.status_code} {r.text[:120]}")
                if card_id:
                    r = await http.get(
                        f"{base}/api/automations/auth-sessions/{card_id}",
                        headers=user_h,
                    )
                    check("reload (proxy) shows the card connected — the "
                          "flip survived the background",
                          r.status_code == 200
                          and r.json().get("status") == "connected",
                          f"{r.status_code} {r.text[:160]}")
                    rows = _agent_db(
                        "SELECT metadata_json FROM messages "
                        "WHERE metadata_json LIKE ?", (f"%{card_id}%",))
                    check("the card's message row was rewritten in place",
                          bool(rows) and '"connected"' in (rows[0][0] or ""),
                          (rows[0][0][:160] if rows else "no row"))

                # ════ GRANT — card live; approve wakes the agent ══════
                section("GRANT — live grant card; the user's approve "
                        "rewrites it on the open socket")
                ws2 = WSClient(user_jwt, "ws2")
                await ws2.connect()
                await ws2.send({"type": "message", "text": (
                    f"{connector_id} is connected now. Next step only: "
                    f"ask my permission for the write this template "
                    f"needs — call automations__request_permission for "
                    f"slack__send_message pinned to the channel with id "
                    f"'C0LIVE1' and label '#brief', mode 'auto', then "
                    f"stop. Do NOT create or arm the automation yet."
                )})
                gcard = await ws2.wait_for(
                    lambda f: f.get("type") == "automation_grant_card",
                    timeout=240,
                )
                check("grant card frame arrived (pending) on the socket",
                      gcard is not None
                      and gcard.get("status") == "pending",
                      "frames: "
                      + str([f.get("type") for f in ws2.frames][-12:]))
                if gcard:
                    gid = gcard["id"]
                    r = await http.post(
                        f"{base}/api/automations/grant-requests/{gid}"
                        f"/approve",
                        headers=user_h, json={"decided_via": "web"},
                    )
                    check("approve on the platform → approved",
                          r.status_code == 200
                          and r.json().get("status") == "approved",
                          f"{r.status_code} {r.text[:160]}")
                    upd = await ws2.wait_for(
                        lambda f: f.get("type") == "automation_grant_card"
                        and f.get("id") == gid
                        and f.get("status") == "approved",
                        timeout=30,
                    )
                    check("approved frame re-broadcast live — same card "
                          "id, rewritten in place", upd is not None)
                await ws2.close()

            # ════ UNDO — real fire, real flush, undo over the wire ════
            section("UNDO — the scheduler fires; undo races the REAL "
                    "flush loop through the proxy")
            r = await http.post(
                f"{base}/api/v1/automations/grant-requests", headers=rpc_h,
                json={"connector_id": "stub", "tool_name": "stub__post",
                      "target": {"kind": "channel", "id": "chan-live-1",
                                 "label": "#live"},
                      "mode": "auto", "summary": "post the live digest"},
            )
            sg = r.json().get("grant") or {}
            r = await http.post(
                f"{base}/api/automations/grant-requests/{sg['id']}/approve",
                headers=user_h, json={"decided_via": "web"},
            )
            check("stub grant staged (RPC) + approved (user)",
                  r.status_code == 200
                  and r.json().get("status") == "approved")

            live_spec = {
                "version": 2,
                "name": "Live stub digest",
                "mode": "auto",
                "trigger": {"sources": [
                    {"id": "feed", "mode": "poll", "connector_id": "stub",
                     "event": "item_created", "poll_interval_s": 5,
                     "dedupe_key": "event.id"},
                ]},
                "steps": [
                    {"id": "post", "connector_id": "stub",
                     "tool": "stub__post",
                     "params": {"channel": "{{grant.target.id}}",
                                "text": "Live: {{event.title}}"},
                     "grant_id": sg["id"]},
                ],
            }
            r = await http.post(f"{base}/api/automations", headers=user_h,
                                json={"spec": live_spec})
            check("v2 automation created through the platform proxy",
                  r.status_code == 200, f"{r.status_code} {r.text[:200]}")
            auto_id = (r.json().get("automation") or {}).get("id")
            r = await http.post(f"{base}/api/automations/{auto_id}/arm",
                                headers=user_h)
            check("armed through the proxy",
                  r.status_code == 200
                  and (r.json().get("automation") or {}).get("status")
                  == "armed", f"{r.status_code} {r.text[:200]}")

            # The REAL RoutineRunner fires the 5-s poll (the R28-D
            # runner fix under test). Catch the FIRST staged write
            # inside its 6-s undo window.
            rows = await _await_agent_rows(
                "SELECT id, status FROM automation_outbox "
                "WHERE automation_id=?", (auto_id,),
                lambda r: any(s == "staged" for _, s in r), timeout=90,
            )
            staged = [i for i, s in rows if s == "staged"]
            check("the real scheduler fired the poll; a write is "
                  "staged in its undo window", bool(staged),
                  str(rows))
            undo_ok = False
            if staged:
                r = await http.post(
                    f"{base}/api/automations/outbox/{staged[0]}/undo",
                    headers=user_h,
                )
                undo_ok = (r.status_code == 200
                           and r.json().get("undone") is True)
                check("undo through the proxy beat the flush",
                      undo_ok, f"{r.status_code} {r.text[:160]}")

            rows = await _await_agent_rows(
                "SELECT id, status FROM automation_outbox "
                "WHERE automation_id=?", (auto_id,),
                lambda r: len(r) >= 3
                and all(s in ("executed", "undone") for _, s in r),
                timeout=60,
            )
            statuses = sorted(s for _, s in rows)
            check("the survivors were flushed for real; the undone row "
                  "stayed dead",
                  statuses == ["executed", "executed", "undone"],
                  str(rows))
            executed = [i for i, s in rows if s == "executed"]
            if executed:
                r = await http.post(
                    f"{base}/api/automations/outbox/{executed[0]}/undo",
                    headers=user_h,
                )
                check("late undo 409s — the write already went out",
                      r.status_code == 409, f"{r.status_code}")
            jrows = await _await_agent_rows(
                "SELECT status, outcome FROM build_jobs "
                "WHERE source_id=? AND job_type='automation_run'",
                (auto_id,),
                lambda r: len(r) >= 3
                and all(st in ("completed", "cancelled") for st, _ in r),
                timeout=30,
            )
            check("run ledger: 2 sent, 1 undone",
                  sorted(o for _, o in jrows)
                  == ["sent", "sent", "undone"], str(jrows))

            # ════ AUTOPAUSE — real failures → the 60-s sweep ══════════
            section("AUTOPAUSE — a revoked grant fails 3 real fires; "
                    "the sweep pauses and speaks ONCE, in the session")
            r = await http.post(
                f"{base}/api/v1/automations/grant-requests", headers=rpc_h,
                json={"connector_id": "stub", "tool_name": "stub__post",
                      "target": {"kind": "channel", "id": "chan-live-2",
                                 "label": "#live2"},
                      "mode": "auto", "summary": "scheduled stub post"},
            )
            sg2 = r.json().get("grant") or {}
            await http.post(
                f"{base}/api/automations/grant-requests/{sg2['id']}"
                f"/approve",
                headers=user_h, json={"decided_via": "web"},
            )
            sched_spec = {
                "name": "Live schedule",
                "trigger": {"mode": "schedule", "schedule": {"every_s": 5}},
                "action": {"connector_id": "stub", "tool": "stub__post",
                           "params_template": {
                               "channel": "{{grant.target.id}}",
                               "text": "tick"},
                           "grant_id": sg2["id"]},
                "mode": "auto",
            }
            r = await http.post(f"{base}/api/automations", headers=user_h,
                                json={"spec": sched_spec})
            sched_id = (r.json().get("automation") or {}).get("id")
            r = await http.post(f"{base}/api/automations/{sched_id}/arm",
                                headers=user_h)
            check("schedule automation armed (fast-lane every_s=5)",
                  r.status_code == 200, f"{r.status_code} {r.text[:200]}")
            jrows = await _await_agent_rows(
                "SELECT id FROM build_jobs WHERE source_id=? "
                "AND status='completed' AND outcome='sent'", (sched_id,),
                lambda r: len(r) >= 1, timeout=90,
            )
            check("the schedule lane fires for real", bool(jrows))
            r = await http.post(
                f"{base}/api/automations/grant-requests/{sg2['id']}"
                f"/revoke", headers=user_h,
            )
            check("grant revoked on the platform (kill switch)",
                  r.status_code == 200
                  and r.json().get("status") == "revoked",
                  f"{r.status_code} {r.text[:160]}")
            arows = await _await_agent_rows(
                "SELECT status, paused_reason, consecutive_failures "
                "FROM automations WHERE id=?", (sched_id,),
                lambda r: bool(r) and r[0][0] == "error", timeout=240,
            )
            check("the REAL sweep auto-paused it to error after 3 "
                  "consecutive failures",
                  bool(arows) and arows[0][0] == "error"
                  and arows[0][1] == "auto_failures"
                  and arows[0][2] >= 3, str(arows))
            r = await http.get(
                f"{base}/api/automations/{sched_id}/thread",
                headers=user_h,
            )
            t_msgs = (r.json() or {}).get("messages") or []
            notices = [m for m in t_msgs
                       if "was paused" in (m.get("content") or "")]
            check("exactly ONE auto-pause notice, IN the session thread",
                  len(notices) == 1,
                  f"{len(notices)} notices in {len(t_msgs)} messages")
            check("the notice carries its navigate chip",
                  bool(notices)
                  and "[[navigate:" in notices[0].get("content", ""),
                  (notices[0].get("content", "")[:160] if notices else ""))

            # ════ ENDPOINTS — thread + memory over the proxy ══════════
            section("ENDPOINTS — session thread and working memory, "
                    "served through the platform")
            r = await http.get(f"{base}/api/automations/{auto_id}/thread",
                               headers=user_h)
            body = r.json() if r.status_code == 200 else {}
            job_cards = [m for m in body.get("messages", [])
                         if m.get("role") == "job"]
            check("thread over the proxy: session id + run cards "
                  "hydrated",
                  r.status_code == 200 and body.get("session_id")
                  and len(job_cards) >= 3
                  and all(c.get("job_status") in ("completed", "cancelled")
                          for c in job_cards),
                  f"{r.status_code} cards={len(job_cards)}")
            r = await http.get(f"{base}/api/automations/{auto_id}/memory",
                               headers=user_h)
            meta = (r.json() or {}).get("metadata") or {} \
                if r.status_code == 200 else {}
            check("working memory over the proxy after terminal runs",
                  r.status_code == 200 and meta.get("last_outcome")
                  and meta.get("last_run_at"),
                  f"{r.status_code} {r.text[:160]}")
            r = await http.get(
                f"{base}/api/automations/{uuid.uuid4()}/memory",
                headers=user_h,
            )
            check("a stranger's automation 404s through the proxy",
                  r.status_code == 404, f"{r.status_code}")

    finally:
        for proc in (a_proc, p_proc):
            proc.terminate()
        for proc in (a_proc, p_proc):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print(f"\n{'PASS' if FAIL == 0 else 'FAIL'}: {PASS} checks passed, "
          f"{FAIL} failed")
    print(f"(logs in {_TMP})")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
