#!/usr/bin/env python3
"""End-to-end proof for Admin Dispatch, across the platform↔agent seam.

WHY THIS EXISTS AS A SCRIPT AND NOT A pytest FILE
    Admin Dispatch is the first feature whose happy path crosses the two
    FastAPI apps: the operator's message is composed on the PLATFORM, but the
    row that makes it appear in the user's chat is written by the TENANT
    AGENT, over HTTP, with `X-Agent-Key`. `run_mode` is process-global and the
    two apps own different halves of one `Base`, so a single pytest process
    can only ever hold one side — which is precisely why every existing test
    of this seam monkeypatches the HTTP client and asserts on the call rather
    than on the result. Nothing in the repo actually runs both.

    So this boots BOTH, as real uvicorn processes on their own SQLite files,
    and drives them with real HTTP requests and a real JWT. Every hop below is
    a genuine network call; nothing is stubbed but APNs (no device to push to)
    and the WS broadcast (no socket connected — the point of asserting on it
    is that `ws_count == 0` must still be a delivery).

WHAT IT PROVES, in order (each step fails loudly with its own message):
    1  an admin can compose a dispatch                      (require_admin, HTTP 201)
    2  the fan-out reaches the tenant over the network       (real X-Agent-Key hop)
    3  the notice is IN the user's chat history              (the reload path)
    4  the operator's payload survived the round trip        (AdminNoticePayload)
    5  the AGENT cannot see it                               (day-context isolation)
    6  the notification was queued on the announcement lane  (data_json field-for-field)
    7  the user's badge counts it                            (GET /notices/state)
    8  reading a `once` notice retracts it from the tenant   (the whole one-time claim)
    9  ...and it is idempotent                               (second read is a no-op)
    10 a `persistent` dispatch opens a thread instead        (and does NOT retract)
    11 the user can reply, and the operator can read it      (the ongoing thread)
    12 the operator can answer, and the user sees it         (with a notification)

USAGE
    cd backend && ./.venv-test/bin/python scripts/e2e_admin_dispatch.py
    (add -v to keep the two servers' logs on stderr)
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERBOSE = "-v" in sys.argv

JWT_SECRET = "e2e-jwt-secret-do-not-ship"
ENCRYPTION_KEY = "test-32-byte-encryption-key--x12"
AGENT_KEY = "e2e-agent-key-" + uuid.uuid4().hex[:8]
# platform_main's lifespan calls credential_crypto.assert_configured() and
# refuses to start without a Fernet key. Generated per run — this process is
# the only thing that will ever hold anything encrypted with it.
import base64 as _b64  # noqa: E402
PLATFORM_ENCRYPTION_KEY = _b64.urlsafe_b64encode(os.urandom(32)).decode()

_step = 0
_failures: list[str] = []

# How long a one-shot DB probe may take before we call it a hang. Generous:
# these are sqlite reads, and the number exists to fail LOUDLY, not to tune.
_PROBE_TIMEOUT_S = 120


def probe(code: str, env: dict) -> subprocess.CompletedProcess:
    """Run a one-shot introspection snippet against one of the two DBs.

    Two things this owns, both learned the hard way on 2026-08-13:

    `subprocess.run` waits for the child to EXIT, not for it to print — and
    aiosqlite's connection worker is a NON-DAEMON thread, so a snippet that
    merely finishes its query leaves the interpreter parked in
    `threading._shutdown` forever. Every assertion had already passed; the
    harness simply never returned from step 6, and because stdout was a pipe
    there was nothing to read either. So the engine is disposed inside the
    same event loop that opened it.

    And the timeout is not belt-and-braces: without it the failure mode of
    ANY future probe is an infinite wait with no output, which is the one
    outcome a proof script must never have. A hang has to look like a
    failure, not like patience.
    """
    if "asyncio.run(m())" not in code:
        raise AssertionError("probe() snippets must end in asyncio.run(m())")
    code = code.replace(
        "asyncio.run(m())",
        "async def _w():\n"
        "    try:\n"
        "        await m()\n"
        "    finally:\n"
        "        from app.db.database import engine\n"
        "        await engine.dispose()\n"
        "asyncio.run(_w())",
    )
    return subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, env=env,
        capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
    )


def step(msg: str) -> None:
    global _step
    _step += 1
    print(f"\n\033[1m{_step:>2}. {msg}\033[0m", flush=True)


def ok(msg: str) -> None:
    print(f"    \033[32m✓\033[0m {msg}", flush=True)


def check(cond: bool, msg: str) -> None:
    if cond:
        ok(msg)
    else:
        _failures.append(msg)
        print(f"    \033[31m✗ {msg}\033[0m", flush=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method: str, url: str, *, token=None, agent_key=None, body=None, expect=None):
    """One request. Returns (status, parsed_json_or_text)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if agent_key:
        req.add_header("X-Agent-Key", agent_key)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        status = e.code
    try:
        payload = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        payload = raw
    if expect is not None and status != expect:
        raise AssertionError(f"{method} {url} → {status} (wanted {expect})\n{raw[:600]}")
    return status, payload


def wait_up(port: int, proc: subprocess.Popen, name: str, timeout=90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"{name} died on boot (exit {proc.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                # The socket is open before the app is ready to route.
                try:
                    http("GET", f"http://127.0.0.1:{port}/health")
                    return
                except Exception:
                    pass
        except OSError:
            pass
        time.sleep(0.4)
    raise SystemExit(f"{name} never came up on :{port}")


def spawn(name: str, module: str, port: int, env_extra: dict) -> subprocess.Popen:
    env = {
        **os.environ,
        "PYTHONPATH": BACKEND,
        "ENVIRONMENT": "test",
        "JWT_SECRET": JWT_SECRET,
        "ENCRYPTION_KEY": ENCRYPTION_KEY,
        "PLATFORM_ENCRYPTION_KEY": PLATFORM_ENCRYPTION_KEY,
        # No APNs credentials → the Live Activity lane records
        # 'apns_not_configured' instead of trying to reach Apple. The queue
        # row is still written, which is what step 6 asserts on.
        "LIVE_ACTIVITY_ENABLED": "false",
        **env_extra,
    }
    out = None if VERBOSE else subprocess.DEVNULL
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", f"{module}:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=BACKEND, env=env, stdout=out, stderr=out,
    )


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="toup-e2e-")
    plat_db = f"sqlite+aiosqlite:///{tmp}/platform.db"
    agent_db = f"sqlite+aiosqlite:///{tmp}/agent.db"
    plat_port, agent_port = free_port(), free_port()
    PLAT = f"http://127.0.0.1:{plat_port}"
    AGENT = f"http://127.0.0.1:{agent_port}"

    user_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    procs: list[subprocess.Popen] = []

    try:
        # ── seed the platform DB before boot: two users + the AgentConfig
        # that points the fan-out at the agent we are about to start ──
        sys.path.insert(0, BACKEND)
        os.environ.update({
            "RUN_MODE": "platform", "ENVIRONMENT": "test",
            "DATABASE_URL": plat_db, "JWT_SECRET": JWT_SECRET,
            "ENCRYPTION_KEY": ENCRYPTION_KEY,
        })
        import asyncio
        from app.db.database import init_db, async_session_maker
        from app.db.models import User, AgentConfig
        from app.services.auth_service import create_access_token

        # Every request below authenticates with a minted JWT, so no password
        # is ever verified — this only satisfies the NOT NULL column. Hashing
        # a real one would drag in passlib/bcrypt, whose pinned pair in
        # .venv-test raises on import ("module 'bcrypt' has no attribute
        # '__about__'"), for nothing this script exercises.
        pwd = "$2b$12$e2e.never.verified.placeholder.hash.value.0000000000"

        async def seed():
            await init_db()
            async with async_session_maker() as db:
                db.add(User(id=admin_id, email="admin@e2e.test", hashed_password=pwd,
                            role="admin", is_active=True, timezone="UTC"))
                db.add(User(id=user_id, email="user@e2e.test", hashed_password=pwd,
                            role="beta_user", is_active=True, timezone="UTC"))
                db.add(AgentConfig(id=str(uuid.uuid4()), user_id=user_id,
                                   agent_url=AGENT, agent_api_key=AGENT_KEY,
                                   deploy_status="active"))
                await db.commit()

        asyncio.run(seed())
        admin_token = create_access_token(admin_id)
        user_token = create_access_token(user_id)

        step("Boot both apps as real processes")
        procs.append(spawn("agent", "agent_main", agent_port, {
            "RUN_MODE": "agent", "DATABASE_URL": agent_db,
            "USER_ID": user_id, "AGENT_API_KEY": AGENT_KEY,
        }))
        procs.append(spawn("platform", "platform_main", plat_port, {
            "RUN_MODE": "platform", "DATABASE_URL": plat_db,
        }))
        wait_up(agent_port, procs[0], "agent")
        wait_up(plat_port, procs[1], "platform")
        ok(f"agent on :{agent_port} (tenant DB), platform on :{plat_port} (platform DB)")

        # ═══ ONE-TIME MODE ═══════════════════════════════════════════════
        step("Admin composes a ONE-TIME dispatch to a single user")
        _, denied = http("POST", f"{PLAT}/api/admin/dispatch", token=user_token,
                         body={"mode": "once", "audience": "user",
                               "target_user_id": user_id, "title": "x", "body": "y"},
                         expect=403)
        ok("a non-admin is refused (403)")

        _, created = http("POST", f"{PLAT}/api/admin/dispatch", token=admin_token,
                          body={"mode": "once", "audience": "user",
                                "target_user_id": user_id,
                                "title": "Scheduled maintenance",
                                "body": "We're upgrading storage on Sunday 03:00 UTC. "
                                        "Nothing you need to do."},
                          expect=201)
        once_id = created["dispatch"]["id"]
        ok(f"dispatch {once_id[:8]} created, status={created['dispatch']['status']}")

        step("The fan-out reaches the tenant agent over the network")
        deadline = time.time() + 30
        detail = None
        while time.time() < deadline:
            _, detail = http("GET", f"{PLAT}/api/admin/dispatch/{once_id}", token=admin_token)
            if detail["dispatch"]["status"] in ("sent", "failed"):
                break
            time.sleep(0.4)
        d, targets = detail["dispatch"], detail["targets"]
        check(d["status"] == "sent", f"dispatch settled to 'sent' (got {d['status']})")
        check(len(targets) == 1, f"exactly one target row (got {len(targets)})")
        t = targets[0]
        check(t["chat_status"] == "delivered",
              f"chat row delivered to the tenant (chat_status={t['chat_status']}, "
              f"err={t.get('last_error')})")
        check(bool(t["chat_message_id"]), "the tenant returned a real message id")
        check(bool(t["notification_id"]), "a notification was queued")

        step("The notice is IN the user's chat history (the reload path)")
        from datetime import datetime, timezone as _tz
        today = datetime.now(_tz.utc).date().isoformat()
        _, msgs = http("GET", f"{AGENT}/api/day-chats/{today}/messages",
                       agent_key=AGENT_KEY, expect=200)
        notices = [m for m in msgs if m.get("admin_notice")]
        check(len(notices) == 1, f"exactly one admin notice in today's thread (got {len(notices)})")

        step("The operator's payload survived the round trip")
        if notices:
            n = notices[0]
            p = n["admin_notice"]
            check(p["dispatch_id"] == once_id, "payload carries the dispatch id")
            check(p["mode"] == "once", f"mode='once' (got {p['mode']!r})")
            check(p["title"] == "Scheduled maintenance", "title round-tripped")
            check(p["sender_name"] == "Toup", f"sender is the operator (got {p['sender_name']!r})")
            check("upgrading storage" in (n["content"] or ""), "the prose is the message content")
            check(n["channel"] == "admin", f"stamped channel='admin' (got {n['channel']!r})")
            check(n["role"] == "assistant", "carried as an assistant-side row")

        step("The AGENT cannot see it — an operator message is not agent context")
        code = (
            "import asyncio, json;"
            "from app.db.database import async_session_maker;"
            "from app.db.models import DayChat;"
            "from sqlalchemy import select;"
            "from app.agent.day_context_loader import load_day_context\n"
            "async def m():\n"
            "    async with async_session_maker() as db:\n"
            "        dc=(await db.execute(select(DayChat))).scalars().first()\n"
            "        if dc is None: print(json.dumps({'n':0,'txt':''})); return\n"
            "        r=await load_day_context(db, dc.id, tz_name='UTC')\n"
            "        txt=' '.join(x.get('content','') for x in r.get('raw_messages',[]))\n"
            "        print(json.dumps({'n':len(r.get('raw_messages',[])),'txt':txt}))\n"
            "asyncio.run(m())"
        )
        env = {**os.environ, "RUN_MODE": "agent", "DATABASE_URL": agent_db,
               "USER_ID": user_id, "AGENT_API_KEY": AGENT_KEY, "PYTHONPATH": BACKEND}
        r = probe(code, env)
        try:
            ctx = json.loads(r.stdout.strip().splitlines()[-1])
            check("upgrading storage" not in ctx["txt"],
                  "load_day_context does NOT contain the operator's message")
        except Exception:
            check(False, f"could not read the agent's day context: {r.stderr[-400:]}")

        step("The notification landed on the announcement lane")
        code = (
            "import asyncio, json;"
            "from app.db.database import async_session_maker;"
            "from app.db.models import NotificationQueue;"
            "from sqlalchemy import select\n"
            "async def m():\n"
            "    async with async_session_maker() as db:\n"
            "        rows=(await db.execute(select(NotificationQueue))).scalars().all()\n"
            "        print(json.dumps([{'kind':x.event_kind,'src':x.source,'idem':x.idempotency_key,"
            "'data':x.data_json,'title':x.title} for x in rows]))\n"
            "asyncio.run(m())"
        )
        env = {**os.environ, "RUN_MODE": "platform", "DATABASE_URL": plat_db, "PYTHONPATH": BACKEND}
        r = probe(code, env)
        rows = json.loads(r.stdout.strip().splitlines()[-1])
        check(len(rows) == 1, f"one queue row (got {len(rows)})")
        if rows:
            q = rows[0]
            check(q["kind"] == "announcement", f"event_kind='announcement' (got {q['kind']!r})")
            check(q["src"] == "platform", "source='platform' (the only kind APNs will trust a deep link from)")
            check(q["idem"] == f"admin-dispatch:{once_id}:{user_id}", "deterministic idempotency key")
            dj = q["data"] or {}
            check(dj.get("kind") == "announcement", "data.kind escapes the autopilot_push toggle")
            check(dj.get("mission_id") == f"admin:{once_id}", "mission id is admin:<dispatch>")
            check(dj.get("cap_exempt") is True, "cap-exempt (an operator notice is not push #11)")
            check(dj.get("deep_link") == f"toup://chat?mission=admin:{once_id}",
                  f"a ONE-TIME card deep-links to the CHAT, where the card is "
                  f"(got {dj.get('deep_link')!r})")
            check(":" not in dj.get("deep_link", "").split("?")[0].removeprefix("toup://"),
                  "no colon in the route segment (Android rebuilds toup://<route>)")

        step("The user's badge counts it")
        _, st = http("GET", f"{PLAT}/api/notices/state", token=user_token, expect=200)
        check(st["unread_notices"] == 1, f"unread_notices=1 (got {st['unread_notices']})")
        check(st["has_thread"] is False, "a one-time notice opens NO thread")
        check(st["thread_unread"] == 0, "and nothing is unread in a thread that does not exist")

        step("Reading a ONE-TIME notice retracts it from the tenant")
        http("POST", f"{PLAT}/api/notices/{once_id}/read", token=user_token, expect=204)
        _, msgs2 = http("GET", f"{AGENT}/api/day-chats/{today}/messages",
                        agent_key=AGENT_KEY, expect=200)
        check(not [m for m in msgs2 if m.get("admin_notice")],
              "the notice is GONE from the tenant's chat history")
        _, st2 = http("GET", f"{PLAT}/api/notices/state", token=user_token, expect=200)
        check(st2["unread_notices"] == 0, "and the badge cleared")
        _, det2 = http("GET", f"{PLAT}/api/admin/dispatch/{once_id}", token=admin_token)
        check(det2["dispatch"]["read_count"] == 1, "the operator sees it was read")
        check(det2["targets"][0]["chat_status"] == "retracted",
              f"the ledger records the retract (got {det2['targets'][0]['chat_status']!r})")

        step("...and reading it again is a no-op, not an error")
        http("POST", f"{PLAT}/api/notices/{once_id}/read", token=user_token, expect=204)
        _, det3 = http("GET", f"{PLAT}/api/admin/dispatch/{once_id}", token=admin_token)
        check(det3["dispatch"]["read_count"] == 1,
              f"read_count still 1, not 2 (got {det3['dispatch']['read_count']})")

        # ═══ PERSISTENT MODE ═════════════════════════════════════════════
        step("Admin composes a PERSISTENT dispatch")
        _, created2 = http("POST", f"{PLAT}/api/admin/dispatch", token=admin_token,
                           body={"mode": "persistent", "audience": "user",
                                 "target_user_id": user_id,
                                 "title": "About your account",
                                 "body": "Your plan changes on 1 September. Reply here if "
                                         "that's a problem.",
                                 "sender_name": "Toup Support"},
                           expect=201)
        pers_id = created2["dispatch"]["id"]
        deadline = time.time() + 30
        while time.time() < deadline:
            _, det = http("GET", f"{PLAT}/api/admin/dispatch/{pers_id}", token=admin_token)
            if det["dispatch"]["status"] in ("sent", "failed"):
                break
            time.sleep(0.4)
        check(det["dispatch"]["status"] == "sent", "it settled to 'sent'")

        _, st3 = http("GET", f"{PLAT}/api/notices/state", token=user_token, expect=200)
        check(st3["has_thread"] is True, "a persistent dispatch OPENS the Admin thread")
        check(st3["thread_unread"] == 1, f"with one unread (got {st3['thread_unread']})")

        _, thread = http("GET", f"{PLAT}/api/notices/thread", token=user_token, expect=200)
        check(len(thread["messages"]) == 1, "the thread holds the operator's message")
        if thread["messages"]:
            m0 = thread["messages"][0]
            check(m0["direction"] == "out", "addressed to the user")
            check(m0["sender_name"] == "Toup Support",
                  f"under the SAME name the card used (got {m0['sender_name']!r})")

        step("A persistent notice stays in the chat when read")
        http("POST", f"{PLAT}/api/notices/{pers_id}/read", token=user_token, expect=204)
        _, msgs3 = http("GET", f"{AGENT}/api/day-chats/{today}/messages",
                        agent_key=AGENT_KEY, expect=200)
        keep = [m for m in msgs3 if (m.get("admin_notice") or {}).get("dispatch_id") == pers_id]
        check(len(keep) == 1, "the card is still there — only `once` retracts")

        step("The user replies, and the operator can read it")
        http("POST", f"{PLAT}/api/notices/thread", token=user_token,
             body={"body": "Yes — I need the old limit until October."}, expect=201)
        _, threads = http("GET", f"{PLAT}/api/admin/dispatch/threads", token=admin_token, expect=200)
        mine = [t for t in threads["threads"] if t["user_id"] == user_id]
        check(len(mine) == 1, "the user appears in the operator's thread list")
        if mine:
            check(mine[0]["unread_in"] == 1, f"with 1 unread reply (got {mine[0]['unread_in']})")
        _, opview = http("GET", f"{PLAT}/api/admin/dispatch/threads/{user_id}",
                         token=admin_token, expect=200)
        bodies = [m["body"] for m in opview["messages"]]
        check(any("old limit until October" in b for b in bodies),
              "the operator reads the user's actual words")

        step("The operator answers, and the user is told")
        http("POST", f"{PLAT}/api/admin/dispatch/threads/{user_id}", token=admin_token,
             body={"body": "Done — you're on the old limit until 31 October."}, expect=201)
        _, thread2 = http("GET", f"{PLAT}/api/notices/thread", token=user_token, expect=200)
        check(any("31 October" in m["body"] for m in thread2["messages"]),
              "the answer is in the user's thread")
        env = {**os.environ, "RUN_MODE": "platform", "DATABASE_URL": plat_db, "PYTHONPATH": BACKEND}
        r = probe(code, env)
        rows = json.loads(r.stdout.strip().splitlines()[-1])
        replies = [x for x in rows if (x["idem"] or "").startswith("admin-thread:")]
        check(len(replies) == 1,
              f"the reply enqueued its own notification (got {len(replies)}) — "
              "an answer nobody is told about is not an answer")

        # ── verdict ──────────────────────────────────────────────────────
        print()
        if _failures:
            print(f"\033[31m✗ {len(_failures)} check(s) FAILED\033[0m")
            for f in _failures:
                print(f"    · {f}")
            return 1
        print("\033[32m✓ end to end: composed on the platform, written by the tenant, "
              "read by the user, retracted, replied to, and answered.\033[0m")
        return 0

    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
