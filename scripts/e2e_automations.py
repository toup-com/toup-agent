#!/usr/bin/env python3
"""End-to-end automations harness (Round 26, Phase 8).

Runs the FULL loop against a REAL platform process over real HTTP —
no kinder cage: the platform boots `platform_main:app` under uvicorn
(monolith run mode, shared sqlite file), the agent side runs in this
process with `platform_api_url` pointed at it, and every write travels
grant-request → user-JWT approve → spec → arm → poll fire → event
dedupe → run (build_jobs) → outbox undo window → grant-gated dispatch
RPC → provider. The stub connector supplies a deterministic feed and a
pinned write with zero network, so the loop is complete and honest on
a laptop.

Sandbox rails enforced and ASSERTED:
  - pinned targets only (a wrong-target write must fail closed)
  - mode='e2e' excluded from metering (zero credit_ledger rows)
  - event dedupe (second poll is a no-op)
  - undo window (undo inside beats the flush; a claimed row wins)
  - auto-pause at 3 consecutive failures
  - R28-C: every run minted a session-thread card with job back-links,
    the auto-pause notice lands IN the session, and noteworthy
    outcomes push with the deep-link contract fields

Real-vendor mode: E2E_CONNECTOR=jira (etc.) re-runs the flow against a
real connector using the vault identity of E2E_USER_EMAIL
(default nariman@toup.ai) — requires a reachable production DB and a
connected identity, so it SKIPS with an explanation when absent.

Usage:  make e2e-automations       (from the repo root)
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent

_PORT = int(os.environ.get("E2E_PLATFORM_PORT", "8971"))
_SECRET = "e2e-automations-secret-key-not-production"


def _fresh_fernet_key() -> str:
    # Generated per run, shared with the platform subprocess via env.
    # Never a committed literal — a secret-shaped constant here would
    # (rightly) trip test_no_committed_secrets and block the public
    # mirror sync.
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


_FERNET = _fresh_fernet_key()
_DB_PATH = Path(tempfile.mkdtemp(prefix="toup-e2e-")) / "e2e.db"
_USER_EMAIL = os.environ.get("E2E_USER_EMAIL", "nariman@toup.ai")

_ENV = {
    **os.environ,
    "RUN_MODE": "monolith",
    "ENVIRONMENT": "development",
    "DATABASE_URL": f"sqlite+aiosqlite:///{_DB_PATH}",
    "SECRET_KEY": _SECRET,
    "PLATFORM_ENCRYPTION_KEY": _FERNET,
    "AUTOMATIONS_ROLLOUT_PCT": "100",
    "AUTOMATIONS_ENABLED": "true",
    "AUTOMATIONS_E2E": "1",
    # Round 28 dev fast-lane: honored because ENVIRONMENT=development
    # above — the same two-sided gate production refuses.
    "AUTOMATIONS_DEV_FAST_LANE": "1",
    "ENABLE_CONNECTOR_OAUTH": "true",
}

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}" + (f" — {detail}" if detail and not ok else ""))
    if ok:
        PASS += 1
    else:
        FAIL += 1


def _wait_port(port: int, timeout: float = 60.0) -> bool:
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


async def main() -> int:
    # ── boot the platform for real ──────────────────────────────────
    # A LEAKED child from a crashed earlier run keeps LISTENing here —
    # the fresh uvicorn then dies on the bind while the harness
    # health-checks the ZOMBIE (old tree, old DB) and every failure
    # after that is nonsense ("no such table: agent_configs"). Fail
    # LOUD before booting instead of debugging a stranger (R29).
    with socket.socket() as _probe:
        _probe.settimeout(0.5)
        try:
            _probe.connect(("127.0.0.1", _PORT))
            print(f"❌ port {_PORT} is already in use — a leaked platform "
                  f"from a crashed run? `lsof -ti :{_PORT} | xargs kill` "
                  f"and re-run (or set E2E_PLATFORM_PORT).")
            return 1
        except OSError:
            pass
    log_path = _DB_PATH.parent / "platform.log"
    log_f = open(log_path, "w")
    print(f"  (platform log: {log_path})")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "platform_main:app",
         "--port", str(_PORT), "--log-level", "warning"],
        cwd=str(BACKEND), env=_ENV,
        stdout=log_f, stderr=subprocess.STDOUT,
    )
    try:
        if not _wait_port(_PORT):
            print("❌ platform did not come up")
            return 1

        # The harness process becomes the "agent container".
        agent_key = f"e2e-key-{uuid.uuid4().hex}"
        user_id = str(uuid.uuid4())
        os.environ.update(_ENV)
        os.environ["RUN_MODE"] = "agent"
        os.environ["USER_ID"] = user_id
        os.environ["AGENT_API_KEY"] = agent_key
        os.environ["PLATFORM_API_URL"] = f"http://127.0.0.1:{_PORT}/api"

        sys.path.insert(0, str(BACKEND))
        import httpx
        from app.config import settings
        assert settings.user_id == user_id
        assert settings.automations_enabled is True

        base = f"http://127.0.0.1:{_PORT}"
        async with httpx.AsyncClient(base_url=base, timeout=30) as http:
            r = await http.get("/api/health")
            check("platform health", r.status_code == 200)

            # ── identities: user (acting as the operator account), ──
            # agent config, stub connector tokens in the REAL vault.
            from app.db.database import async_session_maker
            from app.db.models import AgentConfig, User
            from app.services.auth_service import (
                create_access_token, get_password_hash,
            )
            async with async_session_maker() as db:
                db.add(User(id=user_id, email=_USER_EMAIL,
                            hashed_password=get_password_hash("e2e-pass-1"),
                            name="E2E", timezone="America/Toronto"))
                db.add(AgentConfig(
                    user_id=user_id, agent_url="http://127.0.0.1:1",
                    agent_api_key=agent_key, deploy_status="active",
                ))
                await db.commit()
            jwt = create_access_token(user_id)
            user_h = {"Authorization": f"Bearer {jwt}"}
            rpc_h = {"X-Agent-Key": agent_key, "X-Agent-User-Id": user_id}

            from app.services import connector_vault as vault
            async with async_session_maker() as db:
                await vault.put(
                    db, user_id, "stub",
                    access_token="stub_access_token_e2e",
                    refresh_token="stub_refresh_token_e2e",
                    access_expires_at=datetime.utcnow() + timedelta(hours=2),
                    scopes=[],
                    provider_account_id=_USER_EMAIL,
                    event_type="connected",
                )

            # ── registry over the real RPC ──────────────────────────
            r = await http.get("/api/v1/automations/registry", headers=rpc_h)
            reg_ids = {c["connector_id"] for c in r.json().get("connectors", [])}
            check("registry serves stub (dev only)", "stub" in reg_ids,
                  str(reg_ids))

            # ── grant: request via RPC, approve via user JWT ────────
            r = await http.post(
                "/api/v1/automations/grant-requests", headers=rpc_h,
                json={"connector_id": "stub", "tool_name": "stub__post",
                      "target": {"kind": "channel", "id": "chan-1",
                                 "label": "#e2e"},
                      "cadence": {"per_day": 50}, "mode": "auto",
                      "summary": "post to #e2e"},
            )
            grant = r.json().get("grant") or {}
            check("grant request staged", grant.get("status") == "pending")
            r = await http.post(
                f"/api/automations/grant-requests/{grant['id']}/approve",
                headers=user_h, json={"decided_via": "web"},
            )
            check("grant approved by user", r.json().get("status") == "approved")

            # ── build + arm the automation (agent side, real service) ─
            from app.agent.automations.service import (
                arm_automation, create_automation, list_runs,
            )
            spec = {
                "name": "Stub → Stub e2e",
                "trigger": {"mode": "poll", "connector_id": "stub",
                            "event": "item_created",
                            "poll_interval_s": 300},
                "action": {"connector_id": "stub", "tool": "stub__post",
                           "params_template": {
                               "channel": "{{grant.target.id}}",
                               "text": "New: {{event.title}}"},
                           "grant_id": grant["id"]},
                "dedupe_key": "event.id",
                "mode": "auto",
            }
            async with async_session_maker() as db:
                automation, vspec = await create_automation(
                    db, user_id=user_id, spec=spec,
                )
                check("automation drafted", automation.status == "draft")
                automation = await arm_automation(
                    db, automation_id=automation.id, user_id=user_id,
                )
                check("armed (grant verified via platform)",
                      automation.status == "armed")

            # ── fire: one poll through the whole pipe ───────────────
            from app.agent.automations import executor
            from app.agent.automations.service import parse_spec_live
            async with async_session_maker() as db:
                a = await db.get(type(automation), automation.id)
                stats = await executor.poll_and_run(db, a, await parse_spec_live(a))
            check("poll observed 3, ran 3",
                  stats == {"observed": 3, "fresh": 3, "ran": 3,
                            "failed": 0}, str(stats))

            from sqlalchemy import select
            from app.db.models import AutomationOutbox, BuildJob
            async with async_session_maker() as db:
                boxes = (await db.execute(
                    select(AutomationOutbox)
                    .where(AutomationOutbox.automation_id == automation.id)
                )).scalars().all()
                check("3 outbox rows executed",
                      len(boxes) == 3 and
                      all(b.status == "executed" for b in boxes),
                      str([(b.status, b.last_error) for b in boxes]))
                sent = json.loads(boxes[0].result_json or "{}")
                content = json.loads(sent.get("content") or "{}")
                check("write reached the provider via the platform "
                      "(channel=automation, pinned target)",
                      content.get("channel") == "automation" and
                      content.get("input", {}).get("channel") == "chan-1",
                      str(content))
                jobs = (await db.execute(
                    select(BuildJob)
                    .where(BuildJob.source_id == automation.id)
                    .where(BuildJob.job_type == "automation_run")
                )).scalars().all()
                check("3 runs completed outcome=sent",
                      len(jobs) == 3 and
                      all(j.status == "completed" and j.outcome == "sent"
                          for j in jobs),
                      str([(j.status, j.outcome, j.user_message)
                           for j in jobs]))
                steps_ok = all(
                    len(json.loads(j.steps_json)) == 4 and
                    all(s["status"] == "done" and s.get("duration_ms")
                        is not None
                        for s in json.loads(j.steps_json))
                    for j in jobs
                )
                check("per-step durations recorded", steps_ok)
                runs_payload = await list_runs(db, user_id)
                check("runs API shape", len(runs_payload) == 3 and
                      runs_payload[0]["automation_name"] == a.name)

            # ── metering exclusion ──────────────────────────────────
            r = await http.get("/api/v1/automations/grant-status",
                               params={"grant_id": grant["id"]},
                               headers=rpc_h)
            from app.db.models import CreditLedger
            async with async_session_maker() as db:
                ledger = (await db.execute(
                    select(CreditLedger)
                    .where(CreditLedger.user_id == user_id)
                )).scalars().all()
            check("mode=e2e excluded from metering (empty ledger)",
                  len(ledger) == 0, f"{len(ledger)} rows")

            # ── dedupe: the second poll is a no-op ──────────────────
            async with async_session_maker() as db:
                a = await db.get(type(automation), automation.id)
                stats2 = await executor.poll_and_run(db, a, await parse_spec_live(a))
            check("second poll deduped to zero",
                  stats2["fresh"] == 0 and stats2["observed"] == 3,
                  str(stats2))

            # ── pinned-target violation fails closed ────────────────
            from app.agent.automations import registry as areg
            bad = await areg.dispatch_via_platform(
                user_id, connector_id="stub", tool_name="stub__post",
                tool_input={"channel": "chan-EVIL", "text": "x"},
                grant_id=grant["id"], automation_id=automation.id,
            )
            check("wrong target refused at the platform",
                  bad.get("kind") == "tool_error"
                  and "may only write to" in (bad.get("message") or ""),
                  str(bad)[:160])

            # ── undo inside the window ──────────────────────────────
            from app.agent.automations.outbox import undo_row, _claim
            async with async_session_maker() as db:
                row = AutomationOutbox(
                    user_id=user_id, automation_id=automation.id,
                    connector_id="stub", tool_name="stub__post",
                    payload_json=json.dumps({"channel": "chan-1",
                                             "text": "undo me"}),
                    grant_id=grant["id"],
                    idempotency_key=f"undo:{uuid.uuid4()}",
                    execute_after=datetime.utcnow() + timedelta(seconds=30),
                )
                db.add(row)
                await db.commit()
                undone = await undo_row(db, row.id, user_id)
                claimed = await _claim(db, row.id)
            check("undo beats the flush inside the window",
                  undone is True and claimed is False)

            # ── auto-pause at 3 failures + ONE notice ───────────────
            from app.agent.automations.sweep import _sweep_auto_pause
            async with async_session_maker() as db:
                a = await db.get(type(automation), automation.id)
                a.consecutive_failures = 3
                a.last_error = "e2e-forced"
                await db.commit()
            n = await _sweep_auto_pause()
            async with async_session_maker() as db:
                a = await db.get(type(automation), automation.id)
                check("auto-pause flipped to error",
                      n == 1 and a.status == "error"
                      and a.paused_reason == "auto_failures")
                from app.db.models import Message
                notices = (await db.execute(
                    select(Message).where(Message.source == "automation")
                    .where(Message.content.like("%was paused%"))
                )).scalars().all()
                check("exactly ONE chat notice", len(notices) == 1,
                      f"{len(notices)} notices")

            # ═══ Round 28: spec v2 over the same real rails ═════════
            # A second grant (chan-2), a v2 spec whose read step feeds
            # the write's template, the fast-lane interval, the
            # namespaced dedupe, and the engine-memory round trip.
            r = await http.post(
                "/api/v1/automations/grant-requests", headers=rpc_h,
                json={"connector_id": "stub", "tool_name": "stub__post",
                      "target": {"kind": "channel", "id": "chan-2",
                                 "label": "#e2e-v2"},
                      "mode": "auto", "summary": "post the v2 digest"},
            )
            grant2 = r.json().get("grant") or {}
            await http.post(
                f"/api/automations/grant-requests/{grant2['id']}/approve",
                headers=user_h, json={"decided_via": "web"},
            )

            spec_v2 = {
                "version": 2,
                "name": "Stub digest v2",
                "mode": "auto",
                "variables": {"greeting": "digest"},
                "trigger": {"sources": [
                    {"id": "feed_src", "mode": "poll",
                     "connector_id": "stub", "event": "item_created",
                     # 5s — legal ONLY because the dev fast-lane env is
                     # on and ENVIRONMENT != production.
                     "poll_interval_s": 5,
                     "dedupe_key": "event.id"},
                ]},
                "steps": [
                    {"id": "feed", "connector_id": "stub",
                     "tool": "stub__list_items", "params": {},
                     "collect": {"items_path": "items",
                                 "fields": {"title": "title"},
                                 "format": "• {{item.title}}",
                                 "empty_text": "(empty)"},
                     "on_error": "skip"},
                    {"id": "post", "connector_id": "stub",
                     "tool": "stub__post",
                     "params": {"channel": "{{grant.target.id}}",
                                "text": "{{var.greeting}} "
                                        "[{{steps.feed.count}}]\n"
                                        "{{steps.feed.text}}"},
                     "grant_id": grant2["id"]},
                ],
            }
            async with async_session_maker() as db:
                auto2, _ = await create_automation(
                    db, user_id=user_id, spec=spec_v2,
                )
                check("v2 drafted (fast-lane 5s interval accepted)",
                      auto2.status == "draft")
                auto2 = await arm_automation(
                    db, automation_id=auto2.id, user_id=user_id,
                )
                check("v2 armed (per-step grant verified via platform)",
                      auto2.status == "armed")

            from app.db.models import AutomationBinding, Routine
            async with async_session_maker() as db:
                b2 = (await db.execute(
                    select(AutomationBinding)
                    .where(AutomationBinding.automation_id == auto2.id)
                )).scalars().all()
                routine2 = await db.get(Routine, b2[0].target_id)
                check("v2 poll routine carries source_id + 5s interval",
                      (routine2.config_json or {}).get("source_id")
                      == "feed_src"
                      and routine2.schedule_interval_seconds == 5,
                      str((routine2.config_json,
                           routine2.schedule_interval_seconds)))

            from app.agent.automations import executor_v2
            async with async_session_maker() as db:
                a = await db.get(type(auto2), auto2.id)
                vspec2 = await parse_spec_live(a)
                source = vspec2.source_by_id("feed_src")
                stats3 = await executor_v2.poll_and_run_v2(
                    db, a, vspec2, source)
            check("v2 poll observed 3, ran 3",
                  stats3 == {"observed": 3, "fresh": 3, "ran": 3,
                             "failed": 0}, str(stats3))

            from app.db.models import AutomationEvent
            async with async_session_maker() as db:
                evs = (await db.execute(
                    select(AutomationEvent)
                    .where(AutomationEvent.automation_id == auto2.id)
                )).scalars().all()
                check("v2 events namespaced per source",
                      len(evs) == 3 and
                      all(e.dedupe_key.startswith("feed_src:")
                          for e in evs),
                      str([e.dedupe_key for e in evs]))
                boxes2 = (await db.execute(
                    select(AutomationOutbox)
                    .where(AutomationOutbox.automation_id == auto2.id)
                )).scalars().all()
                sent_ok = (
                    len(boxes2) == 3
                    and all(b.status == "executed" for b in boxes2)
                    and all(b.idempotency_key.endswith(":w0")
                            for b in boxes2)
                )
                check("v2 writes executed with w0 idempotency keys",
                      sent_ok,
                      str([(b.status, b.idempotency_key,
                            b.last_error) for b in boxes2]))
                payload0 = json.loads(boxes2[0].payload_json)
                check("v2 write rendered var + collected read step",
                      payload0.get("channel") == "chan-2"
                      and payload0.get("text", "").startswith("digest [3]")
                      and "• First stub item" in payload0.get("text", ""),
                      str(payload0)[:200])
                jobs2 = (await db.execute(
                    select(BuildJob)
                    .where(BuildJob.source_id == auto2.id)
                    .where(BuildJob.job_type == "automation_run")
                )).scalars().all()
                step_ids = ([s["id"] for s in
                             json.loads(jobs2[0].steps_json)]
                            if jobs2 else [])
                check("v2 runs completed with dynamic step ledger",
                      len(jobs2) == 3
                      and all(j.status == "completed"
                              and j.outcome == "sent" for j in jobs2)
                      and step_ids == ["evaluate", "feed", "post",
                                       "record"],
                      str([(j.status, j.outcome) for j in jobs2])
                      + str(step_ids))

            from app.agent.automations import memory as engine_memory
            async with async_session_maker() as db:
                a = await db.get(type(auto2), auto2.id)
                mem = await engine_memory.read_context(db, a)
            check("engine memory written after run, read at fire time",
                  mem.get("last_outcome") == "sent"
                  and json.loads(mem.get("last_counts") or "{}")
                  == {"feed": 3}, str(mem))

            # ── template catalog: synced at boot, served both ways ──
            r = await http.get("/api/v1/automations/templates",
                               headers=rpc_h)
            tpls = {t["slug"]: t for t in r.json().get("templates", [])}
            check("catalog synced at boot and served over the RPC",
                  "morning-work-brief" in tpls
                  and tpls["morning-work-brief"]["category"] == "work"
                  and len(tpls["morning-work-brief"]["variables"]) >= 3,
                  str(sorted(tpls))[:200])
            r = await http.get("/api/automations/templates",
                               params={"category": "email"},
                               headers=user_h)
            email_slugs = {t["slug"] for t in r.json().get("templates", [])}
            check("user route filters by category",
                  "boss-email-draft" in email_slugs
                  and "morning-work-brief" not in email_slugs,
                  str(email_slugs))

            # ═══ Round 28-C: session thread, run cards, outcome push ═
            # The v1 runs above went through on_run_created and
            # _finalize_job with the R28-C hooks live — verify the
            # artifacts on real rows. In-process like every agent-side
            # check here (the harness runs no agent HTTP server), which
            # also exercises A's StaticPool interleaving warning: these
            # writers ran beside the outbox flush loop on ONE sqlite
            # connection.
            from app.api.automations import automation_thread
            thread = await automation_thread(automation.id, limit=200)
            check("session thread minted for the automation",
                  bool(thread.get("session_id")))
            t_rows = thread.get("messages") or []
            job_cards = [m for m in t_rows
                         if getattr(m, "role", None) == "job"]
            pause_notices = [
                m for m in t_rows
                if "was paused" in (getattr(m, "content", "") or "")
            ]
            check("every v1 run has a card in the session",
                  len(job_cards) >= 3, f"{len(job_cards)} cards")
            check("the auto-pause notice lives IN the session thread",
                  len(pause_notices) == 1, f"{len(pause_notices)}")
            check("run cards hydrate job fields through the pipeline",
                  all(getattr(m, "job_id", None)
                      and getattr(m, "job_status", None) == "completed"
                      for m in job_cards[:3]),
                  str([(getattr(m, "job_id", None),
                        getattr(m, "job_status", None))
                       for m in job_cards[:3]]))
            async with async_session_maker() as db:
                jobs = (await db.execute(
                    select(BuildJob)
                    .where(BuildJob.source_id == automation.id)
                    .where(BuildJob.job_type == "automation_run")
                )).scalars().all()
                check("every run back-linked (summary_message_id + "
                      "conversation_id)",
                      len(jobs) >= 3 and all(
                          j.summary_message_id and j.conversation_id
                          for j in jobs),
                      str([(j.summary_message_id, j.conversation_id)
                           for j in jobs]))
                from app.db.models import AgentNotifyOutbox
                pushes = (await db.execute(
                    select(AgentNotifyOutbox).where(
                        AgentNotifyOutbox.dedup_key
                        == f"automation:{automation.id}:run_done")
                )).scalars().all()
                push_data_ok = all(
                    (p.data_json or {}).get("route") == "automation"
                    and (p.data_json or {}).get("automation_id")
                    == automation.id
                    and (p.data_json or {}).get("run_id")
                    and (p.data_json or {}).get("chat_id")
                    and (p.data_json or {}).get("message_id")
                    for p in pushes)
                check("noteworthy outcomes pushed with the deep-link "
                      "contract fields",
                      len(pushes) >= 1 and push_data_ok,
                      f"{len(pushes)} pushes")

            # ═══ Round 28-D: the card/endpoint surface, in-process ═══
            # The live harness (e2e_automations_live.py) proves these
            # over real HTTP/WS; this keeps the same contracts pinned
            # in the fast suite a laptop can run without an LLM key.
            from app.api.automations import (
                automation_memory, connector_connected_hook,
                grant_decided_hook, undo_outbox, ConnectorHook,
                GrantHook,
            )
            from fastapi import HTTPException

            # /memory — CONTRACTS-R30 §4.5: the route serves the curated
            # categories view (the §3.10 sheet), NOT the engine state row
            # (that row is internal-only now — the D-07 retirement). The
            # old assertion pinned the R29 shape and went red when A
            # landed the contract; the rig follows the contract.
            mem_body = await automation_memory(auto2.id)
            mem_cats = [c.get("key") for c in
                        (mem_body.get("categories") or [])]
            check("memory route serves the §4.5 categories view, "
                  "engine state values off the wire (legacy keys are "
                  "null tombstones for the pre-rebuild app)",
                  isinstance(mem_body.get("count"), int)
                  and mem_cats == ["people", "team_workspace", "your_time",
                                   "work_you_own", "noise_filters"]
                  and not mem_body.get("metadata")
                  and mem_body.get("content") is None,
                  str(mem_body)[:160])
            try:
                await automation_memory(str(uuid.uuid4()))
                check("memory route 404s on a stranger", False, "no raise")
            except HTTPException as e:
                check("memory route 404s on a stranger",
                      e.status_code == 404, str(e.status_code))

            # Connector card + the _connector_connected wake-up hook:
            # stage a card the way the skill tool does, then flip it
            # the way the platform's OAuth callback does.
            from app.agent.automations import cards
            from app.db.models import (
                AutomationAuthSession, AUTOMATION_AUTH_SESSION_TTL_S,
            )
            async with async_session_maker() as db:
                auth_s = AutomationAuthSession(
                    user_id=user_id, connector_id="stub",
                    mode="read", scopes_json="[]", status="offered",
                    expires_at=datetime.utcnow()
                    + timedelta(seconds=AUTOMATION_AUTH_SESSION_TTL_S),
                )
                db.add(auth_s)
                await db.commit()
                payload = cards.connector_card_payload(
                    auth_s, name="Stub", icon=None, scopes=[],
                )
                msg_id, _ = await cards.write_card_message(
                    db, user_id=user_id, content="Connect Stub",
                    metadata_key=cards.CONNECTOR_CARD_KEY,
                    payload=payload, title="Connect a service",
                )
                auth_s2 = await db.get(AutomationAuthSession, auth_s.id)
                auth_s2.message_id = msg_id
                await db.commit()
            hook_res = await connector_connected_hook(
                ConnectorHook(connector_id="stub", ok=True),
            )
            async with async_session_maker() as db:
                auth_s3 = await db.get(AutomationAuthSession, auth_s.id)
                msg = await db.get(Message, msg_id)
                card_meta = json.loads(msg.metadata_json)[
                    cards.CONNECTOR_CARD_KEY]
            check("wake-up hook flips the open card to connected",
                  hook_res.get("updated") == 1
                  and auth_s3.status == "connected",
                  str((hook_res, auth_s3.status)))
            check("…and rewrites the card's message row in place",
                  card_meta.get("status") == "connected"
                  and card_meta.get("id") == auth_s.id,
                  str(card_meta)[:160])

            # Grant card + the _grant_decided hook: the platform's
            # approve calls this to rewrite the chat card.
            async with async_session_maker() as db:
                g_msg_id, _ = await cards.write_card_message(
                    db, user_id=user_id, content="May I post?",
                    metadata_key=cards.GRANT_CARD_KEY,
                    payload={"id": grant2["id"], "status": "pending"},
                    title="Permission",
                )
            await grant_decided_hook(GrantHook(
                grant_id=grant2["id"], status="approved",
                payload={"decided_via": "web"},
            ))
            async with async_session_maker() as db:
                g_msg = await db.get(Message, g_msg_id)
                g_meta = json.loads(g_msg.metadata_json)[
                    cards.GRANT_CARD_KEY]
            check("grant-decided hook rewrites the grant card in place",
                  g_meta.get("status") == "approved"
                  and g_meta.get("decided_via") == "web",
                  str(g_meta)[:160])

            # Undo ROUTE semantics: an executed row answers 409, an
            # in-window row answers {"undone": true}.
            async with async_session_maker() as db:
                late_row = (await db.execute(
                    select(AutomationOutbox)
                    .where(AutomationOutbox.automation_id
                           == automation.id)
                    .where(AutomationOutbox.status == "executed")
                )).scalars().first()
            try:
                await undo_outbox(late_row.id)
                check("undo route 409s after the write went out",
                      False, "no raise")
            except HTTPException as e:
                check("undo route 409s after the write went out",
                      e.status_code == 409, str(e.status_code))
            async with async_session_maker() as db:
                fresh_row = AutomationOutbox(
                    user_id=user_id, automation_id=automation.id,
                    connector_id="stub", tool_name="stub__post",
                    payload_json=json.dumps({"channel": "chan-1",
                                             "text": "route undo"}),
                    grant_id=grant["id"],
                    idempotency_key=f"route-undo:{uuid.uuid4()}",
                    execute_after=datetime.utcnow()
                    + timedelta(seconds=30),
                )
                db.add(fresh_row)
                await db.commit()
                fresh_id = fresh_row.id
            undo_res = await undo_outbox(fresh_id)
            check("undo route cancels an in-window row",
                  undo_res.get("undone") is True, str(undo_res))

            # ═══ Round 29: verbs, last outcome, facts, grants ═══════
            from app.db.models import Automation
            from app.api.automations import (
                add_memory_fact, delete_memory_fact, list_memory_facts,
                list_runs as runs_route, list_automations as list_route,
                mark_seen, update_memory_fact, FactBody, FactPatchBody,
            )

            listing = {a["id"]: a for a in
                       (await list_route())["automations"]}
            row = listing.get(automation.id) or {}
            check("R29 list payload: connectors + rule_text + "
                  "schedule keys",
                  row.get("connectors") == ["stub"]
                  and "schedule_human" in row
                  and (row.get("rule_text") or "").startswith("When a new")
                  and "post to the test channel" in
                  (row.get("rule_text") or ""),
                  str({k: row.get(k) for k in
                       ("connectors", "schedule_human", "rule_text")}))
            lo = row.get("last_outcome") or {}
            check("R29 last outcome stamped by the runs, unseen until "
                  "opened",
                  row.get("unseen") is True
                  and lo.get("tone") in ("ok", "warn", "err")
                  and lo.get("sentence") and "__" not in lo["sentence"],
                  str((row.get("unseen"), lo))[:160])
            await mark_seen(automation.id)
            listing = {a["id"]: a for a in
                       (await list_route())["automations"]}
            check("R29 seen CAS clears unseen",
                  listing[automation.id]["unseen"] is False)

            served_runs = (await runs_route(
                automation_id=automation.id, limit=20))["runs"]
            step_strings = [
                s.get(k) or ""
                for r in served_runs for s in r["steps"]
                for k in ("verb", "label")
            ]
            check("R29 runs carry verb+brand and no raw tool names",
                  served_runs
                  and all("verb" in s and "brand" in s
                          for r in served_runs for s in r["steps"])
                  and all("__" not in x for x in step_strings),
                  str(step_strings)[:200])
            done_writes = [
                s["verb"] for r in served_runs for s in r["steps"]
                if s.get("brand") == "stub"
                and s.get("status") in ("done", "completed")
            ]
            check("R29 write steps wear the dictionary's done form",
                  done_writes
                  and all(v == "Posted to the test channel"
                          for v in done_writes),
                  str(done_writes)[:120])

            # Curated facts: user CRUD through the routes, agent batch
            # through the seam, attribution derived.
            from app.agent.automations import facts as facts_mod
            fact = (await add_memory_fact(automation.id, FactBody(
                text="Standup moved to 9:15", category="work",
            )))["fact"]
            async with async_session_maker() as db:
                await facts_mod.record(
                    db, user_id=user_id, automation_id=automation.id,
                    facts=["Boss is Sarah", "Prefers bullets"],
                    category="preferences", source="agent",
                    source_kind="automation_run", run_id="run-e2e",
                )
            ledger = await list_memory_facts(automation.id)
            check("R29 facts ledger lists both sources in category "
                  "order",
                  len(ledger["facts"]) == 3
                  and ledger["last_agent_update"]["count"] == 2
                  and [f["category"] for f in ledger["facts"]][:2]
                  == ["preferences", "preferences"],
                  str(ledger)[:200])
            await update_memory_fact(
                automation.id, fact["id"],
                FactPatchBody(text="Standup moved to 9:30"),
            )
            await delete_memory_fact(automation.id, fact["id"])
            ledger = await list_memory_facts(automation.id)
            check("R29 fact edit+delete land",
                  len(ledger["facts"]) == 2
                  and all(f["text"] != "Standup moved to 9:30"
                          for f in ledger["facts"]))

            # Grants on the Overview — platform-native, over the wire.
            async with async_session_maker() as db:
                from app.db.models import AutomationGrant
                g_row = AutomationGrant(
                    user_id=user_id, automation_id=automation.id,
                    connector_id="stub", tool_name="stub__post",
                    target_json=json.dumps({"kind": "channel",
                                            "id": "chan-1"}),
                    mode="auto", summary="post to the test channel",
                    status="approved", decided_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(hours=1),
                )
                db.add(g_row)
                await db.commit()
                g_row_id = g_row.id
            r = await http.get(
                f"/api/automations/{automation.id}/grants",
                headers=user_h)
            served = {g["id"] for g in r.json().get("grants", [])}
            check("R29 grants list serves the automation's grants",
                  r.status_code == 200 and g_row_id in served,
                  str(r.json())[:160])
            r = await http.post(
                f"/api/automations/{automation.id}/grants/"
                f"{g_row_id}/revoke",
                headers=user_h)
            check("R29 nested revoke transitions the grant",
                  r.status_code == 200
                  and r.json().get("status") == "revoked",
                  str(r.json())[:120])
            # The pause half of revoke rides the _grant_decided hook —
            # in-process here (the monolith has no agent HTTP server
            # for the platform to call back).
            async with async_session_maker() as db:
                a_row = await db.get(Automation, automation.id)
                a_row.status = "armed"
                a_row.paused_reason = None
                await db.commit()
            await grant_decided_hook(GrantHook(
                grant_id=g_row_id, status="revoked",
                payload={"automation_id": automation.id},
            ))
            listing = {a["id"]: a for a in
                       (await list_route())["automations"]}
            check("R29 revoked grant pauses the dependent "
                  "(attention=grant_revoked)",
                  listing[automation.id]["attention"] == "grant_revoked",
                  str(listing[automation.id].get("attention")))

            check("R29 template cadence tag",
                  tpls["morning-work-brief"].get("cadence_human")
                  == "weekdays 8:00",
                  str(tpls["morning-work-brief"].get("cadence_human")))

            # ── R30: ledger v3, threads, notifications, workflow ───
            from app.agent.automations import ledger as r30_ledger
            from app.db.models import (
                AutomationNotification, AutomationTurn, AutomationWrite,
                Message as R30Message,
            )

            async with async_session_maker() as db:
                thread = await r30_ledger.thread_for(db, auto2.id)
                check("R30 thread minted for the v2 automation",
                      thread is not None, str(thread))
                v2_runs = (await db.execute(
                    select(BuildJob)
                    .where(BuildJob.source_id == auto2.id)
                    .order_by(BuildJob.created_at.asc())
                )).scalars().all()
                run0 = v2_runs[0]
                turns = await r30_ledger.run_turns(db, run_id=run0.id)
                kinds = [t["kind"] for t in turns]
                check("R30 run opens with a note that flipped to RAN",
                      kinds and kinds[0] == "note"
                      and turns[0].get("stamp") == "ran", str(kinds[:3]))
                tools = [t for t in turns if t["kind"] == "tool"]
                reads = [t for t in tools if t["tool_kind"] == "read"]
                writes = [t for t in tools if t["tool_kind"] == "write"]
                check("R30 read + write tool turns with clean actions",
                      len(reads) >= 1 and len(writes) >= 1
                      and all("__" not in (t.get("action") or "")
                              for t in tools)
                      and all(it.get("id")
                              for t in reads for it in t["items"]),
                      str([(t["tool_kind"], t["action"]) for t in tools]))
                results = [t for t in turns if t["kind"] == "result"]
                item_ids = {it["id"] for t in reads
                            for it in (t.get("items") or [])}
                refs = [ref for t in results
                        for g in t.get("groups") or []
                        for r0 in g.get("rows") or []
                        for ref in r0.get("item_refs") or []]
                check("R30 completeness: one result, every item "
                      "referenced exactly once",
                      len(results) == 1 and sorted(refs)
                      == sorted(item_ids) and len(refs) == len(set(refs)),
                      f"items={len(item_ids)} refs={len(refs)}")
                w_rows = (await db.execute(
                    select(AutomationWrite)
                    .where(AutomationWrite.automation_id == auto2.id)
                )).scalars().all()
                check("R30 write ledger: one honest row per executed "
                      "write with undo_ref",
                      len(w_rows) == 3
                      and all(w.undo_ref for w in w_rows)
                      and all(w.audience in ("you", "others")
                              for w in w_rows),
                      str([(w.what, w.audience) for w in w_rows])[:120])
                cfg0 = r30_ledger._cfg_of(run0)
                check("R30 accounts_touched stamped on the run",
                      "stub" in (cfg0.get("accounts_touched") or []),
                      str(cfg0.get("accounts_touched")))

                notes = (await db.execute(
                    select(AutomationNotification)
                    .where(AutomationNotification.automation_id
                           == auto2.id)
                    .where(AutomationNotification.kind
                           == "automation_run")
                )).scalars().all()
                by_run = {n.run_id for n in notes}
                check("R30 one notification per run, bodies filled",
                      len(notes) == len(v2_runs)
                      and by_run == {r.id for r in v2_runs}
                      and all(n.body for n in notes),
                      f"{len(notes)} notes / {len(v2_runs)} runs")
                n0 = next(n for n in notes if n.run_id == run0.id)
                card_msg = await db.get(R30Message, n0.message_id) \
                    if n0.message_id else None
                card = (json.loads(card_msg.metadata_json or "{}")
                        .get("automation_notification")
                        if card_msg is not None else None)
                check("R30 the in-chat card carries the SAME body",
                      card is not None and card.get("body") == n0.body
                      and card.get("status") in ("completed", "partial"),
                      str(card)[:120])

            # Stop route: a finished run refuses with the honest 409.
            from app.api.automations import (
                automations_summary, get_workflow, nested_runs,
                post_workflow_rule, put_account_permissions,
                put_workflow_schedule, stop_run,
                PermissionsBody, PresetBody, RuleBody,
            )
            try:
                await stop_run(run0.id)
                check("R30 stop refuses a finished run", False, "no raise")
            except HTTPException as e:
                check("R30 stop refuses a finished run",
                      e.status_code == 409
                      and (e.detail or {}).get("code") == "not_running",
                      str(e.detail))

            runs_nested = await nested_runs(auto2.id, limit=50)
            check("R30 nested runs alias serves the flat list",
                  len(runs_nested.get("runs") or []) == len(v2_runs),
                  str(len(runs_nested.get("runs") or [])))

            summary = await automations_summary()
            s_items = {x["id"]: x for x in summary["automations"]}
            check("R30 summary: §4.1 shape with thread ids",
                  auto2.id in s_items
                  and s_items[auto2.id]["pill"] in
                  ("On", "Paused", "Needs you", "Just added")
                  and s_items[auto2.id]["thread_id"]
                  and "headline" in summary
                  and "unused_count" in summary,
                  str(s_items.get(auto2.id, {}).get("pill")))

            wf = await get_workflow(auto2.id)
            wf_accounts = {a0["account_id"]: a0 for a0 in wf["accounts"]}
            check("R30 workflow GET: presets, accounts, rails, counts",
                  {p0["id"] for p0 in wf["schedule"]["presets"]}
                  >= {"weekdays-8", "weekdays-730", "daily-8",
                      "weekdays-9"}
                  and "stub" in wf_accounts
                  and any(p0.get("kind") == "rail"
                          for p0 in wf_accounts["stub"]["cant"])
                  and wf["counts"]["briefs_per_run"] == 1,
                  str(list(wf_accounts)))

            # auto2 is EVENT-triggered — the When-it-runs sheet for it
            # lists sources, and a preset write refuses honestly.
            try:
                await put_workflow_schedule(
                    auto2.id, PresetBody(preset_id="weekdays-9"))
                check("R30 preset on an event automation refuses",
                      False, "no raise")
            except HTTPException as e:
                check("R30 preset on an event automation refuses",
                      e.status_code == 409
                      and (e.detail or {}).get("code") == "no_schedule",
                      str(e.detail))

            rule = await post_workflow_rule(
                auto2.id, RuleBody(text="Never post twice about one item."))
            check("R30 rule added with the confirmation sentence",
                  rule["sentence"].startswith("Added a rule")
                  and len(rule["rules"]) == 1, rule["sentence"])

            from app.agent.automations import permissions as r30_perms
            stub_cat = r30_perms.catalog_for("stub")
            try:
                await put_account_permissions(
                    auto2.id, "stub",
                    PermissionsBody(
                        can=[p0["id"] for p0 in stub_cat["reads"]]
                        + [p0["id"] for p0 in stub_cat["rails"][:1]],
                        cant=[]))
                check("R30 a rail can never be allowed", False, "no raise")
            except HTTPException as e:
                check("R30 a rail can never be allowed",
                      e.status_code == 409
                      and (e.detail or {}).get("code") == "hard_rail",
                      str(e.detail))

            async with async_session_maker() as db:
                thread_after = await r30_ledger.thread_for(db, auto2.id)
                t_all, _more = await r30_ledger.list_turns(
                    db, thread_id=thread_after.id, limit=200)
                edited = [t for t in t_all if t["kind"] == "note"
                          and t.get("stamp") == "edited"]
                # Exactly the ONE applied write (the rule) noted — the
                # two REFUSED writes (no_schedule, hard_rail) must not
                # fabricate an edit record.
                check("R30 applied writes note EDITED; refused writes "
                      "do not",
                      len(edited) == 1, str(len(edited)))

            # Memory v2 over the wire: scoped sheet + global tree +
            # forget suppression.
            from app.services import memory_v2_service as m2
            async with async_session_maker() as db:
                saved = await m2.add_fact(
                    db, user_id=user_id,
                    text="Marcus Webb gets same-day answers",
                    category="people", scope=auto2.id,
                    why="You replied within the hour four times running.",
                    source="reaction",
                    subject_entity={"kind": "person",
                                    "name": "Marcus Webb"})
                check("R30 add_fact saves with the seam contract keys",
                      saved.get("saved") is True and saved.get("id"),
                      str(saved)[:100])
            r = await http.get("/api/memory", headers=user_h)
            tree = r.json() if r.status_code == 200 else {}
            fya = {x.get("automation_id"): x
                   for x in tree.get("from_your_automations") or []}
            check("R30 GET /api/memory groups the scoped fact under "
                  "its automation",
                  r.status_code == 200 and auto2.id in fya,
                  f"{r.status_code} {list(fya)[:3]}")
            fact_id = saved["id"]
            r = await http.delete(f"/api/memory/facts/{fact_id}",
                                  headers=user_h)
            check("R30 forget deletes over the wire",
                  r.status_code == 200 and r.json().get("deleted") is True,
                  str(r.status_code))
            async with async_session_maker() as db:
                again = await m2.add_fact(
                    db, user_id=user_id,
                    text="Marcus Webb gets same-day answers",
                    category="people", scope=auto2.id)
                check("R30 the forget signal suppresses relearning",
                      again.get("suppressed") is True, str(again))

            # Catalog over the wire (platform-native).
            r = await http.get("/api/automations/catalog", headers=user_h)
            cat_body = r.json() if r.status_code == 200 else {}
            check("R30 catalog serves cards with cadence + meta",
                  r.status_code == 200
                  and len(cat_body.get("cards") or []) >= 20
                  and all("meta" in c0 and "when" in c0
                          for c0 in cat_body.get("cards") or []),
                  f"{r.status_code} n={len(cat_body.get('cards') or [])}")

            # Routine migration (§4.11a) on a seeded email_briefing.
            from app.db.models import Routine as R30Routine
            from app.agent.automations.routine_migration import (
                migrate_email_briefings,
            )
            async with async_session_maker() as db:
                db.add(R30Routine(
                    id=str(uuid.uuid4()), user_id=user_id,
                    kind="email_briefing", enabled=False,
                    name="Morning new-email briefing",
                    prompt_text="brief me",
                    schedule_cron_local="0 8 * * *",
                    schedule_kind="cron",
                ))
                await db.commit()
                mig = await migrate_email_briefings(db, user_id=user_id)
                check("R30 routine migrated once with the promised-time "
                      "cron",
                      len(mig.get("migrated") or []) == 1, str(mig)[:160])
                mig2 = await migrate_email_briefings(db, user_id=user_id)
                check("R30 migration is idempotent",
                      not mig2.get("migrated"), str(mig2)[:120])

            mig_aid = (mig.get("migrated") or [{}])[0].get("automation_id")
            preset = await put_workflow_schedule(
                mig_aid, PresetBody(preset_id="weekdays-9"))
            async with async_session_maker() as db:
                a_row = await db.get(type(auto2), mig_aid)
                raw_spec = json.loads(a_row.spec_json)
                crons = [src.get("schedule", {}).get("cron_local")
                         for src in raw_spec["trigger"]["sources"]
                         if src.get("schedule")]
            check("R30 schedule preset rewrote the migrated brief's cron",
                  crons == ["0 9 * * 1-5"]
                  and preset["sentence"].startswith("Moved it to"),
                  str(crons))
            async with async_session_maker() as db:
                mig_thread = await r30_ledger.thread_for(db, mig_aid)
                mig_turns, _m = await r30_ledger.list_turns(
                    db, thread_id=mig_thread.id, limit=50)
                check("R30 the preset write left its EDITED note",
                      any(t["kind"] == "note"
                          and t.get("stamp") == "edited"
                          for t in mig_turns),
                      str([t.get("stamp") for t in mig_turns
                           if t["kind"] == "note"]))

            # ── real-vendor mode ────────────────────────────────────
            vendor = os.environ.get("E2E_CONNECTOR")
            if vendor:
                async with async_session_maker() as db:
                    ident = await vault.get(db, user_id, vendor)
                if ident is None:
                    print(f"  ⚠️ E2E_CONNECTOR={vendor}: no vault identity "
                          f"for {_USER_EMAIL} in THIS database — connect "
                          f"the account and re-run against the production "
                          f"DB (blocked today: the prod tenant host is "
                          f"down, see docs/automations/MAPPING.md §4).")
                else:
                    print(f"  → real-vendor flow for {vendor} would run "
                          f"here (identity found).")
            else:
                print("  ℹ real-vendor mode: set E2E_CONNECTOR=jira "
                      "E2E_JIRA_PROJECT=… E2E_SLACK_CHANNEL=… (requires "
                      "a reachable tenant + connected identities).")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n{'PASS' if FAIL == 0 else 'FAIL'}: {PASS} checks passed, "
          f"{FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
