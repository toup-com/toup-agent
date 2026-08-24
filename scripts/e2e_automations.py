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
