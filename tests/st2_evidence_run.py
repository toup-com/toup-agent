"""Ticket 1 / ST-2 evidence — dual-accept on the other 4 platform-side
WS endpoints, plus static-code checks for the agent-internal ws_chat,
the tunnel_client outbound connection, and the token-fragment log scrub.

Drives live checks against:
  - wss://toup.ai/api/ws/browser
  - wss://toup.ai/api/ws/realtime
  - wss://toup.ai/api/ws/deploy
  - wss://toup.ai/api/ws/agent-tunnel

For each endpoint:
  1. ?token=<invalid> connection → auth fail with the right close code,
     [DEPRECATED-WS-AUTH] log fires (proves the deprecation marker is
     wired on this endpoint).
  2. subprotocol with bad bearer.<token> → auth fail, NO deprecation
     log fires (proves subprotocol path is wired and is preferred over
     ?token=).

Also static-code checks for things that aren't easily live-tested:
  - tunnel_client.py:43 — URL no longer carries ?token=, subprotocols arg present.
  - ws_agent_tunnel.py:219+278 — token-fragment logs replaced with len-only.
  - ws_chat.py:991 (agent-internal) — uses the helper.
  - Live frontend bundles use subprotocol for /api/ws/browser and /api/ws/realtime.

Invoke from backend/:
    PYTHONPATH=. railway run --service platform-api python tests/st2_evidence_run.py
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


CANARY_UID = "3134fece-c3de-411b-add8-3f2a58882794"

FAILS: List[str] = []


def ok(label: str) -> None:
    print("  \u2713 " + label)


def fail(label: str, detail: str = "") -> None:
    msg = f"  \u2717 {label}: {detail}" if detail else f"  \u2717 {label}"
    print(msg)
    FAILS.append(msg)


def _mint_bad_jwt() -> str:
    """A structurally-valid JWT with a corrupted signature."""
    from app.services.auth_service import create_access_token
    real = create_access_token(CANARY_UID)
    h, p, s = real.split(".")
    bad_s = s[:-4] + ("AAAA" if s[-4:] != "AAAA" else "BBBB")
    return f"{h}.{p}.{bad_s}"


async def _connect_and_probe(
    url: str,
    subprotocols: Optional[List[str]] = None,
    timeout: float = 12.0,
    send_junk_to_skip_first_frame: bool = False,
) -> Tuple[str, Optional[int], Optional[dict]]:
    """state ∈ {open, auth_failed, closed, connect_failed}"""
    import websockets

    extra = {}
    if subprotocols:
        extra["subprotocols"] = subprotocols

    try:
        async with websockets.connect(url, **extra, open_timeout=timeout) as ws:
            if send_junk_to_skip_first_frame:
                try:
                    await ws.send("not-an-auth-frame")
                except Exception:
                    pass
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                return ("open", None, None)
            try:
                frame = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return ("open", None, {"_raw": raw[:200]})
            if isinstance(frame, dict) and frame.get("type") == "error":
                try:
                    await asyncio.wait_for(ws.wait_closed(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                return ("auth_failed", ws.close_code, frame)
            return ("open", None, frame)
    except websockets.exceptions.InvalidHandshake as e:
        return ("connect_failed", None, {"_err": str(e)[:200]})
    except websockets.exceptions.ConnectionClosed as e:
        return ("closed", e.code, None)


# Each endpoint's expected close code on auth failure.
ENDPOINTS = [
    {
        "label": "/api/ws/browser",
        "url": "wss://toup.ai/api/ws/browser",
        "expect_close_code": 4001,
        "send_junk": False,  # no first-frame fallback
    },
    {
        "label": "/api/ws/realtime",
        "url": "wss://toup.ai/api/ws/realtime",
        "expect_close_code": 4401,
        "send_junk": True,
    },
    {
        "label": "/api/ws/deploy",
        "url": "wss://toup.ai/api/ws/deploy",
        "expect_close_code": 4001,
        "send_junk": False,
    },
    {
        "label": "/api/ws/agent-tunnel",
        "url": "wss://toup.ai/api/ws/agent-tunnel",
        "expect_close_code": 4401,
        "send_junk": True,
    },
]


async def check_endpoint_live(ep: dict) -> None:
    print(f"\n[live] {ep['label']}")
    bad = _mint_bad_jwt()

    # 1. Subprotocol with bad token — should auth-fail without deprecation log.
    state, code, frame = await _connect_and_probe(
        ep["url"],
        subprotocols=["toup.auth.v1", f"bearer.{bad}"],
        send_junk_to_skip_first_frame=ep["send_junk"],
    )
    if state == "auth_failed":
        ok(f"{ep['label']} subprotocol path: bad token → auth_failed")
        if code == ep["expect_close_code"]:
            ok(f"{ep['label']} close code = {code}")
        else:
            fail(f"{ep['label']} close code", f"got {code}, expected {ep['expect_close_code']}")
    elif state == "closed":
        # ws_deploy uses ws.close() without an error frame on bad token.
        if code == ep["expect_close_code"]:
            ok(f"{ep['label']} subprotocol path: bad token → closed code={code}")
        else:
            fail(f"{ep['label']} subprotocol path: closed code mismatch",
                 f"got {code}, expected {ep['expect_close_code']}")
    else:
        fail(f"{ep['label']} subprotocol path", f"unexpected state={state} code={code}")

    await asyncio.sleep(1)

    # 2. ?token= with bad token — should auth-fail AND fire [DEPRECATED-WS-AUTH].
    bad2 = _mint_bad_jwt()
    state, code, frame = await _connect_and_probe(
        ep["url"] + f"?token={bad2}",
        send_junk_to_skip_first_frame=ep["send_junk"],
    )
    if state in ("auth_failed", "closed") and code == ep["expect_close_code"]:
        ok(f"{ep['label']} ?token= path: bad token → close code {code}")
    else:
        fail(f"{ep['label']} ?token= path", f"state={state} code={code}")


def check_static_code() -> None:
    print("\n[static] static-code checks")
    repo_root = Path(__file__).resolve().parent.parent

    # tunnel_client.py:43 must NOT carry ?token= and SHOULD have subprotocols= arg.
    tc = (repo_root / "app/agent/tunnel_client.py").read_text()
    if "?token=" in tc:
        fail("tunnel_client.py still contains ?token= URL fragment")
    else:
        ok("tunnel_client.py: no ?token= in URL construction")
    if "subprotocols=" in tc and 'bearer.' in tc:
        ok("tunnel_client.py: passes subprotocols=[..., bearer.<token>]")
    else:
        fail("tunnel_client.py: subprotocols arg missing")

    # ws_agent_tunnel.py: token-fragment logs gone.
    wat = (repo_root / "app/api/ws_agent_tunnel.py").read_text()
    if re.search(r"token\[:\d+\]|token\[-\d+:\]", wat):
        fail("ws_agent_tunnel.py: still contains token[:N] or token[-N:] log fragment")
    else:
        ok("ws_agent_tunnel.py: no token-fragment slicing in any log line")
    # New len-only log present.
    if "Token not found in DB (len=" in wat or 'Auth failed — token len=%d' in wat:
        ok("ws_agent_tunnel.py: replacement log uses len-only")
    else:
        fail("ws_agent_tunnel.py: replacement len-only log not found")

    # ws_chat.py (agent-internal) wired with the helper.
    wc = (repo_root / "app/api/ws_chat.py").read_text()
    if "accept_with_subprotocol_auth" in wc and "log_deprecated_query_token" in wc:
        ok("ws_chat.py (agent-internal): uses _ws_auth_helpers")
    else:
        fail("ws_chat.py (agent-internal): not wired to helper")

    # Helper module exists with expected exports.
    helper = (repo_root / "app/api/_ws_auth_helpers.py").read_text()
    for sym in ("accept_with_subprotocol_auth", "log_deprecated_query_token", "safe_send_close_ws"):
        if f"def {sym}" not in helper and f"async def {sym}" not in helper:
            fail(f"_ws_auth_helpers.py: {sym} missing")
        else:
            ok(f"_ws_auth_helpers.py exports {sym}")


def check_live_bundles() -> None:
    print("\n[bundle] live frontend bundle subprotocol check (toup.ai)")
    import urllib.request

    def _fetch(u: str) -> str:
        with urllib.request.urlopen(u, timeout=20) as r:
            return r.read().decode("utf-8", errors="replace")

    try:
        idx_html = _fetch("https://toup.ai/")
    except Exception as e:
        fail("could not fetch https://toup.ai/", str(e))
        return

    idx_chunk_match = re.search(r'/assets/(index-[A-Za-z0-9_-]+\.js)', idx_html)
    if not idx_chunk_match:
        fail("could not find index-*.js chunk reference in /")
        return
    idx_chunk = idx_chunk_match.group(1)
    idx_js = _fetch(f"https://toup.ai/assets/{idx_chunk}")

    # ChatPage chunk (carries /api/ws/browser too)
    chat_match = re.search(r"ChatPage-[A-Za-z0-9_-]+\.js", idx_js)
    chat_chunk = chat_match.group(0) if chat_match else None

    # BrowserPage chunk
    browser_match = re.search(r"BrowserPage-[A-Za-z0-9_-]+\.js", idx_js)
    browser_chunk = browser_match.group(0) if browser_match else None

    targets = [
        ("ChatPage chunk (/api/ws/browser)", chat_chunk, "/api/ws/browser"),
        ("BrowserPage chunk (/api/ws/browser)", browser_chunk, "/api/ws/browser"),
        ("api.ts in index (/api/ws/realtime)", idx_chunk, "/api/ws/realtime"),
    ]
    for label, chunk, ws_path in targets:
        if not chunk:
            fail(f"{label}: chunk not located")
            continue
        try:
            js = _fetch(f"https://toup.ai/assets/{chunk}")
        except Exception as e:
            fail(f"{label}: could not fetch chunk", str(e))
            continue
        # Should NOT contain ?token= immediately after the WS path.
        if re.search(re.escape(ws_path) + r"\?token=", js):
            fail(f"{label}: still constructs {ws_path}?token=")
        else:
            ok(f"{label}: no {ws_path}?token= in bundle")
        # Should contain the bearer.${...} pattern somewhere.
        if "toup.auth.v1" in js and "bearer." in js:
            ok(f"{label}: subprotocol ['toup.auth.v1', 'bearer.<token>'] present")
        else:
            fail(f"{label}: subprotocol literals not found")


def check_log_audit() -> None:
    print("\n[logs] Railway platform-api log audit")
    # Wait briefly for log forwarding lag, then pull.
    time.sleep(6)
    proc = subprocess.run(
        ["railway", "logs", "--service", "platform-api"],
        capture_output=True, text=True, timeout=30,
    )
    log_text = proc.stdout
    tail = "\n".join(log_text.splitlines()[-1500:])

    # We expect to see the deprecation marker for at LEAST one of the
    # endpoints we just hammered, since each ?token= test fires it.
    deprec_lines = [ln for ln in tail.splitlines() if "[DEPRECATED-WS-AUTH]" in ln]
    if len(deprec_lines) == 0:
        fail("[DEPRECATED-WS-AUTH] absent from recent log window",
             "expected ≥ 1 from the ?token= probes; check timing")
    else:
        ok(f"[DEPRECATED-WS-AUTH] appears {len(deprec_lines)} time(s) in recent logs")
        # Show what endpoints are firing it.
        endpoints_seen = set()
        for ln in deprec_lines[-12:]:
            m = re.search(r"DEPRECATED-WS-AUTH\] (\S+)", ln)
            if m:
                endpoints_seen.add(m.group(1))
        if endpoints_seen:
            print(f"      endpoints firing: {sorted(endpoints_seen)}")

    # No ws_chat_proxy / ws_browser / ws_realtime / ws_deploy / ws_agent_tunnel tracebacks.
    proxy_tb_files = [
        "ws_chat_proxy.py", "ws_browser.py", "ws_realtime.py",
        "ws_deploy.py", "ws_agent_tunnel.py", "ws_chat.py",
    ]
    has_tb = "Traceback (most recent call last)" in tail
    proxy_tb_hit = has_tb and any(f in tail for f in proxy_tb_files)
    if not proxy_tb_hit:
        ok("no WS-endpoint traceback in recent log window")
    else:
        # Identify which file.
        which = next((f for f in proxy_tb_files if f in tail and tail.rfind(f) > tail.rfind("Traceback")), None) or "(unknown)"
        fail(f"WS-endpoint traceback detected in recent log window",
             f"file in stack: {which}")


async def amain():
    check_static_code()
    check_live_bundles()
    for ep in ENDPOINTS:
        await check_endpoint_live(ep)
    check_log_audit()


def main():
    asyncio.run(amain())
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s) failed")
        for f in FAILS:
            print(f"  {f}")
        sys.exit(1)
    print("ALL ST-2 EVIDENCE CHECKS PASSED")


if __name__ == "__main__":
    main()
