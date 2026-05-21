#!/usr/bin/env python3
"""
toup — Command-line interface for Toup agent platform.

Usage:
    toup send "Hello, what time is it?"
    toup status
    toup doctor
    toup sessions
"""
import argparse
import json
import os
import sys
import time
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("Error: 'requests' package required. Install with: pip install requests")
    sys.exit(1)


DEFAULT_BASE = os.environ.get("TOUP_URL", "http://localhost:8000")
DEFAULT_USER = os.environ.get("TOUP_USER", "admin")
DEFAULT_PASS = os.environ.get("TOUP_PASS", "")


def get_token(base_url: str, username: str, password: str) -> str:
    """Authenticate and return JWT token."""
    resp = requests.post(
        urljoin(base_url, "/api/v1/auth/token"),
        data={"username": username, "password": password},
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"❌ Auth failed: {resp.status_code} — {resp.text[:200]}")
        sys.exit(1)
    return resp.json()["access_token"]


def cmd_send(args):
    """Send a message to the agent and print the response."""
    token = get_token(args.url, args.user, args.password)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "message": args.message,
        "session_id": args.session,
    }

    t0 = time.time()
    resp = requests.post(
        urljoin(args.url, "/api/v1/chat/send"),
        json=payload,
        headers=headers,
        timeout=120,
    )
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"❌ Error {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)

    data = resp.json()
    print(data.get("text", data.get("response", json.dumps(data, indent=2))))
    if args.verbose:
        print(f"\n--- {elapsed:.1f}s | tokens: {data.get('tokens_used', '?')} | session: {data.get('session_id', '?')} ---")


def cmd_status(args):
    """Check system health and status."""
    try:
        resp = requests.get(urljoin(args.url, "/health"), timeout=10)
        data = resp.json()
        print(f"Status: {'✅ OK' if resp.status_code == 200 else '❌ ERROR'}")
        print(f"Version: {data.get('version', '?')}")
        print(f"Uptime: {data.get('uptime', '?')}")
        if "components" in data:
            for comp, info in data["components"].items():
                status_icon = "✅" if info.get("status") in ("ok", "healthy", True) else "⚠️"
                print(f"  {status_icon} {comp}: {info.get('status', info)}")
        if "active_sessions" in data:
            print(f"Active sessions: {data['active_sessions']}")
        if "tools_count" in data:
            print(f"Tools: {data['tools_count']}")
    except requests.ConnectionError:
        print(f"❌ Cannot connect to {args.url}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def cmd_doctor(args):
    """Run diagnostics — check all subsystems."""
    print("🏥 Toup Agent Doctor\n")
    checks = []

    # 1. API reachable
    try:
        resp = requests.get(urljoin(args.url, "/health"), timeout=10)
        checks.append(("API", resp.status_code == 200, f"HTTP {resp.status_code}"))
        health = resp.json() if resp.status_code == 200 else {}
    except Exception as e:
        checks.append(("API", False, str(e)))
        health = {}

    # 2. DB
    db_ok = health.get("components", {}).get("database", {}).get("status") in ("ok", "healthy", True)
    checks.append(("Database", db_ok, "connected" if db_ok else "unreachable"))

    # 3. Auth
    try:
        if args.password:
            token = get_token(args.url, args.user, args.password)
            checks.append(("Auth", True, "token obtained"))
        else:
            checks.append(("Auth", None, "skipped (no password)"))
    except SystemExit:
        checks.append(("Auth", False, "login failed"))

    # 4. Telegram bot
    tg = health.get("components", {}).get("telegram", {})
    tg_ok = tg.get("status") in ("ok", "healthy", True, "running")
    checks.append(("Telegram", tg_ok, tg.get("status", "unknown")))

    # 5. Memory / pgvector
    mem = health.get("components", {}).get("memory", health.get("components", {}).get("pgvector", {}))
    mem_ok = mem.get("status") in ("ok", "healthy", True) if mem else None
    checks.append(("Memory/pgvector", mem_ok, mem.get("status", "unknown") if mem else "not reported"))

    # Print results
    for name, ok, detail in checks:
        if ok is True:
            icon = "✅"
        elif ok is False:
            icon = "❌"
        else:
            icon = "⏭️"
        print(f"  {icon} {name}: {detail}")

    fails = sum(1 for _, ok, _ in checks if ok is False)
    print(f"\n{'❌' if fails else '✅'} {len(checks) - fails}/{len(checks)} checks passed")


def cmd_sessions(args):
    """List active sessions."""
    token = get_token(args.url, args.user, args.password)
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.get(
        urljoin(args.url, "/api/v1/sessions"),
        headers=headers,
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"❌ Error: {resp.text[:200]}")
        sys.exit(1)

    sessions = resp.json()
    if isinstance(sessions, dict):
        sessions = sessions.get("sessions", sessions.get("items", []))

    if not sessions:
        print("No active sessions.")
        return

    for s in sessions[:20]:
        sid = s.get("id", s.get("session_id", "?"))[:8]
        channel = s.get("channel", "?")
        msgs = s.get("message_count", s.get("messages", "?"))
        print(f"  {sid}  {channel:10s}  {msgs} msgs")


def main():
    parser = argparse.ArgumentParser(
        prog="toup",
        description="Toup CLI — interact with the agent platform",
    )
    parser.add_argument("--url", default=DEFAULT_BASE, help="API base URL")
    parser.add_argument("--user", default=DEFAULT_USER, help="Username")
    parser.add_argument("--password", default=DEFAULT_PASS, help="Password")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    sub = parser.add_subparsers(dest="command", help="Command")

    # send
    p_send = sub.add_parser("send", help="Send a message to the agent")
    p_send.add_argument("message", help="The message to send")
    p_send.add_argument("--session", "-s", default=None, help="Session ID")
    p_send.set_defaults(func=cmd_send)

    # status
    p_status = sub.add_parser("status", help="Check system health")
    p_status.set_defaults(func=cmd_status)

    # doctor
    p_doctor = sub.add_parser("doctor", help="Run diagnostics")
    p_doctor.set_defaults(func=cmd_doctor)

    # sessions
    p_sess = sub.add_parser("sessions", help="List active sessions")
    p_sess.set_defaults(func=cmd_sessions)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
