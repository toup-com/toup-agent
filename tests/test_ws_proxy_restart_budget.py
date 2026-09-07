"""Rolling platform restarts versus the installed client's warm-up budget.

What this pins
--------------
uvicorn 0.27.0 (`requirements.platform.txt:7`) ends a graceful shutdown by
walking every live connection and calling `WebSocketProtocol.shutdown()`, which
is `fail_connection(1012)` followed immediately by `transport.close()`
(`uvicorn/protocols/websockets/websockets_impl.py`). build 109 does NOT treat
1012 as a warm-up close — `AGENT_WARMING_CLOSE_CODES` is {4404, 4502, 4503,
4504} (`api.ts:2497`) — so it lands on the bounded abnormal path
(`MAX_ABNORMAL_PRE_REPLY_RETRIES = 2`, `api.ts:2532`), and that counter is
declared once per `sendChatMessage` (`api.ts:3621`), not per attempt. Three
restarts inside ONE send therefore end it with the terminal
"Connection lost — try sending again".

That is not hypothetical: the 2026-09-06 fixes went out as four separate
pushes — Railway deployments at 22:55:58, 22:56:01, 22:58:57, 22:59:25, four
rolling restarts in 207 s — while a send's warm-up loop is now up to ~278 s
long. Nobody was mid-warm-up that night (0 `not_ready` outcomes in the log), so
this is a next-rollout hazard rather than an observed failure.

Why there is no proxy-side fix in this file
-------------------------------------------
The obvious one does not work. `Server.shutdown()` (uvicorn/server.py) closes
the listening sockets and every connection in the same call, and only THEN
awaits `lifespan.shutdown()` — so an ASGI shutdown handler (or a router
`on_shutdown`) fires strictly after the 1012 has been written and the transport
closed, and there is no window in which "new sockets arrive while draining"
either: the listener is already closed, and Railway's load balancer has stopped
routing to the replica before SIGTERM. The only hook early enough is a chained
`signal.signal(SIGTERM, …)` inside the app, and uvicorn installs its own via
`loop.add_signal_handler`; replacing that handler to buy ~0.1 s of warning is
not a change to make to a live two-replica service on the strength of an
unobserved P2. The mitigation is the release process (land a wave as ONE push).

So both halves are pinned here instead: the hazard, and — as the falsifier for
anyone who does build a drain notification later — the fact that ONE
`agent_starting` frame written before the 1012 is enough to make the client
class the restart as warming and survive it.

Run:
    cd backend && PYTHONPATH=. python -m pytest -q tests/test_ws_proxy_restart_budget.py
"""

from __future__ import annotations

import asyncio
import json

import test_ws_proxy_warmup_harness as H


def _restarting_app(real_app, *, restarts: int, warn_before_close: bool,
                    held_virtual_ms: int = 9_000, retry_ms: int = 10_000):
    """Wrap the real ASGI app so the first `restarts` sockets die the way a
    rolling restart kills them: accepted, held, then closed with 1012 — with or
    without a warning frame first. Every later socket reaches the real handler."""
    seen = {"n": 0}

    async def app(scope, receive, send):
        if scope.get("type") != "websocket":
            return await real_app(scope, receive, send)
        seen["n"] += 1
        if seen["n"] > restarts:
            return await real_app(scope, receive, send)
        await receive()                                   # websocket.connect
        await send({"type": "websocket.accept"})
        await asyncio.sleep(H.real_s(held_virtual_ms))    # the hold
        if warn_before_close:
            await send({"type": "websocket.send", "text": json.dumps({
                "type": "agent_starting",
                "message": "Your agent is waking up. Reconnect in a moment.",
                "retry_after_ms": retry_ms,
            })})
        await send({"type": "websocket.close", "code": 1012})

    app.seen = seen  # type: ignore[attr-defined]
    return app


def test_three_restarts_in_one_send_end_it_with_the_terminal_error():
    """THE HAZARD. Two 1012s are absorbed; the third is terminal, and the agent
    behind the fourth socket is perfectly healthy."""
    async def go(mp):
        rig = H.build_rig(mp, ready_at_virtual_ms=0)
        H.configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None)
        app = _restarting_app(rig.app, restarts=3, warn_before_close=False)
        res = await asyncio.wait_for(H.Build109Client(app).send("hello"),
                                     timeout=H.real_s(120_000))
        print(f"\nno warning frame: {res.outcome} after {res.attempts} attempt(s) "
              f"— {res.error}")
        assert res.outcome == "error", res
        assert res.attempts == 3, res.attempts
        assert "Connection lost" in (res.error or ""), res.error
        assert rig.agent.received == [], "the message reached the agent anyway"
    H.run(H._run_with_mp(go))


def test_one_agent_starting_frame_before_the_1012_makes_the_same_restarts_survivable():
    """THE REMEDY, pinned as a falsifier for a future drain notification.

    The frame is the whole difference: it settles the attempt as `warming`
    before the close is read, so the abnormal budget is never touched and the
    send survives to reach the healthy agent on the next socket."""
    async def go(mp):
        rig = H.build_rig(mp, ready_at_virtual_ms=0)
        H.configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None)
        app = _restarting_app(rig.app, restarts=3, warn_before_close=True)
        res = await asyncio.wait_for(H.Build109Client(app).send("hello"),
                                     timeout=H.real_s(200_000))
        print(f"\nwith a warning frame: {res.outcome} after {res.attempts} attempt(s)")
        assert res.outcome == "delivered", res
        assert res.attempts == 4, res.attempts
        assert len(rig.agent.received) == 1, rig.agent.received
        assert json.loads(rig.agent.received[0])["text"] == "hello"
    H.run(H._run_with_mp(go))


def test_two_restarts_are_absorbed_silently():
    """The bound is 2, not 1 or 3 — worth pinning, because a client release that
    changed MAX_ABNORMAL_PRE_REPLY_RETRIES would move the whole hazard and
    nothing else in this repo would notice."""
    async def go(mp):
        rig = H.build_rig(mp, ready_at_virtual_ms=0)
        H.configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None)
        app = _restarting_app(rig.app, restarts=2, warn_before_close=False)
        res = await asyncio.wait_for(H.Build109Client(app).send("hello"),
                                     timeout=H.real_s(120_000))
        assert res.outcome == "delivered", res
        assert res.attempts == 3, res.attempts
        assert len(rig.agent.received) == 1, rig.agent.received
    H.run(H._run_with_mp(go))
