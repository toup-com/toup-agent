"""The two credit-refusal surfaces the Unlimited round split in half.

5.7 of the connector dispatcher and the live-voice pre-flight in
``ws_realtime`` each had ONE refusal shape: "you're out of credits, upgrade
your plan", permanent, with a link to toup.ai/account. The Unlimited round
gave each of them a second branch, and both new branches are about accounts
that cannot run out of anything:

  * **The rate ladder.** An unlimited account is never denied on balance —
    ``check_balance`` is entitlement-blind and its wallet holds 1,000,000 —
    but it CAN be paced. That refusal is not about money, it clears in
    seconds, and telling the user to upgrade a plan they already hold is
    wrong twice: wrong cause, and wrong permanence.

  * **Anti-steering.** The billing link used to ship unconditionally, to
    every client including the iOS app, whose only payers subscribe through
    Apple. App Review 3.1.1: linking an Apple subscriber to an off-app
    purchase page from inside the app is the exposure — and it is useless to
    them anyway, because an Apple subscription is not managed there.

``test_free_tier_unchanged`` proves a free user never reaches either new
branch. This file proves the branches themselves are right, which is the
other half and the half with an App Store consequence attached.

Why this file rather than ``test_connector_dispatcher.py``, where the harness
lives: that file is in ``COVERAGE_DEBT.txt`` ("needs a real database"), so CI
never runs it — two of its tests are red on main right now and have been
invisible. A test in an excused file guards nothing. The fixtures are
imported from it instead, so there is still exactly one harness.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.config import settings
from app.connectors.base import ConnectorOk, ConnectorToolError
from app.db.database import async_session_maker
from app.services import connector_dispatcher as dispatcher

# The dispatcher harness, imported rather than duplicated. `_provision_crypto`
# is autouse in its home module and stays autouse here.
from tests.test_connector_dispatcher import (  # noqa: F401
    _provision_crypto,
    _seed_active_identity,
    alice_user_id,
    make_manifest,
    register_scripted,
)

# No module-level asyncio mark: pytest.ini runs `asyncio_mode = auto`, and
# half the tests here are synchronous AST reads.
BACKEND_DIR = Path(__file__).resolve().parent.parent


def _refusal(reason: str):
    """A ChargeResult refusal carrying `reason`, as check_balance returns."""
    from app.services.credit_service import ChargeResult

    async def _fake(db, user_id, bucket, required):
        return ChargeResult(success=False, reason=reason, balance_after=0.0)
    return _fake


# ── the connector dispatcher's two refusals ──────────────────────────


async def test_a_rate_limited_preflight_is_retryable_and_mentions_no_plan(
    register_scripted, alice_user_id, monkeypatch,
):
    from app.services.credit_exhausted import REASON_RATE_LIMITED
    from app.services.credit_service import credit_service

    _manifest, provider = register_scripted()
    await _seed_active_identity(alice_user_id)
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True)
    monkeypatch.setattr(credit_service, "check_balance",
                        _refusal(REASON_RATE_LIMITED))

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorToolError)
    assert result.retryable is True, (
        "the rate ladder clears in seconds — marking it permanent strands an "
        "unlimited account on a refusal that has already expired"
    )
    low = result.message.lower()
    for word in ("upgrade", "plan", "renewal", "credit"):
        assert word not in low, (
            f"an unlimited account was told about {word!r}: {result.message}"
        )
    assert provider.execute_call_count == 0


async def test_an_insufficient_preflight_keeps_its_permanent_upgrade_copy(
    register_scripted, alice_user_id, monkeypatch,
):
    """The anti-vacuity control. If the split were wired to answer "rate
    limited" for every refusal, the test above would still pass while a free
    user out of integration credits was told to wait a few seconds for
    credits that only come back next month."""
    from app.services.credit_service import credit_service

    _manifest, provider = register_scripted()
    await _seed_active_identity(alice_user_id)
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True)
    monkeypatch.setattr(credit_service, "check_balance",
                        _refusal("insufficient_integration_credits"))

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorToolError)
    assert result.retryable is False
    low = result.message.lower()
    assert "integration credits" in low and "upgrade your plan" in low
    assert provider.execute_call_count == 0


async def test_a_passing_preflight_still_reaches_the_provider(
    register_scripted, alice_user_id, monkeypatch,
):
    """The second control: both tests above would also pass if 5.7 refused
    everything. Enforcement on, balance fine, provider called."""
    _manifest, provider = register_scripted()
    provider._execute_result = ConnectorOk(content='{"ok":true}')
    await _seed_active_identity(alice_user_id)
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorOk)
    assert provider.execute_call_count == 1


# ── ws_realtime anti-steering ────────────────────────────────────────
#
# The pre-flight lives ~3,500 lines inside `realtime_voice_ws`, behind an
# accepted WebSocket, a live OpenAI relay and a session handshake — there is
# no seam to call. So this is asserted on the AST rather than by execution,
# and on the AST rather than by grepping for the URL: the regression that
# matters is not "the string disappeared", it is "the string escaped its
# guard", and only the tree can tell those apart.


def _ws_realtime_tree() -> ast.Module:
    return ast.parse(
        (BACKEND_DIR / "app/api/ws_realtime.py").read_text(encoding="utf-8")
    )


def _billing_url_assignments(tree: ast.Module) -> list[ast.AST]:
    """Every node that puts the account billing URL into a client frame."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and "toup.ai/account" in node.value:
            found.append(node)
    return found


def test_the_billing_link_exists_at_all():
    """Anti-vacuity: the guard test below passes trivially if someone deletes
    the link. Web subscribers manage their plan there and must keep it."""
    assert _billing_url_assignments(_ws_realtime_tree()), (
        "the account billing URL is gone from ws_realtime entirely — web "
        "subscribers have no way to a purchase surface from a voice refusal"
    )


def _has_not_iap_compare(test: ast.AST) -> bool:
    """`<…plan_source…> != "iap"` anywhere inside a guard expression.

    Walked rather than pattern-matched on the top node, because the guard is
    legitimately a BoolOp: it also has to consider WHICH CLIENT is asking, and
    a matcher that only accepts a bare Compare would fail the moment a second,
    stricter condition is added — turning "the rule got stronger" into a red
    test.
    """
    for node in ast.walk(test):
        if not (isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.NotEq)
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "iap"):
            continue
        # The left side is a local in one call site and an attribute in
        # another; what the rule is about is the name, not how it was spelled.
        if "plan_source" in ast.dump(node.left):
            return True
    return False


def test_every_billing_link_is_inside_a_not_iap_guard():
    """App Review 3.1.1. Keyed on `plan_source`, which is where the account
    actually pays, not on a guess at the client."""
    tree = _ws_realtime_tree()
    targets = set(map(id, _billing_url_assignments(tree)))

    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _has_not_iap_compare(node.test):
            continue
        for child in ast.walk(node):
            if id(child) in targets:
                guarded.add(id(child))

    ungated = targets - guarded
    assert not ungated, (
        "a billing URL in ws_realtime.py is NOT inside a "
        "`plan_source != \"iap\"` guard — an Apple subscriber would be "
        "linked to an off-app purchase page from inside the iOS app "
        f"(line(s) {sorted(n.lineno for n in _billing_url_assignments(tree) if id(n) in ungated)})"
    )


def _rate_limited_voice_frame():
    """The pacing frame, together with the `If` that can reach it.

    Both halves matter and only one is obvious. Asserting on the dict alone
    passes for a branch that is dead — `if False:` leaves the literal in the
    file untouched — so the guard is located by its condition
    (`… == REASON_RATE_LIMITED`) and the frame must be found INSIDE it.
    """
    src = (BACKEND_DIR / "app/api/ws_realtime.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if "REASON_RATE_LIMITED" not in ast.dump(node.test):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Dict):
                continue
            keys = {k.value for k in child.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if "type" in keys and "message" in keys:
                return src, child, keys
    return src, None, set()


def test_the_rate_limited_voice_branch_is_reachable():
    """Anti-vacuity for the test below, and a real regression on its own: a
    deleted or dead branch puts an unlimited account back on "you're out of
    Toup credits" for a refusal that is not about credits."""
    _src, frame, _keys = _rate_limited_voice_frame()
    assert frame is not None, (
        "no error frame is reachable from a `REASON_RATE_LIMITED` test in "
        "ws_realtime.py — the pacing branch is gone or unreachable, and an "
        "unlimited account is being told it is out of credits again"
    )


def test_the_rate_limited_voice_frame_carries_no_billing_flag():
    """`billing: True` is what makes the client render a billing surface at
    all (useRealtimeVoice.ts). The pacing refusal must not set it — the
    account it fires for is unlimited, and there is nothing to buy."""
    src, frame, keys = _rate_limited_voice_frame()
    assert frame is not None
    assert "billing" not in keys, (
        "the rate-limited voice frame sets `billing`, which is the flag that "
        "renders a billing surface to an account with nothing to buy"
    )
    text = ast.get_source_segment(src, frame) or ""
    assert "toup.ai/account" not in text, (
        "the pacing frame links to a purchase page"
    )


# ── the chat frame a rate-limited refusal produces ───────────────────


def _rate_limited_response():
    from app.services.credit_exhausted import build_exhausted_response
    from app.services.credit_exhausted import REASON_RATE_LIMITED

    return build_exhausted_response(
        reason=REASON_RATE_LIMITED, bucket="message", balance_after=1_000_000.0,
        plan_id="unlimited", plan_display_name="Unlimited",
        period_end=datetime.now(timezone.utc) + timedelta(days=12),
    )


def test_a_paced_refusal_is_never_a_credit_exhausted_frame():
    """App Store build 109 does not read `cta_hidden` and does not read
    `cta_label`/`cta_url` off the frame at all — it renders its own copy from
    `reason`, and an unrecognised reason falls into `blockedCopy`'s DEFAULT
    branch: headline "This needs a paid plan", CTA "See plans", plus a
    composer replaced by "You're out of credits". To an Unlimited subscriber
    who is merely being paced, and clearable only by a purchase that account
    cannot make.

    So the pacing refusal leaves the billing channel entirely, the way the
    voice path already does.
    """
    from app.services.credit_exhausted import (
        REASON_RATE_LIMITED, response_to_stream_event,
    )

    frame = response_to_stream_event(_rate_limited_response())
    assert frame["type"] == "error", frame
    assert "credit" not in frame["type"]
    assert frame.get("code") == REASON_RATE_LIMITED
    # No key the client could read as a billing surface.
    for key in ("reason", "plan_id", "cta_label", "cta_url", "balance_after",
                "monthly_reset_at", "error"):
        assert key not in frame, f"{key} would re-enter the billing renderer"
    blob = frame["message"].lower()
    for word in ("pricing", "upgrade", "plan", "credit", "$", "limit reached"):
        assert word not in blob, f"{word!r} in a pacing message"


def test_a_real_exhaustion_still_gets_the_card():
    """Anti-vacuity: only the rate ladder leaves the channel."""
    from datetime import datetime as _dt
    from app.services.credit_exhausted import (
        REASON_INSUFFICIENT_MESSAGE, build_exhausted_response,
        response_to_stream_event,
    )

    resp = build_exhausted_response(
        reason=REASON_INSUFFICIENT_MESSAGE, bucket="message", balance_after=0.0,
        plan_id="free", plan_display_name="Free",
        period_end=_dt.now(timezone.utc) + timedelta(days=12),
    )
    frame = response_to_stream_event(resp)
    assert frame["type"] == "credit_exhausted"
    assert frame["reason"] == REASON_INSUFFICIENT_MESSAGE
    assert frame["cta_url"]


def test_the_402_reconstruction_path_cannot_re_enter_the_billing_channel():
    """ws_chat rebuilds a credit frame from the proxy's 402 body when the
    exception is only a string. That second producer must obey the same rule,
    or the fix above holds on one path and leaks on the other."""
    import ast

    src = (BACKEND_DIR / "app/api/ws_chat.py").read_text(encoding="utf-8")
    assert "REASON_RATE_LIMITED" in src, "ws_chat does not know the reason"

    tree = ast.parse(src)
    # Find the assignment `_credit_frame = {"type": "credit_exhausted", **_detail}`
    # and prove it is guarded by a REASON_RATE_LIMITED test, not merely
    # adjacent to one.
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.unparse(node.test)
        if "REASON_RATE_LIMITED" not in test_src:
            continue
        body = "\n".join(ast.unparse(n) for n in node.body)
        orelse = "\n".join(ast.unparse(n) for n in node.orelse)
        if "credit_exhausted" in orelse and "credit_exhausted" not in body:
            found.append(test_src)
    assert found, (
        "the 402-reconstruction branch builds a credit_exhausted frame that is "
        "not inside a REASON_RATE_LIMITED guard"
    )


def test_the_billing_link_is_also_withheld_from_a_client_that_says_it_is_mobile():
    """`plan_source` answers WHO PAYS. It does not answer WHICH APP IS ASKING,
    and 3.1.1 is about the second — it binds hardest for a NON-subscriber, who
    is exactly who a purchase link is aimed at, and whose plan_source reads
    'free', not 'iap'.

    No shipped build sends `client` yet, so this closes the gap only for a
    build that does; the alternative key (`plan_source == 'free'`) would take
    the link away from free WEB users, the one cohort it helps. What must not
    regress is that the parameter is accepted and consulted, or the app has
    nothing to append.
    """
    tree = _ws_realtime_tree()
    src = (BACKEND_DIR / "app/api/ws_realtime.py").read_text(encoding="utf-8")
    assert "client: Optional[str] = Query(None)" in src, (
        "the voice socket no longer accepts a client hint"
    )

    targets = set(map(id, _billing_url_assignments(tree)))
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if not any("client" in n for n in names):
            continue
        for child in ast.walk(node):
            if id(child) in targets:
                guarded.add(id(child))
    assert targets and targets <= guarded, (
        "a billing URL is emitted without consulting the client hint"
    )
