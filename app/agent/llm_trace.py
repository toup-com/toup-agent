"""`x-toup-trace` — the AGENT half of an agent<->platform correlation contract.

WHAT PROBLEM THIS ADDRESSES (and what it does not)

The agent logs `[PERF] llm_total` per LLM iteration; the platform proxy logs
its own per-request line keyed by a platform-local `req_id`. Nothing carries
an identifier ACROSS that boundary, so a slow turn on one side and a slow
upstream call on the other can only be matched by adjacency in time — which
is not a join, and is wrong outright when two requests overlap. This module
mints the one value both sides can print.

THE CONTRACT

    header:  x-toup-trace
    value:   <cmid_h>.<iteration>
    grammar: ^[0-9a-f]{8}\\.[0-9]{1,3}$     (validated by the platform half)

The producer anchors that grammar with `\\A…\\Z` rather than `^…$` — see
`TRACE_VALUE_RE` for why those are not the same rule in Python.

`cmid_h` is the EXISTING 8-hex FNV-1a/32 hash of the turn's `client_msg_id`
— the same function and therefore the same value [TURNTRACE] already logs, so
the agent line, the platform line and the app's own `turnTrace.ts` all agree.
It is resolved at call time (see `_hash`), never re-implemented here: a second
implementation of a join key is a join key that silently stops joining.

`iteration` is the 0-BASED LLM iteration within the turn, so a tool loop's
successive calls are distinguishable. (Note the [PERF] lines print
`iteration + 1`; the trace value is the raw loop index. The trail prints both
on the same line, which is what keeps that off-by-one from being a trap.)

PRIVACY. What goes on the wire is a hex hash and a small integer. No user id,
no message text, no raw client id, no filenames. `cmid_h` is not reversible to
a client id in any useful sense and is already considered safe enough to log
on all three hops.

WHO RECEIVES IT — and why the gate is not only a flag. A header is only
private to us if we know where the request is going, and THIS CONTAINER DOES
NOT ALWAYS TALK TO OUR OWN PROXY. `bundle_client.make_openai_client` points
the SDK at the platform proxy only when `_bundle_active()` (SOURCE, 4f0e9fe1,
`app/services/bundle_client.py:62, :154`); otherwise it returns a client on
the SDK's default base URL — api.openai.com — and when there is no key at all
`openai_agent_service._ensure_client` falls back to `AsyncOpenAI(
api_key="missing")`, also the default base URL. `LLM_MODE=manual` with an
empty `TOUP_TOKEN` is the documented boot state of free-tier / pre-activation
tenants (SOURCE: `app/api/managed_agents.py:48`, `app/api/agent_setup.py:817`,
`app/services/free_tier_activation.py:131`), and BYOK-direct is the designed
alternative. A fleet-wide flag flip therefore reaches containers whose next
request goes straight to a third party. `trace_enabled` uses the bundle path
as a cheap preflight gate. The transport also checks the actual cached
client's base URL immediately before each Responses request, because a failed
bind refresh can leave a direct client behind bundle-mode settings. The
header exists only when that client targets our proxy. (The flag being on
with no bundle path is logged once per process — "enabled but not sent" must
not be silent.)

DEFAULT OFF, and off means ABSENT: no header key, no empty dict, no body
change. Flag-on adds exactly one header to the Responses request and changes
nothing else. An old platform ignores an unknown header; an old agent sends
none — so a mixed fleet is safe in both directions.

WHAT THIS BUYS AND WHAT IT DOES NOT. It creates a join key. It does not make
anything faster, does not explain any latency, and is worthless on its own —
the platform must be logging the validated header (patch B) for the other end
of the join to exist. See the patch notes.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, FrozenSet, Optional

from app.config import settings

logger = logging.getLogger(__name__)

#: The wire header name. Lower-case because that is the spelling the platform
#: half validates against; HTTP header names are case-insensitive, so this is
#: a readability choice, not a correctness one.
LLM_TRACE_HEADER = "x-toup-trace"

#: The platform validates `^[0-9a-f]{8}\.[0-9]{1,3}$`, so the iteration index
#: it can express is 0..999. `agent_max_tool_iterations` is 40 and the
#: per-run ContextVar override has never come near this, but a value the
#: platform would reject is worse than no value: it costs bytes and shows up
#: on the platform side as "header present but malformed", which reads like a
#: bug in the contract rather than a long turn. Past the cap we send nothing.
MAX_TRACE_ITERATION = 999

#: The grammar the platform half enforces, ANCHORED WITH `\A…\Z` rather than
#: `^…$`. In Python `$` also matches immediately before a final newline, so
#: `^[0-9a-f]{8}\.[0-9]{1,3}$` ACCEPTS `"a1b2c3d4.0\n"` — measured, not
#: reasoned (R48-G final review, non-blocking item 1). Nothing the shipped
#: producer mints can end in a newline (`trace_value` builds
#: `f"{hash}.{int}"`), so this is not a live defect; it is only a defect in the
#: thing this pattern is FOR, which is refusing a value some future caller
#: minted differently. A lone LF in a header value is exactly the byte a
#: response-splitting attempt is made of, so the last line of defence must not
#: be the one regex idiom that lets it through.
#:
#: The contract as assigned to the platform half (patch B) is spelled `^…$`.
#: That makes this producer STRICTER than that consumer, which is the safe
#: direction; patch B should be reconciled to `\A…\Z` too, because it LOGS the
#: value. Kept here so the producer is testable against the consumer's rule
#: without importing platform code.
TRACE_VALUE_RE = re.compile(r"\A[0-9a-f]{8}\.[0-9]{1,3}\Z")

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

#: One ERROR per distinct malformed-entry LENGTH, per process — NOT per
#: distinct entry. The latch key is `len(u)`, so two malformed entries of the
#: same length produce one ERROR. That is deliberate (the message names a
#: length and never the entry, so a second line for a second entry of the same
#: length would be a byte-identical repeat), but the name of the set is the
#: only thing that says so: an earlier version of this comment claimed
#: per-entry and was wrong.
_warned_lengths: set = set()

#: One WARNING per process for the flag-on-but-no-bundle-path case. Latched
#: because the alternative is a line per LLM iteration for the life of the
#: container, on a path that is already doing nothing.
_warned_destination: bool = False

_hash_fn: Optional[Callable[[Optional[str]], str]] = None


def _hash() -> Callable[[Optional[str]], str]:
    """Resolve THE `cmid_hash`, never define one.

    Two homes, one implementation. On this tree the function lives in
    `app/api/_turn_trace.py`; R48 patch A moves the body to
    `app/services/cmid.py` and leaves `_turn_trace.cmid_hash` re-exporting it.
    Preferring the services home means that in a combined tree `app/agent/`
    does not import `app/api/` (the wrong layering direction — the platform
    image ships no `app/agent/`), while on this tree alone the api module is
    still the only home and is used.

    Resolved lazily and cached: `app/api/__init__.py` pulls the router graph,
    which the agent process has already imported at boot (agent_main imports
    `app.api` before the runner exists) but a unit test has not. Doing it at
    module import time would put that graph behind `import agent_runner`.
    """
    global _hash_fn
    if _hash_fn is not None:
        return _hash_fn
    try:
        from app.services.cmid import cmid_hash as _fn  # type: ignore
    except Exception:  # noqa: BLE001 — not present on this tree; api is.
        from app.api._turn_trace import cmid_hash as _fn  # type: ignore
    _hash_fn = _fn
    return _fn


def _canary_ids() -> FrozenSet[str]:
    """Parsed per call (tiny), so an ops env change takes effect next turn.

    A non-uuid entry is an ERROR, not a shrug. The failure it guards against
    has already happened once in this investigation: a rollout note carried an
    8-CHARACTER PREFIX as the canary value, because that is what log lines
    print. Matching is exact on the full id, so such an entry matches nobody —
    the flag looks set, the canary is dark, and nothing anywhere is red. The
    entry itself is never logged (it is a user id); only its length is.
    """
    raw = getattr(settings, "llm_trace_header_canary_user_ids", "") or ""
    ids = {u.strip() for u in raw.split(",") if u.strip()}
    for u in ids:
        if _UUID_RE.match(u):
            continue
        key = f"{len(u)}"
        if key in _warned_lengths:
            continue
        _warned_lengths.add(key)
        logger.error(
            "[LLMTRACE] llm_trace_header_canary_user_ids holds an entry that "
            "is not a full user uuid (length=%d, expected 36). Matching is "
            "EXACT on the full id, so this entry can never match and the "
            "canary is silently OFF. The 8-char prefix printed in log lines "
            "is not a user id — resolve the full uuid from the platform "
            "users table.",
            len(u),
        )
    return frozenset(ids)


def _flag_or_canary(user_id: Optional[str]) -> bool:
    """Global-or-canary, the same shape as `stable_prefix_enabled` /
    `channel_envelope_enabled` — agent flags are otherwise fleet-wide, so a
    canary list is the only way to prove this on one tenant first."""
    if bool(getattr(settings, "llm_trace_header", False)):
        return True
    ids = _canary_ids()
    return bool(ids) and bool(user_id) and user_id in ids


def destination_is_platform_proxy() -> bool:
    """Preflight: settings call for a client pointed at OUR proxy.

    Asks `bundle_client._bundle_active` rather than re-deriving
    `llm_mode == "bundle" and toup_token` here, for the same reason `_hash`
    resolves `cmid_hash` instead of re-implementing it: a second copy of a
    condition is a condition that silently stops agreeing. The private name is
    deliberate — it IS the predicate `make_openai_client` branches on, and a
    public paraphrase that drifts from it would mint unnecessary traces. The
    transport still checks the actual cached client, so this settings gate
    never decides alone where the header goes. If bundle_client ever grows a
    public one, move to it and keep the drift test (`tests/test_llm_trace_header.py::
    test_the_destination_gate_is_bundle_clients_own_predicate`).

    Not wrapped in try/except: the only caller is already inside the runner's
    `except Exception -> no trace` guard, so a failure here costs no turn and
    sends no header — fail-safe in the direction that matters.
    """
    from app.services.bundle_client import _bundle_active

    return bool(_bundle_active())


def trace_enabled(user_id: Optional[str]) -> bool:
    """The full gate: someone asked for it AND it is safe to send.

    Order matters for cost, not for correctness: the flag/canary check is
    first so that a container with the flag off — i.e. every container today —
    never even imports `bundle_client` because of this module.
    """
    if not _flag_or_canary(user_id):
        return False
    if destination_is_platform_proxy():
        return True
    # Enabled and NOT sent. Silent would be the wrong answer twice over: an
    # operator running the canary would read "no header on the platform side"
    # as an intermediary stripping it (§7 step 5's discriminator), and the
    # reason it is withheld — this container talks straight to the provider —
    # is exactly the fact worth knowing. Nothing tenant-scoped is logged.
    global _warned_destination
    if not _warned_destination:
        _warned_destination = True
        logger.warning(
            "[LLMTRACE] enabled, but bundle routing is inactive in current "
            "settings (llm_mode=%r, toup_token set=%s), so no x-toup-trace "
            "header is requested: a client built from these settings would "
            "send it straight to the upstream provider, a third party. "
            "Not an error — the join key simply does not exist for this tenant.",
            getattr(settings, "llm_mode", None),
            bool(getattr(settings, "toup_token", "")),
        )
    return False


def trace_value(client_msg_id: Optional[str], iteration: int) -> Optional[str]:
    """The header value for one LLM call, or None to send no header.

    None (never a header) when:
      * there is no `client_msg_id` — routines, channel turns and sub-agent
        runs have none. `cmid_hash(None)` answers the sentinel "00000000",
        which would make every id-less turn in the fleet share one trace
        value: a join key that joins strangers is worse than no join key.
      * the iteration index is not an int in [0, MAX_TRACE_ITERATION] — see
        that constant.

    The returned value is hex-by-construction (`cmid_hash` formats `%08x`), so
    a hostile or malformed `client_msg_id` cannot inject a header value: it
    changes which 8 hex digits come out, and nothing else. The tests assert
    that against control characters, newlines, CRLF, non-ASCII and very long
    inputs rather than trusting the claim.
    """
    if not client_msg_id:
        return None
    try:
        i = int(iteration)
    except (TypeError, ValueError):
        return None
    if i < 0 or i > MAX_TRACE_ITERATION:
        return None
    return f"{_hash()(client_msg_id)}.{i}"


def trace_for_turn(
    user_id: Optional[str],
    client_msg_id: Optional[str],
    iteration: int,
) -> Optional[str]:
    """`trace_value` behind the flag/canary gate. The one call site helper."""
    if not trace_enabled(user_id):
        return None
    return trace_value(client_msg_id, iteration)
