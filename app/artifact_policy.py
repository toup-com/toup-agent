"""ONE definition of what a generated app is allowed to do.

Round 20. Generated apps could not make a sound. The cause was not a bug in
any of them: the platform's artifact response carried **no ``media-src``
directive**, so it fell back to ``default-src 'self'`` — and an artifact runs
on an OPAQUE origin, where ``'self'`` matches nothing. Every ``data:`` and
``blob:`` sound was refused, and the browser reported it as
``NotSupportedError — Failed to load because no supported source was found``,
which is what it says about a **corrupt file**. So it read as a bad asset
rather than as a policy decision, and nothing in the pipeline disagreed.

The reason it survived is the reason this module exists. **The same artifact
had two policies.** The mobile runner writes its own
(``appArtifacts.sandboxCsp``, a ``<meta>``) and has always allowed
``media-src data: blob:``; the web runner sent a response header that did not.
One app, two runners, two answers to "can this make a noise" — and nobody had
written down that they were supposed to agree.

So the policy is defined once, here, and three places read it:

* ``api/artifact_proxy.artifact_headers`` — the response header the browser
  enforces (adds ``frame-ancestors``, which only a real header can carry);
* ``agent/skills/builtins/app_html/verify`` — the PUBLISH GATE, which now
  loads the app under this policy instead of under none. Measured: with the
  gate seeing no policy, this round's bug is invisible to it; with the policy
  applied, the pre-fix directive set produces
  ``blocked: ["media-src blocked data"]`` and the publish is refused. A gate
  that runs the app in a more permissive browser than the user's is a canary
  that cannot fail;
* the mobile runner, by convention — it is a separate repo and cannot import
  this, so :data:`MOBILE_PARITY_NOTE` records the pairing and
  ``test_app_sound_and_look.py`` asserts the directives that must match.

**This module imports nothing but ``app.config``.** The platform image does
not ship ``app/agent/`` (see ``api/llm_proxy.py``'s note) and the agent does
not run the platform's routers, so anything both halves must agree on has to
live above them both. A helper either of them could not import would simply
recreate the divergence one directory up.
"""

from __future__ import annotations

from typing import Optional, Sequence

#: Kept in step with `toup-platform-app`'s `appArtifacts.sandboxCsp`, which
#: cannot import this. When you change a directive here, change it there —
#: the whole failure this module documents was those two drifting apart.
MOBILE_PARITY_NOTE = "src/shared/appArtifacts.ts::sandboxCsp"

#: Directives whose value must be the same on both runners. Not the whole
#: policy: the header can express `frame-ancestors` and a `<meta>` cannot, and
#: the mobile side legitimately says `default-src 'none'` because it names
#: every directive explicitly. These are the ones where a difference means an
#: app that works in one runner and is silently broken in the other.
PARITY_DIRECTIVES = ("media-src", "worker-src", "child-src", "img-src")


def sandbox_csp(cdn: str, *, frame_ancestors: Optional[Sequence[str]] = None) -> str:
    """The policy a generated app runs under.

    ``script-src`` keeps ``'unsafe-inline'`` and ``'unsafe-eval'`` on purpose:
    the whole format is inline script, and Babel-standalone (the supported
    React path) compiles JSX at runtime, which is ``eval``. They are safe to
    grant *here* precisely because everything else is closed — the page has no
    origin worth stealing from, no cookies, and no network.

    ``connect-src 'self'`` reads as permissive and is not: the frame is
    sandboxed without ``allow-same-origin``, so its origin is opaque and
    ``'self'`` matches nothing. It is the tightest expressible value —
    ``'none'`` is equivalent in effect and breaks same-origin debugging of the
    page outside a frame.

    ``media-src`` and ``worker-src``/``child-src`` name ``data:`` and
    ``blob:`` for the same reason ``img-src`` always has: those are bytes the
    page already holds, reachable with no network. Granting them is not a
    loosening — it is the difference between an app that can make a noise or
    run a worker and one that cannot, and both were failing silently on the
    web runner while working on the mobile one.

    ``frame_ancestors`` is header-only; a ``<meta>`` policy ignores it, so the
    gate omits it rather than emitting a directive that does nothing.
    """
    directives = [
        "default-src 'self'",
        f"script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: {cdn}",
        f"style-src 'self' 'unsafe-inline' {cdn}",
        "img-src 'self' data: blob:",
        f"media-src 'self' data: blob: {cdn}",
        "worker-src 'self' blob:",
        "child-src 'self' blob:",
        f"font-src 'self' data: {cdn}",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
    ]
    if frame_ancestors is not None:
        directives.append(f"frame-ancestors {' '.join(frame_ancestors) or chr(39) + 'none' + chr(39)}")
    return "; ".join(directives)


def artifact_cdn_origin() -> str:
    try:
        from app.config import settings
        return getattr(settings, "artifact_cdn_origin", "") or "https://cdnjs.cloudflare.com"
    except Exception:  # pragma: no cover - config must never break the policy
        return "https://cdnjs.cloudflare.com"
