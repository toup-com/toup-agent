"""Env scrubbing for tool/code-engine subprocesses (GATED, opt-in).

The agent's ``exec`` tool and the in-container Claude Code / Codex engines spawn
child shells with the FULL process environment, so one ``env`` / ``printenv``
call dumps every injected secret (docs/security/audit-2026.md EXF-1 / G1).

This helper is the single place that removes platform-internal isolation
secrets from a child's environment. It is **default-off and conservative**:

- ``settings.exec_env_scrub`` defaults ``False`` → the env is returned unchanged
  (exact pre-audit behaviour). Flip it only after validating in staging.
- Even when on, only ``settings.exec_env_scrub_keys`` are dropped. The default
  set is limited to keys **no legitimate tenant/agent exec flow needs**
  (``POOL_ADMIN_TOKEN``, ``AGENT_API_KEY``, ``BRAVE_API_KEY``). ``DATABASE_URL``,
  ``TOUP_TOKEN`` and the tenant's own provider keys are DELIBERATELY excluded
  from the default — the agent legitimately runs ``psql $DATABASE_URL`` via
  ``exec`` and BYO code may read its own provider key. An operator adds those to
  the list only after confirming nothing breaks.

The real EXF-1 fix is infra (don't inject platform secrets into the container
at all); this is the in-repo, non-breaking, opt-in lever toward it.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Mapping, Optional

from app.config import settings

logger = logging.getLogger(__name__)


def scrubbed_environ(base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return a copy of ``base`` (or ``os.environ``) with the configured
    platform-internal secret keys removed when ``exec_env_scrub`` is on.

    Correct for children whose env *replaces* the parent's (the standard
    ``subprocess``/``create_subprocess_shell`` semantics used by the ``exec``
    tool). For a child that LAYERS its env on top of the full parent
    ``os.environ`` (e.g. the Claude Code SDK — see claude_engine), removing a
    key from this dict is not enough; use :func:`scrub_overrides` to blank it.

    Default off → returns the environment unchanged.
    """
    env = dict(base if base is not None else os.environ)
    if not getattr(settings, "exec_env_scrub", False):
        return env
    for key in getattr(settings, "exec_env_scrub_keys", None) or []:
        env.pop(key, None)
    return env


def scrub_overrides(exclude: tuple[str, ...] = ()) -> dict[str, str]:
    """Return ``{key: ""}`` for every scrub key, for children that layer their
    env on top of the parent ``os.environ`` (setting a key to empty is what the
    CLI treats as unset; a mere pop would be re-inherited).

    ``exclude`` lists keys the caller manages itself (e.g. the code engine
    re-injects ``CLAUDE_CODE_OAUTH_TOKEN``), which must not be blanked here.
    Returns ``{}`` when scrubbing is off.
    """
    if not getattr(settings, "exec_env_scrub", False):
        return {}
    return {
        key: ""
        for key in (getattr(settings, "exec_env_scrub_keys", None) or [])
        if key not in exclude
    }


def sandbox_preexec() -> Optional[Callable[[], None]]:
    """Return a ``preexec_fn`` that drops the exec child to the configured
    ``exec_sandbox_user`` (separate, lower-privileged uid), or ``None`` when
    disabled / unresolvable.

    The child then cannot read ``/app`` platform source or the agent process's
    ``/proc/<pid>/environ`` (different uid), while the agent itself is
    unaffected. Requires the parent (agent) to be root or hold CAP_SETUID.
    Returns ``None`` (a no-op — current behaviour) if the user isn't set, the
    platform isn't POSIX, or the user doesn't exist, so enabling it can never
    hard-fail an exec — it just doesn't drop privileges.
    """
    user = (getattr(settings, "exec_sandbox_user", "") or "").strip()
    if not user:
        return None
    try:
        import pwd  # POSIX-only
        pw = pwd.getpwnam(user)
    except Exception as e:  # not POSIX, or user missing in the image
        logger.warning("[exec-sandbox] user %r unavailable (%s) — running exec un-dropped", user, e)
        return None
    uid, gid, home = pw.pw_uid, pw.pw_gid, pw.pw_dir

    def _drop() -> None:  # runs in the forked child, before exec
        os.setgid(gid)
        try:
            os.setgroups([gid])
        except Exception:
            pass
        os.setuid(uid)
        os.environ["HOME"] = home

    return _drop
