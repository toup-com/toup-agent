"""POST /auth/demo is a login backdoor unless gated (G-2, audit 2026-08-09).

The route is unauthenticated, the password is hardcoded in source, and it
mints a real session token for a shared account. It was reachable in
production (mounted via platform_main.py:891) and the demo user exists in
the prod DB (demo@toup.local, created 2026-07-08) — anyone on the internet
could obtain a valid session and the memory/agent/LLM-spend surface behind
it.

Gate: settings.demo_login_enabled, default False → 404 (not 403 — a
disabled backdoor should not advertise its own existence). The gate must
fire BEFORE any DB work.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api import auth as auth_api
from app.config import settings


def test_the_flag_defaults_off():
    """The default in code is the fleet's value — an env var must opt IN."""
    assert settings.model_fields["demo_login_enabled"].default is False


async def test_disabled_demo_login_404s_before_touching_the_db(monkeypatch):
    monkeypatch.setattr(settings, "demo_login_enabled", False)
    with pytest.raises(HTTPException) as exc:
        # db=None proves the gate fires before any query: reaching the DB
        # path would AttributeError on None, not raise a clean 404.
        await auth_api.demo_login(db=None)
    assert exc.value.status_code == 404


async def test_enabled_demo_login_proceeds_to_the_db_path(monkeypatch):
    """Flag on → the original behavior (here: it reaches user lookup)."""
    monkeypatch.setattr(settings, "demo_login_enabled", True)

    seen = {}

    async def fake_get_user_by_email(db, email):
        seen["email"] = email

        class _U:
            id = "demo-user-id"

        return _U()

    monkeypatch.setattr(auth_api, "get_user_by_email", fake_get_user_by_email)
    monkeypatch.setattr(auth_api, "create_access_token", lambda uid: "tok-" + uid)

    token = await auth_api.demo_login(db=None)
    assert seen["email"] == "demo@toup.local"
    assert token.access_token == "tok-demo-user-id"
