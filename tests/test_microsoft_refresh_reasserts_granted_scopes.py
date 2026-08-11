"""Entra's refresh grant needs the granted scopes re-asserted.

The bug this file exists to prevent, from the production trail on
2026-08-08: `microsoft_refresh` sent the literal `"offline_access"` as
its whole `scope`, on the strength of a comment claiming Graph "echoes
back the union of granted scopes". Entra answered

    AADSTS70011: The provided request must include a 'scope' input
    parameter.

so every Microsoft access token died at its one-hour expiry and the
identity went `reauth_required` with no route back. Outlook logged it
twice, once per connect. Nothing caught it: refresh runs on a timer,
the failure is a background event, and no test crossed the token
endpoint.

Teams shares `microsoft_refresh` verbatim, so it would have inherited
the same one-hour lifetime the moment it shipped.
"""

from __future__ import annotations

import inspect

import pytest

from app.connectors import _microsoft_base as msb
from app.connectors.base import RefreshFailed

# The scope list a real, working Outlook identity carried in production.
# Note what is NOT in it: `offline_access`. Entra does not echo that back
# in the granted set, which is exactly why round-tripping the stored list
# unmodified would silently drop the rotating refresh token.
REAL_GRANTED = [
    "profile",
    "openid",
    "email",
    "https://graph.microsoft.com/Mail.Read",
    "https://graph.microsoft.com/Mail.Send",
    "https://graph.microsoft.com/User.Read",
]


def test_the_exact_string_entra_rejected_is_never_sent_again():
    """`offline_access` alone names no resource. That is AADSTS70011."""
    sent = msb._refresh_scope_param(REAL_GRANTED)
    assert sent != "offline_access"
    assert sent.split() != ["offline_access"]


def test_every_granted_resource_scope_is_re_asserted():
    sent = msb._refresh_scope_param(REAL_GRANTED).split()
    for scope in REAL_GRANTED:
        assert scope in sent, f"{scope} dropped from the refresh request"


def test_offline_access_is_re_appended_even_though_entra_never_echoes_it():
    """Without it Entra returns no rotating refresh_token, which turns a
    recoverable one-hour bug into a permanent one at the refresh token's
    own expiry."""
    assert "offline_access" not in REAL_GRANTED  # guards the premise
    assert "offline_access" in msb._refresh_scope_param(REAL_GRANTED).split()


def test_offline_access_is_not_duplicated_when_the_identity_has_it():
    sent = msb._refresh_scope_param([*REAL_GRANTED, "offline_access"]).split()
    assert sent.count("offline_access") == 1


def test_duplicate_and_blank_scopes_collapse():
    sent = msb._refresh_scope_param(
        ["  ", "https://graph.microsoft.com/Mail.Read",
         "https://graph.microsoft.com/Mail.Read", ""]
    ).split()
    assert sent == ["https://graph.microsoft.com/Mail.Read", "offline_access"]


@pytest.mark.parametrize("empty", [None, [], ["offline_access"], ["", "  "]])
def test_no_resource_scope_fails_loudly_instead_of_asking_entra(empty):
    """`offline_access` alone is the precise shape Entra rejects, so an
    identity with no stored resource scope must not spend a round-trip
    to be told so — and must not fall back to the old broken literal."""
    with pytest.raises(RefreshFailed):
        msb._refresh_scope_param(empty)


def test_teams_scopes_survive_the_round_trip():
    """The four delegated permissions the Teams manifest requests."""
    granted = [
        "https://graph.microsoft.com/Team.ReadBasic.All",
        "https://graph.microsoft.com/Channel.ReadBasic.All",
        "https://graph.microsoft.com/Chat.Read",
        "https://graph.microsoft.com/ChatMessage.Send",
    ]
    sent = msb._refresh_scope_param(granted).split()
    assert set(granted).issubset(set(sent))
    assert "offline_access" in sent


# --- wiring -------------------------------------------------------------
# The pure function above can be perfect while nothing calls it with the
# right argument. These read the call sites.


def test_microsoft_refresh_takes_scopes_and_passes_them_to_the_token_call():
    sig = inspect.signature(msb.microsoft_refresh)
    assert "scopes" in sig.parameters
    assert sig.parameters["scopes"].kind is inspect.Parameter.KEYWORD_ONLY
    src = inspect.getsource(msb.microsoft_refresh)
    assert '"scope": _refresh_scope_param(scopes)' in src, (
        "the token request must build its scope from the granted set"
    )
    assert '"scope": "offline_access"' not in src


@pytest.mark.parametrize("connector", ["outlook", "teams"])
def test_the_microsoft_providers_forward_the_granted_scopes(connector):
    """Both share `microsoft_refresh`; a provider that swallows `scopes`
    would reintroduce the bug for its connector alone."""
    mod = __import__(
        f"app.connectors.{connector}.provider", fromlist=["*"],
    )
    provider_cls = next(
        v for k, v in vars(mod).items()
        if isinstance(v, type) and hasattr(v, "refresh")
        and v.__module__ == mod.__name__
    )
    src = inspect.getsource(provider_cls.refresh)
    assert "scopes=scopes" in src, (
        f"{connector} calls microsoft_refresh without forwarding scopes"
    )


def test_the_dispatcher_passes_the_identitys_scopes_not_the_manifests():
    """Entra reissues only already-consented scopes, so sourcing these
    from the manifest would break every existing identity the moment a
    manifest gained a scope."""
    from app.services import connector_dispatcher as cd

    src = inspect.getsource(cd)
    assert "scopes=latest.scopes" in src, (
        "dispatcher must hand provider.refresh the identity's granted scopes"
    )


def test_every_provider_accepts_the_granted_scopes():
    """The dispatcher passes `scopes=` unconditionally, so a provider
    that did not accept it would TypeError at refresh time — in a
    background task, for that connector only, discovered by users.

    Enumerated through the REGISTRY rather than by walking the package
    tree: `pkgutil.iter_modules(...).ispkg` is False for every connector
    but `stub` (they have no `__init__.py`), so a directory walk finds
    one provider and passes vacuously. The registry is also the set the
    dispatcher can actually reach, which is the set that matters.
    """
    from app.services.connector_registry import get_registry

    registry = get_registry()
    if not registry.list_all():
        registry.load_all(include_experimental=True)
    entries = list(registry.list_all())
    # Guards the guard: an empty or near-empty registry would make every
    # assertion below vacuous, which is the exact way this test failed
    # its own mutation run the first time.
    assert len(entries) >= 10, f"registry only yielded {len(entries)} connectors"

    offenders = []
    for entry in entries:
        params = inspect.signature(type(entry.provider).refresh).parameters
        if "scopes" not in params:
            offenders.append(type(entry.provider).__name__)
    assert not offenders, f"providers missing the `scopes` kwarg: {offenders}"
