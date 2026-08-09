"""G-12/G-13 (audit 2026-08-09): two client-side holes on the LLM path.

G-12 — the BYOK OpenAI client was a bare `AsyncOpenAI(api_key=key)`,
riding the SDK's 600s default timeout while every sibling client in
bundle_client caps at 120s. A hung upstream held a BYOK turn (and its
WebSocket) open five times longer than the bundle path would tolerate.

G-13 — `claude-opus-4-7` is the Anthropic CANONICAL
(model_resolver._CANONICAL_ANTHROPIC_MODEL) and had a context-window row
(A8-3 added it for exactly this class of gap) but NO pricing row — with
Anthropic re-enabled it would fall through _calc_cost_cents' conservative
default and bill Opus at Sonnet rates (5x under). The new row is MODELLED
at opus-4-6's rate and says so in source; #509's lesson (correct against
real org billing) applies before trusting any 4-7 figure.
"""

from __future__ import annotations

import inspect
import re


def test_byok_client_has_an_explicit_timeout():
    from app.services import bundle_client

    src = inspect.getsource(bundle_client)
    bare = re.findall(r"AsyncOpenAI\(api_key=key\)", src)
    assert not bare, "the BYOK client is bare again — SDK default is 600s"
    assert re.search(r"AsyncOpenAI\(api_key=key, timeout=120", src), (
        "BYOK timeout must match the module's 120s convention"
    )


def test_the_anthropic_canonical_has_a_pricing_row():
    """Every model the resolver can HAND OUT must be priceable — a missing
    entry does not raise, it silently yields the conservative default."""
    from app.config import settings
    from app.services.model_resolver import _CANONICAL_ANTHROPIC_MODEL

    assert _CANONICAL_ANTHROPIC_MODEL in settings.pricing_per_1k, (
        f"{_CANONICAL_ANTHROPIC_MODEL} is the model default_anthropic_model "
        "falls back to, and it would bill at the fallthrough (Sonnet) rate"
    )
    row = settings.pricing_per_1k[_CANONICAL_ANTHROPIC_MODEL]
    assert row["input"] > settings.pricing_per_1k["claude-sonnet-4-6"]["input"], (
        "an Opus row priced at-or-below Sonnet is almost certainly the "
        "fallthrough value this test exists to prevent"
    )
