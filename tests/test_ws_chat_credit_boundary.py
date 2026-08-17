"""A credit boundary must reach chat as the structured card, never as an error.

The typed path (OutOfCreditsError) is handled inside the runner; these pin the
ws_chat BACKSTOP for raw billing-class exceptions, and the discrimination that
keeps platform-side outages OUT of the user-paywall card.
"""
import inspect

from app.api import ws_chat
from app.api.ws_chat import _friendly_error


class FakeExc(Exception):
    pass


def test_generic_billing_text_is_card_eligible():
    msg = _friendly_error(FakeExc("insufficient_quota: your account has run out"))
    assert "out of Toup credits" in msg


def test_platform_anthropic_outage_is_not_card_eligible():
    # The platform's own Claude billing failing is OUR outage, not the user's
    # wallet — it must never render the user paywall card.
    msg = _friendly_error(FakeExc(
        "Your credit balance is too low to access the Anthropic API. "
        "Please go to Plans & Billing to upgrade or purchase credits."))
    assert "out of Toup credits" not in msg


def test_claude_subscription_exhaustion_is_not_card_eligible():
    msg = _friendly_error(FakeExc("You're out of extra usage. Add more at claude.ai/settings/usage"))
    assert "out of Toup credits" not in msg


def test_backstop_converts_before_the_error_frame():
    src = inspect.getsource(ws_chat)
    block = src.split("user_msg = _friendly_error(e)")[1][:1600]
    # typed first, then the live-state rebuild, and only then the error frame
    assert block.index("isinstance(e, OutOfCreditsError)") < block.index('"out of Toup credits" in user_msg')
    assert block.index("build_exhausted_response()") < block.index('{"type": "error", "message": user_msg}')
    # the card frame is the structured event, not hand-rolled json
    assert "response_to_stream_event" in block
