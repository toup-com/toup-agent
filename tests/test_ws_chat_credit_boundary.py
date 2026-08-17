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
    block = src.split("user_msg = _friendly_error(e)")[1][:3200]
    # typed first, then the live-state rebuild, and only then the error frame
    assert block.index("isinstance(e, OutOfCreditsError)") < block.index('"out of Toup credits" in user_msg')
    assert block.index("build_exhausted_response()") < block.index('{"type": "error", "message": user_msg}')
    # the card frame is the structured event, not hand-rolled json
    assert "response_to_stream_event" in block


def test_detail_extractor_reads_json_and_repr_and_rejects_garbage():
    from app.api.ws_chat import _extract_out_of_credits_detail as ex
    detail = {"error": "out_of_credits", "reason": "insufficient_message_credits",
              "bucket": "message", "monthly_reset_at": "2026-09-05T00:00:00Z"}
    import json
    # JSON body inside prose (the httpx/openai style)
    assert ex(f"Error code: 402 - {json.dumps({'detail': detail})}".replace("'", '"'))["reason"] == "insufficient_message_credits"
    # python-repr (str(dict)) style
    assert ex(f"apologies: {str(detail)} and more")["reason"] == "insufficient_message_credits"
    # bare detail without wrapper
    assert ex(json.dumps(detail))["error"] == "out_of_credits"
    # mentions the phrase but carries no payload
    assert ex("upstream said out_of_credits but nothing else") is None
    assert ex("no credits mentioned here") is None


def test_cold_state_falls_back_to_remote_preflight_in_order():
    import inspect
    from app.api import ws_chat
    src = inspect.getsource(ws_chat)
    block = src.split('_extract_out_of_credits_detail(str(e))')[1][:1200]
    assert block.index('build_exhausted_response()') < block.index('check_balance_remote')
    assert 'check_balance_remote(user_id=user_id)' in block
