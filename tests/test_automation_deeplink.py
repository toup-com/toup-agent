"""Automation run deep link (Round 28-C, platform lane).

The tap target for an automation-run card is COMPUTED server-side from
validated tokens — `automation_deep_link` — never taken from a tenant
payload. Shape agreed with the app round: colon-free host, ids in the
query, `mission` rides along for the tap ACK.
"""

from app.services.live_activity_service import automation_deep_link


def test_builds_the_agreed_shape():
    link = automation_deep_link(
        {"route": "automation", "automation_id": "auto-1", "run_id": "job-9"},
        "job-9",
    )
    assert link == "toup://automation?session=auto-1&run=job-9&mission=job-9"


def test_requires_the_automation_route():
    assert automation_deep_link(
        {"route": "chat", "automation_id": "a", "run_id": "b"}, "b",
    ) is None
    assert automation_deep_link({}, "m") is None


def test_refuses_invalid_ids():
    # Whitespace, oversized, or non-string ids drop the link entirely —
    # never a truncated different id (the _safe_id rule).
    assert automation_deep_link(
        {"route": "automation", "automation_id": "has space", "run_id": "b"},
        "m",
    ) is None
    assert automation_deep_link(
        {"route": "automation", "automation_id": "a", "run_id": "x" * 65},
        "m",
    ) is None
    assert automation_deep_link(
        {"route": "automation", "automation_id": 7, "run_id": "b"}, "m",
    ) is None


def test_url_encodes_id_characters():
    link = automation_deep_link(
        {"route": "automation", "automation_id": "a&b=c", "run_id": "r?1"},
        "m",
    )
    assert link == "toup://automation?session=a%26b%3Dc&run=r%3F1&mission=m"
