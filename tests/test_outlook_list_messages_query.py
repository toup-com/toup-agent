"""The Outlook read builds ONE legal Graph query, never two languages.

Microsoft Graph's message collection rejects `$search` alongside
`$filter` or `$orderby` with a 400, so the shipped combination
(`$orderby=receivedDateTime desc` + `$search="…"`) could never succeed
— every query-bearing Outlook read failed. And `$filter` with
`$orderby` is legal only in the shape "List messages" documents:
the ordered property must lead the filter, or Graph answers
`InefficientFilter`.

Neither is visible in a unit test of the response handling — the
request never gets built anywhere else — so the param builder is
pinned here directly.
"""
from __future__ import annotations

import pytest

from app.connectors.outlook.provider import _list_messages_params


def test_a_plain_read_is_ordered_newest_first():
    params, scan, limit = _list_messages_params({"max_results": 10})
    assert params["$orderby"] == "receivedDateTime desc"
    assert "$search" not in params and "$filter" not in params
    assert (params["$top"], limit, scan) == (10, 10, None)


def test_unread_rides_filter_and_the_sort_property_leads_it():
    params, scan, _ = _list_messages_params({"is_read": False})
    # Rules 1 and 3 of Graph's filter+orderby contract: receivedDateTime
    # is in $orderby, so it must also be in $filter, ahead of isRead.
    assert params["$filter"] == (
        "receivedDateTime ge 1900-01-01T00:00:00Z and isRead eq false"
    )
    assert params["$orderby"] == "receivedDateTime desc"
    assert scan is None
    read_only, _, _ = _list_messages_params({"is_read": True})
    assert read_only["$filter"].endswith("and isRead eq true")


def test_a_search_carries_neither_orderby_nor_filter():
    params, scan, limit = _list_messages_params(
        {"query": "from:alice@contoso.com", "max_results": 10},
    )
    assert params["$search"] == '"from:alice@contoso.com"'
    assert "$orderby" not in params and "$filter" not in params
    assert (scan, limit) == (None, 10)


def test_search_plus_unread_over_fetches_and_filters_the_page_here():
    params, scan, limit = _list_messages_params(
        {"query": "subject:invoice", "is_read": False, "max_results": 10},
    )
    assert "$orderby" not in params and "$filter" not in params
    assert scan is False and limit == 10
    assert params["$top"] == 40  # headroom, so a page of read matches
    assert "isRead" in params["$select"]  # …can still be filtered here


def test_a_quoted_phrase_does_not_break_the_search_expression():
    # Graph documents no escape for a quote inside the $search value.
    params, _, _ = _list_messages_params({"query": 'subject:"year end"'})
    assert params["$search"] == '"subject:year end"'


@pytest.mark.parametrize("top", [0, 999, "junk", None])
def test_top_stays_inside_the_manifest_ceiling(top):
    params, _, limit = _list_messages_params(
        {"max_results": top, "is_read": False, "query": "hello"},
    )
    assert 1 <= params["$top"] <= 50 and 1 <= limit <= 50
