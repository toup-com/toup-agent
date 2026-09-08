"""Persian search asks take the same light, capable path as English asks."""
from __future__ import annotations

import pytest

from app.agent.query_intent import classify_query_intent, filter_tools_by_intent


TOOL_DEFS = [
    {"name": name}
    for name in (
        "web_search", "web_fetch", "browser", "recall_day", "create_job",
        "update_job", "start_mission", "write_file", "generate_pdf",
        "read_file", "gmail__search_emails", "gmail__get_email",
        "calendar__list_events", "grep", "find",
    )
]


@pytest.mark.parametrize("message", [
    "آخرین اخبار هوش مصنوعی را جستجو کن و پنج مورد مهم این ماه را بده",
    "وضعیت آب و هوای امروز تورنتو را پیدا کن",
    "درباره بهترین مدل‌های هوش مصنوعی فعلی با منبع تحقیق کن",
    "لطفا سرچ کن ببین بهترین مدل تولید تصویر کدام است",
    "قیمت آنلاین طلای امروز را جست و جو کن و منبع بده",
])
def test_persian_search_uses_web_intent_without_losing_search_tools(message):
    intent = classify_query_intent(message)
    exposed = {tool["name"] for tool in filter_tools_by_intent(TOOL_DEFS, intent)}
    assert intent.category == "web"
    assert {"web_search", "web_fetch"} <= exposed
    assert "write_file" not in exposed
    assert "generate_pdf" not in exposed


@pytest.mark.parametrize("message", [
    "Search the latest AI news with sources",
    "Find today's Toronto weather online",
])
def test_english_search_keeps_the_same_web_route(message):
    assert classify_query_intent(message).category == "web"


def test_persian_greeting_does_not_become_a_search():
    assert classify_query_intent("سلام چطوری").category == "greeting"


@pytest.mark.parametrize(("message", "required_tools"), [
    (
        "آخرین ایمیل دریافتی از سارا را باز کن و خلاصه کن",
        {"gmail__search_emails", "gmail__get_email"},
    ),
    (
        "برنامهٔ فردای من را بررسی کن و جلساتم را به من بگو",
        {"calendar__list_events"},
    ),
    (
        "فایل گزارش قبلی را بررسی کن و غلط‌های نگارشی آن را اصلاح کن",
        {"read_file", "write_file"},
    ),
    (
        "ایمیل‌های سارا را جستجو کن و آخرین پیامش را خلاصه کن",
        {"gmail__search_emails", "gmail__get_email"},
    ),
    (
        "در فایل‌های من عبارت بودجه را جست‌وجو کن و نام فایل‌های مرتبط را به من بگو",
        {"read_file", "grep", "find"},
    ),
    (
        "جلسه‌های فردای من را در تقویم سرچ کن و ساعت هر جلسه را به من بگو",
        {"calendar__list_events"},
    ),
    (
        "جلسه‌های آنلاین فردای من را در تقویم جستجو کن و ساعت هر جلسه را به من بگو",
        {"calendar__list_events"},
    ),
    (
        "آخرین فایل گزارش قیمت فروش را باز کن و اعداد آن را بررسی کن",
        {"read_file"},
    ),
    (
        "ایمیل‌های سارا را جستجو کن و خبرهای عمومی شرکتش را هم در وب پیدا کن",
        {"gmail__search_emails", "gmail__get_email", "web_search", "web_fetch"},
    ),
    (
        "فایل گزارش فروش من را بخوان و قیمت فعلی طلا را در وب جستجو کن",
        {"read_file", "web_search", "web_fetch"},
    ),
])
def test_owned_persian_data_requests_do_not_lose_the_named_capability(
    message, required_tools,
):
    intent = classify_query_intent(message)
    exposed = {tool["name"] for tool in filter_tools_by_intent(TOOL_DEFS, intent)}
    assert intent.category == "full"
    assert required_tools <= exposed
    assert {"web_search", "web_fetch"} <= exposed
