"""
Toup Data Fetchers — Weather, finance, Wikipedia without a browser.

All use free, open APIs or scraping that doesn't trigger CAPTCHAs:
  - Weather: wttr.in (designed for programmatic access)
  - Finance: yfinance library (scrapes Yahoo Finance)
  - Wikipedia: Wikipedia's own public API (free, no key)
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Weather (wttr.in — designed for CLI/API access, no CAPTCHA)
# ──────────────────────────────────────────────────────────────

async def toup_weather(location: str) -> str:
    """Get weather for a location using wttr.in."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Format=j1 returns JSON, format=3 returns one-line summary
            resp = await client.get(
                f"https://wttr.in/{location}",
                params={"format": "j1"},
                headers={"User-Agent": "curl/7.0"},
            )
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]
        city = area.get("areaName", [{}])[0].get("value", location)
        country = area.get("country", [{}])[0].get("value", "")

        lines = [
            f"Weather for {city}, {country}:",
            f"  Temperature: {current.get('temp_C', '?')}°C / {current.get('temp_F', '?')}°F",
            f"  Feels like: {current.get('FeelsLikeC', '?')}°C",
            f"  Condition: {current.get('weatherDesc', [{}])[0].get('value', '?')}",
            f"  Humidity: {current.get('humidity', '?')}%",
            f"  Wind: {current.get('windspeedKmph', '?')} km/h {current.get('winddir16Point', '')}",
            f"  Visibility: {current.get('visibility', '?')} km",
        ]

        # Add forecast
        forecast = data.get("weather", [])
        if forecast:
            lines.append("")
            lines.append("Forecast:")
            for day in forecast[:3]:
                date = day.get("date", "")
                max_c = day.get("maxtempC", "?")
                min_c = day.get("mintempC", "?")
                desc = day.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "")  # noon
                lines.append(f"  {date}: {min_c}°C - {max_c}°C, {desc}")

        return "\n".join(lines)
    except Exception as e:
        return f"Weather lookup failed: {e}"


# ──────────────────────────────────────────────────────────────
# Finance (yfinance or direct Yahoo Finance scrape)
# ──────────────────────────────────────────────────────────────

async def toup_stock(symbol: str) -> str:
    """Get stock quote for a ticker symbol."""
    try:
        import yfinance as yf
        import asyncio
        # yfinance is sync, run in executor
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
        info = await loop.run_in_executor(None, lambda: ticker.info)

        if not info or "regularMarketPrice" not in info:
            return f"No data found for {symbol}"

        lines = [
            f"{info.get('longName', symbol)} ({info.get('symbol', symbol)})",
            f"  Price: ${info.get('regularMarketPrice', '?')}",
            f"  Change: {info.get('regularMarketChange', '?'):+.2f} ({info.get('regularMarketChangePercent', 0)*100:+.2f}%)" if info.get('regularMarketChange') else "",
            f"  Day range: ${info.get('dayLow', '?')} - ${info.get('dayHigh', '?')}",
            f"  52w range: ${info.get('fiftyTwoWeekLow', '?')} - ${info.get('fiftyTwoWeekHigh', '?')}",
            f"  Market cap: ${info.get('marketCap', 0):,.0f}" if info.get('marketCap') else "",
            f"  Volume: {info.get('volume', 0):,}" if info.get('volume') else "",
        ]
        return "\n".join(l for l in lines if l)

    except ImportError:
        # Fallback: direct Yahoo Finance API
        return await _yahoo_finance_fallback(symbol)
    except Exception as e:
        return f"Stock lookup failed for {symbol}: {e}"


async def _yahoo_finance_fallback(symbol: str) -> str:
    """Fallback stock lookup via Yahoo Finance API (no library needed)."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"interval": "1d", "range": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            data = resp.json()

        meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
        price = meta.get("regularMarketPrice", "?")
        prev = meta.get("previousClose", 0)
        change = float(price) - float(prev) if price != "?" and prev else 0

        return (
            f"{meta.get('longName', symbol)} ({meta.get('symbol', symbol)})\n"
            f"  Price: ${price}\n"
            f"  Change: {change:+.2f}\n"
            f"  Currency: {meta.get('currency', 'USD')}"
        )
    except Exception as e:
        return f"Stock lookup failed for {symbol}: {e}"


# ──────────────────────────────────────────────────────────────
# Wikipedia (public API — free, no key, no CAPTCHA)
# ──────────────────────────────────────────────────────────────

async def toup_wikipedia(query: str, sentences: int = 5) -> str:
    """Look up a topic on Wikipedia using its public API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # First: search for the best matching article
            search_resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 1,
                    "format": "json",
                },
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()
            results = search_data.get("query", {}).get("search", [])
            if not results:
                return f"No Wikipedia article found for '{query}'"

            title = results[0]["title"]

            # Then: get the article summary
            summary_resp = await client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            )
            summary_resp.raise_for_status()
            summary = summary_resp.json()

        extract = summary.get("extract", "")
        url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")

        lines = [f"# {summary.get('title', title)}"]
        if summary.get("description"):
            lines.append(f"_{summary['description']}_")
        lines.append("")
        lines.append(extract)
        if url:
            lines.append(f"\nSource: {url}")
        return "\n".join(lines)

    except Exception as e:
        return f"Wikipedia lookup failed: {e}"
