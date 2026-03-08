---
name: flights
description: Search and compare flights across airlines and OTAs.
---

# Flight Search Skill

## PREFERRED METHOD: Direct URL Navigation

**ALWAYS use URL-based navigation instead of filling the form manually.** The form autocomplete is unreliable in automated browsers.

### One-way flight URL format:
```
https://www.google.com/travel/flights?hl=en#flt=ORIGIN.DEST.YYYY-MM-DD;c:USD;e:1;sd:1;t:f
```

### Round-trip URL format:
```
https://www.google.com/travel/flights?hl=en#flt=ORIGIN.DEST.YYYY-MM-DD*DEST.ORIGIN.YYYY-MM-DD
```

### Examples:
- One-way YYZ→DXB on Mar 27: `https://www.google.com/travel/flights?hl=en#flt=YYZ.DXB.2026-03-27;c:USD;e:1;sd:1;t:f`
- Round-trip YYZ→DXB Mar 27 – Apr 3: `https://www.google.com/travel/flights?hl=en#flt=YYZ.DXB.2026-03-27*DXB.YYZ.2026-04-03`

## STEP-BY-STEP PROCEDURE:

### Step 1: Determine Airport Codes
Map the user's cities to IATA airport codes:
- Toronto → YYZ, Dubai → DXB, London → LHR, New York → JFK
- Paris → CDG, Tokyo → NRT/HND, Singapore → SIN, Hong Kong → HKG
- Los Angeles → LAX, Chicago → ORD, Istanbul → IST, Bangkok → BKK
- Frankfurt → FRA, Amsterdam → AMS, Seoul → ICN, Delhi → DEL
- Sydney → SYD, Doha → DOH, Montreal → YUL, Vancouver → YVR
- If unsure, use the most common airport code for the city.

### Step 2: Construct and Navigate to URL
- Build the URL using the format above with the airport codes and dates
- Use the `navigate` tool to go directly to that URL
- Wait for the page to load

### Step 3: Dismiss Overlays
If you see "Try AI powered Flight Deals" or any promotional overlay, click "Got it", "No thanks", or the X button.

### Step 4: Scroll Down to See Results
- ⚠️ CRITICAL: Flight results are BELOW the search form. You MUST scroll down to see them.
- Use `scroll` with direction "down" and amount 800-1200 to see the flight listings
- Do NOT assume the page has no results — the results are always below the form

### Step 5: Read Results
- Look for flight cards showing airline, price, times, duration, stops
- Look for "Best flights" and "Cheapest" sections
- Extract: airline, price, departure/arrival times, duration, number of stops
- Present the top 3-5 options to the user
- If you see "Explore destinations" instead of flight results, the URL was wrong — re-navigate
- ⚠️ Do NOT call done() with a failure message. If you don't see results, SCROLL DOWN first.

## FALLBACK: Manual Form Entry (only if URL method fails)

If the URL-based approach doesn't show results:

1. Navigate to `https://www.google.com/travel/flights?tfs=CBwQAQ` (one-way)
2. Click "Where from?" → type airport code → **the system will auto-select the airport**
3. Click "Where to?" → type airport code → **the system will auto-select the airport**
4. Click "Departure" → use `select_date` tool → click "Done"
5. Click `button[aria-label="Search"]` or `button[aria-label="Search for flights"]`

## CRITICAL WARNINGS:
- ALWAYS try the URL method first — it's far more reliable
- NEVER click "Sign In" or any Google account link
- NEVER click "Explore" — that's not the Search button
- If you end up on /travel/flights/deals or /travel/explore, navigate back using the correct URL
- Do NOT call done() with a failure message — keep trying
