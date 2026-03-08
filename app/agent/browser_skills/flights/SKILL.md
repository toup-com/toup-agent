---
name: flights
description: Search and compare flights across airlines and OTAs.
---

# Flight Search Skill

## SITE

Navigate to the correct URL based on trip type:
- **One-way**: `https://www.google.com/travel/flights?tfs=CBwQAQ`
- **Round-trip**: `https://www.google.com/travel/flights`
- **Multi-city**: `https://www.google.com/travel/flights?tfs=CBwQAw`

The `tfs=CBwQAQ` parameter sets one-way mode directly in the URL — no need to interact with the trip type dropdown.

## STEP-BY-STEP PROCEDURE:

### Step 1: Navigate
- Navigate to the correct URL above based on trip type
- Wait for the page to load

### Step 2: Dismiss Overlays
If you see "Try AI powered Flight Deals" or any promotional overlay, click "Got it", "No thanks", or the X button. Wait for it to close.

### Step 3: Fill Origin ("Where from?")
- Click the input field labeled "Where from?"
- Type the origin airport code (e.g. "YYZ")
- **WAIT** for the autocomplete dropdown to appear (look at the screenshot!)
- **CLICK** the correct airport suggestion (e.g. "Toronto Pearson International Airport")
- **VERIFY** it appears in the field before moving on
- ⚠️ Do NOT skip clicking the autocomplete! The airport is not set until you click the suggestion.

### Step 4: Fill Destination ("Where to?")
- Click the input field labeled "Where to?" — this is NEXT to origin, NOT the date field
- Type the destination airport code (e.g. "DXB")
- **WAIT** for the autocomplete dropdown to appear (look at the screenshot!)
- **CLICK** the correct airport suggestion (e.g. "Dubai International Airport")
- **VERIFY** it appears in the field
- ⚠️ Do NOT skip clicking the autocomplete! The airport is not set until you click the suggestion.
- ⚠️ Do NOT move to the date field until you have confirmed the destination is set.

### Step 5: Set Departure Date
- Click the "Departure" date field to open the calendar
- Use the `select_date` tool with YYYY-MM-DD format (e.g. "2026-03-27")
- Click "Done" to confirm the date
- ⚠️ ONLY use the `select_date` tool. Do NOT try to click calendar day numbers manually.

### Step 6: Set Return Date (if round-trip)
- Click "Return" field → use `select_date` → click "Done"

### Step 7: Search
- Click the "Search" button — use selector `button[aria-label="Search"]`
- ⚠️ Make sure the calendar is CLOSED (click "Done" first) before clicking Search
- Wait for results to load (3-5 seconds)

### Step 8: Read Results
- Scroll through results to find the best options
- Look for "Best flights" and "Cheapest" sections
- Extract: airline, price, times, duration, number of stops
- Present the top 3-5 options to the user

## COMMON PITFALLS:
- After typing airport code, you MUST wait for and click the autocomplete suggestion
- "Washington, D.C." default origin means the form was reset — the overlay intercepted your input
- The "Explore" button is NOT the search button — look for "Search" specifically
- Do NOT try to click individual date numbers — use the select_date tool
- After filling origin, the NEXT field is "Where to?" (destination), NOT "Departure" (date)
- If the calendar is still open, click "Done" BEFORE clicking Search

## SELECTOR HINTS:
- Origin: input[aria-label*="Where from"]
- Destination: input[aria-label*="Where to"]
- Departure date: input[aria-label="Departure"]
- Return date: input[aria-label="Return"]
- Search button: button[aria-label="Search"]

## CRITICAL WARNINGS:
- NEVER click "Sign In" or any Google account link — stay on the flights form
- NEVER click "Explore" — that's not the Search button
- NEVER click any `button[jsname]` without checking it's actually the Search button
- If you end up on /travel/flights/deals or /travel/flights/explore or /travel/explore, navigate back to the correct flights URL
- If the form shows "Round trip" but you need "One way", navigate to the URL with `?tfs=CBwQAQ`
