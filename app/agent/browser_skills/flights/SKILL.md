---
name: flights
description: Search and compare flights across airlines and OTAs.
---

# Flight Search Skill

## SITE: Navigate to https://www.google.com/travel/flights

## STEP-BY-STEP PROCEDURE:

### Step 1: Dismiss Overlays
If you see "Try AI powered Flight Deals" or any promotional overlay, click "Got it", "No thanks", or the X button. Wait for it to close.

### Step 2: Set Trip Type (TWO-STEP process — CRITICAL)
The trip type selector is a DROPDOWN, not a single button. It requires TWO clicks:

**Step 2a**: Click the button that currently says "Round trip" (this OPENS the dropdown menu).
- Selector: `button[aria-label*="trip" i]` or click the text "Round trip"
- After clicking, a dropdown menu appears with options: "Round trip", "One way", "Multi-city"

**Step 2b**: NOW click "One way" from the OPEN dropdown menu.
- The options are now VISIBLE in the dropdown — click the text "One way"
- Wait for the dropdown to close and verify the button now says "One way"

**COMMON MISTAKE**: Do NOT try to click "One way" without first opening the dropdown.
The text "One way" exists in the DOM but is HIDDEN until the dropdown is open.
If your click fails with "not visible", you forgot Step 2a.

- If round_trip → leave as is (skip Step 2)
- Do NOT click hamburger menus, "Sign In", or navigation links

### Step 3: Fill Origin ("Where from?")
- Click the input field labeled "Where from?"
- Type the origin airport code (e.g. "YYZ")
- Wait for autocomplete dropdown to appear
- Click the correct airport suggestion (e.g. "Toronto Pearson International Airport")
- Verify it appears in the field before moving on

### Step 4: Fill Destination ("Where to?")
- Click the input field labeled "Where to?" — this is NEXT to origin, NOT the date field
- Type the destination airport code (e.g. "DXB")
- Wait for autocomplete → click correct suggestion
- Verify it appears in the field

### Step 5: Set Departure Date
- Click the "Departure" date field to open the calendar
- Use the `select_date` tool with YYYY-MM-DD format
- Click "Done" to confirm

### Step 6: Set Return Date (if round-trip)
- Click "Return" field → use `select_date` → click "Done"

### Step 7: Search
- Click the "Search" button (aria-label="Search")
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

## SELECTOR HINTS:
- Trip type dropdown button: button[aria-label*="trip" i] (click to OPEN dropdown, then click option)
- Origin: input[aria-label*="Where from"]
- Destination: input[aria-label*="Where to"]
- Departure date: input[aria-label="Departure"]
- Return date: input[aria-label="Return"]
- Search button: button[aria-label="Search"]

## CRITICAL WARNINGS:
- NEVER click "Sign In" or any Google account link — stay on the flights form
- NEVER click "Explore" — that's not the Search button
- If you end up on /travel/flights/deals or /travel/flights/explore, navigate back to /travel/flights
- If a dropdown doesn't open after clicking, try scrolling up — the element may be partially hidden
