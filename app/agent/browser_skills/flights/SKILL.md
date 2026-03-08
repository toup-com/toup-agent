---
name: flights
description: Search and compare flights across airlines and OTAs.
---

# Flight Search Skill

## SITE: Navigate to https://www.google.com/travel/flights

## STEP-BY-STEP PROCEDURE:

### Step 1: Dismiss Overlays
If you see "Try AI powered Flight Deals" or any promotional overlay, click "Got it", "No thanks", or the X button. Wait for it to close.

### Step 2: Set Trip Type
Look for the trip type dropdown near the top of the form (says "Round trip" by default).
- If the extracted parameters say trip_type=one_way → click the dropdown → select "One way"
- If round_trip → leave as is
- The dropdown is a small button near the form — do NOT click hamburger/main menu

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
- Origin: input[aria-label*="Where from"]
- Destination: input[aria-label*="Where to"]
- Departure date: input[aria-label="Departure"]
- Return date: input[aria-label="Return"]
- Search button: button[aria-label="Search"]
