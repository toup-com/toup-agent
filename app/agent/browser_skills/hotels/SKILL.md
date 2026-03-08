---
name: hotels
description: Search and compare hotels and accommodation.
---

# Hotel Search Skill

## SITE: Navigate to https://www.google.com/travel/hotels

## STEP-BY-STEP PROCEDURE:

### Step 1: Set Destination
- Click the search/destination field
- Type the city or area name
- Wait for autocomplete → select the correct suggestion

### Step 2: Set Check-in Date
- Click the check-in date field
- Use `select_date` tool with YYYY-MM-DD format
- The calendar should appear — confirm the date

### Step 3: Set Check-out Date
- Click the check-out date field (or it may auto-open after check-in)
- Use `select_date` with check-out date
- Click "Done" to confirm dates

### Step 4: Set Guests (if not default)
- Click the guests/rooms selector
- Adjust adults, children, rooms using +/- buttons
- Click "Done" or close the dropdown

### Step 5: Search
- Click "Search" or the results should auto-load
- Wait for hotel listings to appear

### Step 6: Read Results
- Scroll through listings
- Extract: hotel name, price per night, total price, star rating, guest rating, location
- Note cancellation policies (free cancellation vs non-refundable)
- Present top 5 options sorted by value (considering price + rating)

## ALTERNATIVE SITES:
- **Booking.com**: Navigate to booking.com, use the search form (destination → dates → guests → search)
- **Airbnb**: Navigate to airbnb.com for vacation rentals (different UI — uses map-based search)

## COMMON PITFALLS:
- Some hotel sites show prices per night, others show total — clarify in results
- "Taxes and fees" are often excluded from displayed price
- Cancellation policy affects price — non-refundable is cheaper but risky
- Map view may load by default — switch to list view for easier extraction
