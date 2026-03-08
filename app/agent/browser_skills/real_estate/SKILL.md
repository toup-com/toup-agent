---
name: real_estate
description: Search for homes, apartments, and real estate listings.
---

# Real Estate Search Skill

## SITE: Navigate to https://www.zillow.com or https://www.apartments.com

## PROCEDURE:
1. Navigate to Zillow (for sale) or Apartments.com (for rent)
2. Enter city, zip code, or neighborhood in the search bar
3. Apply filters: bedrooms, bathrooms, price range, home type
4. Browse listings in list view (easier to extract than map view)
5. For each listing extract: address, price, beds/baths, sqft, days on market
6. Click on top listings for more details if needed
7. Present top 5-10 options sorted by relevance

## FOR RENTALS:
- Use apartments.com or zillow.com/homes/for_rent
- Filter by: price range, bedrooms, pet-friendly, laundry, parking

## FOR BUYING:
- Use zillow.com or redfin.com
- Check: price, HOA fees, property tax, school ratings, walkability

## COMMON PITFALLS:
- Listings may be outdated — check "days on market" or listing date
- Price shown may not include HOA, utilities, or parking fees for rentals
- "Coming soon" listings can't be visited or applied for yet
