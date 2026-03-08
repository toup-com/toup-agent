---
name: shopping
description: Search, compare, and buy products online.
---

# Shopping Skill

## SITE SELECTION:
- General products → Amazon (amazon.com)
- Electronics → Best Buy (bestbuy.com) or Newegg (newegg.com)
- Home/furniture → Wayfair (wayfair.com) or IKEA (ikea.com)
- Price comparison → Google Shopping (google.com/shopping)
- Used/deals → eBay (ebay.com)
- If user specifies a store, go directly there

## STEP-BY-STEP PROCEDURE:

### Step 1: Navigate to Store
- Go directly to the appropriate store website
- If "cheapest" or "best deal" is mentioned, start with Google Shopping or Amazon

### Step 2: Search
- Click the search bar (usually at top of page)
- Type the product name or description
- Press Enter or click the search button

### Step 3: Filter Results
- If user specified constraints (price range, brand, rating), apply filters:
  - Price: look for price range filter (slider or min/max inputs)
  - Brand: checkbox filters in sidebar
  - Rating: "4 stars & up" filter
  - Prime/free shipping: toggle if applicable

### Step 4: Browse Results
- Scroll through product listings
- For each promising result, note: product name, price, rating, review count, seller
- If comparing, open the top 2-3 products for detailed specs

### Step 5: Product Details (if needed)
- Click on a product to see full details
- Extract: full name, price, specifications, availability, shipping info
- Check for active deals/coupons

### Step 6: Present Results
- Summarize the best options with prices, ratings, and key specs
- If comparing, present a side-by-side comparison
- Include direct links/product names for the user to find them

## COMMON PITFALLS:
- Amazon "Sponsored" results appear first — these may not be the best match
- Check "Ships from and sold by" — third-party sellers may have different return policies
- Some prices don't include shipping — check total cost
- "List price" vs actual price — the discount percentage can be misleading
- Electronics: check the model year — older models appear in results at lower prices

## SELECTOR HINTS (Amazon):
- Search bar: input#twotabsearchtextbox
- Search button: input#nav-search-submit-button
- Results: div[data-component-type="s-search-result"]
