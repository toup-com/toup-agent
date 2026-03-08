---
name: food_delivery
description: Order food delivery or groceries online.
---

# Food Delivery Skill

## SITE SELECTION:
- Restaurant delivery → Uber Eats (ubereats.com) or DoorDash (doordash.com)
- Groceries → Instacart (instacart.com) or Amazon Fresh
- If user specifies a platform, use that one

## STEP-BY-STEP PROCEDURE:

### Step 1: Set Delivery Address
- The site will ask for delivery address — enter the user's location
- If address is not specified by the user, note this and proceed with the site's default or ask

### Step 2: Browse or Search
- If user wants a specific cuisine or restaurant: use the search bar
- If user wants to browse: look at "Popular near you" or category filters
- Apply filters: cuisine type, price range, delivery time, rating

### Step 3: Select Restaurant
- Click on the restaurant that best matches the user's request
- Check: delivery time, delivery fee, minimum order, rating

### Step 4: Select Items
- Browse the menu
- Click on items to add them
- Handle customization (size, toppings, sides, special instructions)
- Adjust quantities if needed

### Step 5: Review Order
- Check the cart/order summary
- Verify: items, quantities, subtotal, delivery fee, taxes, tip
- Apply promo code if the user has one

### Step 6: Present Summary
- Show the complete order with total price before proceeding to checkout
- Include: items, prices, delivery fee, estimated delivery time

## COMMON PITFALLS:
- Most delivery apps require login — the user may need to authenticate
- Menu items may be unavailable (grayed out or showing "Sold out")
- Delivery fees vary by distance and demand (surge pricing)
- Minimum order amounts — check before adding items
- Some items have required customizations (must select size, protein, etc.)
