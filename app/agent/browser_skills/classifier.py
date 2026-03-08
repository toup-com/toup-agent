"""Fast category classifier — keyword matching with LLM fallback."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    subcategory: str = ""
    params: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Keyword rules — each list is ordered by specificity (multi-word first)
# ---------------------------------------------------------------------------

KEYWORD_RULES: Dict[str, List[str]] = {
    "flights": [
        "round trip", "one way", "one-way", "round-trip", "layover", "nonstop",
        "non-stop", "stopover", "airfare", "airline", "flight", "fly to",
        "fly from", "airport", "boarding pass", "plane ticket",
        # Common airport codes
        "yyz", "dxb", "jfk", "lax", "lhr", "cdg", "sfo", "ord", "atl",
        "ams", "hnd", "nrt", "icn", "sin", "hkg", "bkk", "ist", "muc",
        "fra", "bcn", "fco", "mxp", "iah", "dfw", "sea", "bos", "mia",
        "ewr", "phl", "den", "las", "phx", "msp", "dtw", "yvr", "yul",
    ],
    "hotels": [
        "check-in", "check-out", "checkin", "checkout", "bed and breakfast",
        "vacation rental", "all-inclusive", "all inclusive",
        "hotel", "motel", "hostel", "resort", "airbnb", "vrbo",
        "accommodation", "lodging", "stay at", "book a room", "room for",
        "nights stay",
    ],
    "car_rental": [
        "car rental", "rent a car", "rental car",
        "enterprise", "hertz", "avis", "budget rent",
    ],
    "shopping": [
        "add to cart", "buy online", "free shipping", "best deal",
        "price comparison", "compare prices", "coupon code",
        "buy", "purchase", "shop for", "shopping", "order from",
        "amazon", "walmart", "target", "best buy", "ebay", "etsy",
        "newegg", "wayfair", "costco",
        "cheapest", "best price", "deal on", "discount",
        "product", "gadget", "electronics",
    ],
    "food_delivery": [
        "uber eats", "ubereats", "doordash", "grubhub", "postmates",
        "instacart", "amazon fresh",
        "order food", "deliver food", "food delivery", "takeout",
        "take-out", "get food delivered", "order from restaurant",
    ],
    "restaurants": [
        "opentable", "resy", "reservation for dinner",
        "restaurant", "reservation", "book a table", "table for",
        "dine in", "dinner at", "lunch at", "brunch at",
    ],
    "recipes": [
        "recipe for", "how to cook", "how to bake", "how to make",
        "recipe", "ingredients for", "cooking time",
        "allrecipes", "food network", "epicurious",
    ],
    "real_estate": [
        "zillow", "redfin", "realtor.com", "trulia", "apartments.com",
        "real estate", "house for sale", "apartment for rent",
        "homes for sale", "property", "mortgage rate",
        "rent an apartment", "buy a house", "condo for",
        "apartment", "bedroom", "studio apartment",
    ],
    "jobs": [
        "linkedin jobs", "indeed", "glassdoor", "ziprecruiter",
        "job search", "job listing", "job opening", "career",
        "apply for a job", "find a job", "find me a job", "hiring",
        "resume", "job interview", "salary for", "job as",
        "software engineer", "developer job", "remote job",
    ],
    "healthcare": [
        "zocdoc", "webmd", "mychart", "goodrx",
        "doctor appointment", "find a doctor", "book appointment",
        "symptom", "medication", "prescription", "pharmacy",
        "hospital", "clinic", "dentist", "therapist",
        "health insurance",
    ],
    "finance": [
        "yahoo finance", "google finance", "nerdwallet", "bankrate",
        "stock price", "stock market", "invest in", "credit card",
        "bank account", "savings account", "loan rate", "mortgage",
        "insurance quote", "tax filing", "turbotax",
        "cryptocurrency", "bitcoin", "ethereum",
    ],
    "education": [
        "coursera", "udemy", "khan academy", "edx",
        "online course", "learn to", "tutorial for", "class on",
        "university", "college", "scholarship", "financial aid",
        "study for", "test prep", "duolingo",
    ],
    "entertainment": [
        "imdb", "rotten tomatoes", "justwatch", "netflix",
        "movie", "tv show", "watch", "stream", "trailer",
        "music", "spotify", "youtube", "podcast",
        "game", "steam", "playstation", "xbox", "nintendo",
        "book review", "goodreads",
    ],
    "events": [
        "ticketmaster", "stubhub", "eventbrite", "seatgeek",
        "concert ticket", "sports ticket", "event ticket",
        "buy tickets", "get tickets", "show near me",
        "festival", "conference", "expo",
    ],
    "government": [
        "irs.gov", "uscis", "ssa.gov", "dmv",
        "visa application", "passport renewal", "tax return",
        "government form", "social security", "voter registration",
        "business license", "building permit",
    ],
    "news": [
        "reuters", "ap news", "bbc news", "cnn", "nytimes",
        "breaking news", "latest news", "news about",
        "headline", "current events",
    ],
    "local_services": [
        "thumbtack", "angi", "taskrabbit", "homeadvisor",
        "plumber near me", "electrician near me", "mechanic near me",
        "house cleaning", "lawn care", "moving company",
        "near me", "local",
    ],
    "automotive": [
        "cars.com", "autotrader", "cargurus", "edmunds", "kbb",
        "carfax", "carvana", "truecar",
        "buy a car", "used car", "new car", "car dealer",
        "auto insurance", "vehicle", "truck",
    ],
    "legal": [
        "avvo", "findlaw", "legalzoom", "westlaw", "lexisnexis",
        "lawyer", "attorney", "legal advice", "court filing",
        "trademark", "patent", "copyright", "lawsuit",
    ],
    "technology": [
        "g2.com", "capterra", "producthunt", "techcrunch",
        "software comparison", "saas", "app review",
        "best software for", "tool for",
    ],
    "travel_general": [
        "trip to", "travel to", "visit", "vacation to",
        "itinerary", "things to do in", "attractions in",
        "sightseeing", "tour of",
    ],
}

# Parameter extraction patterns
AIRPORT_CODE_RE = re.compile(r'\b([A-Z]{3})\b')
DATE_RE = re.compile(
    r'(\d{1,2})\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    r'\s*,?\s*(\d{4})',
    re.IGNORECASE,
)
DATE_ISO_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')
DATE_SLASH_RE = re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4})')

MONTH_MAP = {
    "jan": "01", "january": "01", "feb": "02", "february": "02",
    "mar": "03", "march": "03", "apr": "04", "april": "04",
    "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "september": "09",
    "oct": "10", "october": "10", "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}


def _extract_dates(message: str) -> List[str]:
    """Extract dates from message, return as YYYY-MM-DD strings."""
    dates = []
    # ISO format
    for m in DATE_ISO_RE.finditer(message):
        dates.append(m.group(1))
    # "19 March 2026" format
    for m in DATE_RE.finditer(message):
        day = m.group(1).zfill(2)
        month = MONTH_MAP.get(m.group(2).lower(), "01")
        year = m.group(3)
        dates.append(f"{year}-{month}-{day}")
    return dates


# Common 3-letter English words that are NOT airport codes
_FALSE_AIRPORT_CODES = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "LET", "MAY",
    "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID", "GET", "HIM",
    "HIT", "HOT", "MAN", "RAN", "SIT", "TOP", "RED", "HAD", "SET", "BIG",
    "SAY", "SHE", "TOO", "USE", "DAD", "MOM", "DAY", "ANY", "FEW", "GOT",
    "SAT", "RUN", "EAT", "FAR", "END", "FLY", "TRY", "ASK", "MEN", "OWN",
    "PUT", "AGE", "BET", "SAN", "LOS", "BAD", "BUS", "CAR", "DOG", "CAT",
    "BUY", "PAY", "TAX", "JOB", "CUT", "FIT", "WIN", "WON", "BIT", "LOW",
}


def _extract_airports(message: str) -> List[str]:
    """Extract 3-letter airport codes from message, filtering out common English words."""
    candidates = AIRPORT_CODE_RE.findall(message.upper())
    return [c for c in candidates if c not in _FALSE_AIRPORT_CODES]


class CategoryClassifier:
    """Two-stage classifier: keyword matching → LLM fallback."""

    def classify(self, message: str) -> ClassificationResult:
        lower = message.lower()
        # Pre-tokenize for word-boundary matching
        words_set = set(re.findall(r'\b\w+\b', lower))

        # Score each category by keyword matches
        scores: Dict[str, float] = {}
        for category, keywords in KEYWORD_RULES.items():
            score = 0
            for kw in keywords:
                # Short keywords (≤4 chars) must match as whole words to avoid
                # false positives like "lax" in "galaxy" or "sea" in "search"
                if len(kw) <= 4 and " " not in kw:
                    if kw in words_set:
                        score += 1.0
                elif kw in lower:
                    weight = 1.5 if " " in kw else 1.0
                    score += weight
            if score > 0:
                scores[category] = score

        if not scores:
            return ClassificationResult(category="general", confidence=0.3)

        best = max(scores, key=scores.get)
        best_score = scores[best]

        # Normalize confidence (2+ keyword matches = high confidence)
        confidence = min(0.95, 0.5 + best_score * 0.15)

        # Extract parameters based on category
        params = {}
        if best == "flights":
            params = self._extract_flight_params(message)
        elif best in ("hotels", "restaurants"):
            dates = _extract_dates(message)
            if dates:
                params["dates"] = dates
        elif best == "travel_general":
            # Could be flights, hotels, or general travel — check for flight indicators
            airports = _extract_airports(message)
            if len(airports) >= 2:
                best = "flights"
                params = self._extract_flight_params(message)

        # Determine subcategory
        subcategory = self._get_subcategory(best, lower)

        return ClassificationResult(
            category=best,
            confidence=confidence,
            subcategory=subcategory,
            params=params,
        )

    def _extract_flight_params(self, message: str) -> Dict:
        params: Dict = {}
        airports = _extract_airports(message)
        if len(airports) >= 2:
            params["origin"] = airports[0]
            params["destination"] = airports[1]
        elif len(airports) == 1:
            params["destination"] = airports[0]

        dates = _extract_dates(message)
        if dates:
            params["departure_date"] = dates[0]
            if len(dates) >= 2:
                params["return_date"] = dates[1]
                params["trip_type"] = "round_trip"
            else:
                params["trip_type"] = "one_way"
        else:
            params["trip_type"] = "one_way"  # Default if no date found

        return params

    def _get_subcategory(self, category: str, lower_msg: str) -> str:
        if category == "flights":
            if "business class" in lower_msg or "first class" in lower_msg:
                return "premium"
            if "multi" in lower_msg and "city" in lower_msg:
                return "multi_city"
            return ""
        if category == "shopping":
            if "compare" in lower_msg:
                return "comparison"
            return ""
        if category == "food_delivery":
            if "grocer" in lower_msg:
                return "groceries"
            return "restaurant_delivery"
        return ""
