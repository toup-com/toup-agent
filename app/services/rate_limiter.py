"""
In-memory rate limiter for login attempts.

Tracks failed login attempts by IP + email combination.
Allows 5 attempts per 5-minute window. Resets on successful login.
"""

import time
from collections import defaultdict
from typing import Optional


class LoginRateLimiter:
    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, list[float]] = defaultdict(list)

    def _key(self, ip: str, email: str) -> str:
        return f"{ip}:{email.lower()}"

    def check(self, ip: str, email: str) -> Optional[int]:
        """Returns None if allowed, or seconds until retry if rate-limited."""
        key = self._key(ip, email)
        now = time.time()
        cutoff = now - self.window_seconds
        self._attempts[key] = [t for t in self._attempts[key] if t > cutoff]
        if len(self._attempts[key]) >= self.max_attempts:
            oldest = self._attempts[key][0]
            return int(oldest + self.window_seconds - now) + 1
        return None

    def record(self, ip: str, email: str):
        """Record a failed login attempt."""
        self._attempts[self._key(ip, email)].append(time.time())

    def clear(self, ip: str, email: str):
        """Clear attempts on successful login."""
        self._attempts.pop(self._key(ip, email), None)


login_rate_limiter = LoginRateLimiter()
