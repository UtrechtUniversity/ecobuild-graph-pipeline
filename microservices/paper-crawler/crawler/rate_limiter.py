"""Throttling strategies for crawlers hitting rate/quota-limited APIs.

Each crawler picks the strategy matching its API's quota shape and calls
.wait() before every outbound request.
"""
from time import monotonic, sleep
from typing import Protocol


class RateLimiter(Protocol):
    def wait(self) -> None:
        """Blocks as needed before the next request goes out."""
        ...


class FixedDelay:
    """Sleeps a fixed number of seconds after every request.

    Fits a roughly-per-second quota (e.g. Semantic Scholar).
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def wait(self) -> None:
        sleep(self.seconds)


class EvenSpacing:
    """Spreads a quota evenly across a period (e.g. Scopus's ~20k requests/week).

    # ponytail: even spacing, not a true token bucket with burst capacity —
    # good enough for a background crawl that isn't racing anything. Add
    # burst support if that changes.
    """

    def __init__(self, quota: int, period_seconds: float) -> None:
        self.interval = period_seconds / quota
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.interval - (monotonic() - self._last)
            if remaining > 0:
                sleep(remaining)
        self._last = monotonic()
