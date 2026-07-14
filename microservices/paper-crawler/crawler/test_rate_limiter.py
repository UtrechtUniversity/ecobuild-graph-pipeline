"""Assert-based self-check for rate_limiter.py.

Run directly: `poetry run python -m crawler.test_rate_limiter`
"""
from unittest.mock import patch

from crawler.rate_limiter import EvenSpacing, FixedDelay


def test_fixed_delay_sleeps_the_configured_amount() -> None:
    limiter = FixedDelay(2.5)
    with patch("crawler.rate_limiter.sleep") as mock_sleep:
        limiter.wait()
    mock_sleep.assert_called_once_with(2.5)


def test_even_spacing_does_not_sleep_on_first_call() -> None:
    limiter = EvenSpacing(quota=10, period_seconds=100)  # interval = 10s
    with patch("crawler.rate_limiter.sleep") as mock_sleep:
        limiter.wait()
    mock_sleep.assert_not_called()


def test_even_spacing_sleeps_remaining_interval_on_next_call() -> None:
    limiter = EvenSpacing(quota=10, period_seconds=100)  # interval = 10s
    times = iter([0.0, 3.0, 3.0])  # __init__ doesn't call monotonic; wait() calls do
    with patch("crawler.rate_limiter.monotonic", side_effect=times), \
         patch("crawler.rate_limiter.sleep") as mock_sleep:
        limiter.wait()  # _last = 0.0, no sleep (first call)
        limiter.wait()  # elapsed = 3.0, sleeps remaining 7.0
    mock_sleep.assert_called_once_with(7.0)


def main() -> None:
    test_fixed_delay_sleeps_the_configured_amount()
    test_even_spacing_does_not_sleep_on_first_call()
    test_even_spacing_sleeps_remaining_interval_on_next_call()
    print("crawler.rate_limiter self-check passed")


if __name__ == "__main__":
    main()
