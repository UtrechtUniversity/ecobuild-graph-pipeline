"""Assert-based self-check for the start/stop/status state machine in service.py.

Run directly: `poetry run python -m crawler.test_service`
"""
import asyncio
import threading
import time
from unittest.mock import patch

from crawler import service


def _slow_crawl(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        time.sleep(0.05)


def _failing_crawl(stop_event: threading.Event) -> None:
    raise RuntimeError("boom")


def main() -> None:
    with patch("crawler.service.run_crawl", _slow_crawl):
        assert asyncio.run(service.get_status()) == {"status": "idle", "error": None}

        assert asyncio.run(service.start()) == {"status": "running", "error": None}
        assert asyncio.run(service.get_status()) == {"status": "running", "error": None}

        # Starting again while already running is a no-op.
        assert asyncio.run(service.start()) == {"status": "running", "error": None}

        assert asyncio.run(service.stop()) == {"status": "stopped", "error": None}
        service._thread.join(timeout=1)
        assert asyncio.run(service.get_status()) == {"status": "stopped", "error": None}

    # Reset shared state between scenarios.
    service._status = "idle"
    service._stop_event = threading.Event()

    with patch("crawler.service.run_crawl", _failing_crawl):
        assert asyncio.run(service.start()) == {"status": "running", "error": None}
        service._thread.join(timeout=1)
        status = asyncio.run(service.get_status())
        assert status == {"status": "error", "error": "boom"}, status

    print("crawler.service self-check passed")


if __name__ == "__main__":
    main()
