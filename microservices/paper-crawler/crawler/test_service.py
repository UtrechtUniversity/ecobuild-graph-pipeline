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
        assert asyncio.run(service.get_status()) == {"status": "idle"}

        assert asyncio.run(service.start()) == {"status": "running"}
        assert asyncio.run(service.get_status()) == {"status": "running"}

        # Starting again while already running is a no-op.
        assert asyncio.run(service.start()) == {"status": "running"}

        assert asyncio.run(service.stop()) == {"status": "stopped"}
        service._thread.join(timeout=1)
        assert asyncio.run(service.get_status()) == {"status": "stopped"}

    # Reset shared state between scenarios.
    service._status = "idle"
    service._stop_event = threading.Event()

    with patch("crawler.service.run_crawl", _failing_crawl):
        assert asyncio.run(service.start()) == {"status": "running"}
        service._thread.join(timeout=1)
        assert asyncio.run(service.get_status()) == {"status": "error"}

    print("crawler.service self-check passed")


if __name__ == "__main__":
    main()
