"""Assert-based self-check for the papers/extraction logic in main.py.

Run directly: `poetry run python -m test_main`
"""
from unittest.mock import patch

import main


class FakeCursor:
    """In-memory stand-in for a psycopg cursor, just enough for these queries."""

    def __init__(self, papers):
        self._papers = papers
        self._result = None

    def execute(self, sql, params=()):
        sql = sql.strip()
        if sql.startswith("SELECT id, title"):
            self._result = [
                (p["id"], p["title"], p["authors"], p["query"], p["open_access"], p["pdf_url"])
                for p in self._papers
            ]
        elif sql.startswith("SELECT pdf_url"):
            match = next((p for p in self._papers if p["id"] == params[0]), None)
            self._result = (match["pdf_url"],) if match else None

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    """Just enough of psycopg's Connection/cursor context-manager protocol for tests."""

    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self._cursor


def main_() -> None:
    # get_papers() reports "pending" for papers with no extraction attempt yet.
    main.extraction_status.clear()
    cursor = FakeCursor([
        {"id": 1, "title": "A", "authors": ["X"], "query": "q", "open_access": True, "pdf_url": "http://x/a.pdf"},
    ])
    assert main.get_papers(cursor) == [{
        "id": 1, "title": "A", "authors": ["X"], "query": "q", "open_access": True,
        "pdf_url": "http://x/a.pdf", "extraction_status": "pending", "extraction_error": None,
    }]

    # Happy path: download succeeds, extraction service accepts it -> "done",
    # and its result JSON is stored for later retrieval.
    main.extraction_status.clear()
    main.extraction_results.clear()
    cursor = FakeCursor([{"id": 1, "title": "A", "authors": [], "query": "q", "open_access": True, "pdf_url": "http://x/a.pdf"}])
    with patch("main.get_db_connection", lambda: FakeConnection(cursor)), \
         patch("main.download_pdf", lambda url: b"%PDF-1.4 fake"), \
         patch("main.notify_extraction", lambda paper_id, pdf_bytes: {"Intro": [{"label": "x"}]}):
        main.run_extraction(1)
    assert main.extraction_status[1] == {"status": "done", "error": None}
    assert main.extraction_results[1] == {"Intro": [{"label": "x"}]}

    # No pdf_url on record -> "failed" without ever calling the network.
    main.extraction_status.clear()
    cursor = FakeCursor([{"id": 2, "title": "B", "authors": [], "query": "q", "open_access": False, "pdf_url": None}])
    with patch("main.get_db_connection", lambda: FakeConnection(cursor)):
        main.run_extraction(2)
    assert main.extraction_status[2]["status"] == "failed"

    # Download succeeds but knowledge-extraction is unreachable -> "failed", not "extracting" forever.
    def _unreachable(paper_id, pdf_bytes):
        raise ConnectionRefusedError("no such service")

    main.extraction_status.clear()
    cursor = FakeCursor([{"id": 3, "title": "C", "authors": [], "query": "q", "open_access": True, "pdf_url": "http://x/c.pdf"}])
    with patch("main.get_db_connection", lambda: FakeConnection(cursor)), \
         patch("main.download_pdf", lambda url: b"%PDF-1.4 fake"), \
         patch("main.notify_extraction", _unreachable):
        main.run_extraction(3)
    assert main.extraction_status[3]["status"] == "failed"
    assert "unreachable" in main.extraction_status[3]["error"]

    print("backend papers/extraction self-check passed")


if __name__ == "__main__":
    main_()
