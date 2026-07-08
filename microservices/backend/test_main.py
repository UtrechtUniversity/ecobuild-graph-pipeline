"""Assert-based self-check for the papers/extraction logic in main.py.

Run directly: `poetry run python -m test_main`
"""
from datetime import datetime
from unittest.mock import patch

import main


class FakeCursor:
    """In-memory stand-in for a psycopg cursor, just enough for these queries."""

    def __init__(self, papers, runs=None):
        self._papers = papers
        self._runs = runs if runs is not None else []
        self._next_run_id = max((r["id"] for r in self._runs), default=0) + 1
        self._result = None

    def _latest_run(self, paper_id):
        matches = [r for r in self._runs if r["paper_id"] == paper_id]
        return matches[-1] if matches else None

    def execute(self, sql, params=()):
        sql = " ".join(sql.split())
        if sql.startswith("SELECT p.id, p.title"):
            rows = []
            for p in self._papers:
                run = self._latest_run(p["id"])
                rows.append((
                    p["id"], p["title"], p["authors"], p["query"], p["open_access"], p["pdf_url"],
                    run["status"] if run else None,
                    run["error"] if run else None,
                    run["started_at"] if run else None,
                ))
            self._result = rows
        elif sql.startswith("SELECT pdf_url"):
            match = next((p for p in self._papers if p["id"] == params[0]), None)
            self._result = (match["pdf_url"],) if match else None
        elif sql.startswith("INSERT INTO extraction_runs"):
            paper_id, status = params
            run_id = self._next_run_id
            self._next_run_id += 1
            self._runs.append({
                "id": run_id, "paper_id": paper_id, "status": status,
                "error": None, "raw_result": None, "started_at": datetime.now(), "finished_at": None,
            })
            self._result = (run_id,)
        elif sql.startswith("UPDATE extraction_runs"):
            status, error, raw_result, finished_at, run_id = params
            run = next(r for r in self._runs if r["id"] == run_id)
            run["status"] = status
            run["error"] = error
            if raw_result is not None:
                run["raw_result"] = raw_result.obj
            if finished_at is not None:
                run["finished_at"] = finished_at
        elif sql.startswith("SELECT raw_result FROM extraction_runs"):
            done_runs = [r for r in self._runs if r["paper_id"] == params[0] and r["status"] == "done"]
            self._result = (done_runs[-1]["raw_result"],) if done_runs else None

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
    # get_papers() reports "pending" for papers with no extraction run yet.
    cursor = FakeCursor([
        {"id": 1, "title": "A", "authors": ["X"], "query": "q", "open_access": True, "pdf_url": "http://x/a.pdf"},
    ])
    assert main.get_papers(cursor) == [{
        "id": 1, "title": "A", "authors": ["X"], "query": "q", "open_access": True,
        "pdf_url": "http://x/a.pdf", "extraction_status": "pending", "extraction_error": None,
        "extraction_started_at": None,
    }]

    # Happy path: download succeeds, extraction service accepts it -> "done",
    # and its result JSON is persisted to the (fake) database, not just held in memory.
    cursor = FakeCursor([{"id": 1, "title": "A", "authors": [], "query": "q", "open_access": True, "pdf_url": "http://x/a.pdf"}])
    run_id = main.create_run(cursor, 1, "downloading")
    with patch("main.get_db_connection", lambda: FakeConnection(cursor)), \
         patch("main.download_pdf", lambda url: b"%PDF-1.4 fake"), \
         patch("main.notify_extraction", lambda paper_id, pdf_bytes: {"Intro": [{"label": "x"}]}):
        main.run_extraction(run_id, 1)
    run = cursor._latest_run(1)
    assert run["status"] == "done"
    assert run["raw_result"] == {"Intro": [{"label": "x"}]}
    assert run["finished_at"] is not None

    # No pdf_url on record -> "failed" without ever calling the network.
    cursor = FakeCursor([{"id": 2, "title": "B", "authors": [], "query": "q", "open_access": False, "pdf_url": None}])
    run_id = main.create_run(cursor, 2, "downloading")
    with patch("main.get_db_connection", lambda: FakeConnection(cursor)):
        main.run_extraction(run_id, 2)
    assert cursor._latest_run(2)["status"] == "failed"

    # Download succeeds but knowledge-extraction is unreachable -> "failed", not "extracting" forever.
    def _unreachable(paper_id, pdf_bytes):
        raise ConnectionRefusedError("no such service")

    cursor = FakeCursor([{"id": 3, "title": "C", "authors": [], "query": "q", "open_access": True, "pdf_url": "http://x/c.pdf"}])
    run_id = main.create_run(cursor, 3, "downloading")
    with patch("main.get_db_connection", lambda: FakeConnection(cursor)), \
         patch("main.download_pdf", lambda url: b"%PDF-1.4 fake"), \
         patch("main.notify_extraction", _unreachable):
        main.run_extraction(run_id, 3)
    run = cursor._latest_run(3)
    assert run["status"] == "failed"
    assert "unreachable" in run["error"]

    print("backend papers/extraction self-check passed")


if __name__ == "__main__":
    main_()
