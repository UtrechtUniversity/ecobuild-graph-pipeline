"""Assert-based self-check for the search_queries DB helpers in main.py.

Run directly: `poetry run python -m crawler.test_main`
"""
import threading
from unittest.mock import Mock, patch

from crawler.main import add_query, get_queries, handle_query, remove_query


class FakeCursor:
    """In-memory stand-in for a psycopg cursor, just enough for these three queries."""

    def __init__(self):
        self._rows = []
        self._next_id = 1
        self._result = None
        self._rowcount = 0

    def execute(self, sql, params=()):
        sql = sql.strip()
        if sql.startswith("SELECT"):
            self._result = list(self._rows)
        elif sql.startswith("INSERT"):
            row = (self._next_id, params[0])
            self._rows.append(row)
            self._next_id += 1
            self._result = row
        elif sql.startswith("DELETE"):
            query_id = params[0]
            before = len(self._rows)
            self._rows = [r for r in self._rows if r[0] != query_id]
            self._rowcount = before - len(self._rows)

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result

    @property
    def rowcount(self):
        return self._rowcount


def main() -> None:
    cursor = FakeCursor()

    assert get_queries(cursor) == []

    created = add_query(cursor, "Green roof effect on evaporation")
    assert created == {"id": 1, "query": "Green roof effect on evaporation"}
    assert get_queries(cursor) == [{"id": 1, "query": "Green roof effect on evaporation"}]

    add_query(cursor, "rainwater harvesting effectiveness morocco")
    assert len(get_queries(cursor)) == 2

    assert remove_query(cursor, 1) is True
    assert get_queries(cursor) == [{"id": 2, "query": "rainwater harvesting effectiveness morocco"}]

    assert remove_query(cursor, 999) is False

    print("crawler.main search_queries self-check passed")


def test_handle_query_sleeps_between_every_request() -> None:
    """Rate limit is cumulative across all requests, including paginated ones."""
    pages = [
        Mock(status_code=200, json=lambda: {"total": 2, "data": [], "next": 1}),
        Mock(status_code=200, json=lambda: {"total": 2, "data": []}),  # no "next": last page
    ]
    with patch("crawler.main.session.get", side_effect=pages) as mock_get, \
         patch("crawler.main.sleep") as mock_sleep:
        result = handle_query(Mock(), "test query", threading.Event())

    assert result is None
    assert mock_get.call_count == 2
    assert mock_sleep.call_count == 2

    print("crawler.main handle_query rate-limit self-check passed")


if __name__ == "__main__":
    main()
    test_handle_query_sleeps_between_every_request()
