"""Assert-based self-check for the search_queries DB helpers in main.py.

Run directly: `poetry run python -m crawler.test_main`
"""
import threading
from unittest.mock import Mock, patch

from crawler.main import SOURCE, add_query, get_queries, handle_query, record_match, remove_query, update_query


class FakeCursor:
    """In-memory stand-in for a psycopg cursor, just enough for these three queries."""

    def __init__(self):
        self._rows = []  # (id, query, source, design_strategy, ecosystem_service)
        self._next_id = 1
        self._result = None
        self._rowcount = 0

    def execute(self, sql, params=()):
        sql = sql.strip()
        if sql.startswith("SELECT"):
            source = params[0]
            self._result = [(r[0], r[1], r[3], r[4]) for r in self._rows if r[2] == source]
        elif sql.startswith("INSERT"):
            query, source, design_strategy, ecosystem_service = params
            row = (self._next_id, query, source, design_strategy, ecosystem_service)
            self._rows.append(row)
            self._next_id += 1
            self._result = (row[0], row[1], row[3], row[4])
        elif sql.startswith("DELETE"):
            query_id, source = params
            before = len(self._rows)
            self._rows = [r for r in self._rows if not (r[0] == query_id and r[2] == source)]
            self._rowcount = before - len(self._rows)
        elif sql.startswith("UPDATE"):
            query, design_strategy, ecosystem_service, query_id, source = params
            for i, row in enumerate(self._rows):
                if row[0] == query_id and row[2] == source:
                    self._rows[i] = (query_id, query, source, design_strategy, ecosystem_service)
                    self._result = (query_id, query, design_strategy, ecosystem_service)
                    return
            self._result = None

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

    created = add_query(cursor, "Green roof effect on evaporation", "Green roof", "Evaporative cooling")
    assert created == {
        "id": 1, "query": "Green roof effect on evaporation",
        "design_strategy": "Green roof", "ecosystem_service": "Evaporative cooling",
    }
    assert get_queries(cursor) == [created]

    add_query(cursor, "rainwater harvesting effectiveness morocco")
    assert len(get_queries(cursor)) == 2

    assert remove_query(cursor, 1) is True
    assert get_queries(cursor) == [{
        "id": 2, "query": "rainwater harvesting effectiveness morocco",
        "design_strategy": None, "ecosystem_service": None,
    }]

    assert remove_query(cursor, 999) is False

    add_query(cursor, "original text")
    row_id = get_queries(cursor)[-1]["id"]
    updated = {
        "id": row_id, "query": "edited text",
        "design_strategy": "Rainwater cistern", "ecosystem_service": "Water availability",
    }
    assert update_query(cursor, row_id, "edited text", "Rainwater cistern", "Water availability") == updated
    assert get_queries(cursor)[-1] == updated
    assert update_query(cursor, 999, "nope") is None

    # a row belonging to another crawler's source is invisible to this one
    cursor._rows.append((9001, "scopus-only query", "scopus", None, None))
    assert all(q["id"] != 9001 for q in get_queries(cursor))
    assert remove_query(cursor, 9001) is False
    assert update_query(cursor, 9001, "nope") is None

    print("crawler.main search_queries self-check passed")


def test_handle_query_sleeps_between_every_request() -> None:
    """Rate limit is cumulative across all requests, including paginated ones."""
    pages = [
        Mock(status_code=200, json=lambda: {"total": 2, "data": [], "token": "page2"}),
        Mock(status_code=200, json=lambda: {"total": 2, "data": []}),  # no "token": last page
    ]
    with patch("crawler.main.session.get", side_effect=pages) as mock_get, \
         patch("crawler.main.rate_limiter.wait") as mock_wait:
        result = handle_query(Mock(), 1, "test query", threading.Event())

    assert result is None
    assert mock_get.call_count == 2
    assert mock_wait.call_count == 2
    assert "token" not in mock_get.call_args_list[0].kwargs["params"]
    assert mock_get.call_args_list[1].kwargs["params"]["token"] == "page2"

    print("crawler.main handle_query rate-limit self-check passed")


def test_handle_query_tolerates_zero_results() -> None:
    """S2 omits the "data" field entirely (not data: []) when a query has zero hits."""
    page = Mock(status_code=200, json=lambda: {"total": 0})
    with patch("crawler.main.session.get", return_value=page), \
         patch("crawler.main.rate_limiter.wait"):
        result = handle_query(Mock(), 1, "no hits query", threading.Event())

    assert result is None

    print("crawler.main handle_query zero-results self-check passed")


class FakePapersCursor:
    """Minimal stand-in for testing record_match's resolve-then-link SQL pattern."""

    def __init__(self, existing_paper_id: int):
        self.existing_paper_id = existing_paper_id
        self.executed = []

    def execute(self, sql, params=()):
        self.executed.append((sql.strip(), params))

    def fetchone(self):
        return (self.existing_paper_id,)


def test_record_match_links_query_to_resolved_paper_id() -> None:
    """Whatever paper_id the dedup SELECT resolves to (new insert or pre-existing
    row from a doi/source+external_id conflict) is what gets linked to the query."""
    cursor = FakePapersCursor(existing_paper_id=42)
    record_match(cursor, query_id=7, doi="10.1/x", external_id="abc")

    select_sql, select_params = cursor.executed[0]
    assert select_sql.startswith("SELECT id FROM papers")
    assert select_params == ("10.1/x", SOURCE, "abc")

    insert_sql, insert_params = cursor.executed[1]
    assert insert_sql.startswith("INSERT INTO paper_queries")
    assert insert_params == (42, 7)

    print("crawler.main record_match self-check passed")


if __name__ == "__main__":
    main()
    test_handle_query_sleeps_between_every_request()
    test_handle_query_tolerates_zero_results()
    test_record_match_links_query_to_resolved_paper_id()
