"""Assert-based self-check for crawler/main.py.

Run directly: `poetry run python -m crawler.test_main`
"""
import threading
from unittest.mock import Mock, patch

from crawler.main import (
    _as_list, _entry_url, _scopus_id, _to_bool, _to_int,
    add_query, get_queries, handle_query, remove_query, update_query, write_to_db,
)


class FakeCursor:
    """In-memory stand-in for a psycopg cursor, just enough for search_queries + papers + paper_queries writes."""

    def __init__(self):
        self._rows = []  # (id, query, source, design_strategy, ecosystem_service)
        self._next_id = 1
        self._result = None
        self._rowcount = 0
        self.papers_written = []
        self.paper_queries_written = []
        self._next_paper_id = 1
        self._last_paper_id = None

    def execute(self, sql, params=()):
        sql = sql.strip()
        if sql.startswith("SELECT id, query, design_strategy, ecosystem_service FROM search_queries"):
            source = params[0]
            self._result = [(r[0], r[1], r[3], r[4]) for r in self._rows if r[2] == source]
        elif sql.startswith("INSERT INTO search_queries"):
            query, source, design_strategy, ecosystem_service = params
            row = (self._next_id, query, source, design_strategy, ecosystem_service)
            self._rows.append(row)
            self._next_id += 1
            self._result = (row[0], row[1], row[3], row[4])
        elif sql.startswith("INSERT INTO papers"):
            self.papers_written.append(params)
            self._last_paper_id = self._next_paper_id
            self._next_paper_id += 1
        elif sql.startswith("SELECT id FROM papers"):
            self._result = (self._last_paper_id,)
        elif sql.startswith("INSERT INTO paper_queries"):
            self.paper_queries_written.append(params)
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


def test_search_queries_scoped_to_scopus_source() -> None:
    cursor = FakeCursor()

    assert get_queries(cursor) == []
    created = add_query(cursor, "green infrastructure runoff", "Green infrastructure", "Stormwater retention")
    assert created == {
        "id": 1, "query": "green infrastructure runoff",
        "design_strategy": "Green infrastructure", "ecosystem_service": "Stormwater retention",
    }
    assert get_queries(cursor) == [created]

    updated = {
        "id": 1, "query": "edited",
        "design_strategy": "Green infrastructure", "ecosystem_service": "Stormwater retention",
    }
    assert update_query(cursor, 1, "edited", "Green infrastructure", "Stormwater retention") == updated
    assert remove_query(cursor, 1) is True
    assert remove_query(cursor, 999) is False

    # a row belonging to another crawler's source is invisible to this one
    cursor._rows.append((9001, "semantic-scholar-only query", "semantic_scholar", None, None))
    assert all(q["id"] != 9001 for q in get_queries(cursor))
    assert remove_query(cursor, 9001) is False

    print("crawler.main search_queries self-check passed")


def test_helpers_normalize_scopus_json_quirks() -> None:
    # single-item collections come back as a bare dict, not a list
    assert _as_list(None) == []
    assert _as_list({"a": 1}) == [{"a": 1}]
    assert _as_list([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]

    assert _to_bool(None) is None
    assert _to_bool(True) is True
    assert _to_bool("true") is True
    assert _to_bool("false") is False

    assert _to_int(None) is None
    assert _to_int("") is None
    assert _to_int("42") == 42

    assert _scopus_id({"dc:identifier": "SCOPUS_ID:85012345678"}) == "85012345678"
    assert _scopus_id({"dc:identifier": "85012345678"}) == "85012345678"

    entry = {"link": [{"@ref": "self", "@href": "https://api.elsevier.com/x"}, {"@ref": "scopus", "@href": "https://scopus.com/y"}]}
    assert _entry_url(entry) == "https://scopus.com/y"
    assert _entry_url({"link": {"@ref": "scopus", "@href": "https://scopus.com/z"}}) == "https://scopus.com/z"
    assert _entry_url({}) is None

    print("crawler.main helpers self-check passed")


def test_write_to_db_prefers_full_abstract_authors_over_search_snippet() -> None:
    cursor = FakeCursor()
    entry = {
        "dc:identifier": "SCOPUS_ID:1", "dc:title": "A Paper", "dc:creator": "Doe J.",
        "prism:doi": "10.1/x", "prism:publicationName": "Journal of Things",
        "citedby-count": "3", "openaccessFlag": "true", "prism:coverDate": "2021-05-01",
        "link": [{"@ref": "scopus", "@href": "https://scopus.com/1"}],
    }
    abstract_response = {
        "coredata": {"dc:description": "Full abstract text."},
        "authors": {"author": {"preferred-name": {"ce:indexed-name": "Doe, John"}}},
    }
    write_to_db(cursor, 7, "test query", entry, abstract_response)

    assert len(cursor.papers_written) == 1
    source, external_id, title, authors, url, doi, venue, citation_count, year, abstract, pdf_url, open_access, query = cursor.papers_written[0]
    assert source == "scopus"
    assert external_id == "1"
    assert authors == ["Doe, John"]  # full name from abstract retrieval, not the "Doe J." snippet
    assert citation_count == 3
    assert year == 2021
    assert open_access is True
    assert pdf_url is None
    assert abstract == "Full abstract text."

    # the match is recorded against the resolved paper, linked to the query that found it
    assert cursor.paper_queries_written == [(cursor._last_paper_id, 7)]

    print("crawler.main write_to_db self-check passed")


def test_write_to_db_falls_back_to_search_snippet_when_abstract_unavailable() -> None:
    cursor = FakeCursor()
    entry = {"dc:identifier": "SCOPUS_ID:2", "dc:title": "B Paper", "dc:creator": "Roe J."}
    write_to_db(cursor, 7, "test query", entry, {})  # abstract retrieval failed/unentitled

    _, _, _, authors, *_ = cursor.papers_written[0]
    assert authors == ["Roe J."]

    print("crawler.main write_to_db fallback self-check passed")


def test_handle_query_paginates_and_rate_limits_search_and_abstract_calls() -> None:
    """Rate limit applies to both the search page requests and the per-paper abstract calls."""
    page_1 = Mock(status_code=200, json=lambda: {"search-results": {
        "opensearch:totalResults": "2",
        "entry": [{"dc:identifier": "SCOPUS_ID:1", "dc:title": "One"}],
    }})
    page_2 = Mock(status_code=200, json=lambda: {"search-results": {
        "opensearch:totalResults": "2",
        "entry": [{"dc:identifier": "SCOPUS_ID:2", "dc:title": "Two"}],
    }})
    abstract_response = Mock(status_code=200, json=lambda: {"abstracts-retrieval-response": {}})

    with patch("crawler.main.session.get", side_effect=[page_1, abstract_response, page_2, abstract_response]) as mock_get, \
         patch("crawler.main.rate_limiter.wait") as mock_wait, \
         patch("crawler.main.PAGE_SIZE", 1):  # forces a second page fetch with only 2 total results
        result = handle_query(FakeCursor(), 7, "test query", threading.Event())

    assert result is None
    assert mock_get.call_count == 4  # 2 search pages + 2 abstract lookups
    assert mock_wait.call_count == 4  # rate limiter guards every one of those calls

    print("crawler.main handle_query pagination/rate-limit self-check passed")


def test_handle_query_skips_empty_result_placeholder() -> None:
    empty = Mock(status_code=200, json=lambda: {"search-results": {
        "opensearch:totalResults": "0",
        "entry": [{"error": "Result set was empty"}],
    }})
    with patch("crawler.main.session.get", return_value=empty), \
         patch("crawler.main.rate_limiter.wait"):
        result = handle_query(FakeCursor(), 7, "no matches query", threading.Event())

    assert result is None
    print("crawler.main empty-results self-check passed")


def main() -> None:
    test_search_queries_scoped_to_scopus_source()
    test_helpers_normalize_scopus_json_quirks()
    test_write_to_db_prefers_full_abstract_authors_over_search_snippet()
    test_write_to_db_falls_back_to_search_snippet_when_abstract_unavailable()
    test_handle_query_paginates_and_rate_limits_search_and_abstract_calls()
    test_handle_query_skips_empty_result_placeholder()


if __name__ == "__main__":
    main()
