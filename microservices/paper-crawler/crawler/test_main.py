"""Assert-based self-check for the search_queries DB helpers in main.py.

Run directly: `poetry run python -m crawler.test_main`
"""
from crawler.main import add_query, get_queries, remove_query


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


if __name__ == "__main__":
    main()
