import os
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import psycopg

from typing import Dict, Optional

from .config import QUOTA_PERIOD_SECONDS, WEEKLY_QUOTA
from .crawler_logger import logger
from .rate_limiter import EvenSpacing

# Initialize environment variables
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
api_key = os.getenv("SCOPUS_API_KEY")
insttoken = os.getenv("SCOPUS_INSTTOKEN")

# search_queries is shared across crawlers (one row per source); this crawler
# only ever touches its own rows.
SOURCE = "scopus"


def get_connection() -> psycopg.Connection:
    return psycopg.connect(
        host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password
    )


def get_queries(cursor: psycopg.Cursor) -> list[dict]:
    """Returns this crawler's configured search queries as {id, query} dicts."""
    cursor.execute("SELECT id, query FROM search_queries WHERE source = %s ORDER BY id", (SOURCE,))
    return [{"id": row[0], "query": row[1]} for row in cursor.fetchall()]


def add_query(cursor: psycopg.Cursor, query: str) -> dict:
    cursor.execute(
        "INSERT INTO search_queries (query, source) VALUES (%s, %s) RETURNING id, query",
        (query, SOURCE),
    )
    row = cursor.fetchone()
    return {"id": row[0], "query": row[1]}


def remove_query(cursor: psycopg.Cursor, query_id: int) -> bool:
    """Returns True if a query was deleted, False if query_id didn't exist."""
    cursor.execute("DELETE FROM search_queries WHERE id = %s AND source = %s", (query_id, SOURCE))
    return cursor.rowcount > 0


def update_query(cursor: psycopg.Cursor, query_id: int, query: str) -> dict | None:
    """Returns the updated {id, query}, or None if query_id didn't exist."""
    cursor.execute(
        "UPDATE search_queries SET query = %s WHERE id = %s AND source = %s RETURNING id, query",
        (query, query_id, SOURCE),
    )
    row = cursor.fetchone()
    return {"id": row[0], "query": row[1]} if row else None


headers = {"X-ELS-APIKey": api_key or "", "Accept": "application/json"}
if insttoken:
    headers["X-ELS-Insttoken"] = insttoken

SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
ABSTRACT_URL = "https://api.elsevier.com/content/abstract/scopus_id/{}"
PAGE_SIZE = 25

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503],
    raise_on_status=False,  # return the failing response instead of raising, so existing status_code handling still applies
)))

# Scopus's quota is weekly (not per-second like S2), so it needs a different
# throttling shape — see rate_limiter.EvenSpacing.
rate_limiter = EvenSpacing(WEEKLY_QUOTA, QUOTA_PERIOD_SECONDS)

logger.info("Scopus crawler initialized")


def _as_list(value):
    """Scopus's JSON API collapses single-item lists to a bare object — normalize back to a list."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _to_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _to_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _scopus_id(entry: Dict) -> str:
    """Strips the 'SCOPUS_ID:' prefix dc:identifier carries."""
    identifier = entry["dc:identifier"]
    return identifier.split(":", 1)[1] if ":" in identifier else identifier


def _entry_url(entry: Dict) -> Optional[str]:
    for link in _as_list(entry.get("link")):
        if link.get("@ref") == "scopus":
            return link.get("@href")
    return None


def fetch_abstract(external_id: str) -> Dict:
    """Fetches the Abstract Retrieval record for one paper (full abstract + full author list).

    Returns {} if the call fails (e.g. missing entitlement) — callers fall back
    to what the search result already had.
    """
    rate_limiter.wait()
    response = session.get(ABSTRACT_URL.format(external_id), headers=headers, timeout=30)
    if response.status_code != 200:
        logger.warning(f"Abstract retrieval failed for {external_id}: HTTP {response.status_code}")
        return {}
    return response.json().get("abstracts-retrieval-response", {})


def _abstract_text(abstract_response: Dict) -> Optional[str]:
    return (abstract_response.get("coredata") or {}).get("dc:description")


def _abstract_authors(abstract_response: Dict) -> list[str]:
    names = []
    for author in _as_list((abstract_response.get("authors") or {}).get("author")):
        preferred = author.get("preferred-name") or {}
        name = preferred.get("ce:indexed-name") or author.get("ce:indexed-name")
        if name:
            names.append(name)
    return names


def handle_query(cursor: psycopg.Cursor, query_id: int, query: str, stop_event: threading.Event) -> int | None:
    """Gathers all papers corresponding to the specified query.

    Returns the failing HTTP status code if a request failed and the query
    had to be abandoned, None otherwise (including if stopped early).
    """
    logger.info(f"Handling output of query '{query}'")
    start = 0

    while not stop_event.is_set():
        query_params = {"query": query, "start": start, "count": PAGE_SIZE}

        rate_limiter.wait()
        response = session.get(SEARCH_URL, params=query_params, headers=headers, timeout=30)

        if response.status_code != 200:
            logger.error(f"Request failed, giving up on query '{query}'. Status code: {response.status_code}")
            return response.status_code

        results = response.json().get("search-results", {})
        total = _to_int(results.get("opensearch:totalResults")) or 0
        entries = _as_list(results.get("entry"))

        for entry in entries:
            if stop_event.is_set():
                break
            if "dc:identifier" not in entry:
                continue  # Scopus returns a single {"error": "..."} entry when a query matches nothing
            logger.debug(f"Found paper: {entry.get('dc:title')}")
            abstract_response = fetch_abstract(_scopus_id(entry))
            write_to_db(cursor, query_id, query, entry, abstract_response)

        start += PAGE_SIZE
        if start >= total or not entries:
            break

    return None


def write_to_db(cursor: psycopg.Cursor, query_id: int, query: str, entry: Dict, abstract_response: Dict) -> None:
    """Writes a search result, enriched with its abstract-retrieval data, to the document database."""
    template = (
        "INSERT INTO papers (source, external_id, title, authors, url, doi, venue, citation_count, abstract, pdf_url, open_access, query) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"
    )
    external_id = _scopus_id(entry)
    title = entry.get("dc:title")
    authors = _abstract_authors(abstract_response) or ([entry["dc:creator"]] if entry.get("dc:creator") else [])
    url = _entry_url(entry)
    doi = entry.get("prism:doi")
    venue = entry.get("prism:publicationName")
    citation_count = _to_int(entry.get("citedby-count"))
    abstract = _abstract_text(abstract_response)
    open_access = _to_bool(entry.get("openaccessFlag"))
    pdf_url = None  # Scopus is an abstract/citation index, not a full-text host — see README

    cursor.execute(template, (
        SOURCE, external_id, title, authors, url, doi, venue, citation_count, abstract, pdf_url, open_access, query,
    ))
    record_match(cursor, query_id, doi, external_id)


def record_match(cursor: psycopg.Cursor, query_id: int, doi: str | None, external_id: str) -> None:
    """Links this query to whichever papers row now represents it — the one just
    inserted, or (if a doi/source+external_id conflict dropped the insert) the
    pre-existing one — so per-query match counts stay accurate even for papers
    a different source (or an earlier query) already found."""
    cursor.execute(
        "SELECT id FROM papers WHERE (doi IS NOT NULL AND doi = %s) OR (source = %s AND external_id = %s) LIMIT 1",
        (doi, SOURCE, external_id),
    )
    paper_id = cursor.fetchone()[0]
    cursor.execute(
        "INSERT INTO paper_queries (paper_id, query_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (paper_id, query_id),
    )


def run_crawl(stop_event: threading.Event) -> None:
    """Runs one full crawl pass over all configured queries.

    Cooperatively exits early if `stop_event` is set, so callers can run this
    on a background thread and stop it from the outside.
    """
    with get_connection() as connection:
        with connection.cursor() as cursor:
            queries = get_queries(cursor)

            for q in queries:
                if stop_event.is_set():
                    break

                failure_status = handle_query(cursor, q["id"], q["query"], stop_event)
                connection.commit()
                if failure_status is not None:
                    raise RuntimeError(f"request failed (HTTP {failure_status})")


if __name__ == "__main__":
    run_crawl(threading.Event())
