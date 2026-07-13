import os
import threading
from psycopg.errors import PipelineStatus
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import psycopg

from time import sleep
from typing import Dict

from .config import RATE_LIMIT
from .crawler_logger import logger

# Initialize environment variables
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
api_key = os.getenv('SS_API_KEY')

# search_queries is shared across crawlers (one row per source); this crawler
# only ever touches its own rows.
SOURCE = "semantic_scholar"


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

headers = {"x-api-key": api_key}
url = "https://api.semanticscholar.org/graph/v1/paper/search"

# ponytail: urllib3's Retry already honors S2's Retry-After header on 429s,
# so a plain Session+adapter covers backoff without hand-rolled retry logic.
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503],
    raise_on_status=False,  # return the failing response instead of raising, so existing status_code handling still applies
)))

logger.info("Crawler initialized")


def handle_query(cursor: psycopg.Cursor, query: str, stop_event: threading.Event) -> int | None:
    """Gathers all papers corresponding to the specified query.

    Returns the failing HTTP status code if a request failed and the query
    had to be abandoned, None otherwise (including if stopped early).
    """
    logger.info(f"Handling output of query '{query}'")
    # (re)set offset as 0 to start at first results
    offset = 0

    while not stop_event.is_set():

        query_params = {
            "fields": "paperId,title,year,url,abstract,citationCount,isOpenAccess,openAccessPdf,authors",
            "query": query,
            "offset": offset,
        }

        # send request
        response = session.get(url, params=query_params, headers=headers, timeout=30)
        # rate limit is cumulative across all requests, so sleep after every one (including the last page)
        sleep(RATE_LIMIT)

        if response.status_code == 200:
            logger.debug(f"Request succesful")

            response = response.json()
            logger.debug(response)
            total_responses = response['total']
            logger.debug(f"Found {total_responses} responses")

            for paper in response["data"]: 
                # store in DB
                logger.debug(f"Found paper: {paper['title']}")
                write_to_db(cursor, query, paper)

            # semantic scholar API should provide us with a 'next' if there are more pages left
            try:  
                offset = response["next"]
            except KeyError: # if there are no more pages left, excit loop for this query
                break
        else:
            logger.error(f"Request failed, giving up on query '{query}'. Status code: {response.status_code} ")
            return response.status_code

    return None


def write_to_db(cursor: psycopg.Cursor, query: str, paper: Dict) -> None:
    """Writes paper and current query to the document database"""
    template = (
        "INSERT INTO papers (ss_id, title, authors, url, abstract, pdf_url, open_access, query) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (ss_id) DO NOTHING"
    )
    ss_id = paper["paperId"]
    title = paper["title"]
    authors = [author["name"] for author in paper.get("authors") or []]
    url = paper["url"]
    abstract = paper["abstract"]
    open_access = paper["isOpenAccess"]
    pdf_url = extract_pdf_url(paper.get("openAccessPdf"))

    cursor.execute(template, (ss_id, title, authors, url, abstract, pdf_url, open_access, query))

def extract_pdf_url(open_access_pdf: dict | None) -> str | None:
    """Extracts the pdf url from the openAccessPdf field, if present"""
    if not open_access_pdf:
        return None
    return open_access_pdf.get("url")


def run_crawl(stop_event: threading.Event) -> None:
    """Runs one full crawl pass over all configured queries.

    Cooperatively exits early if `stop_event` is set, so callers can run this
    on a background thread and stop it from the outside.
    """
    with get_connection() as connection:
        with connection.cursor() as cursor:
            queries = [q["query"] for q in get_queries(cursor)]

            for query in queries:
                if stop_event.is_set():
                    break

                failure_status = handle_query(cursor, query, stop_event)
                if failure_status == 429:
                    raise RuntimeError("rate limiting")
                elif failure_status is not None:
                    raise RuntimeError(f"request failed (HTTP {failure_status})")


if __name__ == "__main__":
    run_crawl(threading.Event())

                    




