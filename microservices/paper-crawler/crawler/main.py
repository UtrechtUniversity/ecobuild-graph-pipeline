import os
import threading
from psycopg.errors import PipelineStatus
import requests
import psycopg

from time import sleep
from typing import Dict

from .config import QUERIES, RATE_LIMIT
from .crawler_logger import logger

# Initialize environment variables
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
api_key = os.getenv('SS_API_KEY')

headers = {"x-api-key": api_key}
url = "https://api.semanticscholar.org/graph/v1/paper/search"

logger.info("Crawler initialized")


def handle_query(cursor: psycopg.Cursor, query: str, stop_event: threading.Event) -> bool:
    """Gathers all papers corresponding to the specified query.

    Returns False if a request failed and the query had to be abandoned,
    True otherwise (including if stopped early).
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
        response = requests.get(url, params=query_params, headers=headers)

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
            return False

    return True


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
    with psycopg.connect(
        host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password
    ) as connection:
        with connection.cursor() as cursor:

            for query in QUERIES:
                if stop_event.is_set():
                    break

                if not handle_query(cursor, query, stop_event):
                    raise RuntimeError(f"Crawl failed on query '{query}'")

                # sleep to respect rate limit
                sleep(RATE_LIMIT)


if __name__ == "__main__":
    run_crawl(threading.Event())

                    




