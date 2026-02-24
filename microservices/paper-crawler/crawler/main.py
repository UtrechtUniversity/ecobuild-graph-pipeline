import os
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
db_name = os.getenv("DB_NAME")
api_key = os.getenv('SS_API_KEY')

headers = {"x-api-key": api_key}
url = "https://api.semanticscholar.org/graph/v1/paper/search"

logger.info("Crawler initialized")


def handle_query(cursor: psycopg.Cursor, query: str):
    """Gathers all papers corresponding to the specified query"""
    logger.info(f"Handling output of query '{query}'")
    # (re)set offset as 0 to start at first results
    offset = 0

    while True:

        query_params = {
            "fields": "paperId,title,year,url,abstract,citationCount,isOpenAccess,openAccessPdf",
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
            logger.debug(f"Request failed. Status code: {response.status_code} ")


def write_to_db(cursor: psycopg.Cursor, query: str, paper: Dict) -> None:
    """Writes paper and current query to the document database"""
    template ="INSERT INTO papers (title, authors, url, abstract, pdf_url, keywords, query) values (%s, %s, %s, %s, %s, %s, %s, %s)"
    title = paper["title"]
    authors = paper["authors"]
    url = paper["url"]
    abstract = paper["abstract"]
    open_access = paper["isOpenAccess"]
    pdf_url = extract_pdf_url(paper["openAccessPdf"])

    cursor.execute(template, (title, authors, url, abstract, pdf_url, open_access, query))

def extract_pdf_url(input: str) -> str:
    """Extracts the pdf url from the openAccesPdf field"""
    raise NotImplementedError





if __name__ == "__main__":
    with psycopg.connect(f"dbname={db_name} user={db_user}") as connection:
        with connection.cursor() as cursor:

            for query in QUERIES:

                handle_query(cursor, query)

                # sleep to respect rate limit
                sleep(RATE_LIMIT)

                # Only one iteration
                # quit()

                    




