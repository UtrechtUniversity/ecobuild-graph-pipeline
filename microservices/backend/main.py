# main.py

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException # Import HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Literal
from uuid import uuid4
from datetime import datetime, timedelta
import asyncio
import os
import json
import shutil
import urllib.request
import urllib.error
import psycopg
from psycopg.types.json import Jsonb
import neo4j

# Import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS middleware
origins = [
    "http://localhost:3000",  # Allow your frontend's origin
    "http://127.0.0.1:3000", # Sometimes localhost resolves to 127.0.0.1, good to include both
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of allowed origins
    allow_credentials=True,         # Allow cookies to be included in cross-origin HTTP requests
    allow_methods=["*"],            # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],            # Allow all headers
)

# In-memory experiment registry (replace with Postgres later)
experiment_queue = []
running_experiment = None
experiment_results = {}

EXPERIMENTS_DIR = "experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True) # Ensure the directory exists

class ExperimentMetadata(BaseModel):
    id: str
    status: str  # queued, running, completed
    created_at: datetime
    config_path: str
    result_path: str | None = None

# --- DUMMY DATA INITIALIZATION FUNCTIONS ---
# (Keep these as they were)
def create_dummy_experiment_files(uid: str, config_content: dict, result_content: dict | None = None):
    config_path = os.path.join(EXPERIMENTS_DIR, f"{uid}_config.json")
    result_path = os.path.join(EXPERIMENTS_DIR, f"{uid}_result.json")

    with open(config_path, "w") as f:
        json.dump(config_content, f, indent=2)

    if result_content:
        with open(result_path, "w") as f:
            json.dump(result_content, f, indent=2)
    return config_path, result_path

def initialize_dummy_data():
    global experiment_queue, running_experiment, experiment_results

    # Clear previous state if server reloads to avoid duplicates in memory
    experiment_queue = []
    running_experiment = None
    experiment_results = {}

    print(f"[{datetime.now().isoformat()}] Initializing dummy data...")

    # 1. Past Experiments (completed)
    past_id_1 = str(uuid4())
    past_config_1 = {
        "model": "KnowledgeBERT_v1.0",
        "schema": "Neo4j_Schema_Paper_Author",
        "prompt_template": "template_alpha",
        "data_source": "arXiv_2024_Q1",
        "max_relations": 10
    }
    past_result_1 = {
        "id": past_id_1,
        "finished_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
        "config": past_config_1,
        "metrics": {
            "precision": 0.88,
            "recall": 0.82,
            "f1_score": 0.85,
            "nodes_extracted": 1500,
            "edges_extracted": 3200,
            "runtime_seconds": 3600
        },
        "graph": {
            "nodes": [
                {"id": "paper1", "labels": ["Paper"], "properties": {"title": "Scalable Knowledge Graph Extraction"}},
                {"id": "author1", "labels": ["Author"], "properties": {"name": "Alice Smith"}},
                {"id": "concept1", "labels": ["Concept"], "properties": {"name": "Knowledge Graph"}},
                {"id": "concept2", "labels": ["Concept"], "properties": {"name": "Natural Language Processing"}}
            ],
            "relationships": [
                {"id": "rel1", "startNode": "paper1", "endNode": "author1", "type": "AUTHORED_BY"},
                {"id": "rel2", "startNode": "paper1", "endNode": "concept1", "type": "MENTIONS"},
                {"id": "rel3", "startNode": "concept1", "endNode": "concept2", "type": "RELATED_TO"}
            ]
        }
    }
    create_dummy_experiment_files(past_id_1, past_config_1, past_result_1)
    experiment_results[past_id_1] = past_result_1
    print(f"Added past experiment: {past_id_1}")

    past_id_2 = str(uuid4())
    past_config_2 = {
        "model": "FineTuneBERT_v0.5",
        "schema": "Simple_Schema_Entities",
        "prompt_template": "template_beta",
        "data_source": "PubMed_Abstracts_2023",
        "max_entities": 200
    }
    past_result_2 = {
        "id": past_id_2,
        "finished_at": (datetime.utcnow() - timedelta(days=1, hours=10)).isoformat(),
        "config": past_config_2,
        "metrics": {
            "error_message": "Model convergence failed due to invalid hyperparams.",
            "status": "failed",
            "runtime_seconds": 600
        },
        "graph": {"nodes": [], "edges": []}
    }
    create_dummy_experiment_files(past_id_2, past_config_2, past_result_2)
    experiment_results[past_id_2] = past_result_2
    print(f"Added failed experiment: {past_id_2}")


    # 2. Queued Experiments
    queued_id_1 = str(uuid4())
    queued_config_1 = {
        "model": "GPT4_Custom",
        "schema": "Complex_Medical_KG",
        "prompt_template": "template_gamma",
        "data_source": "Clinical_Trials_Phase3",
        "temperature": 0.7
    }
    config_path_q1, result_path_q1 = create_dummy_experiment_files(queued_id_1, queued_config_1)
    experiment_queue.append(
        ExperimentMetadata(
            id=queued_id_1,
            status="queued",
            created_at=datetime.utcnow() - timedelta(hours=2),
            config_path=config_path_q1,
            result_path=result_path_q1,
        )
    )
    print(f"Added queued experiment: {queued_id_1}")

    queued_id_2 = str(uuid4())
    queued_config_2 = {
        "model": "T5_FineTune",
        "schema": "Financial_Reports_KG",
        "prompt_template": "template_delta",
        "data_source": "SEC_Filings_Q2_2025",
        "embedding_dim": 768
    }
    config_path_q2, result_path_q2 = create_dummy_experiment_files(queued_id_2, queued_config_2)
    experiment_queue.append(
        ExperimentMetadata(
            id=queued_id_2,
            status="queued",
            created_at=datetime.utcnow() - timedelta(minutes=45),
            config_path=config_path_q2,
            result_path=result_path_q2,
        )
    )
    print(f"Added queued experiment: {queued_id_2}")

initialize_dummy_data()

# --- ENDPOINTS ---

@app.post("/queue")
async def queue_experiment(
    background_tasks: BackgroundTasks,
    config_file: UploadFile = File(...)
):
    uid = str(uuid4())
    created_at = datetime.utcnow()
    config_path = os.path.join(EXPERIMENTS_DIR, f"{uid}_config.json")
    result_path = os.path.join(EXPERIMENTS_DIR, f"{uid}_result.json")

    with open(config_path, "wb") as f:
        shutil.copyfileobj(config_file.file, f)

    metadata = ExperimentMetadata(
        id=uid,
        status="queued",
        created_at=created_at,
        config_path=config_path,
        result_path=result_path,
    )

    experiment_queue.append(metadata)
    background_tasks.add_task(run_next_experiment)
    print(f"[{datetime.now().isoformat()}] Queued new experiment: {uid}")
    return JSONResponse(content={"id": uid, "status": "queued"})


@app.delete("/queue/{experiment_id}") # <--- NEW ENDPOINT
async def remove_from_queue(experiment_id: str):
    global experiment_queue

    # Find the experiment in the queue
    found_index = -1
    for i, exp in enumerate(experiment_queue):
        if exp.id == experiment_id:
            found_index = i
            break

    if found_index == -1:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found in queue.")

    # Remove from queue
    removed_experiment = experiment_queue.pop(found_index)
    print(f"[{datetime.now().isoformat()}] Removed experiment '{experiment_id}' from queue.")

    # Clean up associated files
    try:
        if os.path.exists(removed_experiment.config_path):
            os.remove(removed_experiment.config_path)
            print(f"[{datetime.now().isoformat()}] Removed config file: {removed_experiment.config_path}")
        if removed_experiment.result_path and os.path.exists(removed_experiment.result_path):
            os.remove(removed_experiment.result_path) # Though queued items shouldn't have results yet
            print(f"[{datetime.now().isoformat()}] Removed result file: {removed_experiment.result_path}")
    except OSError as e:
        print(f"[{datetime.now().isoformat()}] Error cleaning up files for {experiment_id}: {e}")
        # We don't raise an HTTP error for file cleanup failure, as the main task (removing from queue) succeeded.

    return JSONResponse(content={"message": f"Experiment '{experiment_id}' removed from queue."})


@app.get("/status")
async def get_status():
    return {
        "running": running_experiment.model_dump() if running_experiment else None,
        "queue": [e.model_dump() for e in experiment_queue],
        "completed": list(experiment_results.values()),
    }


@app.get("/experiments/{experiment_id}")
async def get_experiment_details(experiment_id: str):
    result = experiment_results.get(experiment_id)
    if not result:
        if running_experiment and running_experiment.id == experiment_id:
            return JSONResponse(status_code=202, content={"status": running_experiment.status, "id": running_experiment.id})
        for q_exp in experiment_queue:
            if q_exp.id == experiment_id:
                 return JSONResponse(status_code=202, content={"status": q_exp.status, "id": q_exp.id})
        raise HTTPException(status_code=404, detail="Experiment not found or not yet completed") # Use HTTPException for 404
    return result


CRAWLER_URLS = {
    "semantic_scholar": os.getenv("SS_CRAWLER_URL", "http://crawler:8000"),
    "scopus": os.getenv("SCOPUS_CRAWLER_URL", "http://scopus-crawler:8000"),
}


def _crawler_request(source: str, method: str, path: str, body: dict | None = None) -> dict:
    if source not in CRAWLER_URLS:
        raise HTTPException(status_code=404, detail=f"Unknown crawler source '{source}'")
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(f"{CRAWLER_URLS[source]}{path}", method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=e.code, detail=json.loads(e.read()).get("detail", str(e)))
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"Crawler unreachable: {e}")


@app.get("/crawlers/{source}/status")
async def get_crawler_status(source: str):
    return await asyncio.to_thread(_crawler_request, source, "GET", "/status")


@app.post("/crawlers/{source}/start")
async def start_crawler(source: str):
    return await asyncio.to_thread(_crawler_request, source, "POST", "/start")


@app.post("/crawlers/{source}/stop")
async def stop_crawler(source: str):
    return await asyncio.to_thread(_crawler_request, source, "POST", "/stop")


class QueryCreate(BaseModel):
    query: str
    design_strategy: str | None = None
    ecosystem_service: str | None = None


@app.get("/crawlers/{source}/queries")
async def list_crawler_queries(source: str):
    return await asyncio.to_thread(_crawler_request, source, "GET", "/queries")


@app.post("/crawlers/{source}/queries")
async def create_crawler_query(source: str, body: QueryCreate):
    return await asyncio.to_thread(_crawler_request, source, "POST", "/queries", body.model_dump())


@app.put("/crawlers/{source}/queries/{query_id}")
async def update_crawler_query(source: str, query_id: int, body: QueryCreate):
    return await asyncio.to_thread(_crawler_request, source, "PUT", f"/queries/{query_id}", body.model_dump())


@app.delete("/crawlers/{source}/queries/{query_id}")
async def delete_crawler_query(source: str, query_id: int):
    return await asyncio.to_thread(_crawler_request, source, "DELETE", f"/queries/{query_id}")


# --- PAPERS & KNOWLEDGE EXTRACTION ---

def get_db_connection() -> psycopg.Connection:
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


PAPERS_DIR = "downloaded_papers"
os.makedirs(PAPERS_DIR, exist_ok=True)

# Manual fallback for papers automated download can't reach (bot-walled
# publishers, etc.): drop/upload a PDF named "{paper_id}.pdf" here and the
# watcher below picks it up, runs it through the normal extraction pipeline,
# and files it under extracted/ once done.
MANUAL_DIR = "manual_uploads"
MANUAL_EXTRACTED_DIR = os.path.join(MANUAL_DIR, "extracted")
os.makedirs(MANUAL_EXTRACTED_DIR, exist_ok=True)
MANUAL_POLL_SECONDS = 5

EXTRACTION_URL = os.getenv("EXTRACTION_URL", "http://knowledge-extraction:8001")


def _is_reachable(url: str, timeout: float = 2.0) -> bool:
    """A response, even a non-2xx one, still means the process is up."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


@app.get("/health")
async def health():
    sources = list(CRAWLER_URLS)
    reachability = await asyncio.gather(
        *(asyncio.to_thread(_is_reachable, f"{CRAWLER_URLS[source]}/status") for source in sources),
        asyncio.to_thread(_is_reachable, EXTRACTION_URL),
    )
    *crawler_reachability, knowledge_extraction = reachability
    return {
        "backend": True,
        "knowledge_extraction": knowledge_extraction,
        **dict(zip(sources, crawler_reachability)),
    }

def _papers_search_filter(q: str | None) -> tuple[str, list]:
    if not q:
        return "", []
    where_clause = """WHERE (
        p.title ILIKE %s OR p.abstract ILIKE %s OR p.query ILIKE %s
        OR EXISTS (SELECT 1 FROM unnest(p.authors) a WHERE a ILIKE %s)
    )"""
    needle = f"%{q}%"
    return where_clause, [needle, needle, needle, needle]


def count_papers(cursor: psycopg.Cursor, q: str | None = None) -> int:
    where_clause, params = _papers_search_filter(q)
    cursor.execute(f"SELECT COUNT(*) FROM papers p {where_clause}", params)
    return cursor.fetchone()[0]


def get_papers(cursor: psycopg.Cursor, limit: int = 50, offset: int = 0, q: str | None = None) -> list[dict]:
    where_clause, params = _papers_search_filter(q)
    cursor.execute(f"""
        SELECT p.id, p.title, p.authors, p.query, p.open_access, p.pdf_url,
               p.source, p.external_id, p.url, p.doi, p.venue, p.citation_count, p.abstract,
               p.relevance_checked, p.relevant, p.created_at,
               r.status, r.error, r.started_at
        FROM papers p
        LEFT JOIN LATERAL (
            SELECT status, error, started_at
            FROM extraction_runs
            WHERE paper_id = p.id
            ORDER BY id DESC
            LIMIT 1
        ) r ON true
        {where_clause}
        ORDER BY p.id
        LIMIT %s OFFSET %s
    """, (*params, limit, offset))
    papers = []
    for (paper_id, title, authors, query, open_access, pdf_url, source, external_id, url, doi, venue, citation_count, abstract,
         relevance_checked, relevant, created_at, status, error, started_at) in cursor.fetchall():
        papers.append({
            "id": paper_id,
            "title": title,
            "authors": authors or [],
            "query": query,
            "open_access": open_access,
            "pdf_url": pdf_url,
            "source": source,
            "external_id": external_id,
            "url": url,
            "doi": doi,
            "venue": venue,
            "citation_count": citation_count,
            "abstract": abstract,
            "relevance_checked": relevance_checked,
            "relevant": relevant,
            "created_at": created_at.isoformat() if created_at else None,
            "extraction_status": status or "pending",
            "extraction_error": error,
            "extraction_started_at": started_at.isoformat() if started_at else None,
        })
    return papers


def get_paper_counts_by_query_field(cursor: psycopg.Cursor, field: str) -> list[dict]:
    """Counts distinct matched papers per query's design_strategy/ecosystem_service — the meta study's per-DS/ES totals."""
    assert field in ("design_strategy", "ecosystem_service"), field
    cursor.execute(f"""
        SELECT sq.{field} AS name, COUNT(DISTINCT pq.paper_id) AS paper_count
        FROM search_queries sq
        JOIN paper_queries pq ON pq.query_id = sq.id
        WHERE sq.{field} IS NOT NULL
        GROUP BY sq.{field}
        ORDER BY paper_count DESC, name
    """)
    return [{"name": name, "paper_count": count} for name, count in cursor.fetchall()]


# DS->ES mapping lives in a separate, externally-hosted Neo4j knowledge graph
# (not the local `neo4j` service) — cached in-process since it's a slow-moving
# taxonomy and the external server shouldn't be hit on every page load.
_META_NEO4J_TTL = timedelta(minutes=10)
_meta_neo4j_cache: dict = {"pairs": None, "fetched_at": None}


def _get_meta_neo4j_driver() -> neo4j.Driver:
    return neo4j.GraphDatabase.driver(
        os.environ["META_NEO4J_URI"],
        auth=(os.environ["META_NEO4J_USER"], os.environ["META_NEO4J_PASSWORD"]),
    )


def get_design_strategy_ecosystem_service_pairs() -> list[tuple[str, str]]:
    """Fetches (design_strategy, ecosystem_service) pairs from the (DesignStrategy)-[:GENERATES]->(EcosystemService) graph."""
    fetched_at = _meta_neo4j_cache["fetched_at"]
    if fetched_at is not None and datetime.now() - fetched_at < _META_NEO4J_TTL:
        return _meta_neo4j_cache["pairs"]

    driver = _get_meta_neo4j_driver()
    try:
        with driver.session(database=os.environ["META_NEO4J_DATABASE"]) as session:
            result = session.run(
                "MATCH (d:DesignStrategy)-[:GENERATES]->(e:EcosystemService) RETURN d.name AS ds, e.name AS es"
            )
            pairs = [(record["ds"], record["es"]) for record in result]
    finally:
        driver.close()

    _meta_neo4j_cache["pairs"] = pairs
    _meta_neo4j_cache["fetched_at"] = datetime.now()
    return pairs


def get_ecosystem_service_paper_counts(cursor: psycopg.Cursor) -> list[dict]:
    """Map-reduces each design strategy's paper count onto the ecosystem service(s) it GENERATES."""
    ds_counts = {row["name"]: row["paper_count"] for row in get_paper_counts_by_query_field(cursor, "design_strategy")}

    es_totals: dict[str, int] = {}
    for ds, es in get_design_strategy_ecosystem_service_pairs():
        count = ds_counts.get(ds)
        if count:
            es_totals[es] = es_totals.get(es, 0) + count

    return [
        {"name": name, "paper_count": count}
        for name, count in sorted(es_totals.items(), key=lambda item: (-item[1], item[0]))
    ]


def get_paper_counts_by_venue(cursor: psycopg.Cursor) -> list[dict]:
    """Counts papers per publication venue — the meta study's per-venue totals."""
    cursor.execute("""
        SELECT venue AS name, COUNT(*) AS paper_count
        FROM papers
        WHERE venue IS NOT NULL
        GROUP BY venue
        ORDER BY paper_count DESC, name
    """)
    return [{"name": name, "paper_count": count} for name, count in cursor.fetchall()]


def get_paper_year_counts(cursor: psycopg.Cursor, field: str, value: str) -> list[dict]:
    """Papers-per-year for a single design_strategy/ecosystem_service value — the meta study's per-DS/ES timeline.

    An ecosystem_service isn't tagged on search_queries directly (see
    get_ecosystem_service_paper_counts); it's rolled up from whichever design
    strategies the DS->ES graph says generate it.
    """
    assert field in ("design_strategy", "ecosystem_service"), field
    if field == "design_strategy":
        design_strategies = [value]
    else:
        design_strategies = [ds for ds, es in get_design_strategy_ecosystem_service_pairs() if es == value]
        if not design_strategies:
            return []

    cursor.execute("""
        SELECT p.year AS year, COUNT(DISTINCT p.id) AS paper_count
        FROM search_queries sq
        JOIN paper_queries pq ON pq.query_id = sq.id
        JOIN papers p ON p.id = pq.paper_id
        WHERE sq.design_strategy = ANY(%s) AND p.year IS NOT NULL
        GROUP BY p.year
        ORDER BY p.year
    """, (design_strategies,))
    return [{"year": year, "paper_count": count} for year, count in cursor.fetchall()]


def create_run(cursor: psycopg.Cursor, paper_id: int, status: str) -> int:
    """Starts a new extraction_runs row for a paper, returning its id."""
    cursor.execute(
        "INSERT INTO extraction_runs (paper_id, status) VALUES (%s, %s) RETURNING id",
        (paper_id, status),
    )
    return cursor.fetchone()[0]


def update_run(
    cursor: psycopg.Cursor,
    run_id: int,
    *,
    status: str,
    error: str | None = None,
    raw_result: dict | None = None,
    raw_text: str | None = None,
    finished: bool = False,
) -> None:
    """Updates an extraction_runs row's status/error/result, stamping finished_at once terminal."""
    cursor.execute(
        """
        UPDATE extraction_runs
        SET status = %s,
            error = %s,
            raw_result = COALESCE(%s, raw_result),
            raw_text = COALESCE(%s, raw_text),
            finished_at = COALESCE(%s, finished_at)
        WHERE id = %s
        """,
        (status, error, Jsonb(raw_result) if raw_result is not None else None, raw_text,
         datetime.now() if finished else None, run_id),
    )


def _update_run(run_id: int, **fields) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            update_run(cursor, run_id, **fields)


_TAG_FIELDS = [
    "extraction_run_id", "tag_type", "group_id", "field", "value", "category",
    "anchor_text", "context", "match_score", "rationale", "verified", "extra_data",
    "page_number", "bbox",
]


def _tag(*, review_status: str = "pending", added_manually: bool = False, **fields) -> dict:
    """A tags-table row, defaulted to None for any column a given extractor doesn't use."""
    tag = dict.fromkeys(_TAG_FIELDS)
    tag.update(fields)
    tag["review_status"] = review_status
    tag["added_manually"] = added_manually
    return tag


def labels_to_tags(extraction_run_id: int, result: dict) -> list[dict]:
    """Flattens {"labels": {section: [decision, ...]}} into one tag row per label decision."""
    return [
        _tag(
            extraction_run_id=extraction_run_id, tag_type="label",
            value=decision.get("label"), anchor_text=decision.get("anchor_text"),
            context=decision.get("context"), match_score=decision.get("match_score"),
            rationale=decision.get("rationale"), verified=decision.get("verdict") == "YES",
            page_number=decision.get("page_number"), bbox=decision.get("bbox"),
        )
        for decisions in result.get("labels", {}).values()
        for decision in decisions
    ]


def entities_to_tags(extraction_run_id: int, result: dict) -> list[dict]:
    """Flattens {"entities": [{field: {value, context, context_verified, ...}}]} into one tag row per verified field.

    Each entity's fields share a group_id (their index in the list) so the
    review UI can cluster them back into a single card.
    """
    tags = []
    for group_id, entity in enumerate(result.get("entities", [])):
        for field_name, field_data in entity.items():
            if not isinstance(field_data, dict) or "value" not in field_data:
                continue
            tags.append(_tag(
                extraction_run_id=extraction_run_id, tag_type="entity", group_id=group_id,
                field=field_name, value=field_data.get("value"), context=field_data.get("context"),
                match_score=field_data.get("context_match_score"), verified=field_data.get("context_verified"),
                page_number=field_data.get("page_number"), bbox=field_data.get("bbox"),
            ))
    return tags


def design_strategies_to_tags(extraction_run_id: int, result: dict) -> list[dict]:
    """Flattens {"design_strategies": [...]} into one tag row per strategy."""
    return [
        _tag(
            extraction_run_id=extraction_run_id, tag_type="design_strategy",
            value=strategy.get("name"), anchor_text=strategy.get("anchor_text"),
            context=strategy.get("context"), match_score=strategy.get("anchor_match_score"),
            verified=strategy.get("anchor_verified"),
            extra_data={
                "implementation_details": strategy.get("implementation_details"),
                "vocab_top_matches": strategy.get("vocab_top_matches"),
            },
            page_number=strategy.get("page_number"), bbox=strategy.get("bbox"),
        )
        for strategy in result.get("design_strategies", [])
    ]


def ecosystem_services_to_tags(extraction_run_id: int, result: dict) -> list[dict]:
    """Flattens {"ecosystem_services": [...]} into one tag row per service."""
    return [
        _tag(
            extraction_run_id=extraction_run_id, tag_type="ecosystem_service",
            value=service.get("name"), category=service.get("category"),
            anchor_text=service.get("anchor_text"), context=service.get("context"),
            match_score=service.get("anchor_match_score"), verified=service.get("anchor_verified"),
            extra_data={"vocab_top_matches": service.get("vocab_top_matches")},
            page_number=service.get("page_number"), bbox=service.get("bbox"),
        )
        for service in result.get("ecosystem_services", [])
    ]


def extraction_result_to_tags(extraction_run_id: int, result: dict) -> list[dict]:
    """Maps every extractor's output in a completed run's result JSON into tag rows."""
    return (
        labels_to_tags(extraction_run_id, result)
        + entities_to_tags(extraction_run_id, result)
        + design_strategies_to_tags(extraction_run_id, result)
        + ecosystem_services_to_tags(extraction_run_id, result)
    )


def insert_tags(cursor: psycopg.Cursor, tags: list[dict]) -> None:
    for tag in tags:
        params = {
            **tag,
            "extra_data": Jsonb(tag["extra_data"]) if tag.get("extra_data") is not None else None,
            "bbox": Jsonb(tag["bbox"]) if tag.get("bbox") is not None else None,
        }
        cursor.execute(
            """
            INSERT INTO tags (extraction_run_id, tag_type, group_id, field, value, category,
                               anchor_text, context, match_score, rationale, verified, extra_data,
                               page_number, bbox, review_status, added_manually)
            VALUES (%(extraction_run_id)s, %(tag_type)s, %(group_id)s, %(field)s, %(value)s, %(category)s,
                    %(anchor_text)s, %(context)s, %(match_score)s, %(rationale)s, %(verified)s, %(extra_data)s,
                    %(page_number)s, %(bbox)s, %(review_status)s, %(added_manually)s)
            """,
            params,
        )


def _insert_tags(tags: list[dict]) -> None:
    if not tags:
        return
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            insert_tags(cursor, tags)


def get_latest_done_run_id(cursor: psycopg.Cursor, paper_id: int) -> int | None:
    cursor.execute(
        "SELECT id FROM extraction_runs WHERE paper_id = %s AND status = 'done' ORDER BY id DESC LIMIT 1",
        (paper_id,),
    )
    row = cursor.fetchone()
    return row[0] if row else None


_TAG_READ_COLUMNS = [
    "id", "tag_type", "group_id", "field", "value", "category", "anchor_text", "context",
    "match_score", "rationale", "verified", "extra_data", "page_number", "bbox",
    "review_status", "edited_value", "added_manually",
]


def get_tags(cursor: psycopg.Cursor, extraction_run_id: int) -> list[dict]:
    cursor.execute(
        f"""
        SELECT {", ".join(_TAG_READ_COLUMNS)}
        FROM tags WHERE extraction_run_id = %s ORDER BY id
        """,
        (extraction_run_id,),
    )
    return [dict(zip(_TAG_READ_COLUMNS, row)) for row in cursor.fetchall()]


def get_paper(cursor: psycopg.Cursor, paper_id: int) -> dict | None:
    cursor.execute("SELECT pdf_url FROM papers WHERE id = %s", (paper_id,))
    row = cursor.fetchone()
    return {"pdf_url": row[0]} if row else None


def download_pdf(pdf_url: str) -> bytes:
    """Downloads PDF bytes, raising ValueError if the response isn't a PDF.

    # ponytail: some publishers (MDPI, Elsevier, ...) 403 a self-identifying bot
    # User-Agent outright; a normal browser UA + Accept header clears that for
    # publishers that only check headers. It won't help against an IP-reputation
    # block from a datacenter egress — that needs a different network path, not
    # a code fix.
    """
    request = urllib.request.Request(pdf_url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/pdf,text/html,*/*",
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()
    if content[:5] != b"%PDF-":
        raise ValueError("downloaded content is not a PDF")
    return content


def notify_extraction(paper_id: int, pdf_bytes: bytes) -> dict:
    """Hands the downloaded PDF off to the knowledge-extraction service, returning its result JSON."""
    request = urllib.request.Request(
        f"{EXTRACTION_URL}/extract?paper_id={paper_id}",
        method="POST",
        data=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )
    # Runs LLM/embedding calls over the whole paper, so this is minutes, not seconds.
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def pdf_path(paper_id: int) -> str:
    return os.path.join(PAPERS_DIR, f"{paper_id}.pdf")


def run_extraction(run_id: int, paper_id: int) -> None:
    """Downloads a paper's PDF (skipping if already on disk) and hands it to knowledge-extraction, tracking status."""
    path = pdf_path(paper_id)

    if os.path.exists(path):
        with open(path, "rb") as f:
            pdf_bytes = f.read()
    else:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                paper = get_paper(cursor, paper_id)

        if not paper or not paper["pdf_url"]:
            _update_run(run_id, status="failed", error="No PDF URL available for this paper", finished=True)
            return

        try:
            pdf_bytes = download_pdf(paper["pdf_url"])
        except Exception as e:
            _update_run(run_id, status="failed", error=f"Download failed: {e}", finished=True)
            return

        with open(path, "wb") as f:
            f.write(pdf_bytes)

    _update_run(run_id, status="extracting")

    try:
        result = notify_extraction(paper_id, pdf_bytes)
    except Exception as e:
        _update_run(run_id, status="failed", error=f"Knowledge extraction unreachable: {e}", finished=True)
        return

    _update_run(run_id, status="done", raw_result=result, raw_text=result.get("raw_text"), finished=True)
    _insert_tags(extraction_result_to_tags(run_id, result))


def _run_in_progress(paper_id: int) -> bool:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM extraction_runs WHERE paper_id = %s AND status IN ('downloading', 'extracting') LIMIT 1",
                (paper_id,),
            )
            return cursor.fetchone() is not None


def _process_manual_upload(filename: str) -> None:
    """Runs one manual_uploads/{paper_id}.pdf through the normal pipeline, then files it under extracted/."""
    paper_id = int(filename[: -len(".pdf")])
    src_path = os.path.join(MANUAL_DIR, filename)

    if _run_in_progress(paper_id):
        return  # already being handled (this tick or the normal download flow) — leave it for next time

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            run_id = create_run(cursor, paper_id, "downloaded")

    # run_extraction() skips downloading when pdf_path(paper_id) already exists,
    # so staging the manual file there reuses the whole pipeline unchanged.
    cached_path = pdf_path(paper_id)
    shutil.copy(src_path, cached_path)
    try:
        run_extraction(run_id, paper_id)
    finally:
        os.remove(cached_path)

    shutil.move(src_path, os.path.join(MANUAL_EXTRACTED_DIR, filename))


def _process_manual_uploads() -> None:
    for filename in sorted(os.listdir(MANUAL_DIR)):
        if not filename.lower().endswith(".pdf") or not filename[: -len(".pdf")].isdigit():
            continue  # not a "{paper_id}.pdf" — leave whatever this is alone
        try:
            _process_manual_upload(filename)
        except Exception as e:
            print(f"[manual_upload_watcher] failed on {filename}: {e}")


async def manual_upload_watcher() -> None:
    while True:
        try:
            await asyncio.to_thread(_process_manual_uploads)
        except Exception as e:
            print(f"[manual_upload_watcher] tick failed: {e}")
        await asyncio.sleep(MANUAL_POLL_SECONDS)


@app.on_event("startup")
async def start_manual_upload_watcher():
    asyncio.create_task(manual_upload_watcher())


class PaperExtractRequest(BaseModel):
    paper_ids: list[int]


@app.get("/papers")
async def list_papers(limit: int = 50, offset: int = 0, q: str | None = None):
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                return get_papers(cursor, limit, offset, q)
    except psycopg.OperationalError as e:
        raise HTTPException(status_code=502, detail=f"Document database unreachable: {e}")


@app.get("/papers/count")
async def papers_count(q: str | None = None):
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                return {"total": count_papers(cursor, q)}
    except psycopg.OperationalError as e:
        raise HTTPException(status_code=502, detail=f"Document database unreachable: {e}")


@app.get("/stats/design-strategies")
async def design_strategy_stats():
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                return get_paper_counts_by_query_field(cursor, "design_strategy")
    except psycopg.OperationalError as e:
        raise HTTPException(status_code=502, detail=f"Document database unreachable: {e}")


@app.get("/stats/ecosystem-services")
async def ecosystem_service_stats():
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                return get_ecosystem_service_paper_counts(cursor)
    except psycopg.OperationalError as e:
        raise HTTPException(status_code=502, detail=f"Document database unreachable: {e}")
    except (neo4j.exceptions.Neo4jError, neo4j.exceptions.DriverError) as e:
        raise HTTPException(status_code=502, detail=f"Metastudy knowledge graph unreachable: {e}")


@app.get("/stats/venues")
async def venue_stats():
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                return get_paper_counts_by_venue(cursor)
    except psycopg.OperationalError as e:
        raise HTTPException(status_code=502, detail=f"Document database unreachable: {e}")


@app.get("/stats/papers-by-year")
async def papers_by_year_stats(field: Literal["design_strategy", "ecosystem_service"], value: str):
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                return get_paper_year_counts(cursor, field, value)
    except psycopg.OperationalError as e:
        raise HTTPException(status_code=502, detail=f"Document database unreachable: {e}")
    except (neo4j.exceptions.Neo4jError, neo4j.exceptions.DriverError) as e:
        raise HTTPException(status_code=502, detail=f"Metastudy knowledge graph unreachable: {e}")


@app.post("/papers/extract")
async def extract_papers(body: PaperExtractRequest, background_tasks: BackgroundTasks):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            for paper_id in body.paper_ids:
                status = "downloaded" if os.path.exists(pdf_path(paper_id)) else "downloading"
                run_id = create_run(cursor, paper_id, status)
                background_tasks.add_task(run_extraction, run_id, paper_id)
    return {"queued": body.paper_ids}


@app.get("/papers/{paper_id}/results")
async def get_paper_results(paper_id: int):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            run_id = get_latest_done_run_id(cursor, paper_id)
            if run_id is None:
                raise HTTPException(status_code=404, detail="No completed extraction results for this paper")
            tags = get_tags(cursor, run_id)
    return {"tags": tags}


def get_raw_text(cursor: psycopg.Cursor, extraction_run_id: int) -> str | None:
    cursor.execute("SELECT raw_text FROM extraction_runs WHERE id = %s", (extraction_run_id,))
    row = cursor.fetchone()
    return row[0] if row else None


@app.get("/papers/{paper_id}/text")
async def get_paper_text(paper_id: int):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            run_id = get_latest_done_run_id(cursor, paper_id)
            text = get_raw_text(cursor, run_id) if run_id is not None else None
    if text is None:
        raise HTTPException(status_code=404, detail="No extracted text for this paper")
    return {"text": text}


def update_tag_review_status(cursor: psycopg.Cursor, tag_id: int, status: str) -> bool:
    """Sets a tag's review_status (accepted/rejected are both soft — the row is never deleted). Returns whether the tag existed."""
    cursor.execute("UPDATE tags SET review_status = %s WHERE id = %s RETURNING id", (status, tag_id))
    return cursor.fetchone() is not None


class TagReviewRequest(BaseModel):
    status: Literal["accepted", "rejected"]


@app.post("/tags/{tag_id}/review")
async def review_tag(tag_id: int, body: TagReviewRequest):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            found = update_tag_review_status(cursor, tag_id, body.status)
    if not found:
        raise HTTPException(status_code=404, detail="Tag not found")
    return {"id": tag_id, "review_status": body.status}


def update_tag_value(cursor: psycopg.Cursor, tag_id: int, edited_value: str) -> bool:
    """Records a reviewer's correction without touching the original extractor value. Returns whether the tag existed."""
    cursor.execute(
        "UPDATE tags SET edited_value = %s, review_status = 'edited' WHERE id = %s RETURNING id",
        (edited_value, tag_id),
    )
    return cursor.fetchone() is not None


class TagEditRequest(BaseModel):
    value: str


@app.post("/tags/{tag_id}/edit")
async def edit_tag(tag_id: int, body: TagEditRequest):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            found = update_tag_value(cursor, tag_id, body.value)
    if not found:
        raise HTTPException(status_code=404, detail="Tag not found")
    return {"id": tag_id, "edited_value": body.value, "review_status": "edited"}


def add_manual_tag(cursor: psycopg.Cursor, extraction_run_id: int, tag_type: str, value: str) -> dict:
    """A reviewer-added tag needs no review (a human just typed it), so it starts 'accepted'."""
    tag = _tag(
        extraction_run_id=extraction_run_id, tag_type=tag_type, value=value,
        review_status="accepted", added_manually=True,
    )
    insert_tags(cursor, [tag])
    return tag


class TagCreateRequest(BaseModel):
    tag_type: str
    value: str


@app.post("/papers/{paper_id}/tags")
async def create_paper_tag(paper_id: int, body: TagCreateRequest):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            run_id = get_latest_done_run_id(cursor, paper_id)
            if run_id is None:
                raise HTTPException(status_code=404, detail="No completed extraction results for this paper")
            add_manual_tag(cursor, run_id, body.tag_type, body.value)
            tags = get_tags(cursor, run_id)
    return {"tags": tags}


@app.get("/papers/{paper_id}/pdf")
async def get_paper_pdf(paper_id: int):
    path = pdf_path(paper_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No PDF on disk for this paper")
    return FileResponse(path, media_type="application/pdf")


@app.post("/papers/{paper_id}/upload")
async def upload_paper_pdf(paper_id: int, file: UploadFile = File(...)):
    """Manual fallback for when automated download fails: drops the PDF into
    manual_uploads/, where the watcher (see manual_upload_watcher) picks it up
    and runs it through the normal extraction pipeline."""
    content = await file.read()
    if content[:5] != b"%PDF-":
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF")
    with open(os.path.join(MANUAL_DIR, f"{paper_id}.pdf"), "wb") as f:
        f.write(content)
    return {"queued": paper_id}


# Dummy long-running task
def run_next_experiment():
    global running_experiment, experiment_queue, experiment_results
    if running_experiment is not None or not experiment_queue:
        return

    experiment = experiment_queue.pop(0)
    running_experiment = experiment
    experiment.status = "running"
    print(f"[{datetime.now().isoformat()}] Started running experiment: {experiment.id}")

    import time
    time.sleep(10)

    is_successful = (sum(ord(c) for c in experiment.id) % 2) == 0

    if is_successful:
        result_data = {
            "id": experiment.id,
            "finished_at": datetime.utcnow().isoformat(),
            "config": json.load(open(experiment.config_path)),
            "metrics": {
                "precision": round(0.75 + (uuid4().int % 20) / 100, 2),
                "recall": round(0.70 + (uuid4().int % 25) / 100, 2),
                "f1_score": round(0.72 + (uuid4().int % 23) / 100, 2),
                "nodes_extracted": 100 + (uuid4().int % 200),
                "edges_extracted": 200 + (uuid4().int % 400),
                "runtime_seconds": 10
            },
            "graph": {
                "nodes": [{"id": f"node-{i}", "labels": ["Concept"], "properties": {"name": f"Concept {i}"}} for i in range(5)],
                "relationships": [{"id": f"rel-{i}", "startNode": "node-0", "endNode": f"node-{i+1}", "type": "HAS_RELATION"} for i in range(4)]
            },
        }
    else:
        result_data = {
            "id": experiment.id,
            "finished_at": datetime.utcnow().isoformat(),
            "config": json.load(open(experiment.config_path)),
            "metrics": {
                "error_message": "Simulated failure: Failed to connect to external service.",
                "status": "failed",
                "runtime_seconds": 10
            },
            "graph": {"nodes": [], "edges": []},
        }

    with open(experiment.result_path, "w") as f:
        json.dump(result_data, f, indent=2)

    experiment.status = "completed"
    experiment_results[experiment.id] = result_data
    print(f"[{datetime.now().isoformat()}] Completed experiment: {experiment.id} (Success: {is_successful})")
    running_experiment = None

    run_next_experiment()
