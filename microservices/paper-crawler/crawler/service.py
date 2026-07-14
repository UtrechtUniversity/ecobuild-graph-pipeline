import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg.errors import UniqueViolation

from .crawler_logger import logger
from .main import add_query, get_connection, get_queries, remove_query, run_crawl, update_query

app = FastAPI()


class QueryCreate(BaseModel):
    query: str
    design_strategy: str | None = None
    ecosystem_service: str | None = None


_lock = threading.Lock()
_stop_event = threading.Event()
_thread: threading.Thread | None = None
_status = "idle"  # idle | running | stopped | error
_error: str | None = None


def _run() -> None:
    global _status, _error
    try:
        run_crawl(_stop_event)
        _status = "stopped" if _stop_event.is_set() else "idle"
    except Exception as exc:
        logger.exception("Crawl run failed")
        _error = str(exc)
        _status = "error"


@app.get("/status")
async def get_status():
    return {"status": _status, "error": _error}


@app.post("/start")
async def start():
    global _thread, _status, _error
    with _lock:
        if _status == "running":
            return {"status": _status, "error": _error}
        _stop_event.clear()
        _status = "running"
        _error = None
        _thread = threading.Thread(target=_run, daemon=True)
        _thread.start()
    return {"status": _status, "error": _error}


@app.post("/stop")
async def stop():
    global _status
    with _lock:
        if _status != "running":
            return {"status": _status, "error": _error}
        _stop_event.set()
        _status = "stopped"
    return {"status": _status, "error": _error}


@app.get("/queries")
async def list_queries():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            return get_queries(cursor)


@app.post("/queries")
async def create_query(body: QueryCreate):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")
    with get_connection() as connection:
        with connection.cursor() as cursor:
            try:
                return add_query(cursor, query, body.design_strategy, body.ecosystem_service)
            except UniqueViolation:
                raise HTTPException(status_code=409, detail="query already exists")


@app.put("/queries/{query_id}")
async def edit_query(query_id: int, body: QueryCreate):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")
    with get_connection() as connection:
        with connection.cursor() as cursor:
            try:
                updated = update_query(cursor, query_id, query, body.design_strategy, body.ecosystem_service)
            except UniqueViolation:
                raise HTTPException(status_code=409, detail="query already exists")
    if not updated:
        raise HTTPException(status_code=404, detail=f"query '{query_id}' not found")
    return updated


@app.delete("/queries/{query_id}")
async def delete_query(query_id: int):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            deleted = remove_query(cursor, query_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"query '{query_id}' not found")
    return {"message": f"query '{query_id}' removed"}
