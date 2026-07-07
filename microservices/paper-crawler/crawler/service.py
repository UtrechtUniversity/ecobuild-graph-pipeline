import threading

from fastapi import FastAPI

from .crawler_logger import logger
from .main import run_crawl

app = FastAPI()

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
