import threading

from fastapi import FastAPI

from .crawler_logger import logger
from .main import run_crawl

app = FastAPI()

_lock = threading.Lock()
_stop_event = threading.Event()
_thread: threading.Thread | None = None
_status = "idle"  # idle | running | stopped | error


def _run() -> None:
    global _status
    try:
        run_crawl(_stop_event)
        _status = "stopped" if _stop_event.is_set() else "idle"
    except Exception:
        logger.exception("Crawl run failed")
        _status = "error"


@app.get("/status")
async def get_status():
    return {"status": _status}


@app.post("/start")
async def start():
    global _thread, _status
    with _lock:
        if _status == "running":
            return {"status": _status}
        _stop_event.clear()
        _status = "running"
        _thread = threading.Thread(target=_run, daemon=True)
        _thread.start()
    return {"status": _status}


@app.post("/stop")
async def stop():
    global _status
    with _lock:
        if _status != "running":
            return {"status": _status}
        _stop_event.set()
        _status = "stopped"
    return {"status": _status}
