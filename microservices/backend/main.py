# main.py

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException # Import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime, timedelta
import os
import json
import shutil

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
