# EcoBuild Graph Pipeline (AUTO-KG-LIT)

This repo orchestrates a pipeline that turns a folder of academic PDFs about
sustainable building design (green roofs, living walls, ecosystem services,
etc.) into a Neo4j knowledge graph, with a small web UI for tracking
extraction experiments.

## Components

| Path | What it does |
|---|---|
| `orchestration/` | Docker Compose setup that wires everything together, plus experiment-running scripts (`run_experiments.py`) |
| `microservices/paper-crawler/` | Crawls Semantic Scholar for papers and stores metadata in the document database |
| `microservices/document-database/` | Postgres schema/init for crawled paper metadata |
| `microservices/knowledge-extraction/` | Converts PDFs to text, labels sections, and extracts entities / design strategies / ecosystem services using an LLM (via Ollama) and `llama-index` |
| `microservices/backend/` | FastAPI service backing the frontend (experiment queue/results) |
| `frontend/` | React + Bun "Experiment Dashboard" UI |

Everything is run via Docker Compose from `orchestration/`. Some pieces
(`backend`, `frontend`, `knowledge-extraction`) currently use **dummy/placeholder
data** and are not yet fully wired end-to-end — see "Current status" below.

## Prerequisites

- Docker + Docker Compose v2
- Python 3.12 (for local scripts in `orchestration/`)
- An Ollama instance with an LLM model (e.g. `qwen3.5:35b-a3b`) and an
  embedding model (e.g. `embeddinggemma`) — either local or remote

## Getting started

1. Clone the microservice repos into `microservices/` (only needed if they
   aren't already present):
   ```bash
   chmod +x orchestration/utils/clone.sh
   ./orchestration/utils/clone.sh
   ```

2. Set up the Python venv for `orchestration/` scripts (e.g.
   `run_experiments.py`):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   poetry -C orchestration install
   ```

3. Create `orchestration/.env` from `orchestration/.env.template` and fill in:
   - `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
   - `NEO4J_USER`, `NEO4J_PASSWORD`
   - `SS_API_KEY` (Semantic Scholar API key, for the crawler)
   - `OLLAMA_HOST`, `OLLAMA_LLM_MODEL`, `OLLAMA_EMBEDDING_MODEL`

   Each microservice may also need its own `.env` (see that service's
   `.env.template`).

## Running the stack

All commands below are run from `orchestration/`.

### Database + Neo4j (always needed)

```bash
docker compose up -d document-db neo4j
```
- Postgres is exposed on host port **5433** (container port 5432; 5432 is
  often already taken by other local Postgres instances)
- Neo4j browser UI: http://localhost:7474, bolt: `bolt://localhost:7687`

### Frontend + backend (experiment dashboard)

```bash
docker compose up -d --build api-backend frontend
```
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

### Knowledge extraction

The `extract` profile builds and runs the `knowledge-extraction` service,
which reads PDFs and writes preprocessed text / labels / extraction results.

**With a local Ollama** (GPU, via `docker-compose.gpu.yml`, or CPU by default):
```bash
docker compose --profile extract up --build knowledge-extraction
```

**With a remote Ollama** (e.g. accessed via an SSH tunnel to
`localhost:<port>`, or directly over the network): set `OLLAMA_HOST` in
`.env` (use `http://host.docker.internal:<port>` if the host is reachable
from your machine but not from inside the container), then:
```bash
docker compose -f docker-compose.yml -f docker-compose.remote-ollama.yml \
  --profile extract up --build knowledge-extraction
```
This override also mounts an external folder of PDFs as the input directory
— edit the volume path in `docker-compose.remote-ollama.yml` to point at
your PDF folder.

Output (raw text, section labels, extraction JSON, reports) is written to
`microservices/knowledge-extraction/test_papers/preprocessed/`.

### Paper crawler

```bash
docker compose --profile crawl up --build crawler
```

## Current status / known gaps

- `knowledge-extraction/src/main.py` currently stops after the section-labeling
  step (`raise SystemError("Exiting after label extraction.")`); entity/design
  strategy/ecosystem-service extraction and the Neo4j write-back exist but are
  not yet reconnected.
- `microservices/backend` serves dummy in-memory experiment data on startup —
  it is not yet connected to Postgres/Neo4j or to the `knowledge-extraction`
  output.
- `paper-crawler` and `document-database` are present but not verified
  end-to-end against the current schema.
