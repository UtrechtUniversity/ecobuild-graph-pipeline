# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that builds a knowledge graph about nature-based building design (green
roofs/walls, ecosystem services) by crawling academic papers, extracting structured
entities from them with an LLM, and storing the results in Neo4j/Postgres. It is a
monorepo: each subdirectory under `microservices/` was originally its own git repo
(see `orchestration/utils/clone.sh`, which still references separate
`git.science.uu.nl` remotes and a `document-parser` repo that isn't checked out
here) but everything now lives together in this one repo — treat `clone.sh`/`pull.sh`
as vestigial, not the current workflow.

## Architecture / data flow

`orchestration/docker-compose.yml` is the hub that wires everything together via
Compose profiles:

1. **`paper-crawler`** (profile `crawl`) — queries the Semantic Scholar API and
   writes paper metadata into the `papers` table in **`document-db`** (Postgres,
   schema in `microservices/document-database/init.sql`).
2. **`knowledge-extraction`** (profile `extract`) — the core pipeline, in
   `microservices/knowledge-extraction/src/`:
   - `paper_preprocessor.py` — PDF → markdown/text via `pymupdf4llm`, splits into
     canonical sections.
   - `paper_labeler.py` — LLM-based labeling of sections (via LlamaIndex + Ollama).
   - `entity_extractor.py`, `design_strategy_extractor.py`,
     `ecosystem_service_extractor.py` — pull structured entities (buildings, design
     strategies, ecosystem services) out of section text using the LLM.
   - `context_resolver.py` — verifies each extracted item's anchor text actually
     appears in the source document (anchor verification), rather than trusting the
     LLM's claim blindly.
   - `entity_resolution.py` — fuzzy/embedding matching of extracted items against a
     fixed vocabulary (via `rapidfuzz` + Ollama embeddings).
   - Talks to **Ollama** (`OLLAMA_HOST`, `OLLAMA_LLM_MODEL`, `OLLAMA_EMBEDDING_MODEL`)
     for both the LLM and embeddings, and to **Neo4j** for graph storage (Neo4j
     integration is still stubbed out — see gotcha below).
   - Outputs per paper into `test_papers/preprocessed/`: `*_raw.md`, `*_sections.json`,
     `*_labels.json`, `*_extraction.json`, `*_report.txt`.
3. **`api-backend`** (`microservices/backend`, runs by default, no profile) — FastAPI
   service, single file `main.py`. Two independent experiment-tracking paths exist
   side by side, don't confuse them:
   - In-memory dummy queue (`POST /queue`, `GET /status`) — fake pass/fail runner
     seeded with `initialize_dummy_data()` on every boot, state lost on restart. This
     is scaffolding, not real.
   - Postgres-backed uploads (`POST /experiments/upload`) — the real path. Accepts
     the knowledge-extraction output files above (matched by filename suffix,
     grouped by paper name) and persists them to the `experiments` table.
4. **`frontend`** (`microservices` sibling `frontend/`, runs by default) — React 19 +
   Bun "Experiment Dashboard": `ExperimentList`/`ExperimentDetail` read from the
   backend's dummy+DB-backed experiment list, `AddExperimentRun`/`ExperimentUploader`
   post to `/experiments/upload`.

### Known gotcha in knowledge-extraction

`src/main.py`'s pipeline currently exits early: after the labeling step it hits
`raise SystemError("Exiting after label extraction.")` (line ~181) before reaching
entity/design-strategy/ecosystem-service extraction, anchor resolution, and the
Neo4j write — that code is currently unreachable dead code mid-refactor, not a bug
to silently "fix" without checking with whoever's mid-change on it.

## Common commands

Everything below assumes `orchestration/.env` (copied from `.env.template`) plus a
per-service `.env` (see each `microservices/*/​.env.template`) — the `POSTGRES_*`,
`SS_API_KEY`, `NEO4J_PASSWORD`, `OLLAMA_*` vars are consumed by
`orchestration/docker-compose.yml` and passed into containers.

```bash
# Start just the DB (Postgres + Neo4j)
cd orchestration && docker compose up -d document-db neo4j

# Crawl papers
docker compose --profile crawl up --build

# Run extraction (CPU, default)
docker compose --profile extract up --build
# GPU:
docker compose --profile extract -f docker-compose.yml -f docker-compose.gpu.yml up --build
# Remote Ollama instead of local container:
docker compose -f docker-compose.yml -f docker-compose.remote-ollama.yml --profile extract up --build knowledge-extraction

# Backend (FastAPI), from microservices/backend
make setup   # poetry install + copy .env.template
make db      # starts document-db via orchestration compose
make dev     # uvicorn main:app --reload --port 8000

# Frontend, from frontend/
bun install
bun dev      # dev server
bun build    # production bundle
```

Inspecting the DB while `document-db` is running:
```bash
docker exec -it orchestration-document-db-1 psql -U <db-username> -d <db-name>
```

Resetting the DB (destructive — only if you mean it):
```bash
docker compose down
docker volume rm orchestration_pgdata orchestration_neo4j_data
```

There is currently no automated test suite in any service — verify changes by
running the relevant service directly.

## Workflow

Work happens on feature branches off `main`, merged via pull request — don't commit
directly to `main`.
