# backend

FastAPI service backing the [frontend](../../frontend) Experiment Dashboard. It tracks
experiment runs and serves them at `http://localhost:8000` (see the
[orchestration repo](../../orchestration) for how it's deployed alongside the other services).

Two experiment sources exist side by side:

- **In-memory queue/dummy runner** — `POST /queue` accepts a config file, queues it, and a
  background task "runs" it (10s sleep, then a coin-flip pass/fail with fabricated metrics).
  This is scaffolding/demo data, not a real experiment runner. State resets whenever the
  process restarts, and `initialize_dummy_data()` seeds a couple of fake past/queued
  experiments on every boot.
- **Postgres-backed uploads** — `POST /experiments/upload` accepts real output files from a
  [knowledge-extraction](../knowledge-extraction) run and persists them to the `experiments`
  table. This is the real path the "Add Experiment Run" page in the frontend uses.

## Running

Needs a reachable Postgres instance (the `document-db` service in `orchestration/docker-compose.yml`).

```bash
poetry install
DB_HOST=localhost DB_PORT=5433 DB_USER=... DB_PASSWORD=... DB_NAME=... \
  poetry run uvicorn main:app --reload --port 8000
```

Or via the orchestration stack (`api-backend` service, runs by default, no profile needed):

```bash
cd ../../orchestration
docker compose up -d document-db api-backend
```

### Environment variables

| Variable | Default | Notes |
|---|---|---|
| `DB_HOST` | `document-db` | Postgres host |
| `DB_PORT` | `5432` | Postgres port |
| `DB_USER` | — | required |
| `DB_PASSWORD` | — | required |
| `DB_NAME` | — | required |

## Endpoints

| Method & path | Description |
|---|---|
| `POST /queue` | Upload a config file, queue a (dummy) experiment run. |
| `DELETE /queue/{id}` | Remove a queued experiment and delete its files. |
| `GET /status` | Currently running experiment, queue, and completed experiments (DB + in-memory). |
| `GET /experiments/{id}` | Full detail (config, metrics, graph) for one experiment. |
| `POST /experiments/upload` | Register a completed knowledge-extraction run from its output files. |

### `POST /experiments/upload`

Accepts multipart `files[]`. Files are matched by suffix and grouped by paper name
(everything before the suffix):

| Suffix | Stored as |
|---|---|
| `_raw.md` | raw markdown |
| `_plain.txt` | plain text |
| `_sections.json` | sections (parsed JSON) |
| `_labels.json` | labels (parsed JSON) |
| `_extraction.json` | extraction (parsed JSON) |
| `_report.txt` | report |

Unrecognized files are silently skipped. The resulting record's `metrics.total_labels` sums
the length of every list under each paper's `labels` object.
