.PHONY: install dev stop

install:
	poetry install --no-root

# One-shot extraction pipeline (not a long-lived server). Needs document-db,
# neo4j and ollama, and reads PDFs from a path only mounted inside the
# container, so this runs via the shared compose file instead of poetry directly.
dev:
	docker compose -f ../../orchestration/docker-compose.yml --profile extract up --build knowledge-extraction

# Only stops this service's own containers — document-db is shared with
# backend/crawler, so it's left running.
stop:
	-docker compose -f ../../orchestration/docker-compose.yml --profile extract stop knowledge-extraction ollama neo4j
