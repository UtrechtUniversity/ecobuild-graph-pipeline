.PHONY: install dev

install:
	poetry install --no-root

# One-shot extraction pipeline (not a long-lived server). Needs document-db,
# neo4j and ollama, and reads PDFs from a path only mounted inside the
# container, so this runs via the shared compose file instead of poetry directly.
dev:
	docker compose -f ../../orchestration/docker-compose.yml --profile extract up --build knowledge-extraction
