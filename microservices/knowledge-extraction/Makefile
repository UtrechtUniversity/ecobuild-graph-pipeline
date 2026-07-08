.PHONY: install dev stop

COMPOSE := docker compose -f ../../orchestration/docker-compose.yml
STOP_SERVICES := knowledge-extraction ollama neo4j

# Set to 0 to use the local "ollama" container instead of a remote/host Ollama
# (see orchestration/docker-compose.remote-ollama.yml and the OLLAMA_HOST /
# OLLAMA_LLM_MODEL / OLLAMA_EMBEDDING_MODEL vars in orchestration/.env).
REMOTE_OLLAMA ?= 1
ifeq ($(REMOTE_OLLAMA),1)
COMPOSE += -f ../../orchestration/docker-compose.remote-ollama.yml
STOP_SERVICES := knowledge-extraction neo4j
endif

install:
	poetry install --no-root

# One-shot extraction pipeline (not a long-lived server). Needs document-db,
# neo4j and ollama, and reads PDFs from a path only mounted inside the
# container, so this runs via the shared compose file instead of poetry directly.
dev:
	$(COMPOSE) --profile extract up --build knowledge-extraction

# Only stops this service's own containers — document-db is shared with
# backend/crawler, so it's left running.
stop:
	-$(COMPOSE) --profile extract stop $(STOP_SERVICES)
