.PHONY: install up down dev stop status extract

install:
	$(MAKE) -C microservices/backend install
	$(MAKE) -C microservices/paper-crawler install
	$(MAKE) -C microservices/knowledge-extraction install
	$(MAKE) -C frontend install

# Postgres etc., via the orchestration compose file.
up:
	docker compose -f orchestration/docker-compose.yml up -d document-db

down:
	docker compose -f orchestration/docker-compose.yml down

# Starts db + backend + crawler + frontend + knowledge-extraction in the background.
# extraction is a one-shot batch pipeline (builds its own image, needs neo4j +
# ollama via the "extract" compose profile), not a long-lived server, but it's
# backgrounded the same way so `make dev` brings everything up in one go.
# `stop` targets the local processes by command substring, and the extraction
# containers by compose profile, so no pidfile bookkeeping.
dev: up
	$(MAKE) -C microservices/backend dev >/tmp/backend.log 2>&1 &
	$(MAKE) -C microservices/paper-crawler dev >/tmp/crawler.log 2>&1 &
	$(MAKE) -C frontend dev >/tmp/frontend.log 2>&1 &
	$(MAKE) extract >/tmp/extract.log 2>&1 &
	@echo "backend:  http://localhost:8000  (log: /tmp/backend.log)"
	@echo "crawler:  http://localhost:8001  (log: /tmp/crawler.log)"
	@echo "frontend: http://localhost:3000  (log: /tmp/frontend.log)"
	@echo "extract:  (log: /tmp/extract.log)"

stop:
	-pkill -f 'uvicorn main:app'
	-pkill -f 'uvicorn crawler.service:app'
	-pkill -f 'bun --hot'
	$(MAKE) -C microservices/knowledge-extraction stop
	@echo "stopped"

status:
	@ss -ltnp 2>/dev/null | grep -E ':(3000|8000|8001)\b' || echo "nothing on 3000/8000/8001"
	@docker ps --format '{{.Names}}: {{.Ports}}'

# One-shot batch pipeline, not a long-lived server. Needs neo4j + ollama,
# hence the "extract" compose profile. Also callable on its own via `make extract`.
extract:
	$(MAKE) -C microservices/knowledge-extraction dev
