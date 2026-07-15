.PHONY: install up down dev stop status extract deploy

install:
	$(MAKE) -C microservices/backend install
	$(MAKE) -C microservices/paper-crawler install
	$(MAKE) -C microservices/scopus-crawler install
	$(MAKE) -C microservices/knowledge-extraction install
	$(MAKE) -C frontend install

# Postgres etc., via the orchestration compose file.
up:
	docker compose -f orchestration/docker-compose.yml up -d document-db

down:
	docker compose -f orchestration/docker-compose.yml down

# Starts db + backend + crawler + frontend + knowledge-extraction in the background.
# extraction runs via Docker (builds its own image, needs neo4j + ollama via
# the "extract" compose profile) rather than poetry, so it's backgrounded the
# same way so `make dev` brings everything up in one go.
# `stop` targets the local processes by command substring, and the extraction
# containers by compose profile, so no pidfile bookkeeping.
dev: up
	$(MAKE) -C microservices/backend dev >/tmp/backend.log 2>&1 &
	$(MAKE) -C microservices/paper-crawler dev >/tmp/crawler.log 2>&1 &
	$(MAKE) -C microservices/scopus-crawler dev >/tmp/scopus-crawler.log 2>&1 &
	$(MAKE) -C frontend dev >/tmp/frontend.log 2>&1 &
	$(MAKE) extract >/tmp/extract.log 2>&1 &
	@echo "backend:        http://localhost:8000  (log: /tmp/backend.log)"
	@echo "crawler:        http://localhost:8001  (log: /tmp/crawler.log)"
	@echo "scopus-crawler: http://localhost:8003  (log: /tmp/scopus-crawler.log)"
	@echo "frontend:       http://localhost:3000  (log: /tmp/frontend.log)"
	@echo "extract:        http://localhost:8002  (log: /tmp/extract.log)"

stop:
	-pkill -f 'uvicorn main:app'
	-pkill -f 'uvicorn crawler.service:app'
	-pkill -f 'bun --hot'
	$(MAKE) -C microservices/knowledge-extraction stop
	@echo "stopped"

status:
	@ss -ltnp 2>/dev/null | grep -E ':(3000|8000|8001|8002|8003)\b' || echo "nothing on 3000/8000/8001/8002/8003"
	@docker ps --format '{{.Names}}: {{.Ports}}'

# Serves the extraction API via Docker. Needs neo4j + ollama, hence the
# "extract" compose profile. Also callable on its own via `make extract`.
extract:
	$(MAKE) -C microservices/knowledge-extraction dev

# Production: everything containerized. No ports published for
# frontend/api-backend — the already-running ecobuild-caddy-1 on the deploy
# host reaches them over the shared "ecobuild-edge" network instead (see
# orchestration/Caddyfile.snippet). `docker network create` is idempotent,
# so this is safe to run every time, not just the first. The local ollama
# container is opt-in only (profile "local-ollama", not requested here) —
# see OLLAMA_HOST in .env.template.
# `make down` tears this back down too — it stops whatever containers exist
# for this project regardless of which profile started them.
deploy:
	docker network create ecobuild-edge 2>/dev/null || true
	docker compose -f orchestration/docker-compose.yml --profile extract up -d --build
