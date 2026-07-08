.PHONY: install dev

install:
	poetry install --no-root

# Runs in /bin/sh regardless of your login shell, so this works from fish too.
dev: install
	set -a; . ./.env; set +a; poetry run uvicorn crawler.service:app --host 0.0.0.0 --port 8001
