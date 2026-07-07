.PHONY: setup db dev

setup:
	poetry install --no-root
	test -f .env || cp .env.template .env

# Starts just the Postgres dependency (document-db) via the orchestration compose file.
db:
	cd ../../orchestration && docker compose up -d document-db

dev:
	set -a && . ./.env && set +a && poetry run uvicorn main:app --reload --port 8000
