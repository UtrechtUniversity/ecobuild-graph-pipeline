.PHONY: install dev

install:
	env -u VIRTUAL_ENV poetry install --no-root

# Runs in /bin/sh regardless of your login shell, so this works from fish too.
# -u VIRTUAL_ENV: ignore any venv activated in the parent shell (e.g. by an
# IDE) so poetry always targets this project's own .venv, not a stray one.
dev: install
	set -a; . ./.env; set +a; env -u VIRTUAL_ENV poetry run uvicorn crawler.service:app --host 0.0.0.0 --port 8001
