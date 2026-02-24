FROM python:3.12-slim-bookworm AS builder

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python - \
    && ${POETRY_HOME}/bin/poetry self add poetry-plugin-export

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root

# NEW LINE: Copy your entire application source code into the builder stage
# This ensures that main.py and other files are available when /app is copied to runtime.
COPY . .

FROM python:3.12-slim-bookworm AS runtime

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
