FROM python:3.12-slim

# Install netcat (and cleanup)
RUN apt-get update && \
    apt-get install -y netcat-openbsd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install Poetry
RUN pip install poetry==1.8.3

# Set working directory inside container
WORKDIR /app

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml ./

# Install dependencies (no dev)
RUN poetry install --only main

# Copy source code of app
COPY src ./src
COPY scripts ./scripts
COPY test_papers ./test_papers

# Make sure logs show up immediately
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

# Serve the extraction API (POST /extract); the batch CLI (src.main) is still
# runnable directly for local testing over the test_papers directory.
ENTRYPOINT ["poetry", "run", "uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8001", "--no-access-log"]

