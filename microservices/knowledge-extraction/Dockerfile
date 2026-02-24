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

# Run the pipeline main module via poetry
ENTRYPOINT ["poetry", "run", "python", "-m", "src.main"]

