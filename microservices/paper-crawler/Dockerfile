FROM python:3.12-slim

# Install Poetry
RUN pip install poetry==1.8.3

# Set working directory inside container
WORKDIR /app

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev)
RUN poetry install --no-dev

# Copy your app source code
COPY crawler ./crawler

# Make sure logs show up immediately
ENV PYTHONUNBUFFERED=1

# Run the crawler main module via poetry
ENTRYPOINT ["poetry", "run", "python", "-m", "crawler.main"]
