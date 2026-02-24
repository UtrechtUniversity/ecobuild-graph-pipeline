#!/bin/bash

# Remote base directory (on server)
REMOTE_ORCHESTRATION="~/AUTO-KG-LIT/orchestration"
REMOTE_BASE="~/AUTO-KG-LIT/microservices"

# Loop over each microservice folder locally
for dir in microservices/*/ ; do
  env_file="${dir%.}/.env"

  if [ -f "$env_file" ]; then
    echo "Copying $env_file to server..."

    # Use scp to copy the .env file to the server
    scp "$env_file" "auto_kg_lit:${REMOTE_BASE}/$(basename "$dir")/.env"

    if [ $? -eq 0 ]; then
      echo "Copied $(basename "$dir")/.env successfully"
    else
      echo "Failed to copy $(basename "$dir")/.env" >&2
    fi
  else
    echo "No .env file found in $dir, skipping."
  fi
done

# Copy the orchestration .env file
echo "Copying ./orchestration/.env to server..."
scp "./orchestration/.env" "auto_kg_lit:${REMOTE_ORCHESTRATION}/.env"

if [ $? -eq 0 ]; then
  echo "Copied ./orchestration/.env successfully"
else
  echo "Failed to copy ./orchestration/.env" >&2
fi

