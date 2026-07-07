#!/bin/bash

# Go to the orchestration root folder from utils
cd "$(dirname "$0")/../.." || { echo "Failed to go to orchestration root"; exit 1; }

REPOS=(
  "paper-crawler git@git.science.uu.nl:auto-kg-lit/paper-crawler.git"
  "document-database git@git.science.uu.nl:auto-kg-lit/document-database.git"
  "knowledge-extraction git@git.science.uu.nl:auto-kg-lit/knowledge-extraction.git"
  "document-parser git@git.science.uu.nl:auto-kg-lit/document-parser.git"
  "backend git@git.science.uu.nl:auto-kg-lit/backend.git"
)

BASE_DIR="./microservices"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR" || exit 1

for entry in "${REPOS[@]}"; do
  read -r dir url <<< "$entry"
  if [ -d "$dir" ]; then
    echo "$dir already exists, skipping clone."
  else
    git clone "$url" "$dir"
  fi
done
