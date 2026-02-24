#!/bin/bash

cd "$(dirname "$0")/../.." || { echo "Failed to go to orchestration root"; exit 1; }

BASE_DIR="../microservices"
cd "$BASE_DIR" || { echo "Directory $BASE_DIR not found"; exit 1; }

for dir in */; do
  if [ -d "$dir/.git" ]; then
    echo "Pulling latest in $dir"
    cd "$dir" || continue
    git pull
    cd ..
  else
    echo "Skipping $dir — not a git repo"
  fi
done
