#!/bin/sh
# Fills in the placeholders index.html ships with, from env vars set on the
# container (see orchestration/docker-compose.yml's frontend service). Safe
# to no-op: if API_BASE/BASE_PATH aren't set, sed just doesn't match anything.
set -e

INDEX=/usr/share/nginx/html/index.html

if [ -n "$API_BASE" ]; then
  sed -i "s#__API_BASE__#${API_BASE}#" "$INDEX"
fi
if [ -n "$BASE_PATH" ]; then
  sed -i "s#__BASE_PATH__#${BASE_PATH}#" "$INDEX"
fi

exec nginx -g 'daemon off;'
