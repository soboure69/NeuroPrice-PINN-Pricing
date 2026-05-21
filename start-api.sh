#!/usr/bin/env sh
set -eu

PORT="${PORT:-8000}"
echo "Starting NeuroPrice API on 0.0.0.0:${PORT}"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"
