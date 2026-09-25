#!/bin/sh
set -e

export PYTHONPATH="/app"

exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-8000}"
