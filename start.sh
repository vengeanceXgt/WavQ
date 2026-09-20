#!/bin/bash
export PYTHONPATH="/app"

# Start FastAPI API in the background
uvicorn rf_analyzer.api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit UI in the foreground
streamlit run rf_analyzer/gui/app.py --server.port 8501 --server.address 0.0.0.0
