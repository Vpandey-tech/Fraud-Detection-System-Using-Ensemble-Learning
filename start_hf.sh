#!/bin/bash

# 1. Start the FASTAPI Backend in the background (&)
# We run it on port 8000 so the frontend can find it at http://localhost:8000
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for the backend to start
sleep 5

# 2. Start the STREAMLIT Frontend
# Hugging Face expects the app to run on port 7860
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
