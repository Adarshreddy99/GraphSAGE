#!/bin/bash
# 🤖 GraphSAGE Launch Orchestrator

echo "🚀 Starting FastAPI Recommender Engine..."
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 &

# Wait for the model to load into memory
sleep 8

echo "📊 Starting Gradio Web Interface..."
python src/ui/app_gradio.py
