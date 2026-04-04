#!/bin/bash

# 1. Start the FastAPI Brain in the background
echo "🚀 Starting FastAPI Recommender Engine..."
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 &

# 2. Wait a few seconds for the model to load
sleep 5

# 3. Start the Gradio Face in the foreground
echo "📊 Starting Gradio Web Interface..."
python src/ui/app_gradio.py
