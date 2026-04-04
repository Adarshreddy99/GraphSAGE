import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add parent directory to path to resolve 'recommender'
sys.path.append(os.path.dirname(__file__))

try:
    from recommender import Recommender
except ImportError:
    from .recommender import Recommender
from loguru import logger
import time

app = FastAPI(
    title="GraphSAGE Paper Recommender",
    description="High-performance citation recommendation engine using GraphSAGE and HNSW FAISS.",
    version="1.0.0"
)

# Global Recommender Instance
rec = None

# 1. Pydantic Models for Input/Output
class RecRequest(BaseModel):
    query: str = Field(..., example="Deep learning for graph neural networks and node classification")
    k: int = Field(default=10, ge=1, le=50)
    lambda_param: float = Field(default=0.7, ge=0.0, le=1.0)

class PaperResponse(BaseModel):
    paper_id: int
    title: str
    year: int
    subject: str
    score: float

class HealthResponse(BaseModel):
    status: str
    model_version: str
    total_papers: int
    uptime: float

# 2. Lifecycle Events
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    global rec
    logger.info("Server starting: Loading models and building index...")
    rec = Recommender()
    logger.info("Serving engine is healthy and live.")

# 3. Endpoints
@app.get("/health", response_model=HealthResponse)
async def health():
    if rec is None:
        raise HTTPException(status_code=503, detail="Model is still loading...")
    
    return {
        "status": "healthy",
        "model_version": "1.0.0",
        "total_papers": len(rec.df),
        "uptime": time.time() - start_time
    }

@app.post("/recommend", response_model=List[PaperResponse])
async def recommend(request: RecRequest):
    if rec is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")
    
    try:
        results = rec.recommend(
            query_text=request.query,
            k=request.k,
            lambda_param=request.lambda_param
        )
        return results
    except Exception as e:
        logger.error(f"Recommendation Failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Inference error occurred.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
